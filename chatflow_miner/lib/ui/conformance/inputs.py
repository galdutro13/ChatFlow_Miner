from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any, Iterable

from collections.abc import Mapping

import streamlit as st

from chatflow_miner.lib.conformance.utils import ensure_marking_obj
from chatflow_miner.lib.utils import load_dataset

def _init_synthesize_clicked_state():
    if 'synthesize_clicked' not in st.session_state:
        st.session_state.synthesize_clicked = False

def _toggle_synthesize_clicked():
    if not st.session_state.synthesize_clicked:
        st.session_state.synthesize_clicked = True

def _init_minerar_clicked_state():
    if 'minerar_clicked' not in st.session_state:
        st.session_state.minerar_clicked = False

def _toggle_minerar_clicked():
    if not st.session_state.minerar_clicked:
        st.session_state.minerar_clicked = True


def _init_normative_model_state() -> None:
    if "normative_model" not in st.session_state:
        st.session_state.normative_model = {
            "net": None,
            "initial_marking": None,
            "final_marking": None,
            "source": None,
        }

def _init_reference_log_state():
    if "reference_log_state" not in st.session_state:
        st.session_state.reference_log_state = {"log": None, "name": None, "info": None}

def _reset_normative_model_state() -> None:
    _init_normative_model_state()
    st.session_state.normative_model.update(
        {"net": None, "initial_marking": None, "final_marking": None, "source": None}
    )


def _import_optional(module_name: str) -> Any | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


def _get_reference_log_state() -> dict[str, Any]:
    """Return the canonical reference-log state stored in session_state.

    Legacy top-level keys (reference_log, reference_log_name, reference_log_info)
    previously mirrored the same information; they are migrated once into the
    structured ``reference_log_state`` mapping to avoid dual sources of truth
    across reruns.
    """

    _init_reference_log_state()

    state: dict[str, Any] = st.session_state.reference_log_state

    legacy_keys = (
        ("reference_log", "log"),
        ("reference_log_name", "name"),
        ("reference_log_info", "info"),
    )
    for legacy_key, state_key in legacy_keys:
        if legacy_key in st.session_state and state.get(state_key) is None:
            state[state_key] = st.session_state.get(legacy_key)
        st.session_state.pop(legacy_key, None)

    # Reassign to ensure Streamlit tracks in-place updates for reruns
    st.session_state.reference_log_state = state

    return st.session_state.reference_log_state


def _clear_reference_log() -> None:
    state = _get_reference_log_state()
    state.update({"log": None, "name": None, "info": None})
    _reset_normative_model_state()


def _persist_reference_log(log: Any, load_info: Mapping[str, Any]) -> None:
    state = _get_reference_log_state()
    state.update({"log": log, "name": load_info.get("file_name"), "info": load_info})
    _reset_normative_model_state()


def _persist_normative_model(net: Any, im: Any, fm: Any, *, source: str) -> None:
    _init_normative_model_state()
    st.session_state.normative_model.update(
        {
            "net": net,
            "initial_marking": ensure_marking_obj(net, im),
            "final_marking": ensure_marking_obj(net, fm),
            "source": source,
        }
    )


def _render_petri_preview(net: Any, im: Any, fm: Any) -> None:
    pn_visualizer = _import_optional("pm4py.visualization.petri_net.visualizer")
    if pn_visualizer is None:
        st.warning("Pré-visualização indisponível: pm4py não encontrado")
        return

    with st.expander("Visualizar modelo gerado", expanded=False):
        gviz = pn_visualizer.apply(net, im, fm, variant=pn_visualizer.Variants.WO_DECORATION)
        st.graphviz_chart(gviz.source)


def _load_model_from_upload(uploaded: Any) -> None:
    pm4py = _import_optional("pm4py")
    bpmn_importer = _import_optional("pm4py.objects.bpmn.importer")
    pnml_importer = _import_optional("pm4py.objects.petri.importer")
    ptml_importer = _import_optional("pm4py.objects.process_tree.importer")

    if pm4py is None or bpmn_importer is None or pnml_importer is None or ptml_importer is None:
        st.error("pm4py é necessário para carregar modelos. Certifique-se de que a dependência está instalada.")
        return

    ext = Path(uploaded.name).suffix.lower()
    file_bytes = uploaded.read()

    with st.status("Processando arquivo enviado", expanded=False) as status:
        buffer = io.BytesIO(file_bytes)
        try:
            if ext == ".pnml":
                status.update(label="Importando PNML...")
                net, im, fm = pnml_importer.apply(buffer)
            elif ext == ".bpmn":
                status.update(label="Importando BPMN...")
                bpmn_graph = bpmn_importer.apply(buffer)
                status.update(label="Convertendo BPMN -> Rede de Petri")
                net, im, fm = pm4py.convert_to_petri_net(bpmn_graph)
            elif ext == ".ptml":
                status.update(label="Importando Process Tree...")
                process_tree = ptml_importer.apply(buffer)
                status.update(label="Convertendo árvore -> Rede de Petri")
                net, im, fm = pm4py.convert_to_petri_net(process_tree)
            else:
                raise ValueError("Formato de arquivo não suportado. Use pnml, bpmn ou ptml.")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Falha ao carregar modelo", state="error")
            st.error(f"Erro ao processar arquivo: {exc}")
            return

        status.update(label="Modelo carregado", state="complete")
        _persist_normative_model(net, im, fm, source="upload")
        _render_petri_preview(net, im, fm)


def _render_reference_log_uploader(*, key_suffix: str = "default") -> None:
    st.info(
        "Envie um log de referência (CSV ou XES) para minerar o modelo normativo.",
        icon="📁",
    )
    sep_key = f"reference_log_sep_{key_suffix}"
    sep = st.selectbox(
        "Separador do CSV",
        options=[",", ";", "\t"],
        index=0,
        key=sep_key,
    )
    uploader_key = f"reference_log_uploader_{key_suffix}"
    uploaded = st.file_uploader(
        "Carregar log de referência (CSV/XES)",
        type=["csv", "xes"],
        accept_multiple_files=False,
        key=uploader_key,
    )
    if uploaded is not None:
        load_info = {"sep": sep, "file_name": uploaded.name}
        log = load_dataset(uploaded, load_info)
        _persist_reference_log(log, load_info)
        st.rerun()


def _discover_from_log(*, noise_threshold: float) -> None:
    reference_log = _get_reference_log_state().get("log")
    if reference_log is None or len(reference_log) == 0:
        st.warning("Nenhum log de referência disponível para mineração")
        return

    pm4py = _import_optional("pm4py")
    if pm4py is None:
        st.error("pm4py é necessário para minerar modelos. Instale a dependência para continuar.")
        return

    with st.spinner("Executando Inductive Miner..."):
        net, im, fm = pm4py.discover_petri_net_inductive(
            reference_log, noise_threshold=noise_threshold
        )
    _persist_normative_model(net, im, fm, source="log_full")
    _render_petri_preview(net, im, fm)


def _normalize_variant_activities(variant: Iterable[Any]) -> list[str]:
    """Ensure all activities are converted to string labels.

    Variants coming from pm4py statistics return ``Trace`` objects with
    ``Event`` entries; manual variants arrive as lists of strings. This helper
    extracts ``concept:name`` when available and falls back to ``str`` to avoid
    mixing objects inside the synthetic log.
    """

    normalized: list[str] = []
    for activity in variant:
        name: Any | None = None
        if isinstance(activity, Mapping):
            name = activity.get("concept:name")
        elif hasattr(activity, "get"):
            try:
                name = activity.get("concept:name")  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                name = None

        normalized.append(str(name) if name is not None else str(activity))

    return normalized


def _build_synthetic_log_from_variants(variants: Iterable[Iterable[Any]]) -> Any:
    from pm4py.objects.log.obj import Event, EventLog, Trace

    log = EventLog()
    for variant in variants:
        activities = _normalize_variant_activities(variant)
        trace = Trace([Event({"concept:name": act}) for act in activities])
        log.append(trace)
    return log


def _discover_from_variants(selected_variants: list[str], manual_traces: list[str]) -> None:
    pm4py = _import_optional("pm4py")
    variants_get = _import_optional("pm4py.statistics.variants.log.get")
    if pm4py is None or variants_get is None:
        st.error("pm4py é necessário para minerar variantes. Instale a dependência para continuar.")
        return

    reference_log = _get_reference_log_state().get("log")
    if reference_log is None or len(reference_log) == 0:
        st.warning("Nenhum log de referência disponível para minerar variantes")
        return

    event_log = pm4py.convert_to_event_log(reference_log)
    variants_map = variants_get.get_variants(event_log)

    selected_variant_traces: list[list[str]] = []
    for variant_label in selected_variants:
        variant_traces = variants_map.get(tuple(variant_label.split(" -> ")))
        if variant_traces:
            selected_variant_traces.append(
                [
                    str(event.get("concept:name"))
                    for event in variant_traces[0]
                    if "concept:name" in event
                ]
            )

    manual_variant_traces: list[list[str]] = []
    for line in manual_traces:
        parts = [p.strip() for p in line.split("->") if p.strip()]
        if parts:
            manual_variant_traces.append(parts)

    all_variants = selected_variant_traces + manual_variant_traces
    if not all_variants:
        st.warning("Selecione ao menos uma variante ou informe traços manualmente")
        return

    synthetic_log = _build_synthetic_log_from_variants(all_variants)
    with st.spinner("Minerando log sintético..."):
        net, im, fm = pm4py.discover_petri_net_inductive(synthetic_log)
    _persist_normative_model(net, im, fm, source="variants")
    _render_petri_preview(net, im, fm)


def _reference_log_loaded() -> bool:
    return _get_reference_log_state().get("log") is not None


@st.fragment
def _render_model_upload_tab() -> None:
    uploaded = st.file_uploader(
        "Carregar modelo (pnml, bpmn, ptml)",
        type=["pnml", "bpmn", "ptml"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        _load_model_from_upload(uploaded)


@st.fragment
def _render_discovery_tab() -> None:
    _init_minerar_clicked_state()
    if not _reference_log_loaded():
        _render_reference_log_uploader(key_suffix="discovery")
    if _reference_log_loaded():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("O modelo será minerado a partir do log de referência carregado.")
        with c2:
            noise = st.slider(
                "noise_threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                disabled=not _reference_log_loaded(),
            )
        if st.button(
            "Minerar", use_container_width=True, disabled=not _reference_log_loaded()
        ):
            _toggle_minerar_clicked()
            # Precisamos adicionar esse rerun devido ao uso de st.fragment em _render_variant_tab.
            # Já que a função que nos chama é um fragment, os elementos dela são atualizados independentemente
            # do resto do app. Ou seja, mesmo a função mudando o estado global do app, como ela é executada
            # independentemente, os outros objetos não "percebem" a mudança do estado.
            st.rerun(scope="app")
        if st.session_state.minerar_clicked and _reference_log_loaded():
            _discover_from_log(noise_threshold=noise)


@st.fragment
def _render_variant_tab() -> None:
    _init_synthesize_clicked_state()
    if not _reference_log_loaded():
        _render_reference_log_uploader(key_suffix="variant")
        selected: list[str] = []
        manual_lines: list[str] = []
    else:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Selecione variantes do log de referência")
                variants_options: list[str] = []
                pm4py = _import_optional("pm4py")
                variants_get = _import_optional("pm4py.statistics.variants.log.get")
                if pm4py is None or variants_get is None:
                    st.warning("pm4py é necessário para extrair variantes")
                else:
                    try:
                        event_log = pm4py.convert_to_event_log(_get_reference_log_state().get("log"))
                        variants_map = variants_get.get_variants(event_log)
                        variants_options = [" -> ".join(map(str, v)) for v in variants_map.keys()]
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Não foi possível extrair variantes: {exc}")
                selected = st.multiselect(
                    "Variantes", options=variants_options, disabled=not _reference_log_loaded()
                )
            with c2:
                st.caption("Ou insira traços manualmente (um por linha)")
                manual_input = st.text_area(
                    "Traços",
                    placeholder="A -> B -> C\nA -> C -> D",
                    height=120,
                    disabled=not _reference_log_loaded(),
                )
                manual_lines = [line for line in manual_input.splitlines() if line.strip()]
    if st.button(
        "Sintetizar Modelo",
        use_container_width=True,
        disabled=not _reference_log_loaded()
    ):
        _toggle_synthesize_clicked()
        # Precisamos adicionar esse rerun devido ao uso de st.fragment em _render_variant_tab.
        # Já que a função que nos chama é um fragment, os elementos dela são atualizados independentemente
        # do resto do app. Ou seja, mesmo a função mudando o estado global do app, como ela é executada
        # independentemente, os outros objetos não "percebem" a mudança do estado.
        st.rerun(scope="app")

    if st.session_state.synthesize_clicked and _reference_log_loaded():
        _discover_from_variants(selected_variants=selected, manual_traces=manual_lines)


@st.fragment
def render_button_fragment() -> None:
    reference_state = _get_reference_log_state()
    reference_name = reference_state.get("name")

    if _reference_log_loaded() and reference_name:
        info_col, action_col = st.columns([6, 1])
        with info_col:
            st.caption(f"Usando log de referência: {reference_name}")
        with action_col:
            if st.button("Remover", type="tertiary", key="remove-reference-log"):
                _clear_reference_log()
                st.rerun()

def render_normative_model_selector(_execution_log: Any | None = None) -> None:
    _init_normative_model_state()
    render_button_fragment()


    st.markdown("### Modelo normativo")


    tabs = st.tabs(["Por arquivo", "Por descoberta (Log Completo)", "Por variante"])

    with tabs[0]:
        _render_model_upload_tab()

    with tabs[1]:
        _render_discovery_tab()

    with tabs[2]:
        _render_variant_tab()

    if st.session_state.normative_model["net"] is not None:
        st.success("Modelo normativo pronto para análise")
