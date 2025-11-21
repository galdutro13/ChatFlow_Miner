from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import Any, Iterable

from collections.abc import Mapping

import streamlit as st

from chatflow_miner.lib.conformance.utils import ensure_marking_obj

def _init_sintetizar_modelo_state():
    """Função para iniciar o estado do botão de sintetizar modelo"""
    if "sintetizar_modelo" not in st.session_state:
        st.session_state.sintetizar_modelo = False

def _toggle_sistetizar_modelo():
    """Muda o estado de sintetizar modelo para True"""
    st.session_state.sintetizar_modelo = True

def _init_normative_model_state() -> None:
    if "normative_model" not in st.session_state:
        st.session_state.normative_model = {
            "net": None,
            "initial_marking": None,
            "final_marking": None,
            "source": None,
        }


def _import_optional(module_name: str) -> Any | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


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


def _discover_from_log(log_df: Any, *, noise_threshold: float) -> None:
    if log_df is None or len(log_df) == 0:
        st.warning("Nenhum log disponível para mineração")
        return

    pm4py = _import_optional("pm4py")
    if pm4py is None:
        st.error("pm4py é necessário para minerar modelos. Instale a dependência para continuar.")
        return

    with st.spinner("Executando Inductive Miner..."):
        net, im, fm = pm4py.discover_petri_net_inductive(log_df, noise_threshold=noise_threshold)
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


def _discover_from_variants(
    log_df: Any, selected_variants: list[str], manual_traces: list[str]
) -> None:
    pm4py = _import_optional("pm4py")
    variants_get = _import_optional("pm4py.statistics.variants.log.get")
    if pm4py is None or variants_get is None:
        st.error("pm4py é necessário para minerar variantes. Instale a dependência para continuar.")
        return

    if log_df is None or len(log_df) == 0:
        st.warning("Nenhum log base disponível para minerar variantes")
        return

    event_log = pm4py.convert_to_event_log(log_df)
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


def render_normative_model_selector(log_df: Any | None = None) -> None:
    _init_normative_model_state()
    st.markdown("### Modelo normativo")
    tabs = st.tabs(["Por arquivo", "Por descoberta (Log Completo)", "Por variante"])

    with tabs[0]:
        uploaded = st.file_uploader(
            "Carregar modelo (pnml, bpmn, ptml)",
            type=["pnml", "bpmn", "ptml"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            _load_model_from_upload(uploaded)

    with tabs[1]:
        with st.form("normative_discovery_form"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.caption("Selecione o log para mineração")
                st.selectbox(
                    "Log disponível",
                    options=["Log carregado"] if log_df is not None else ["Nenhum log"],
                    disabled=log_df is None,
                    key="normative_discovery_log_selector",
                )
            with c2:
                noise = st.slider("noise_threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            submitted = st.form_submit_button("Minerar", use_container_width=True)
            if submitted:
                _discover_from_log(log_df, noise_threshold=noise)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Selecione variantes do log base")
            variants_options: list[str] = []
            if log_df is not None:
                pm4py = _import_optional("pm4py")
                variants_get = _import_optional("pm4py.statistics.variants.log.get")
                if pm4py is None or variants_get is None:
                    st.warning("pm4py é necessário para extrair variantes")
                else:
                    try:
                        event_log = pm4py.convert_to_event_log(log_df)
                        variants_map = variants_get.get_variants(event_log)
                        variants_options = [" -> ".join(map(str, v)) for v in variants_map.keys()]
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Não foi possível extrair variantes: {exc}")
            selected = st.multiselect("Variantes", options=variants_options)
        with c2:
            st.caption("Ou insira traços manualmente (um por linha)")
            manual_input = st.text_area(
                "Traços", placeholder="A -> B -> C\nA -> C -> D", height=120
            )
        st.button("Sintetizar Modelo", use_container_width=True, on_click=_toggle_sistetizar_modelo)
        sintetizar_modelo = st.session_state.get("sintetizar_modelo", False)
        if sintetizar_modelo:
            manual_lines = [line for line in manual_input.splitlines() if line.strip()]
            _discover_from_variants(log_df, selected_variants=selected, manual_traces=manual_lines)

    if st.session_state.normative_model["net"] is not None:
        st.success("Modelo normativo pronto para análise")
