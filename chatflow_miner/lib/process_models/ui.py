from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import pandas as pd
import streamlit as st

from ..event_log.view import EventLogView
from ..state.manager import set_selected_model
from .base import BaseProcessModel
from .dfg import DFGModel, PerformanceDFGModel
from .petri_net import PetriNetModel
from .view import ProcessModelView

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Naming utilities (pure)
# -----------------------------


def normalize_name(name: str) -> str:
    """Normaliza o nome para comparação: strip + casefold."""
    return name.strip().casefold()


def is_valid_name(name: str) -> bool:
    """Nome é válido se não for vazio após strip."""
    return bool(name and name.strip())


def name_is_unique(name: str, existing: Iterable[str]) -> bool:
    """Compara de forma case-insensitive e ignorando espaços nas extremidades."""
    normalized = normalize_name(name)
    existing_norm = {normalize_name(n) for n in existing}
    return normalized not in existing_norm


# -----------------------------
# Model generation and render
# -----------------------------

_MODEL_FACTORY: Mapping[str, type[BaseProcessModel]] = {
    "dfg": DFGModel,
    "performance-dfg": PerformanceDFGModel,
    "performance_dfg": PerformanceDFGModel,
    "petri-net": PetriNetModel,
}
_CANONICAL_KEYS = ("dfg", "performance-dfg", "petri-net")


def _resolve_model(model: BaseProcessModel | str | None) -> BaseProcessModel:
    if model is None:
        return DFGModel()
    if isinstance(model, BaseProcessModel):
        return model

    key = str(model).strip().casefold()
    try:
        model_cls = _MODEL_FACTORY[key]
    except KeyError as exc:  # pragma: no cover - defensive path
        available = "', '".join(_CANONICAL_KEYS)
        raise ValueError(
            "Tipo de modelo inválido. Forneça uma instância ou uma das chaves: "
            f"'{available}'."
        ) from exc
    return model_cls()


def generate_process_model(
    log_view: EventLogView,
    model: BaseProcessModel | str | None = None,
) -> ProcessModelView:
    """Gera um :class:`ProcessModelView` a partir de um log de eventos.

    ``model`` pode ser uma instância pronta de ``BaseProcessModel`` ou uma
    string correspondente às chaves do mapeamento local. O comportamento padrão
    permanece utilizando ``DFGModel`` quando nenhum valor é informado.
    """

    resolved = _resolve_model(model)
    return ProcessModelView(log_view=log_view, model=resolved)


def render_process_graph(view: ProcessModelView) -> None:
    """Renderiza o grafo do modelo a partir de uma ProcessModelView.

    Usa o método to_graphviz() da view para gerar a visualização.
    """
    gviz = view.to_graphviz(bgcolor="white", rankdir="LR")
    st.graphviz_chart(gviz, width="stretch")


# -----------------------------
# Persistência em session_state
# -----------------------------


def save_model(name: str, view: ProcessModelView) -> None:
    """Salva o modelo no registry, atualiza seleção e fecha o diálogo.

    Regras:
      - Não salva nomes vazios
      - Não sobrescreve nomes existentes (case-insensitive, strip)
    Em sucesso, seleciona o modelo e fecha o diálogo via st.rerun().
    """
    if not is_valid_name(name):
        raise ValueError("Insira um nome não vazio para salvar o modelo.")

    # Nomes existentes (exclui placeholder se presente)
    existing_names = list(st.session_state.process_models.names)
    placeholder = "Criar novo modelo de processo..."
    existing_names = [n for n in existing_names if n != placeholder]

    if not name_is_unique(name, existing_names):
        raise ValueError("Já existe um modelo com este nome. Escolha outro nome.")

    st.session_state.process_models.add(name, view)
    set_selected_model(name)

    # Fechar o diálogo
    st.rerun()


# -----------------------------
# UI: diálogo e áreas principais
# -----------------------------


@st.dialog("Modelo de processos gerado com sucesso", dismissible=True)
def show_generated_model_dialog() -> None:
    """Dialogo para exibir a visualização e permitir salvar com um nome."""
    model_view = st.session_state.get("latest_generated_model")
    if model_view is None:
        st.info("Nenhum modelo gerado nesta execução.")
        return

    render_process_graph(model_view)

    with st.form(key="dialog.form.save"):
        name = st.text_input("Nome do modelo", key="dialog.name")
        submitted = st.form_submit_button("Salvar", width="stretch")
        if submitted:
            try:
                with st.spinner("Salvando modelo..."):
                    save_model(name, model_view)
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(
                    "Ocorreu um erro ao salvar o modelo. Verifique o nome e tente novamente."
                )
                st.exception(exc)


@st.fragment
def render_saved_model_ui(selected_name: str) -> None:
    """Renderiza o modelo salvo na área principal, substituindo a UI de filtros."""
    from ..state.manager import get_process_model  # import local para evitar ciclos

    view = get_process_model(selected_name)
    if view is None:
        st.error("Modelo selecionado não encontrado.")
        return

    options = ["Horizontal", "Vertical"]
    rankdir = st.segmented_control(
        "Orientação do grafo",
        options,
        selection_mode="single",
        default=options[0],
    )

    try:
        match rankdir:
            case "Horizontal":
                gviz = view.to_graphviz(bgcolor="white", rankdir="LR")
                st.graphviz_chart(gviz, width="stretch")
            case "Vertical":
                gviz = view.to_graphviz(bgcolor="white", rankdir="TB")
                st.graphviz_chart(gviz, width="stretch")
            case _:
                st.error("Você deve selecionar uma orientação para o modelo.")
    except Exception as exc:
        st.error("Falha ao renderizar o modelo salvo.")
        st.exception(exc)

    st.divider()
    st.subheader("Métricas de qualidade do modelo")

    try:
        with st.spinner("Calculando métricas..."):
            metrics = view.quality_metrics()
    except NotImplementedError:
        st.info("Métricas de qualidade não disponíveis para este tipo de modelo.")
        return
    except Exception:
        st.warning("Não foi possível calcular as métricas de qualidade do modelo.")
        LOGGER.exception(
            "Falha ao calcular métricas de qualidade para o modelo '%s'", selected_name
        )
        return

    rows = []
    ordered_keys = [
        ("fitness", "Fitness"),
        ("precision", "Precisão"),
        ("generalization", "Generalização"),
        ("simplicity", "Simplicidade"),
    ]
    for key, label in ordered_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            display = f"{value:.3f}"
        else:
            display = "N/A"
        rows.append({"Métrica": label, "Valor": display})

    metrics_df = pd.DataFrame(rows).set_index("Métrica")
    with st.popover("Ajuda"):
        st.markdown("**Fitness:** Quão bem o modelo consegue reproduzir o comportamento registrado no log de eventos.\n\n**Precisão:** Quão pouco comportamento extra o modelo permite além do que de fato aparece no log de eventos.\n\n**Generalização:** Quão bem o modelo permite comportamentos plausíveis, porém não observados, sem apenas memorizar o log.\n\n**Simplicidade:** Quão compacto e fácil de entender é o modelo, mantendo a capacidade de explicar o log.")

    st.table(metrics_df)
