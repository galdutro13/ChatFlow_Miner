from __future__ import annotations

from typing import Iterable
import streamlit as st

from .dfg import DFGModel
from .view import ProcessModelView
from ..event_log.view import EventLogView
from ..state.manager import set_selected_model


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

def generate_process_model(log_view: EventLogView) -> ProcessModelView:
    """Gera um ProcessModelView a partir de um EventLogView.

    Retorna uma ProcessModelView que combina o log_view com um DFGModel.
    O modelo será computado lazy quando necessário.
    """
    model = DFGModel()
    return ProcessModelView(log_view=log_view, model=model)


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
                st.error("Ocorreu um erro ao salvar o modelo. Verifique o nome e tente novamente.")
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
        # gviz = view.to_graphviz(bgcolor="white", rankdir="LR")
        # st.graphviz_chart(gviz, width="stretch")
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


