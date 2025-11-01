import streamlit as st

from chatflow_miner.lib.state import get_selected_model
from chatflow_miner.lib.ui.filters.streamlit_fragments import filter_section
from chatflow_miner.lib.ui.process_models import render_saved_model_ui


def model_discovery(disabled: bool):
    model_names = list(st.session_state.process_models.names)

    selected_index = 0
    current_selected = (
        get_selected_model()
    )  # Computamos o nome do modelo selecionado ou None aqui
    if current_selected is not None:
        try:
            selected_index = model_names.index(current_selected)
        except ValueError:
            selected_index = 0

    st.selectbox(
        label="**Modelos de processo**",
        options=model_names,
        index=selected_index,
        key="selected_model",  # Referenciando o estado da sessão st.session_state.selected_model
        placeholder=None,
        help="Modelos de processos gerados podem ser encontrados aqui."
    )

    st.divider(width="stretch")

    if current_selected is None:
        # Exibe interface de filtros; desabilita quando não há log carregado
        filter_section(disabled=not disabled)
    else:
        # Exibe modelo salvo selecionado independentemente de haver log carregado
        render_saved_model_ui(current_selected)
