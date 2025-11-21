import streamlit as st

from chatflow_miner.lib.ui.dashboard import (
    model_discovery,
    render_agent_analysis,
    render_conformance_analysis,
    render_exploratory_analysis,
)
from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (
    get_log_eventos,
    initialize_session_state,
    open_input_dialog,
    reset_log_eventos,
)

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()

st.title("ChatFlow Miner")

col1, col2 = st.columns(2)

with col1:
    # Pré-computar na linha abaixo para não precisar chamar get_log_eventos() duas vezes
    disabled = (
        load_info := get_log_eventos(which="load_info")
    ) is not None  # Desabilitar se já houver um arquivo carregado
    col1.button(
        "Carregar log de eventos",
        on_click=open_input_dialog,
        type="primary",
        disabled=disabled,
    )
    if st.session_state.input_dialog:
        input_dataset()

with col2:
    # load_info já foi pré-computado acima
    if load_info is not None:
        txt_col, btn_col = st.columns([6, 1])
        with txt_col:
            st.text(f"Usando o arquivo: {load_info['file_name']}")
        with btn_col:
            if st.button("Remover", type="tertiary", key="remove-log"):
                reset_log_eventos()
                st.rerun()
    else:
        st.text("Nenhum arquivo carregado.")

tab_discover, tab_anexp, tab_conformance, tab_anagent = st.tabs([
    "Descoberta de modelo de processo",
    "Análise exploratória",
    "Conformidade",
    "Análise de agentes",
])
with tab_discover:
    model_discovery(disabled=disabled)
with tab_anexp:
    render_exploratory_analysis()
with tab_conformance:
    render_conformance_analysis()
with tab_anagent:
    render_agent_analysis()
