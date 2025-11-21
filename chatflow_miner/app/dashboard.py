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


def maybe_show_discovery_toast() -> str | None:
    log_counter = st.session_state.get("log_load_counter", 0)
    last_toast_counter = st.session_state.get("last_toast_log_counter", 0)
    initial_shown = st.session_state.get("initial_discovery_toast_shown", False)

    if not initial_shown:
        if log_counter == 0:
            st.toast(
                "Envie um novo log de eventos para gerar ou visualizar modelos.",
                icon="ℹ️",
            )
            st.session_state.initial_discovery_toast_shown = True
            st.session_state.last_toast_log_counter = log_counter
            return "initial"

        st.session_state.initial_discovery_toast_shown = True
        last_toast_counter = -1

    if log_counter > last_toast_counter:
        st.toast("Pronto para gerar ou visualizar modelos", icon="✅")
        st.session_state.last_toast_log_counter = log_counter
        return "ready"

    return None


col1, col2 = st.columns(2)

with col1:
    # Pré-computar na linha abaixo para não precisar chamar get_log_eventos() duas vezes
    disabled = (
        load_info := get_log_eventos(which="load_info")
    ) is not None  # Desabilitar se já houver um arquivo carregado
    st.button(
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

tab_discover, tab_anexp, tab_anagent, tab_conformance = st.tabs([
    "Descoberta de modelo de processo",
    "Análise exploratória",
    "Análise de agentes",
    "Análise de Conformidade"
])
with tab_discover:
    maybe_show_discovery_toast()
    model_discovery(disabled=disabled)
with tab_anexp:
    render_exploratory_analysis()
with tab_anagent:
    render_agent_analysis()
with tab_conformance:
    render_conformance_analysis()
