import streamlit as st

from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (initialize_session_state,
                                    open_input_dialog,
                                    close_input_dialog,
                                    get_log_eventos)

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()

col1, col2 = st.columns(2)
# TODO: seletor de modelos de processos
# model_picker = st.selectbox()

with col1:
    col1.button("Carregar", on_click=open_input_dialog)
    if st.session_state.input_dialog:
        input_dataset()

with col2:
    if (load_info := get_log_eventos(which="load_info")) is not None:
        st.text(f"Usando o arquivo: {load_info['file_name']}")
    else:
        st.text("Nenhum arquivo carregado.")

