import streamlit as st

from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (initialize_session_state,
                                    open_input_dialog,
                                    get_log_eventos)
from chatflow_miner.lib.filters.streamlit_fragments import filter_section

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()


col1, col2 = st.columns(2)
# Seletor de modelos de processos com o primeiro item (padrão) selecionado sendo o placeholder para criar novo modelo.
model_names = list(st.session_state.process_models.names)
model_selector = st.selectbox(label="Modelos de processo", options=model_names, width=400)

with col1:
    col1.button("Carregar", on_click=open_input_dialog)
    if st.session_state.input_dialog:
        input_dataset()

with col2:
    if (load_info := get_log_eventos(which="load_info")) is not None:
        st.text(f"Usando o arquivo: {load_info['file_name']}")
    else:
        st.text("Nenhum arquivo carregado.")

if get_log_eventos() is not None:
    filter_section()