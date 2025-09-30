import streamlit as st

from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (
    initialize_session_state,
    open_input_dialog,
    get_log_eventos,
    get_selected_model,
    reset_log_eventos,
)
from chatflow_miner.lib.filters.streamlit_fragments import filter_section
from chatflow_miner.lib.process_models.ui import render_saved_model_ui

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()

st.sidebar.title("ChatFlow Miner")
st.sidebar.write("Descoberta de modelo de processo")
st.sidebar.write("Análise exploratória")
st.sidebar.write("Análise de agentes")

col1, col2 = st.columns(2)

with col1:
    # Pré-computar na linha abaixo para não precisar chamar get_log_eventos() duas vezes
    disabled = (load_info := get_log_eventos(which="load_info")) is not None # Desabilitar se já houver um arquivo carregado
    col1.button("Carregar log de eventos", on_click=open_input_dialog, type="primary", disabled=disabled)
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

model_names = list(st.session_state.process_models.names)

selected_index = 0
current_selected = get_selected_model() # Computamos o nome do modelo selecionado ou None aqui
if current_selected is not None:
    try:
        selected_index = model_names.index(current_selected)
    except ValueError:
        selected_index = 0

st.selectbox(
    label="Modelos de processo",
    options=model_names,
    index=selected_index,
    key="selected_model", # Referenciando o estado da sessão st.session_state.selected_model
    placeholder=None,
)

if current_selected is None:
    # Exibe interface de filtros; desabilita quando não há log carregado
    filter_section(disabled=not disabled)
else:
    # Exibe modelo salvo selecionado independentemente de haver log carregado
    render_saved_model_ui(current_selected)
        