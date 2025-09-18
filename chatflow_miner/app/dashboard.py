import streamlit as st

from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (
    initialize_session_state,
    open_input_dialog,
    get_log_eventos,
)
from chatflow_miner.lib.filters.streamlit_fragments import filter_section
from chatflow_miner.lib.process_models.ui import render_saved_model_ui

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()


col1, col2 = st.columns(2)

# Seletor de modelos de processos: placeholder seguido dos nomes salvos
placeholder = "Criar novo modelo de processo..."
model_names = list(st.session_state.process_models.names)
if not model_names or model_names[0] != placeholder:
    # Garantir placeholder na primeira posição quando vazio
    if len(model_names) == 0:
        st.session_state.process_models.add(placeholder, None)
        model_names = [placeholder]

selected_index = 0
if st.session_state.selected_model is not None:
    try:
        selected_index = model_names.index(st.session_state.selected_model)
    except ValueError:
        selected_index = 0

selected_name = st.selectbox(
    label="Modelos de processo",
    options=model_names,
    index=selected_index,
    key="selector.selected",
    placeholder=None,
)

# Sincroniza seleção no estado
if selected_name == placeholder:
    st.session_state.selected_model = None
else:
    st.session_state.selected_model = selected_name

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
    if st.session_state.selected_model is None:
        # Exibe interface de filtros para criar novo
        filter_section()
    else:
        # Exibe modelo salvo selecionado
        render_saved_model_ui(st.session_state.selected_model)
        