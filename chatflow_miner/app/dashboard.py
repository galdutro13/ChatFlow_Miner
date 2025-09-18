import streamlit as st

from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (
    initialize_session_state,
    open_input_dialog,
    get_log_eventos,
    get_selected_model,
    set_selected_model,
)
from chatflow_miner.lib.filters.streamlit_fragments import filter_section
from chatflow_miner.lib.process_models.ui import render_saved_model_ui

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()


col1, col2 = st.columns(2)

# Seletor de modelos de processos: placeholder seguido dos nomes salvos
placeholder = "Criar novo modelo de processo..."
model_names = list(st.session_state.process_models.names)

selected_index = 0
current_selected = get_selected_model()
if current_selected is not None:
    try:
        selected_index = model_names.index(current_selected)
    except ValueError:
        selected_index = 0

selected_name = st.selectbox(
    label="Modelos de processo",
    options=model_names,
    index=selected_index,
    key="selector.selected",
    placeholder=None,
)

with col1:
    col1.button("Carregar", on_click=open_input_dialog)
    if st.session_state.input_dialog:
        input_dataset()

with col2:
    if (load_info := get_log_eventos(which="load_info")) is not None:
        st.text(f"Usando o arquivo: {load_info['file_name']}")
    else:
        st.text("Nenhum arquivo carregado.")


# Sincroniza seleção no estado
if selected_name == placeholder:
    set_selected_model(None)
else:
    set_selected_model(selected_name)

if get_log_eventos() is not None:
    if get_selected_model() is None:
        # Exibe interface de filtros para criar novo
        filter_section()
    else:
        # Exibe modelo salvo selecionado
        render_saved_model_ui(get_selected_model())
        