import streamlit as st

from chatflow_miner.lib.inputs import input_dataset
from chatflow_miner.lib.state import (initialize_session_state,
                                    open_input_dialog,
                                    close_input_dialog,
                                    get_log_eventos)
from chatflow_miner.lib.filters import AgentFilter, EventLogView
from chatflow_miner.lib.process_models import ProcessModelView, DFGModel

st.set_page_config(page_title="ChatFlow Miner", layout="wide")
initialize_session_state()

@st.fragment
def filter():
    st.write("Filtro de dados - Em construção")
    options = ["ai", "human", "ambos"]
    filter_selection = st.segmented_control("Filtro de AGENTE", options, selection_mode="single", default=options[2])
    event_log_view = EventLogView(base_df=get_log_eventos(which="log_eventos"))

    match filter_selection:
        case "ai":
            agent_filter = AgentFilter(agent="ai")
            event_log_view = event_log_view.filter(agent_filter)
        case "human":
            agent_filter = AgentFilter(agent="human")
            event_log_view = event_log_view.filter(agent_filter)
        case "ambos":
            pass
    st.dataframe(event_log_view.compute())

    if st.button("Gerar"):
        process_model_view = ProcessModelView(log_view=event_log_view, model=DFGModel())
        dfg = process_model_view.to_graphviz()
        st.graphviz_chart(dfg)

col1, col2 = st.columns(2)
# TODO: abaixo das duas colunas, mostrar seletor de modelos de processos com o primeiro item (padrão) selecionado sendo o placeholder para criar novo modelo.
model_selector_placeholder = "Criar novo modelo de processo..."
model_selector = st.selectbox(label="Modelos de processo", options=[model_selector_placeholder], width=400)

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
    filter()