import streamlit as st

from chatflow_miner.lib.state import get_log_eventos
from chatflow_miner.lib.filters import AgentFilter, EventLogView
from chatflow_miner.lib.process_models import ProcessModelView, DFGModel


@st.fragment
def filter_section():
    """Fragmento reutilizável para seção de filtros em Streamlit."""
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

