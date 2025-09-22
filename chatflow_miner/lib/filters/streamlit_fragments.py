import streamlit as st
import pandas as pd

from chatflow_miner.lib.state import get_log_eventos
from chatflow_miner.lib.filters.builtins import AgentFilter
from chatflow_miner.lib.filters.view import EventLogView
from chatflow_miner.lib.process_models.ui import (
    generate_process_model,
    show_generated_model_dialog,
)


@st.fragment
def filter_section(*, disabled: bool = False):
    """Fragmento reutilizável para seção de filtros em Streamlit."""
    st.write("Filtro de dados - Em construção")
    options = ["chatbot", "cliente", "ambos"]
    filter_selection = st.segmented_control(
        "Filtro de AGENTE",
        options,
        selection_mode="single",
        default=options[2],
        disabled=disabled,
    )

    base_df = get_log_eventos(which="log_eventos")
    if base_df is None:
        base_df = pd.DataFrame()
    event_log_view = EventLogView(base_df=base_df)

    match filter_selection:
        case "chatbot":
            agent_filter = AgentFilter(agent="ai")
            event_log_view = event_log_view.filter(agent_filter)
        case "cliente":
            agent_filter = AgentFilter(agent="human")
            event_log_view = event_log_view.filter(agent_filter)
        case "ambos":
            pass
    st.dataframe(event_log_view.compute())

    # Área inferior com botão à direita
    _, right_col = st.columns([6, 1])
    with right_col:
        if st.button("Gerar", key="filters.generate", disabled=disabled):
            try:
                with st.spinner("Gerando modelo..."):
                    model_data = generate_process_model(event_log_view)
                    st.session_state.latest_generated_model = model_data
                # Abre diálogo para visualização e salvamento
                show_generated_model_dialog()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("Falha ao gerar o modelo de processo.")
                st.exception(exc)

