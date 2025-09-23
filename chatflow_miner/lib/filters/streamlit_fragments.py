import streamlit as st
import pandas as pd

from chatflow_miner.lib.state import get_log_eventos
from chatflow_miner.lib.filters.builtins import AgentFilter, CaseFilter
from chatflow_miner.lib.event_log.view import EventLogView
from chatflow_miner.lib.process_models.ui import (
    generate_process_model,
    show_generated_model_dialog,
)
from chatflow_miner.lib.aggregations import CaseVariantAggregator, CaseAggView

@st.fragment
def filter_section(*, disabled: bool = False):
    """Fragmento reutilizável para seção de filtros em Streamlit."""
    st.write("Filtro de dados - Em construção")
    event_log_view = filter_by_agents(disabled)
    event_log_view = filter_by_variants(event_log_view, disabled) or event_log_view

    st.dataframe(event_log_view.compute())

    # Área inferior com botão à direita
    _, right_col = st.columns([6, 1])
    with right_col:
        if st.button("Gerar", key="filters.generate", disabled=disabled):
            try:
                with st.spinner("Gerando modelo..."):
                    view = generate_process_model(event_log_view)
                    st.session_state.latest_generated_model = view
                # Abre diálogo para visualização e salvamento
                show_generated_model_dialog()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("Falha ao gerar o modelo de processo.")
                st.exception(exc)


def filter_by_variants(event_view: EventLogView, disabled: bool):
    """
    Exibe e aplica um filtro de variantes de casos sobre um EventLogView usando Streamlit.

    Parâmetros:
        event_view (EventLogView): A visão do log de eventos a ser filtrada.
        disabled (bool): Se True, desabilita a interação do filtro na interface.

    Retorna:
        EventLogView ou None: Um novo EventLogView filtrado pelas variantes selecionadas, ou None se não houver dados.
    """
    df = get_log_eventos(which="log_eventos")
    if df is None:
        st.multiselect("Filtro de VARIANTE", [], [], disabled=disabled); return None
    result = CaseAggView(base_df=df).with_aggregator(CaseVariantAggregator()).compute()
    gid = lambda k: getattr(result[k], "variant_id", k)
    seen = set()
    variants = [v for v in result if (vid := gid(v)) not in seen and not seen.add(vid)]
    variants.sort(key=lambda v: result[v].frequency, reverse=True)
    fmt = lambda v: f"freq={result[v].frequency} | {result[v].variant_id} | " \
                    f"{result[v].variant}"
    selected = st.multiselect("Filtro de VARIANTE", variants, [], format_func=fmt,
                              disabled=disabled)
    if selected:
        cid = {gid(v) for v in selected}; ks = [k for k in result if gid(k) in cid]
        if ks: return event_view.filter(CaseFilter(case_ids=ks))
    return event_view



def filter_by_agents(disabled: bool) -> EventLogView:
    """
    Exibe e aplica um filtro de agente sobre o EventLogView usando Streamlit.

    Parâmetros:
        disabled (bool): Se True, desabilita a interação do filtro na interface.

    Retorna:
        EventLogView: Um EventLogView filtrado pelo agente selecionado ('chatbot', 'cliente' ou 'ambos').
    """
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
    return event_log_view
