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


def filter_by_variants(event_view: EventLogView,  disabled: bool):
    base_df = get_log_eventos(which="log_eventos")
    if base_df is None:
        st.multiselect(
            "Filtro de VARIANTE",
            options=[],
            default=[],
            disabled=disabled
        )
        return None

    var_agg = CaseAggView(base_df=base_df).with_aggregator(CaseVariantAggregator())
    result = var_agg.compute()
    seen = set()
    variants = [v for v in result if (vid := getattr(result[v], "variant_id", v)) not in seen and not seen.add(vid)]

    # Reordenar a lista com base em frequência decrescente
    variants.sort(key=lambda v: result[v].frequency, reverse=True)

    filter_selection = st.multiselect(
        "Filtro de VARIANTE",
        options=variants,
        default=[],
        format_func=lambda variant: f"freq={result[variant].frequency} | {result[variant].variant_id} | {result[variant].variant}",
        disabled=disabled,
    )

    if filter_selection:
        # coletar os variant_id das variantes selecionadas
        selected_variant_ids = {getattr(result[v], "variant_id", v) for v in filter_selection}
        # obter os case_ids cujas variantes têm variant_id selecionado
        case_ids = [
            case_id
            for case_id in result
            if getattr(result[case_id], "variant_id", case_id) in selected_variant_ids
        ]
        if case_ids:
            case_filter = CaseFilter(case_ids=case_ids)
            return event_view.filter(case_filter)

    return event_view



def filter_by_agents(disabled: bool) -> EventLogView:
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

