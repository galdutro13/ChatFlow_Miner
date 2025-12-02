from typing import Any

import pandas as pd
import streamlit as st

from chatflow_miner.lib.aggregations import (
    CaseAggView,
    CaseDateAggregator,
    CaseVariantAggregator,
)
from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_START_TS,
)
from chatflow_miner.lib.event_log.view import EventLogView
from chatflow_miner.lib.filters.base.exceptions import FilterError
from chatflow_miner.lib.filters.builtins import (
    AgentFilter,
    CaseFilter,
    DirectlyFollowsFilter,
    EventuallyFollowsFilter,
    TimeWindowFilter,
)
from chatflow_miner.lib.ui.process_models import (
    generate_process_model,
    show_generated_model_dialog,
)
from chatflow_miner.lib.state import get_log_eventos


def filter_section(*, disabled: bool = False, preview_area=None):
    """Fragmento reutilizável para seção de filtros em Streamlit."""
    if "selected_variants" not in st.session_state:
        st.session_state.selected_variants = []

    df = get_log_eventos(which="log_eventos")
    base_df = df if df is not None else pd.DataFrame()
    event_log_view = EventLogView(base_df=base_df)
    filters_disabled = disabled or base_df.empty

    with st.expander("Filtros", expanded=True):
        with st.expander("Agente", expanded=False):
            st.info(
                "Foque a análise nos atendimentos do chatbot, do humano ou de ambos para comparar padrões de condução.",
                icon="ℹ️",
            )
            event_log_view = filter_by_agents(event_log_view, filters_disabled)
        with st.expander("Variantes", expanded=False):
            st.info(
                "Priorize as variantes mais frequentes para ver quais caminhos impactam volume ou gargalos.",
                icon="ℹ️",
            )
            event_log_view = filter_by_variants(event_log_view, filters_disabled) or event_log_view
        with st.expander("Janela Temporal", expanded=False):
            st.info(
                "Restringe o período analisado para evidenciar tendências sazonais ou incidentes pontuais.",
                icon="ℹ️",
            )
            event_log_view = temporal_filter(event_log_view, filters_disabled) or event_log_view

        with st.expander("Relações Directly/Eventually Follows", expanded=False):
            st.info(
                "Defina relações Directly/Eventually Follows para testar hipóteses de sequência, como 'Login precede Pagamento'.",
                icon="ℹ️",
            )
            event_log_view = activity_relationship_filter(event_log_view, filters_disabled) or event_log_view

        selected_model_type = model_type_selector(filters_disabled)

        generate_model(filters_disabled, event_log_view, selected_model_type)

    render_preview(event_log_view, preview_area)


def generate_model(
    disabled: bool, event_log_view: EventLogView | None, selected_model_type
):
    """
    Gera o modelo de processo quando o botão "Gerar" é acionado na UI do Streamlit.
    """
    running = st.session_state.get("processing_model", False)
    button_disabled = disabled or running
    if st.button("Gerar", key="filters.generate", disabled=button_disabled):
        if event_log_view is None:
            st.warning("Carregue um log para gerar o modelo de processo.")
            return
        st.session_state.processing_model = True
        st.session_state.processing_error = False
        try:
            with st.status("Gerando modelo...", state="running", expanded=True) as status:
                status.write("Pré-processando log e filtros...")
                view = generate_process_model(event_log_view, model=selected_model_type)

                status.write("Minerando modelo selecionado...")
                view.compute()

                status.write("Preparando visualização e métricas...")
                st.session_state.latest_generated_model = view
                status.update(label="Modelo gerado com sucesso", state="complete", expanded=False)
            show_generated_model_dialog()
        except ValueError as exc:
            st.session_state.processing_error = True
            st.error(str(exc))
        except Exception as exc:
            st.session_state.processing_error = True
            st.error("Falha ao gerar o modelo de processo.")
            st.exception(exc)
        finally:
            st.session_state.processing_model = False


def model_type_selector(disabled: bool) -> Any:
    """
    Exibe um seletor de tipo de modelo de processo na interface Streamlit e retorna
    a opção atualmente selecionada armazenada em `st.session_state`.

    Comportamento:
    - Inicializa `st.session_state['process_model_type']` com o valor padrão "dfg"
      caso não exista ou se o valor presente não estiver entre as opções válidas.
    - Renderiza um controle de rádio (streamlit) com as opções suportadas e
      um rótulo legível para cada opção.

    Parâmetros:
        disabled (bool): Se True, desabilita a interação do controle na UI.

    Retorno:
        Any: A string correspondente ao tipo de modelo selecionado. Valores possíveis:
            - "dfg": DFG (fluxo de atividades)
            - "performance-dfg": Performance DFG (DFG de performance)
            - "petri-net": Petri Net

    Efeitos colaterais:
    - Modifica `st.session_state['process_model_type']` quando necessário para
      garantir um valor válido.
    """
    if "process_model_type" not in st.session_state:
        st.session_state["process_model_type"] = "dfg"

    process_model_options = ["dfg", "performance-dfg", "petri-net"]
    if st.session_state["process_model_type"] not in process_model_options:
        st.session_state["process_model_type"] = "dfg"

    labels = {
        "dfg": "DFG",
        "performance-dfg": "Performance DFG",
        "petri-net": "Petri Net",
    }
    st.radio(
        label="Process model",
        options=process_model_options,
        format_func=lambda opt: labels.get(opt, opt),
        key="process_model_type",
        disabled=disabled,
        help="Selecione o tipo de modelo de processo que deseje gerar."
    )
    selected_model_type = st.session_state["process_model_type"]
    return selected_model_type


def filter_by_variants(event_view: EventLogView, disabled: bool) -> EventLogView | None:
    """
    Exibe e aplica um filtro de variantes de casos sobre um EventLogView usando Streamlit.
    """
    df = get_log_eventos(which="log_eventos")
    if df is None or df.empty:
        st.data_editor(
            pd.DataFrame(columns=["variant_id", "frequency", "variant", "selected"]),
            disabled=True,
            hide_index=True,
        )
        st.session_state.selected_variants = []
        return None

    variants_per_case = (
        CaseAggView(base_df=df).with_aggregator(CaseVariantAggregator()).compute()
    )
    if not variants_per_case:
        st.info("Nenhuma variante disponível para seleção.")
        st.session_state.selected_variants = []
        return event_view

    case_ids_by_variant: dict[str, list[str]] = {}
    variant_info_by_id: dict[str, Any] = {}
    for case_id, info in variants_per_case.items():
        variant_info_by_id[info.variant_id] = info
        case_ids_by_variant.setdefault(info.variant_id, []).append(case_id)

    rows = [
        {
            "variant_id": info.variant_id,
            "frequency": info.frequency,
            "variant": info.variant,
            "selected": info.variant_id in st.session_state.selected_variants,
        }
        for info in sorted(
            variant_info_by_id.values(), key=lambda itm: itm.frequency, reverse=True
        )
    ]
    variants_df = pd.DataFrame(rows)

    edited = st.data_editor(
        variants_df,
        use_container_width=True,
        hide_index=True,
        disabled=disabled,
        column_config={
            "selected": st.column_config.CheckboxColumn("Selecionar", help="Selecione variantes para manter"),
            "variant_id": st.column_config.TextColumn("ID da variante", width="small"),
            "frequency": st.column_config.NumberColumn("Frequência", format="%d", width="small"),
            "variant": st.column_config.TextColumn("Atividades"),
        },
        key="variant-selector",
    )

    selected_ids: list[str] = []
    if isinstance(edited, pd.DataFrame) and not edited.empty:
        selected_ids = edited.loc[edited["selected"], "variant_id"].astype(str).tolist()
    st.session_state.selected_variants = selected_ids

    if selected_ids:
        case_ids = [
            case
            for variant_id in selected_ids
            for case in case_ids_by_variant.get(variant_id, [])
        ]
        if case_ids:
            return event_view.filter(CaseFilter(case_ids=case_ids))
    return event_view


def filter_by_agents(event_view: EventLogView, disabled: bool) -> EventLogView:
    """Exibe e aplica um filtro de agente sobre o EventLogView usando Streamlit."""

    sel = st.segmented_control(
        "Filtrar por AGENTE",
        ["chatbot", "cliente", "ambos"],
        selection_mode="single",
        default="ambos",
        disabled=disabled,
    )
    if disabled:
        return event_view

    agent = {"chatbot": "ai", "cliente": "human"}.get(sel)
    return event_view.filter(AgentFilter(agent=agent)) if agent else event_view


def temporal_filter(event_view: EventLogView, disabled: bool) -> EventLogView | None:
    """Aplica filtro temporal com janela unificada."""
    df = get_log_eventos(which="log_eventos")

    if df is None or df.empty:
        empty_date = pd.to_datetime("1970-01-01").date()
        st.slider(
            "Intervalo de datas",
            value=(empty_date, empty_date),
            format="YYYY-MM-DD",
            disabled=True,
        )
        return None

    if COLUMN_START_TS not in df.columns:
        st.info("O log carregado não possui timestamps para aplicar filtro temporal.")
        return event_view

    date_options = sorted(
        {
            pd.to_datetime(d).date()
            for d in CaseAggView(base_df=df)
            .with_aggregator(CaseDateAggregator())
            .compute()
            .values()
            if d
        }
    )
    if not date_options:
        st.info("Nenhuma data disponível para filtrar.")
        return event_view

    start_default = pd.to_datetime(date_options[0]).date()
    end_default = pd.to_datetime(date_options[-1]).date()
    start_date, end_date = st.slider(
        "Intervalo de datas",
        min_value=start_default,
        max_value=end_default,
        value=(start_default, end_default),
        format="YYYY-MM-DD",
        disabled=disabled,
    )

    if end_date < start_date:
        st.error(
            "Falha ao gerar o filtro temporal: a data final não pode ser anterior à data inicial."
        )
        return None

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = (
        pd.to_datetime(end_date).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    series = df.get(COLUMN_START_TS)
    if series is not None and pd.api.types.is_datetime64tz_dtype(series.dtype):
        tz = series.dt.tz
        start_ts = start_ts if start_ts.tzinfo else start_ts.tz_localize(tz)
        end_ts = end_ts if end_ts.tzinfo else end_ts.tz_localize(tz)

    filtered_view = event_view.filter(TimeWindowFilter(start=start_ts, end=end_ts))
    filtered_df = filtered_view.compute()
    num_cases = filtered_df[COLUMN_CASE_ID].nunique() if COLUMN_CASE_ID in filtered_df else len(filtered_df)
    st.metric("Casos filtrados", num_cases)
    return filtered_view


def activity_relationship_filter(
    event_view: EventLogView, disabled: bool
) -> EventLogView | None:
    """Renderiza filtros avançados baseados em relações entre atividades."""

    df = get_log_eventos(which="log_eventos")
    if df is None or COLUMN_ACTIVITY not in df.columns or df.empty:
        st.data_editor(
            pd.DataFrame(columns=["Predecessora", "Sucessora", "Tipo"]),
            disabled=True,
            hide_index=True,
            num_rows="dynamic",
        )
        return None

    activities = sorted(
        {
            str(value).strip()
            for value in df[COLUMN_ACTIVITY].dropna().unique().tolist()
            if str(value).strip()
        }
    )

    if not activities:
        st.info("O log carregado não possui atividades disponíveis para filtrar.")
        return None

    base_rules = st.session_state.get("advanced-rules")
    if not isinstance(base_rules, pd.DataFrame):
        base_rules = pd.DataFrame(columns=["Predecessora", "Sucessora", "Tipo"])

    edited = st.data_editor(
        base_rules,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        disabled=disabled,
        column_config={
            "Predecessora": st.column_config.SelectboxColumn(
                "Predecessora", options=activities, required=False
            ),
            "Sucessora": st.column_config.SelectboxColumn(
                "Sucessora", options=activities, required=False
            ),
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo", options=["Directly Follows", "Eventually Follows"],
                required=False,
            ),
        },
        key="advanced-rules",
    )

    if disabled or edited is None or edited.empty:
        return None

    cleaned = edited.dropna(subset=["Predecessora", "Sucessora", "Tipo"], how="any")
    cleaned = cleaned.drop_duplicates(subset=["Predecessora", "Sucessora", "Tipo"])

    filters = []
    for _, row in cleaned.iterrows():
        if row["Predecessora"] == row["Sucessora"]:
            st.warning("Escolha atividades diferentes para predecessor e sucessora.")
            continue
        if row["Tipo"] == "Eventually Follows":
            filters.append(EventuallyFollowsFilter(row["Predecessora"], row["Sucessora"]))
        else:
            filters.append(DirectlyFollowsFilter(row["Predecessora"], row["Sucessora"]))

    if not filters:
        return None

    try:
        return event_view.filter(filters)
    except FilterError as exc:
        st.error(str(exc))
        return None


def render_preview(event_log_view: EventLogView, preview_area) -> None:
    target = preview_area if preview_area is not None else st
    with target:
        try:
            with st.spinner("Atualizando pré-visualização…"):
                preview_df = event_log_view.compute()
        except Exception:
            preview_df = event_log_view.compute()
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                COLUMN_CASE_ID: st.column_config.TextColumn(
                    "Caso", width="small", help="Identificador do caso"
                )
            },
        )
