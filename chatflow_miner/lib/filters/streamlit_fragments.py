from typing import Any

import pandas as pd
import streamlit as st

from chatflow_miner.lib.aggregations import (
    CaseAggView,
    CaseDateAggregator,
    CaseVariantAggregator,
)
from chatflow_miner.lib.constants import COLUMN_ACTIVITY, COLUMN_START_TS
from chatflow_miner.lib.event_log.view import EventLogView
from chatflow_miner.lib.filters.base.exceptions import FilterError
from chatflow_miner.lib.filters.builtins import (
    AgentFilter,
    CaseFilter,
    DirectlyFollowsFilter,
    EventuallyFollowsFilter,
    TimeWindowFilter,
)
from chatflow_miner.lib.process_models.ui import (
    generate_process_model,
    show_generated_model_dialog,
)
from chatflow_miner.lib.state import get_log_eventos


@st.fragment
def filter_section(*, disabled: bool = False):
    """Fragmento reutilizável para seção de filtros em Streamlit."""
    st.markdown("Filtro de dados - <u>Em construção</u>", unsafe_allow_html=True)
    event_log_view = filter_by_agents(disabled)
    event_log_view = filter_by_variants(event_log_view, disabled) or event_log_view
    event_log_view = temporal_filter(event_log_view, disabled) or event_log_view

    selected_model_type = process_model_selector(disabled)

    event_log_view = advanced_filter(event_log_view, disabled) or event_log_view

    st.dataframe(event_log_view.compute())

    # Área inferior com botão à direita
    generate_model(disabled, event_log_view, selected_model_type)


def generate_model(
    disabled: bool, event_log_view: EventLogView | None, selected_model_type
):
    """
    Gera o modelo de processo quando o botão "Gerar" é acionado na UI do Streamlit.

    Parâmetros:
        disabled (bool): Se True, o botão de geração é desabilitado.
        event_log_view (EventLogView | None): A visão do log de eventos a ser usada como base
            para a geração do modelo. Pode ser None, caso em que a geração pode falhar
            dependendo da implementação de generate_process_model.
        selected_model_type (Any): Identificador do tipo de modelo a ser gerado (ex.: "dfg",
            "performance-dfg", "petri-net").

    Comportamento:
        - Renderiza um botão na coluna direita da UI. Ao ser clicado:
            * Exibe um spinner "Gerando modelo...".
            * Chama generate_process_model(event_log_view, model=selected_model_type).
            * Armazena o modelo gerado em st.session_state['latest_generated_model'].
            * Abre um diálogo de visualização/salvamento via show_generated_model_dialog().
        - Em caso de ValueError, mostra a mensagem de erro ao usuário.
        - Em caso de exceção genérica, mostra uma mensagem de falha e imprime a exceção.

    Efeitos colaterais:
        - Modifica st.session_state['latest_generated_model'].
        - Produz saídas visuais na UI do Streamlit (spinner, mensagens de erro, diálogo).

    Observações:
        - A função é voltada apenas para execução dentro de um contexto Streamlit (não retorna valor significativo).
    """
    _, right_col = st.columns([6, 1])
    with right_col:
        if st.button("Gerar", key="filters.generate", disabled=disabled):
            try:
                with st.spinner("Gerando modelo..."):
                    view = generate_process_model(
                        event_log_view, model=selected_model_type
                    )
                    st.session_state.latest_generated_model = view
                # Abre diálogo para visualização e salvamento
                show_generated_model_dialog()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("Falha ao gerar o modelo de processo.")
                st.exception(exc)


def process_model_selector(disabled: bool) -> Any:
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
    )
    selected_model_type = st.session_state["process_model_type"]
    return selected_model_type


def filter_by_variants(event_view: EventLogView, disabled: bool) -> EventLogView | None:
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
        st.multiselect("Filtro de VARIANTE", [], [], disabled=disabled)
        return None
    result = CaseAggView(base_df=df).with_aggregator(CaseVariantAggregator()).compute()

    def gid(key: str) -> str:
        return getattr(result[key], "variant_id", key)

    seen = set()
    variants = [v for v in result if (vid := gid(v)) not in seen and not seen.add(vid)]
    variants.sort(key=lambda v: result[v].frequency, reverse=True)

    def fmt(variant_key: str) -> str:
        return (
            f"freq={result[variant_key].frequency} | "
            f"{result[variant_key].variant_id} | "
            f"{result[variant_key].variant}"
        )

    selected = st.multiselect(
        "Filtro de VARIANTE", variants, [], format_func=fmt, disabled=disabled
    )
    if selected:
        cid = {gid(v) for v in selected}
        ks = [k for k in result if gid(k) in cid]
        if ks:
            return event_view.filter(CaseFilter(case_ids=ks))
    return event_view


def filter_by_agents(disabled: bool) -> EventLogView:
    """
    Exibe e aplica um filtro de agente sobre o EventLogView usando Streamlit.

    Parâmetros:
        disabled (bool): Se True, desabilita a interação do filtro na interface.

    Retorna:
        EventLogView: Um EventLogView filtrado pelo agente selecionado ('chatbot', 'cliente' ou 'ambos').
    """
    sel = st.segmented_control(
        "Filtro de AGENTE",
        ["chatbot", "cliente", "ambos"],
        selection_mode="single",
        default="ambos",
        disabled=disabled,
    )
    df = (
        d if (d := get_log_eventos(which="log_eventos")) is not None else pd.DataFrame()
    )
    view = EventLogView(base_df=df)
    agent = {"chatbot": "ai", "cliente": "human"}.get(sel)
    return view.filter(AgentFilter(agent=agent)) if agent else view


def temporal_filter(event_view: EventLogView, disabled: bool) -> EventLogView | None:
    """
    Apresenta controles de filtro temporal na interface Streamlit e aplica um filtro
    de janela temporal (TimeWindowFilter) sobre a EventLogView fornecida.

    Comportamento:
    - Se não houver dados de eventos disponíveis (get_log_eventos retorna None),
      renderiza controles desabilitados e retorna None.
    - Calcula opções de data a partir de agregação por caso (CaseDateAggregator),
      apresenta sliders para data inicial e final, valida que a data final não seja
      anterior à inicial e converte as datas selecionadas para timestamps
      (incluindo o final do dia para a data final).
    - Se a série de timestamps do log possuir informação de fuso (tz-aware),
      ajusta os timestamps selecionados para o mesmo fuso horário.
    - Retorna a EventLogView filtrada pelo TimeWindowFilter correspondente.

    Parâmetros:
        event_view (EventLogView): visão atual do log de eventos a ser filtrada.
        disabled (bool): se True, desabilita os controles da interface Streamlit.

    Retorno:
        EventLogView ou None: a EventLogView filtrada pela janela temporal, ou None
        se não houver dados de evento a serem filtrados.

    Observações:
    - A função exibe mensagens de erro na UI quando a data final é anterior à inicial.
    - A janela temporal é inclusiva: a data final é ajustada para o último microssegundo do dia.
    """
    st.markdown("Filtro temporal", unsafe_allow_html=True)
    df = get_log_eventos(which="log_eventos")

    if df is None:
        col1, col2 = st.columns(2)
        for lbl, val, col in [("Data inicial", 1970, col1), ("Data final", 2025, col2)]:
            with col:
                st.select_slider(
                    lbl, options=[1970, 1997, 2025], value=val, disabled=disabled
                )
        return None

    date_options = sorted(
        {
            d
            for d in CaseAggView(base_df=df)
            .with_aggregator(CaseDateAggregator())
            .compute()
            .values()
            if d
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.select_slider(
            "Data inicial", date_options, date_options[0], disabled=disabled
        )
    with col2:
        end_date = st.select_slider(
            "Data final", date_options, date_options[-1], disabled=disabled
        )

    if end_date < start_date:
        st.error(
            "Falha ao gerar o filtro temporal: a data final não pode ser anterior à data inicial."
        )

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

    return event_view.filter(TimeWindowFilter(start=start_ts, end=end_ts))


def advanced_filter(event_view: EventLogView, disabled: bool) -> EventLogView | None:
    """Renderiza filtros avançados baseados em relações entre atividades."""

    relation_options = ("Eventually Follows", "Directly Follows")

    with st.expander("Filtros Avançados"):
        st.markdown(
            "Selecione pares de atividades para manter apenas os casos em que a"
            " sucessora ocorre após a predecessora. Use 'Directly Follows' para"
            " exigir que a transição seja imediata, sem eventos intermediários."
        )

        df = get_log_eventos(which="log_eventos")
        if df is None or COLUMN_ACTIVITY not in df.columns or df.empty:
            st.radio(
                label="Tipo de relação",
                options=relation_options,
                index=0,
                disabled=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    label="Predecessora",
                    options=[""],
                    index=0,
                    disabled=True,
                )
            with col2:
                st.selectbox(
                    label="Sucessora",
                    options=[""],
                    index=0,
                    disabled=True,
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

        relation_type = st.radio(
            label="Tipo de relação",
            options=relation_options,
            index=0,
            disabled=disabled,
        )

        placeholder = "Selecione uma atividade"
        options = [placeholder, *activities]

        col1, col2 = st.columns(2)
        with col1:
            predecessor = st.selectbox(
                label="Predecessora",
                options=options,
                index=0,
                disabled=disabled,
            )
        with col2:
            successor = st.selectbox(
                label="Sucessora",
                options=options,
                index=0,
                disabled=disabled,
            )

        if disabled:
            return None

        if predecessor == placeholder or successor == placeholder:
            st.warning("Selecione uma atividade para cada posição da relação.")
            return None

        if predecessor == successor:
            st.warning("Escolha atividades diferentes para predecessor e sucessora.")
            return None

        if relation_type == "Eventually Follows":
            flt = EventuallyFollowsFilter(predecessor, successor)
        else:
            flt = DirectlyFollowsFilter(predecessor, successor)

        try:
            return event_view.filter(flt)
        except FilterError as exc:
            st.error(str(exc))
            return None
