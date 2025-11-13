"""Fragmento de análise de agentes focado em interações da IA."""
from __future__ import annotations

from typing import List

import pandas as pd
import pm4py
import streamlit as st

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_AGENT,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_RESOURCE,
    COLUMN_START_TS,
)
from chatflow_miner.lib.event_log import EventLogView
from chatflow_miner.lib.state import get_log_eventos
from chatflow_miner.lib.ui.process_models.streamlit_fragments import (
    generate_process_model,
)


def _normalise_resource_tokens(value: str) -> List[str]:
    tokens = [token.strip() for token in value.split(",")]
    return [token for token in tokens if token]


@st.fragment
def render_agent_analysis() -> None:
    """Renderiza visualizações sobre recursos utilizados pelo agente IA."""

    log_df = get_log_eventos(which="log_eventos")
    if log_df is None or log_df.empty:
        st.info("Nenhum log carregado.")
        return

    missing_columns = [
        column
        for column in (COLUMN_AGENT, COLUMN_RESOURCE, COLUMN_ACTIVITY)
        if column not in log_df.columns
    ]
    if missing_columns:
        st.info("Log carregado não possui dados suficientes para análise de agentes.")
        return

    ai_mask = (
        log_df[COLUMN_AGENT]
        .astype(str)
        .str.lower()
        .eq("ai")
    )
    ai_df = log_df.loc[ai_mask].copy()
    if ai_df.empty:
        st.info("Não há eventos associados ao agente IA.")
        return

    timestamp_columns = [
        column
        for column in (COLUMN_START_TS, COLUMN_END_TS)
        if column in ai_df.columns
    ]
    for column in timestamp_columns:
        ai_df[column] = pd.to_datetime(
            ai_df[column], errors="coerce", format="mixed"
        )

    ai_df = ai_df.dropna(subset=[COLUMN_RESOURCE])
    if ai_df.empty:
        st.info("Não há recursos associados aos eventos do agente IA.")
        return

    normalised_per_event = (
        ai_df[COLUMN_RESOURCE]
        .astype(str)
        .apply(_normalise_resource_tokens)
    )
    normalised_per_event = normalised_per_event[normalised_per_event.map(bool)]
    if normalised_per_event.empty:
        st.info("Não há recursos válidos para os eventos do agente IA.")
        return

    ai_df = ai_df.loc[normalised_per_event.index]
    ai_df = ai_df.assign(normalised_resources=normalised_per_event)

    exploded_resources = ai_df.explode("normalised_resources").rename(
        columns={"normalised_resources": "resource"}
    )
    resource_counts = (
        exploded_resources["resource"].value_counts()
        .rename_axis("Recurso")
        .rename("Frequência")
    )
    resources_df = resource_counts.reset_index()

    st.subheader("Recursos utilizados pelo agente IA")
    st.bar_chart(resources_df.set_index("Recurso")[["Frequência"]])
    st.dataframe(resources_df, use_container_width=True)

    st.subheader("Tendência de uso por recurso")
    if COLUMN_START_TS not in exploded_resources.columns:
        st.info(
            "O log não possui coluna de início necessária para análise temporal de recursos."
        )
    else:
        resources_with_start = exploded_resources.dropna(
            subset=[COLUMN_START_TS]
        ).copy()
        if resources_with_start.empty:
            st.info(
                "Não há registros com timestamp para gerar tendências de recursos."
            )
        else:
            timestamp_resource_counts = (
                resources_with_start["resource"].value_counts()
            )
            if timestamp_resource_counts.empty:
                st.info(
                    "Não há dados suficientes para gerar séries temporais dos recursos selecionados."
                )
            else:
                freq_options = {"Diário": "D", "Semanal": "W-MON"}
                freq_label = st.radio(
                    "Granularidade",
                    list(freq_options.keys()),
                    horizontal=True,
                    key="agent_analysis_freq",
                )
                chart_label = st.radio(
                    "Visualização",
                    ["Área", "Linha"],
                    horizontal=True,
                    key="agent_analysis_chart_type",
                )

                max_resources = max(1, min(len(timestamp_resource_counts), 15))
                top_n_default = min(5, max_resources)
                top_n = st.slider(
                    "Quantidade de recursos mais frequentes",
                    min_value=1,
                    max_value=max_resources,
                    value=top_n_default,
                )

                top_resources = (
                    timestamp_resource_counts.index[:top_n].tolist()
                )

                freq = freq_options[freq_label]
                grouped = (
                    resources_with_start.set_index(COLUMN_START_TS)
                    .sort_index()
                    .groupby([pd.Grouper(freq=freq), "resource"])
                    .size()
                )

                if grouped.empty:
                    st.info(
                        "Não há dados suficientes para gerar séries temporais dos recursos selecionados."
                    )
                else:
                    timeseries = grouped.unstack(fill_value=0)
                    if not timeseries.empty:
                        full_index = pd.date_range(
                            start=timeseries.index.min(),
                            end=timeseries.index.max(),
                            freq=freq,
                        )
                        timeseries = timeseries.reindex(full_index, fill_value=0)

                    selected_timeseries = timeseries.reindex(
                        columns=top_resources, fill_value=0
                    )
                    if selected_timeseries.empty:
                        st.info(
                            "Os recursos selecionados não possuem ocorrências no período analisado."
                        )
                    else:
                        if chart_label == "Área":
                            st.area_chart(
                                selected_timeseries,
                                use_container_width=True,
                            )
                        else:
                            st.line_chart(
                                selected_timeseries,
                                use_container_width=True,
                            )

    missing_process_columns = [
        column
        for column in (COLUMN_CASE_ID, COLUMN_START_TS, COLUMN_END_TS)
        if column not in exploded_resources.columns
    ]
    if missing_process_columns:
        st.info(
            "O log não possui colunas suficientes para gerar o modelo de processo por recurso."
        )
    else:
        resource_process_df = exploded_resources.dropna(
            subset=[COLUMN_CASE_ID, COLUMN_START_TS, COLUMN_END_TS]
        ).copy()

        if resource_process_df.empty:
            st.info(
                "Não há dados suficientes para gerar o modelo de processo por recurso."
            )
        else:
            resource_process_df[COLUMN_RESOURCE] = (
                resource_process_df["resource"].astype("string").str.strip()
            )
            resource_process_df = resource_process_df.dropna(
                subset=[COLUMN_RESOURCE]
            )
            resource_process_df = resource_process_df.loc[
                resource_process_df[COLUMN_RESOURCE].ne("")
            ]

            if resource_process_df.empty:
                st.info(
                    "Não há recursos válidos para gerar o modelo de processo por recurso."
                )
            else:
                columns_to_keep = [
                    column
                    for column in (
                        COLUMN_CASE_ID,
                        COLUMN_START_TS,
                        COLUMN_END_TS,
                        COLUMN_RESOURCE,
                    )
                    if column in resource_process_df.columns
                ]
                resource_process_df = resource_process_df[columns_to_keep]

                try:
                    formatted_resource_log = pm4py.format_dataframe(
                        resource_process_df,
                        case_id=COLUMN_CASE_ID,
                        activity_key=COLUMN_RESOURCE,
                        timestamp_key=COLUMN_END_TS,
                        start_timestamp_key=COLUMN_START_TS,
                    )
                except Exception as exc:
                    st.error(
                        "Falha ao preparar o log para gerar o modelo de processo por recurso."
                    )
                    st.exception(exc)
                else:
                    try:
                        resource_log_view = EventLogView(formatted_resource_log)
                        process_model_view = generate_process_model(resource_log_view)
                        graphviz = process_model_view.to_graphviz(
                            bgcolor="white", rankdir="LR"
                        )

                        col_subheader, col_metrics = st.columns(2, gap="large")
                        with col_subheader:
                            st.subheader("Modelo de processo por recurso")
                        with col_metrics:
                            metrics = process_model_view.quality_metrics()
                            rows = []
                            ordered_keys = [
                                ("fitness", "Fitness"),
                                ("precision", "Precisão"),
                                ("generalization", "Generalização"),
                                ("simplicity", "Simplicidade"),
                            ]
                            for key, label in ordered_keys:
                                value = metrics.get(key)
                                if isinstance(value, (int, float)):
                                    display = f"{value:.3f}"
                                else:
                                    display = "N/A"
                                rows.append({"Métrica": label, "Valor": display})
                            metrics_df = pd.DataFrame(rows).set_index("Métrica")
                            with st.popover("Métricas do Modelo"):
                                st.table(metrics_df)

                        st.graphviz_chart(graphviz, use_container_width=True)
                    except Exception as exc:
                        st.error(
                            "Falha ao gerar ou renderizar o modelo de processo por recurso."
                        )
                        st.exception(exc)

    combination_labels = normalised_per_event.apply(
        lambda tokens: ", ".join(sorted(set(tokens)))
    )
    combination_labels = combination_labels[combination_labels != ""]
    if combination_labels.empty:
        st.info("Não foi possível identificar combinações de recursos para o agente IA.")
        st.caption(
            "As combinações representam sinergias entre recursos atribuídos a um mesmo evento do agente IA."
        )
        return

    combination_counts = (
        combination_labels.value_counts()
        .rename_axis("Combinação")
        .rename("Frequência")
    )
    combination_df = combination_counts.reset_index()

    st.subheader("Combinações de recursos do agente IA")
    st.dataframe(combination_df, use_container_width=True)

    st.caption(
        "As combinações representam sinergias entre recursos atribuídos a um mesmo evento do agente IA."
    )

    activity_resource_counts = (
        exploded_resources.groupby([COLUMN_ACTIVITY, "resource"])
        .size()
        .rename("Frequência")
        .reset_index()
    )

    if activity_resource_counts.empty:
        return

    pivot_table = activity_resource_counts.pivot_table(
        index=COLUMN_ACTIVITY,
        columns="resource",
        values="Frequência",
        fill_value=0,
        aggfunc="sum",
    )

    if pivot_table.empty:
        return

    activity_totals = pivot_table.sum(axis=1).sort_values(ascending=False)
    resource_totals = pivot_table.sum(axis=0).sort_values(ascending=False)
    pivot_table = pivot_table.loc[activity_totals.index, resource_totals.index]

    st.subheader("Atividades x recursos do agente IA")
    available_activities = list(activity_totals.index)
    available_resources = list(resource_totals.index)

    selected_activities = st.multiselect(
        "Filtrar atividades",
        options=available_activities,
        default=available_activities,
    )
    selected_resources = st.multiselect(
        "Filtrar recursos",
        options=available_resources,
        default=available_resources,
    )

    filtered_table = pivot_table
    if selected_activities:
        filtered_table = filtered_table.loc[filtered_table.index.intersection(selected_activities)]
    if selected_resources:
        filtered_table = filtered_table.loc[:, filtered_table.columns.intersection(selected_resources)]

    if filtered_table.empty:
        st.info("Nenhum dado disponível para os filtros selecionados.")
        return

    st.dataframe(
        filtered_table.style.background_gradient(axis=None),
        use_container_width=True,
    )
