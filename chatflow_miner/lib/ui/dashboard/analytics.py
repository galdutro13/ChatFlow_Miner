"""Helpers e fragmento de análise exploratória."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from chatflow_miner.lib.aggregations import (
    CaseAggView,
    CaseDateAggregator,
    CaseDurationAggregator,
    CaseVariantAggregator,
    NormalizeTimestampsOp,
    VariantInfo,
)
from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_START_TS,
)
from chatflow_miner.lib.state import get_log_eventos


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas de timestamp sem mutar ``df`` original."""

    ts_columns = [c for c in (COLUMN_START_TS, COLUMN_END_TS) if c in df.columns]
    normalized = df.copy()
    for col in ts_columns:
        normalized[col] = pd.to_datetime(
            normalized[col], errors="coerce", format="mixed"
        )
    return normalized


def compute_log_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Computa métricas agregadas do log."""

    total_events = int(len(df))
    total_cases = int(df[COLUMN_CASE_ID].nunique()) if COLUMN_CASE_ID in df else 0

    overview: dict[str, Any] = {
        "total_events": total_events,
        "total_cases": total_cases,
        "attributes": sorted(df.columns.tolist()),
        "start_date": None,
        "end_date": None,
        "avg_events_per_case": 0.0,
    }

    if total_cases:
        overview["avg_events_per_case"] = total_events / total_cases

    if not df.empty and COLUMN_CASE_ID in df.columns and COLUMN_START_TS in df.columns:
        view = (
            CaseAggView(df)
            .with_aux(NormalizeTimestampsOp())
            .with_aggregator(CaseDateAggregator())
        )
        case_dates = pd.Series(view.compute()).dropna()
        if not case_dates.empty:
            overview["start_date"] = str(case_dates.min())
            overview["end_date"] = str(case_dates.max())

    return overview


def compute_activity_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Conta frequência por atividade."""

    if COLUMN_ACTIVITY not in df.columns or df.empty:
        return pd.DataFrame(columns=["Atividade", "Frequência"])

    counts = (
        df[COLUMN_ACTIVITY]
        .dropna()
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Atividade")
        .reset_index(name="Frequência")
    )
    return counts


def compute_variant_frames(
    df: pd.DataFrame, *, ignore_syst: bool = True, top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna dataframes com variantes (top-N e completo)."""

    if df.empty:
        empty = pd.DataFrame(columns=["variant_id", "variant", "frequency", "length"])
        return empty.copy(), empty

    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseVariantAggregator(ignore_syst=ignore_syst))
    )
    result = view.compute()
    if not result:
        empty = pd.DataFrame(columns=["variant_id", "variant", "frequency", "length"])
        return empty.copy(), empty

    unique_infos: dict[str, VariantInfo] = {}
    for info in result.values():
        unique_infos[info.variant_id] = info

    full_df = pd.DataFrame(
        [
            {
                "variant_id": info.variant_id,
                "variant": info.variant,
                "frequency": info.frequency,
                "length": info.length,
            }
            for info in unique_infos.values()
        ]
    )
    full_df = full_df.sort_values(
        by=["frequency", "length", "variant_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    top_df = full_df.head(top_n).reset_index(drop=True)
    return top_df, full_df


def compute_case_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Computa duração por caso em minutos."""

    required = {COLUMN_CASE_ID, COLUMN_START_TS, COLUMN_END_TS}
    if not required.issubset(df.columns) or df.empty:
        return pd.DataFrame(columns=[COLUMN_CASE_ID, "duracao", "duracao_minutos"])

    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseDurationAggregator())
    )
    durations = view.compute()
    if not durations:
        return pd.DataFrame(columns=[COLUMN_CASE_ID, "duracao", "duracao_minutos"])

    durations_df = (
        pd.Series(durations, name="duracao")
        .rename_axis(COLUMN_CASE_ID)
        .reset_index()
    )
    durations_df["duracao_minutos"] = durations_df["duracao"].dt.total_seconds().div(60)
    return durations_df[[COLUMN_CASE_ID, "duracao", "duracao_minutos"]]


def compute_events_per_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Agrupa eventos por período usando ``freq`` (ex.: ``"D"`` ou ``"W-MON"``)."""

    if df.empty or COLUMN_START_TS not in df.columns:
        return pd.DataFrame(columns=["periodo", "eventos"])

    normalized = df.copy()
    normalized[COLUMN_START_TS] = pd.to_datetime(
        normalized[COLUMN_START_TS], errors="coerce", format="mixed"
    )
    normalized = normalized.dropna(subset=[COLUMN_START_TS])
    if normalized.empty:
        return pd.DataFrame(columns=["periodo", "eventos"])

    counts = (
        normalized.set_index(COLUMN_START_TS)
        .sort_index()
        .groupby(pd.Grouper(freq=freq))
        .size()
        .asfreq(freq, fill_value=0)
    )
    counts_df = counts.reset_index().rename(
        columns={COLUMN_START_TS: "periodo", 0: "eventos"}
    )
    return counts_df


@st.fragment
def render_exploratory_analysis() -> None:
    """Renderiza a aba de análise exploratória."""

    log_df = get_log_eventos(which="log_eventos")
    if log_df is None or log_df.empty:
        st.info("Nenhum log carregado.")
        return

    with st.spinner("Calculando estatísticas..."):
        normalized = normalize_log(log_df)
        overview = compute_log_overview(normalized)
        activity_hist = compute_activity_histogram(normalized)
        top_variants, all_variants = compute_variant_frames(normalized)
        durations_df = compute_case_durations(normalized)
        events_daily = compute_events_per_period(normalized, "D")
        events_weekly = compute_events_per_period(normalized, "W-MON")

    st.subheader("Visão geral do log")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Casos", f"{overview['total_cases']:,}".replace(",", "."))
    metric_cols[1].metric("Eventos", f"{overview['total_events']:,}".replace(",", "."))
    metric_cols[2].metric(
        "Eventos por caso",
        f"{overview['avg_events_per_case']:.2f}" if overview["total_cases"] else "0",
    )

    interval_parts = []
    if overview.get("start_date"):
        interval_parts.append(overview["start_date"])
    if overview.get("end_date"):
        interval_parts.append(overview["end_date"])
    if interval_parts:
        st.caption(f"Intervalo do log: {' – '.join(interval_parts)}")

    if overview["attributes"]:
        st.caption(
            "Atributos disponíveis: " + ", ".join(str(attr) for attr in overview["attributes"])
        )

    st.divider()

    st.subheader("Frequência de atividades")
    if activity_hist.empty:
        st.info("Não foi possível calcular a frequência de atividades.")
    else:
        chart_df = activity_hist.set_index("Atividade")
        st.bar_chart(chart_df)

    st.divider()

    st.subheader("Variantes de processo")
    if all_variants.empty:
        st.info("Nenhuma variante encontrada.")
    else:
        max_top = int(max(1, min(len(all_variants), 25)))
        default_top = min(10, max_top)
        top_n = st.slider(
            "Exibir Top N variantes",
            min_value=1,
            max_value=max_top,
            value=default_top,
            help="Selecione quantas variantes mais frequentes deseja visualizar.",
        )
        top_df = all_variants.head(top_n)
        chart_variants = top_df.set_index("variant")[["frequency"]]
        chart_variants = chart_variants.rename(columns={"frequency": "Frequência"})
        st.bar_chart(chart_variants)

        display_df = all_variants.rename(
            columns={
                "variant_id": "ID",
                "variant": "Variante",
                "frequency": "Frequência",
                "length": "Comprimento",
            }
        )
        st.dataframe(display_df, use_container_width=True)

    st.divider()

    st.subheader("Duração por caso")
    if durations_df.empty:
        st.info("Não há informações suficientes para calcular a duração dos casos.")
    else:
        stats = durations_df["duracao_minutos"].describe().to_frame(name="Minutos")
        st.table(stats)

        hist = (
            pd.cut(durations_df["duracao_minutos"], bins=10)
            .value_counts()
            .sort_index()
            .rename_axis("Intervalo (min)")
            .to_frame(name="Casos")
        )
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)

    st.divider()

    st.subheader("Eventos por período")
    if events_daily.empty and events_weekly.empty:
        st.info("Sem timestamps para calcular eventos por período.")
    else:
        option = st.radio(
            "Granularidade",
            ("Diário", "Semanal"),
            index=0,
            horizontal=True,
        )
        period_df = events_daily if option == "Diário" else events_weekly
        if period_df.empty:
            st.info("Sem dados suficientes para a granularidade selecionada.")
        else:
            chart_period = period_df.set_index("periodo")[["eventos"]]
            chart_period = chart_period.rename(columns={"eventos": "Eventos"})
            st.area_chart(chart_period)

