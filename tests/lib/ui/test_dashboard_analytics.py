import pandas as pd

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_AGENT,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_EVENT_ID,
    COLUMN_START_TS,
)
from chatflow_miner.lib.ui.dashboard.analytics import (
    compute_activity_histogram,
    compute_case_durations,
    compute_events_per_period,
    compute_log_overview,
    compute_variant_frames,
    normalize_log,
)


def _sample_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COLUMN_CASE_ID: "C1",
                COLUMN_EVENT_ID: 0,
                COLUMN_ACTIVITY: "Saudação",
                COLUMN_START_TS: "2024-01-01 10:00:00",
                COLUMN_END_TS: "2024-01-01 10:05:00",
                COLUMN_AGENT: "syst",
                "canal": "chat",
            },
            {
                COLUMN_CASE_ID: "C1",
                COLUMN_EVENT_ID: 1,
                COLUMN_ACTIVITY: "Atendimento",
                COLUMN_START_TS: "2024-01-01 10:05:00",
                COLUMN_END_TS: "2024-01-01 10:10:00",
                COLUMN_AGENT: "ai",
                "canal": "chat",
            },
            {
                COLUMN_CASE_ID: "C2",
                COLUMN_EVENT_ID: 0,
                COLUMN_ACTIVITY: "Saudação",
                COLUMN_START_TS: "2024-01-02 09:00:00",
                COLUMN_END_TS: "2024-01-02 09:05:00",
                COLUMN_AGENT: "syst",
                "canal": "email",
            },
            {
                COLUMN_CASE_ID: "C2",
                COLUMN_EVENT_ID: 1,
                COLUMN_ACTIVITY: "Atendimento",
                COLUMN_START_TS: "2024-01-02 09:05:00",
                COLUMN_END_TS: "2024-01-02 09:15:00",
                COLUMN_AGENT: "human",
                "canal": "email",
            },
            {
                COLUMN_CASE_ID: "C3",
                COLUMN_EVENT_ID: 0,
                COLUMN_ACTIVITY: "Saudação",
                COLUMN_START_TS: "2024-01-03 12:00:00",
                COLUMN_END_TS: "2024-01-03 12:10:00",
                COLUMN_AGENT: "syst",
                "canal": "chat",
            },
            {
                COLUMN_CASE_ID: "C3",
                COLUMN_EVENT_ID: 1,
                COLUMN_ACTIVITY: "Encerramento",
                COLUMN_START_TS: "2024-01-03 12:10:00",
                COLUMN_END_TS: "2024-01-03 12:30:00",
                COLUMN_AGENT: "human",
                "canal": "chat",
            },
        ]
    )


def test_normalize_log_returns_new_dataframe():
    df = _sample_log()
    normalized = normalize_log(df)

    assert normalized is not df
    assert normalized[COLUMN_START_TS].dtype == "datetime64[ns]"
    assert df[COLUMN_START_TS].dtype == object


def test_compute_log_overview_uses_case_dates():
    df = normalize_log(_sample_log())

    overview = compute_log_overview(df)

    assert overview["total_cases"] == 3
    assert overview["total_events"] == 6
    assert overview["start_date"] == "2024-01-01"
    assert overview["end_date"] == "2024-01-03"
    assert overview["avg_events_per_case"] == 2
    assert overview["attributes"] == sorted(df.columns.tolist())


def test_compute_activity_histogram_orders_by_frequency():
    df = normalize_log(_sample_log())

    hist = compute_activity_histogram(df)

    assert list(hist["Atividade"]) == ["Saudação", "Atendimento", "Encerramento"]
    assert list(hist["Frequência"]) == [3, 2, 1]


def test_compute_variant_frames_returns_sorted_frames():
    df = normalize_log(_sample_log())

    top_df, full_df = compute_variant_frames(df, top_n=5)

    assert len(full_df) == 2
    assert top_df.iloc[0]["frequency"] == 2
    assert full_df.iloc[0]["variant"] == "Atendimento"
    assert int(full_df.iloc[0]["length"]) == 1


def test_compute_case_durations_in_minutes():
    df = normalize_log(_sample_log())

    durations_df = compute_case_durations(df)

    assert set(durations_df.columns) == {
        COLUMN_CASE_ID,
        "duracao",
        "duracao_minutos",
    }
    assert durations_df.loc[durations_df[COLUMN_CASE_ID] == "C1", "duracao_minutos"].iloc[0] == 10
    assert durations_df.loc[durations_df[COLUMN_CASE_ID] == "C3", "duracao_minutos"].iloc[0] == 30


def test_compute_events_per_period_daily_and_weekly():
    df = normalize_log(_sample_log())

    daily = compute_events_per_period(df, "D")
    weekly = compute_events_per_period(df, "W-MON")

    assert list(daily["eventos"]) == [2, 2, 2]
    assert pd.to_datetime(daily["periodo"]).tolist()[0].date().isoformat() == "2024-01-01"

    assert len(weekly) == 2
    assert int(weekly.iloc[0]["eventos"]) == 2
    assert int(weekly.iloc[1]["eventos"]) == 4
