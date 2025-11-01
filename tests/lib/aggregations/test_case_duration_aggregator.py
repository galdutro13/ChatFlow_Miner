import pandas as pd

from chatflow_miner.lib.aggregations import (
    CaseAggView,
    CaseDurationAggregator,
    NormalizeTimestampsOp,
)
from chatflow_miner.lib.constants import (
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_EVENT_ID,
    COLUMN_START_TS,
)


def _build_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COLUMN_CASE_ID: "A",
                COLUMN_EVENT_ID: 0,
                COLUMN_START_TS: "2024-01-01 10:00:00",
                COLUMN_END_TS: "2024-01-01 10:05:00",
            },
            {
                COLUMN_CASE_ID: "A",
                COLUMN_EVENT_ID: 1,
                COLUMN_START_TS: "2024-01-01 10:05:00",
                COLUMN_END_TS: "2024-01-01 10:15:00",
            },
            {
                COLUMN_CASE_ID: "B",
                COLUMN_EVENT_ID: 0,
                COLUMN_START_TS: "2024-01-01 11:00:00",
                COLUMN_END_TS: "2024-01-01 11:07:30",
            },
        ]
    )


def test_case_duration_returns_timedelta_per_case():
    df = _build_df()
    original = df.copy(deep=True)

    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseDurationAggregator())
    )
    result = view.compute()

    assert set(result.keys()) == {"A", "B"}
    assert result["A"] == pd.Timedelta(minutes=15)
    assert result["B"] == pd.Timedelta(minutes=7, seconds=30)

    pd.testing.assert_frame_equal(df, original)


def test_case_duration_missing_end_timestamp_uses_start():
    df = _build_df()
    df.loc[df[COLUMN_CASE_ID] == "B", COLUMN_END_TS] = None

    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseDurationAggregator())
    )
    result = view.compute()

    assert result["B"] == pd.Timedelta(0)


def test_case_duration_with_all_invalid_timestamps_returns_zero():
    df = pd.DataFrame(
        [
            {
                COLUMN_CASE_ID: "C",
                COLUMN_EVENT_ID: 0,
                COLUMN_START_TS: None,
                COLUMN_END_TS: None,
            }
        ]
    )

    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseDurationAggregator())
    )
    result = view.compute()

    assert result["C"] == pd.Timedelta(0)
