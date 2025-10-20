from __future__ import annotations

import pandas as pd
import pm4py
import pytest

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_START_TS,
)
from chatflow_miner.lib.filters.base import MissingColumnsError
from chatflow_miner.lib.filters.builtins import (
    DirectlyFollowsFilter,
    EventuallyFollowsFilter,
)


def _build_formatted_log() -> pd.DataFrame:
    records = [
        {  # case-1: eventually follows relation only
            COLUMN_CASE_ID: "case-1",
            COLUMN_ACTIVITY: "A",
            COLUMN_START_TS: "2020-01-01T10:00:00",
            COLUMN_END_TS: "2020-01-01T10:01:00",
        },
        {
            COLUMN_CASE_ID: "case-1",
            COLUMN_ACTIVITY: "X",
            COLUMN_START_TS: "2020-01-01T10:02:00",
            COLUMN_END_TS: "2020-01-01T10:03:00",
        },
        {
            COLUMN_CASE_ID: "case-1",
            COLUMN_ACTIVITY: "C",
            COLUMN_START_TS: "2020-01-01T10:04:00",
            COLUMN_END_TS: "2020-01-01T10:05:00",
        },
        {  # case-2: missing successor
            COLUMN_CASE_ID: "case-2",
            COLUMN_ACTIVITY: "A",
            COLUMN_START_TS: "2020-01-02T11:00:00",
            COLUMN_END_TS: "2020-01-02T11:01:00",
        },
        {
            COLUMN_CASE_ID: "case-2",
            COLUMN_ACTIVITY: "B",
            COLUMN_START_TS: "2020-01-02T11:02:00",
            COLUMN_END_TS: "2020-01-02T11:03:00",
        },
        {  # case-3: contains direct transition A -> C at the end
            COLUMN_CASE_ID: "case-3",
            COLUMN_ACTIVITY: "A",
            COLUMN_START_TS: "2020-01-03T12:00:00",
            COLUMN_END_TS: "2020-01-03T12:01:00",
        },
        {
            COLUMN_CASE_ID: "case-3",
            COLUMN_ACTIVITY: "B",
            COLUMN_START_TS: "2020-01-03T12:02:00",
            COLUMN_END_TS: "2020-01-03T12:03:00",
        },
        {
            COLUMN_CASE_ID: "case-3",
            COLUMN_ACTIVITY: "A",
            COLUMN_START_TS: "2020-01-03T12:04:00",
            COLUMN_END_TS: "2020-01-03T12:05:00",
        },
        {
            COLUMN_CASE_ID: "case-3",
            COLUMN_ACTIVITY: "C",
            COLUMN_START_TS: "2020-01-03T12:06:00",
            COLUMN_END_TS: "2020-01-03T12:07:00",
        },
        {  # case-4: minimal direct relation A -> C
            COLUMN_CASE_ID: "case-4",
            COLUMN_ACTIVITY: "A",
            COLUMN_START_TS: "2020-01-04T13:00:00",
            COLUMN_END_TS: "2020-01-04T13:01:00",
        },
        {
            COLUMN_CASE_ID: "case-4",
            COLUMN_ACTIVITY: "C",
            COLUMN_START_TS: "2020-01-04T13:02:00",
            COLUMN_END_TS: "2020-01-04T13:03:00",
        },
    ]

    df = pd.DataFrame.from_records(records)
    df[COLUMN_START_TS] = pd.to_datetime(df[COLUMN_START_TS])
    df[COLUMN_END_TS] = pd.to_datetime(df[COLUMN_END_TS])
    return pm4py.format_dataframe(
        df,
        case_id=COLUMN_CASE_ID,
        activity_key=COLUMN_ACTIVITY,
        timestamp_key=COLUMN_END_TS,
        start_timestamp_key=COLUMN_START_TS,
    )


def test_eventually_follows_filter_keeps_cases_with_relation():
    df = _build_formatted_log()
    flt = EventuallyFollowsFilter("A", "C")

    mask = flt.mask(df)

    kept_cases = set(df.loc[mask, COLUMN_CASE_ID])
    assert kept_cases == {"case-1", "case-3", "case-4"}

    discarded_cases = set(df.loc[~mask, COLUMN_CASE_ID])
    assert discarded_cases == {"case-2"}


def test_directly_follows_filter_requires_immediate_relation():
    df = _build_formatted_log()
    flt = DirectlyFollowsFilter("A", "C")

    mask = flt.mask(df)

    kept_cases = set(df.loc[mask, COLUMN_CASE_ID])
    assert kept_cases == {"case-3", "case-4"}

    discarded_cases = set(df.loc[~mask, COLUMN_CASE_ID])
    assert discarded_cases == {"case-1", "case-2"}


@pytest.mark.parametrize(
    "flt",
    [EventuallyFollowsFilter("A", "C"), DirectlyFollowsFilter("A", "C")],
)
def test_relation_filters_raise_on_missing_columns(flt: EventuallyFollowsFilter) -> None:
    df = pd.DataFrame({COLUMN_CASE_ID: ["case-1", "case-2"], COLUMN_ACTIVITY: ["A", "C"]})

    with pytest.raises(MissingColumnsError):
        flt.mask(df)
