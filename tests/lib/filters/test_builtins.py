import pandas as pd
import pytest

from chatflow_miner.lib.filters.base import MissingColumnsError
from chatflow_miner.lib.filters.builtins import (
    AgentFilter,
    CaseHasActivityFilter,
    TimeWindowFilter,
)


def test_agent_filter_includes_syst_by_default():
    df = pd.DataFrame(
        {
            "AGENTE": ["AI", "human", "syst"],
            "CASE_ID": [1, 1, 2],
        }
    )

    f = AgentFilter("ai")
    mask = f.mask(df)
    assert mask.tolist() == [True, False, True]


def test_agent_filter_respects_include_syst_false():
    df = pd.DataFrame(
        {
            "AGENTE": ["AI", "human", "syst"],
            "CASE_ID": [1, 1, 2],
        }
    )

    f = AgentFilter("human", include_syst=False)
    mask = f.mask(df)
    assert mask.tolist() == [False, True, False]


def test_agent_filter_raises_on_invalid_agent_value():
    with pytest.raises(ValueError):
        AgentFilter("bot")


def test_agent_filter_raises_missing_columns_when_column_missing():
    df = pd.DataFrame({"CASE_ID": [1, 2, 3]})
    f = AgentFilter("ai")
    with pytest.raises(MissingColumnsError):
        f.mask(df)


def test_case_has_activity_filter_keeps_all_events_of_matching_cases():
    df = pd.DataFrame(
        {
            "CASE_ID": [1, 1, 2, 3, 3],
            "ACTIVITY": ["X", "target", "Y", "target", "Z"],
        }
    )

    f = CaseHasActivityFilter("target")
    mask = f.mask(df)
    assert mask.tolist() == [True, True, False, True, True]


def test_case_has_activity_filter_raises_when_required_columns_missing():
    df = pd.DataFrame({"CASE_ID": [1, 2, 3]})
    f = CaseHasActivityFilter("anything")
    with pytest.raises(MissingColumnsError):
        f.mask(df)


def test_time_window_filter_touches_mode_intersection_behavior():
    df = pd.DataFrame(
        {
            "START_TIMESTAMP": [
                "2021-01-01T10:00:00",
                "2021-01-02T10:00:00",
                "2021-01-01T09:00:00",
            ],
            "END_TIMESTAMP": [
                "2021-01-01T11:00:00",
                "2021-01-02T11:00:00",
                pd.NaT,
            ],
        }
    )
    df["START_TIMESTAMP"] = pd.to_datetime(df["START_TIMESTAMP"])
    df["END_TIMESTAMP"] = pd.to_datetime(df["END_TIMESTAMP"])

    f = TimeWindowFilter(
        start="2021-01-01T10:30:00", end="2021-01-01T10:45:00", mode="touches"
    )
    mask = f.mask(df)
    assert mask.tolist() == [True, False, False]


def test_time_window_filter_inside_mode_requires_full_containment():
    df = pd.DataFrame(
        {
            "START_TIMESTAMP": ["2021-01-01T10:00:00", "2021-01-01T09:00:00"],
            "END_TIMESTAMP": ["2021-01-01T11:00:00", "2021-01-01T09:30:00"],
        }
    )
    df["START_TIMESTAMP"] = pd.to_datetime(df["START_TIMESTAMP"])
    df["END_TIMESTAMP"] = pd.to_datetime(df["END_TIMESTAMP"])

    f = TimeWindowFilter(
        start="2021-01-01T09:30:00", end="2021-01-01T11:30:00", mode="inside"
    )
    mask = f.mask(df)
    assert mask.tolist() == [True, False]


def test_time_window_filter_works_with_missing_end_timestamp_column():
    df = pd.DataFrame(
        {"START_TIMESTAMP": ["2021-01-01T10:00:00", "2021-01-02T10:00:00"]}
    )
    df["START_TIMESTAMP"] = pd.to_datetime(df["START_TIMESTAMP"])

    f = TimeWindowFilter(
        start="2021-01-01T09:00:00", end="2021-01-01T11:00:00", mode="touches"
    )
    mask = f.mask(df)
    assert mask.tolist() == [True, False]


def test_time_window_filter_raises_when_start_column_missing():
    df = pd.DataFrame({"OTHER": [1, 2, 3]})
    f = TimeWindowFilter(start=None, end=None)
    with pytest.raises(MissingColumnsError):
        f.mask(df)
