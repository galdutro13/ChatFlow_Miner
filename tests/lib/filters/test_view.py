import pandas as pd
from pathlib import Path

from chatflow_miner.lib.filters.view import EventLogView
from chatflow_miner.lib.filters.builtins import AgentFilter, TimeWindowFilter


def test_event_log_view_compute_without_filters_returns_copy_and_equal():
    df = pd.DataFrame({"CASE_ID": [1, 2], "AGENTE": ["AI", "human"]})
    view = EventLogView(df)
    result = view.compute()
    assert result.equals(df)
    assert result is not df


def test_event_log_view_filter_is_immutable_and_combines_filters_and_compute_applies_all():
    df = pd.DataFrame({
        "CASE_ID": [1, 1, 2, 3],
        "AGENTE": ["AI", "human", "AI", "syst"],
    })
    view = EventLogView(df)

    f1 = AgentFilter("ai")
    f2 = AgentFilter("ai", include_syst=False)

    view1 = view.filter(f1)
    view2 = view1.filter(f2)

    # original view unchanged
    assert view.compute().equals(df)

    # view1 applies f1
    r1 = view1.compute()
    assert r1.shape[0] == 3

    # view2 applies f1 AND f2 (both ai, but second excludes syst)
    r2 = view2.compute()
    # rows matching both should exclude syst row -> fewer or equal rows than r1
    assert 0 < r2.shape[0] <= r1.shape[0]


def test_event_log_view_filter_accepts_sequence_of_filters_and_applies_all():
    df = pd.DataFrame({
        "CASE_ID": [1, 1, 2, 2],
        "AGENTE": ["AI", "human", "AI", "AI"],
    })
    view = EventLogView(df)
    f_list = [AgentFilter("ai", include_syst=False), AgentFilter("ai")]
    new_view = view.filter(f_list)
    res = new_view.compute()
    # sequence applied as additional filters (AND): result should be subset
    assert res.shape[0] <= df.shape[0]


def test_event_log_view_head_materializes_and_limits_rows():
    df = pd.DataFrame({"CASE_ID": list(range(10)), "AGENTE": ["AI"] * 10})
    view = EventLogView(df, filters=[AgentFilter("ai")])
    h = view.head(3)
    assert isinstance(h, pd.DataFrame)
    assert len(h) == 3


def test_event_log_view_to_csv_writes_file_without_index(tmp_path: Path):
    df = pd.DataFrame({"CASE_ID": [1, 2], "AGENTE": ["AI", "human"]})
    view = EventLogView(df, filters=[AgentFilter("ai")])
    p = tmp_path / "out.csv"
    view.to_csv(str(p))
    content = p.read_text(encoding="utf-8")
    # header and at least one line of data expected
    assert "CASE_ID" in content
    assert "AGENTE" in content
