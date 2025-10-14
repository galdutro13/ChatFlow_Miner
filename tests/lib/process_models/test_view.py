from typing import Any

import pandas as pd
import pm4py


def _build_formatted_log() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "CASE_ID": ["c1", "c1", "c2", "c2"],
            "ACTIVITY": ["A", "B", "A", "C"],
            "START_TIMESTAMP": pd.to_datetime(
                [
                    "2021-01-01 09:00",
                    "2021-01-01 10:00",
                    "2021-01-02 09:00",
                    "2021-01-02 10:00",
                ]
            ),
            "END_TIMESTAMP": pd.to_datetime(
                [
                    "2021-01-01 09:05",
                    "2021-01-01 10:05",
                    "2021-01-02 09:05",
                    "2021-01-02 10:05",
                ]
            ),
        }
    )
    return pm4py.format_dataframe(
        raw,
        case_id="CASE_ID",
        activity_key="ACTIVITY",
        timestamp_key="END_TIMESTAMP",
        start_timestamp_key="START_TIMESTAMP",
    )


def test_quality_metrics_returns_expected_keys_and_caches():
    from chatflow_miner.lib.event_log.view import EventLogView
    from chatflow_miner.lib.process_models.dfg import DFGModel
    from chatflow_miner.lib.process_models.view import ProcessModelView

    log = _build_formatted_log()
    view = EventLogView(log)
    model_view = ProcessModelView(log_view=view, model=DFGModel())

    metrics = model_view.quality_metrics()

    assert set(metrics.keys()) == {
        "fitness",
        "precision",
        "generalization",
        "simplicity",
    }
    for value in metrics.values():
        assert value is None or isinstance(value, float)

    # segunda chamada deve reutilizar o cache
    assert model_view.quality_metrics() is metrics


def test_quality_metrics_on_empty_log_returns_none_values():
    from chatflow_miner.lib.event_log.view import EventLogView
    from chatflow_miner.lib.process_models.dfg import DFGModel
    from chatflow_miner.lib.process_models.view import ProcessModelView

    raw = pd.DataFrame(
        columns=["CASE_ID", "ACTIVITY", "START_TIMESTAMP", "END_TIMESTAMP"]
    )
    raw["START_TIMESTAMP"] = pd.to_datetime(raw["START_TIMESTAMP"])
    raw["END_TIMESTAMP"] = pd.to_datetime(raw["END_TIMESTAMP"])
    formatted = pm4py.format_dataframe(
        raw,
        case_id="CASE_ID",
        activity_key="ACTIVITY",
        timestamp_key="END_TIMESTAMP",
        start_timestamp_key="START_TIMESTAMP",
    )

    empty_view = EventLogView(formatted)
    model_view = ProcessModelView(log_view=empty_view, model=DFGModel())

    metrics = model_view.quality_metrics()

    assert metrics == {
        "fitness": None,
        "precision": None,
        "generalization": None,
        "simplicity": None,
    }


def test_to_graphviz_converts_log_once_and_passes_event_log(monkeypatch):
    from pm4py.visualization.dfg import visualizer as dfg_visualizer

    from chatflow_miner.lib.event_log.view import EventLogView
    from chatflow_miner.lib.process_models import dfg as dfg_module
    from chatflow_miner.lib.process_models import view as view_module
    from chatflow_miner.lib.process_models.dfg import DFGModel
    from chatflow_miner.lib.process_models.view import ProcessModelView

    formatted = _build_formatted_log()
    event_log_view = EventLogView(formatted)
    model_view = ProcessModelView(log_view=event_log_view, model=DFGModel())

    dfg_tuple = ({("A", "B"): 1}, {"A": 1}, {"B": 1})

    discover_calls: list[pd.DataFrame] = []

    def fake_discover_dfg(df):
        discover_calls.append(df)
        return dfg_tuple

    convert_calls: list[pd.DataFrame] = []
    event_log_obj = {"event_log": True}

    def fake_convert(df):
        convert_calls.append(df)
        return event_log_obj

    viz_calls: list[dict[str, Any]] = []

    def fake_apply(dfg, log=None, parameters=None, variant=None, **kwargs):
        viz_calls.append(
            {
                "dfg": dfg,
                "log": log,
                "parameters": parameters,
                "variant": variant,
            }
        )
        return {"gviz": len(viz_calls)}

    monkeypatch.setattr(dfg_module.pm4py, "discover_dfg", fake_discover_dfg)
    monkeypatch.setattr(view_module.pm4py, "convert_to_event_log", fake_convert)
    monkeypatch.setattr(dfg_visualizer, "apply", fake_apply)

    first = model_view.to_graphviz(bgcolor="white")
    assert first == {"gviz": 1}
    assert len(discover_calls) == 1
    pd.testing.assert_frame_equal(discover_calls[0], formatted)
    assert viz_calls[0]["log"] is event_log_obj
    assert len(convert_calls) == 1
    pd.testing.assert_frame_equal(convert_calls[0], formatted)

    second = model_view.to_graphviz(bgcolor="white")
    assert second is first
    assert len(convert_calls) == 1
    assert len(viz_calls) == 1

    third = model_view.to_graphviz(rankdir="TB")
    assert third == {"gviz": 2}
    assert len(convert_calls) == 2
    assert viz_calls[-1]["log"] is event_log_obj
