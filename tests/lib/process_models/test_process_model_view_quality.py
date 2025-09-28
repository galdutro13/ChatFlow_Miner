import pandas as pd
import pm4py
import pytest
import sys
import importlib


def _sample_log_df() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01 00:00:00")
    data = {
        "CASE_ID": ["c1", "c1", "c1", "c2", "c2"],
        "EVENT_ID": [1, 2, 3, 4, 5],
        "ACTIVITY": ["Start", "Chat", "End", "Start", "End"],
        "START_TIMESTAMP": [
            base,
            base + pd.Timedelta(minutes=1),
            base + pd.Timedelta(minutes=2),
            base + pd.Timedelta(minutes=3),
            base + pd.Timedelta(minutes=4),
        ],
        "END_TIMESTAMP": [
            base + pd.Timedelta(seconds=30),
            base + pd.Timedelta(minutes=1, seconds=30),
            base + pd.Timedelta(minutes=2, seconds=30),
            base + pd.Timedelta(minutes=3, seconds=30),
            base + pd.Timedelta(minutes=4, seconds=30),
        ],
    }
    df = pd.DataFrame(data)
    formatted = pm4py.format_dataframe(
        df,
        case_id="CASE_ID",
        activity_key="ACTIVITY",
        timestamp_key="END_TIMESTAMP",
        start_timestamp_key="START_TIMESTAMP",
    )
    return formatted


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("chatflow_miner.lib.process_models.dfg", "DFGModel"),
        ("chatflow_miner.lib.process_models.petri_net", "PetriNetModel"),
    ],
)
def test_quality_metrics_are_computed_and_cached(module_path, class_name):
    df = _sample_log_df()
    from chatflow_miner.lib.event_log.view import EventLogView
    from chatflow_miner.lib.process_models.view import ProcessModelView

    log_view = EventLogView(df)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    view = ProcessModelView(log_view=log_view, model=model_cls())

    metrics = view.quality_metrics()

    expected_keys = {"fitness", "precision", "generalization", "simplicity"}
    assert expected_keys.issubset(metrics.keys())

    for key in expected_keys:
        value = metrics[key]
        assert value is None or isinstance(value, float)

    cached_metrics = view.quality_metrics()
    assert cached_metrics is metrics

    # Limpa o módulo para que testes que usam stubs consigam reimportar com monkeypatch
    sys.modules.pop(module_path, None)
    importlib.invalidate_caches()
