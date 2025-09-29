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
        "soundness",
    }
    for value in metrics.values():
        assert value is None or isinstance(value, (float, bool))

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

    assert metrics["soundness"] is None
    assert metrics["fitness"] is None
    assert metrics["precision"] is None
    assert metrics["generalization"] is None
    assert metrics["simplicity"] is None or isinstance(metrics["simplicity"], float)


def test_quality_metrics_preserves_boolean_values():
    from chatflow_miner.lib.process_models.base import BaseProcessModel
    from chatflow_miner.lib.process_models.view import ProcessModelView

    class _BoolModel(BaseProcessModel):
        def compute(self, df):
            return object()

        def quality_metrics(self, df, model):
            return {
                "fitness": 0.123,
                "precision": 0.456,
                "generalization": 0.789,
                "simplicity": 0.321,
                "soundness": True,
            }

    dummy_df = pd.DataFrame({})
    view = ProcessModelView(log_view=dummy_df, model=_BoolModel())

    metrics = view.quality_metrics()

    assert metrics["soundness"] is True
    assert isinstance(metrics["fitness"], float)

