import importlib
import sys

from .test_dfg import TempModulePatcher, make_graphviz_stub, make_pm4py_stub


def test_resolve_model_supports_performance_alias_and_default():
    frequency_tuple = (
        {("A", "B"): 1},
        {"A": 1},
        {"B": 1},
    )
    pm4py_stub = make_pm4py_stub(frequency_tuple)
    graphviz_stub = make_graphviz_stub()

    mapping = {
        "pm4py": pm4py_stub,
        "pm4py.visualization": pm4py_stub.visualization,
        "pm4py.visualization.dfg": pm4py_stub.visualization.dfg,
        "pm4py.visualization.dfg.visualizer": pm4py_stub.visualization.dfg.visualizer,
        "graphviz": graphviz_stub,
    }

    with TempModulePatcher(mapping):
        sys.modules.pop("chatflow_miner.lib.process_models.dfg", None)
        sys.modules.pop("chatflow_miner.lib.process_models.ui", None)
        importlib.invalidate_caches()

        from chatflow_miner.lib.process_models.dfg import DFGModel, PerformanceDFGModel
        from chatflow_miner.lib.process_models.ui import _resolve_model

        default_model = _resolve_model(None)
        assert isinstance(default_model, DFGModel)

        perf_model = _resolve_model("performance-dfg")
        assert isinstance(perf_model, PerformanceDFGModel)

        perf_model_snake = _resolve_model("performance_dfg")
        assert isinstance(perf_model_snake, PerformanceDFGModel)

    sys.modules.pop("chatflow_miner.lib.process_models.dfg", None)
    sys.modules.pop("chatflow_miner.lib.process_models.ui", None)
    importlib.invalidate_caches()
