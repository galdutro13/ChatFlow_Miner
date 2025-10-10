import importlib
import sys

import pandas as pd

from .test_dfg import TempModulePatcher, make_graphviz_stub, make_pm4py_stub


def test_performance_dfgmodel_compute_uses_performance_discovery():
    df = pd.DataFrame({"CASE_ID": [1, 1, 2], "ACTIVITY": ["A", "B", "C"]})
    frequency_tuple = (
        {("A", "B"): 1},
        {"A": 1},
        {"B": 1},
    )
    performance_tuple = (
        {("A", "B"): {"performance": 1.5}},
        {"A": {"sojourn": 0.2}},
        {"B": {"sojourn": 0.1}},
    )

    pm4py_stub = make_pm4py_stub(frequency_tuple, performance_return=performance_tuple)
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
        importlib.invalidate_caches()

        from chatflow_miner.lib.process_models.dfg import PerformanceDFGModel

        model = PerformanceDFGModel()
        result = model.compute(df)

    assert result == performance_tuple
    sys.modules.pop("chatflow_miner.lib.process_models.dfg", None)
    importlib.invalidate_caches()


def test_performance_dfgmodel_to_graphviz_uses_performance_variant():
    df = pd.DataFrame({"CASE_ID": [1, 1, 2], "ACTIVITY": ["A", "B", "C"]})
    performance_tuple = (
        {("A", "B"): {"performance": 1.5}},
        {"A": {"sojourn": 0.2}},
        {"B": {"sojourn": 0.1}},
    )

    pm4py_stub = make_pm4py_stub(performance_tuple, performance_return=performance_tuple)
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
        importlib.invalidate_caches()

        from chatflow_miner.lib.process_models.dfg import PerformanceDFGModel

        model = PerformanceDFGModel()
        computed = model.compute(df)
        gviz = model.to_graphviz(computed, bgcolor="navy", rankdir="TB", max_num_edges=3)

    assert getattr(gviz, "kind", None) == "gviz"
    params = getattr(gviz, "params", {})
    assert params.get("bgcolor") == "navy"
    assert params.get("rankdir") == "TB"
    assert params.get("maxNoOfEdgesInDiagram") == 3
    variant = getattr(gviz, "variant", None)
    assert variant is pm4py_stub.visualization.dfg.visualizer.Variants.PERFORMANCE
    assert getattr(gviz, "log", "__missing__") is None
    sys.modules.pop("chatflow_miner.lib.process_models.dfg", None)
    importlib.invalidate_caches()
