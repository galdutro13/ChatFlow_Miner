import types
import pandas as pd
class TempModulePatcher:
    """
    Context manager to temporarily inject/patch entries in sys.modules.
    Avoids importing heavy deps like pm4py/graphviz in unit tests.
    """
    def __init__(self, mapping):
        self.mapping = mapping
        self._originals = {}

    def __enter__(self):
        import sys
        for name, module in self.mapping.items():
            self._originals[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys
        for name in self.mapping.keys():
            if self._originals[name] is None and name in sys.modules:
                del sys.modules[name]
            else:
                sys.modules[name] = self._originals[name]


def make_pm4py_stub(discover_return, performance_return=None):
    pm4py = types.ModuleType("pm4py")
    # Mark as a package so nested imports are permitted
    pm4py.__path__ = []

    def discover_dfg(df):  # signature compatible with usage in DFGModel
        return discover_return

    pm4py.discover_dfg = discover_dfg
    
    def discover_performance_dfg(df):
        if performance_return is not None:
            return performance_return
        return discover_return

    pm4py.discover_performance_dfg = discover_performance_dfg
    def convert_to_event_log(df):
        return {"converted_df": df}

    pm4py.convert_to_event_log = convert_to_event_log

    # Also provide visualization dfg module structure for to_graphviz
    vis_dfg = types.ModuleType("pm4py.visualization.dfg")
    visualizer = types.ModuleType("pm4py.visualization.dfg.visualizer")

    class _Variants:
        class _FrequencyVariant:
            value = types.SimpleNamespace(Parameters=types.SimpleNamespace(
                FORMAT="FORMAT",
                START_ACTIVITIES="START_ACTIVITIES",
                END_ACTIVITIES="END_ACTIVITIES",
                TIMESTAMP_KEY="TIMESTAMP_KEY",
                START_TIMESTAMP_KEY="START_TIMESTAMP_KEY",
            ))

        FREQUENCY = _FrequencyVariant()
        class _PerformanceVariant:
            value = types.SimpleNamespace(Parameters=types.SimpleNamespace(
                FORMAT="FORMAT",
                START_ACTIVITIES="START_ACTIVITIES",
                END_ACTIVITIES="END_ACTIVITIES",
                TIMESTAMP_KEY="TIMESTAMP_KEY",
                START_TIMESTAMP_KEY="START_TIMESTAMP_KEY",
            ))

        PERFORMANCE = _PerformanceVariant()

    def apply_stub(dfg, log=None, parameters=None, variant=None, **kwargs):
        return types.SimpleNamespace(
            kind="gviz",
            dfg=dfg,
            params=parameters,
            variant=variant,
            log=log,
        )

    visualizer.Variants = _Variants
    visualizer.apply = apply_stub

    # Expose modules in package hierarchy
    pm4py.visualization = types.ModuleType("pm4py.visualization")
    pm4py.visualization.dfg = vis_dfg
    vis_dfg.visualizer = visualizer

    return pm4py


def make_graphviz_stub():
    graphviz = types.ModuleType("graphviz")
    Digraph = type("Digraph", (), {})
    graphviz.Digraph = Digraph
    return graphviz


def test_dfgmodel_compute_returns_pm4py_discover_output():
    df = pd.DataFrame({"CASE_ID": [1, 1, 2], "ACTIVITY": ["A", "B", "C"]})
    expected = (
        {("A", "B"): 1, ("B", "C"): 1},
        {"A": 1},
        {"C": 1},
    )

    pm4py_stub = make_pm4py_stub(expected)
    graphviz_stub = make_graphviz_stub()

    mapping = {
        "pm4py": pm4py_stub,
        "pm4py.visualization": pm4py_stub.visualization,
        "pm4py.visualization.dfg": pm4py_stub.visualization.dfg,
        "pm4py.visualization.dfg.visualizer": pm4py_stub.visualization.dfg.visualizer,
        "graphviz": graphviz_stub,
    }
    with TempModulePatcher(mapping):
        from chatflow_miner.lib.process_models.dfg import DFGModel
        model = DFGModel()
        result = model.compute(df)

    assert result == expected


def test_dfgmodel_to_graphviz_returns_visualizer_output_structure():
    df = pd.DataFrame({"CASE_ID": [1, 1, 2], "ACTIVITY": ["A", "B", "C"]})
    dfg_tuple = (
        {("A", "B"): 1, ("B", "C"): 1},
        {"A": 1},
        {"C": 1},
    )

    pm4py_stub = make_pm4py_stub(dfg_tuple)
    graphviz_stub = make_graphviz_stub()

    mapping = {
        "pm4py": pm4py_stub,
        "pm4py.visualization": pm4py_stub.visualization,
        "pm4py.visualization.dfg": pm4py_stub.visualization.dfg,
        "pm4py.visualization.dfg.visualizer": pm4py_stub.visualization.dfg.visualizer,
        "graphviz": graphviz_stub,
    }
    with TempModulePatcher(mapping):
        from chatflow_miner.lib.process_models.dfg import DFGModel
        model = DFGModel()
        computed = model.compute(df)
        gviz = model.to_graphviz(computed, bgcolor="black", rankdir="TB", max_num_edges=5)

    # gviz should be the object returned by our stub apply()
    assert getattr(gviz, "kind", None) == "gviz"
    # parameters were threaded through
    params = getattr(gviz, "params", {})
    assert params.get("bgcolor") == "black"
    assert params.get("rankdir") == "TB"
    assert params.get("maxNoOfEdgesInDiagram") == 5

    # Ensure variant provided is FREQUENCY and log defaults to None when não fornecido
    variant = getattr(gviz, "variant", None)
    assert variant is not None
    assert getattr(gviz, "log", "__missing__") is None
