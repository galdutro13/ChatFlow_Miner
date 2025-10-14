import types

import pandas as pd


class TempModulePatcher:
    """Context manager para injetar stubs em ``sys.modules`` temporariamente."""

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

        for name in self.mapping:
            original = self._originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def make_pm4py_stub(expected_model, *, expected_rankdir=None):
    pm4py = types.ModuleType("pm4py")
    pm4py.__path__ = []

    pm4py.discover_petri_net_inductive = lambda df, **kwargs: expected_model

    algo_mod = types.ModuleType("pm4py.algo")
    discovery_mod = types.ModuleType("pm4py.algo.discovery")
    inductive_mod = types.ModuleType("pm4py.algo.discovery.inductive")
    algorithm_mod = types.ModuleType("pm4py.algo.discovery.inductive.algorithm")

    def apply_stub(df, **kwargs):
        return expected_model

    algorithm_mod.apply = apply_stub

    vis_pkg = types.ModuleType("pm4py.visualization")
    petri_net_mod = types.ModuleType("pm4py.visualization.petri_net")
    visualizer_mod = types.ModuleType("pm4py.visualization.petri_net.visualizer")

    parameters_ns = types.SimpleNamespace(
        FORMAT="FORMAT",
        RANKDIR="set_rankdir",
    )

    class _Variants:
        class _WoDecoration:
            value = types.SimpleNamespace(
                Parameters=parameters_ns,
            )

        WO_DECORATION = _WoDecoration()

    def visualizer_apply(net, initial_marking, final_marking, parameters, variant):
        if expected_rankdir is not None:
            assert parameters.get(parameters_ns.RANKDIR) == expected_rankdir
            assert "rankdir" not in parameters
        return types.SimpleNamespace(
            kind="gviz",
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
            params=parameters,
            variant=variant,
        )

    visualizer_mod.Variants = _Variants
    visualizer_mod.apply = visualizer_apply

    pm4py.algo = algo_mod
    pm4py.visualization = vis_pkg

    algo_mod.discovery = discovery_mod
    discovery_mod.inductive = inductive_mod
    inductive_mod.algorithm = algorithm_mod

    vis_pkg.petri_net = petri_net_mod
    petri_net_mod.visualizer = visualizer_mod

    return {
        "pm4py": pm4py,
        "pm4py.algo": algo_mod,
        "pm4py.algo.discovery": discovery_mod,
        "pm4py.algo.discovery.inductive": inductive_mod,
        "pm4py.algo.discovery.inductive.algorithm": algorithm_mod,
        "pm4py.visualization": vis_pkg,
        "pm4py.visualization.petri_net": petri_net_mod,
        "pm4py.visualization.petri_net.visualizer": visualizer_mod,
    }


def test_petrinetmodel_compute_returns_inductive_output():
    df = pd.DataFrame({"CASE_ID": ["c1"], "ACTIVITY": ["A"]})
    expected = (
        types.SimpleNamespace(name="net"),
        types.SimpleNamespace(marking="i"),
        types.SimpleNamespace(marking="f"),
    )

    mapping = make_pm4py_stub(expected)

    with TempModulePatcher(mapping):
        from chatflow_miner.lib.process_models.petri_net import PetriNetModel

        model = PetriNetModel()
        result = model.compute(df)

    assert result == expected


def test_petrinetmodel_to_graphviz_forwards_kwargs():
    df = pd.DataFrame({"CASE_ID": ["c1"], "ACTIVITY": ["A"]})
    expected_model = (
        types.SimpleNamespace(name="net"),
        types.SimpleNamespace(marking="i"),
        types.SimpleNamespace(marking="f"),
    )

    mapping = make_pm4py_stub(expected_model, expected_rankdir="TB")

    with TempModulePatcher(mapping):
        from chatflow_miner.lib.process_models.petri_net import PetriNetModel

        model = PetriNetModel()
        computed = model.compute(df)
        gviz = model.to_graphviz(
            computed,
            bgcolor="black",
            rankdir="TB",
            format="png",
            extra_option=123,
        )

    assert getattr(gviz, "kind", None) == "gviz"
    params = getattr(gviz, "params", {})
    assert params.get("bgcolor") == "black"
    assert params.get("set_rankdir") == "TB"
    assert "rankdir" not in params
    assert params.get("FORMAT") == "png"
    assert params.get("extra_option") == 123
