from __future__ import annotations

import types

import pytest

from chatflow_miner.lib.conformance.alignments import (
    aggregate_alignment_results,
    apply_alignments,
)


class TempModulePatcher:
    def __init__(self, mapping: dict[str, types.ModuleType]):
        self.mapping = mapping
        self._originals: dict[str, types.ModuleType | None] = {}

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


def make_pm4py_alignments_stub(
    return_value, *, include_petri_net: bool = True, include_legacy: bool = False
):
    pm4py_module = types.ModuleType("pm4py")
    pm4py_module.__path__ = []

    algo_mod = types.ModuleType("pm4py.algo")
    conformance_mod = types.ModuleType("pm4py.algo.conformance")
    alignments_mod = types.ModuleType("pm4py.algo.conformance.alignments")
    petri_net_mod = types.ModuleType("pm4py.algo.conformance.alignments.petri_net")
    algorithm_mod = types.ModuleType(
        "pm4py.algo.conformance.alignments.petri_net.algorithm"
    )

    def apply(log, net, im, fm):
        return return_value

    algorithm_mod.apply = apply

    xes_module = types.ModuleType("pm4py.util.xes_constants")
    xes_module.DEFAULT_NAME_KEY = "concept:name"

    mapping = {
        "pm4py": pm4py_module,
        "pm4py.algo": algo_mod,
        "pm4py.algo.conformance": conformance_mod,
        "pm4py.algo.conformance.alignments": alignments_mod,
        "pm4py.util.xes_constants": xes_module,
    }

    if include_petri_net:
        mapping.update(
            {
                "pm4py.algo.conformance.alignments.petri_net": petri_net_mod,
                "pm4py.algo.conformance.alignments.petri_net.algorithm": algorithm_mod,
            }
        )

    if include_legacy:
        mapping["pm4py.algo.conformance.alignments.algorithm"] = algorithm_mod

    return mapping


def test_apply_alignments_forwards_arguments_and_returns():
    expected = [
        {"fitness": 1.0, "cost": 0},
    ]
    mapping = make_pm4py_alignments_stub(expected)

    with TempModulePatcher(mapping):
        results = apply_alignments(log=[[]], net="n", im="i", fm="f")

    assert results == expected


def test_apply_alignments_falls_back_to_legacy_module():
    expected = [{"fitness": 0.5}]
    mapping = make_pm4py_alignments_stub(
        expected, include_petri_net=False, include_legacy=True
    )

    with TempModulePatcher(mapping):
        results = apply_alignments(log=[[]], net="n", im="i", fm="f")

    assert results == expected


def test_aggregate_alignment_results_counts_moves_and_costs():
    mapping = make_pm4py_alignments_stub([])
    log = [
        [{"concept:name": "A"}, {"concept:name": "B"}],
        [{"concept:name": "A"}, {"concept:name": "C"}],
        [{"concept:name": "A"}, {"concept:name": "B"}],
    ]
    alignment_results = [
        {
            "fitness": 0.9,
            "cost": 1.0,
            "alignment": [(("A", "A"), "sync_move"), (("B", ">>"), "model_move")],
        },
        {
            "fitness": 0.5,
            "cost": 3.0,
            "alignment": [(("A", "A"), "sync_move"), ((">>", "C"), "log_move")],
        },
        {
            "fitness": 0.8,
            "cost": 2.0,
            "alignment": [(("A", "A"), "sync_move"), (("B", "B"), "sync_move")],
        },
    ]

    with TempModulePatcher(mapping):
        df = aggregate_alignment_results(log, alignment_results)

    assert set(df.columns) == {
        "variant",
        "frequency",
        "fitness",
        "cost",
        "n_log_moves",
        "n_model_moves",
        "n_sync_moves",
    }

    ab_row = df[df["variant"] == "A -> B"].iloc[0]
    ac_row = df[df["variant"] == "A -> C"].iloc[0]

    assert ab_row["frequency"] == 2
    assert ab_row["fitness"] == pytest.approx(0.85)
    assert ab_row["cost"] == pytest.approx(1.5)
    assert ab_row["n_model_moves"] == 1
    assert ab_row["n_sync_moves"] == 3

    assert ac_row["n_log_moves"] == 1
    assert ac_row["cost"] == 3.0
