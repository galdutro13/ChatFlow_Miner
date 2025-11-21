from __future__ import annotations

import types

import pytest

from chatflow_miner.lib.conformance.token_replay import (
    aggregate_token_replay_results,
    apply_token_replay,
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


def make_pm4py_token_replay_stub(return_value):
    pm4py_module = types.ModuleType("pm4py")
    pm4py_module.__path__ = []

    algo_mod = types.ModuleType("pm4py.algo")
    conformance_mod = types.ModuleType("pm4py.algo.conformance")
    tokenreplay_mod = types.ModuleType("pm4py.algo.conformance.tokenreplay")
    algorithm_mod = types.ModuleType("pm4py.algo.conformance.tokenreplay.algorithm")

    def apply(log, net, im, fm):
        return return_value

    algorithm_mod.apply = apply

    xes_constants = types.SimpleNamespace(DEFAULT_NAME_KEY="concept:name")
    xes_module = types.ModuleType("pm4py.util.xes_constants")
    xes_module.DEFAULT_NAME_KEY = xes_constants.DEFAULT_NAME_KEY

    return {
        "pm4py": pm4py_module,
        "pm4py.algo": algo_mod,
        "pm4py.algo.conformance": conformance_mod,
        "pm4py.algo.conformance.tokenreplay": tokenreplay_mod,
        "pm4py.algo.conformance.tokenreplay.algorithm": algorithm_mod,
        "pm4py.util.xes_constants": xes_module,
    }


def test_apply_token_replay_forwards_arguments_and_returns():
    expected = [
        {"trace_fitness": 1.0, "missing_tokens": 0, "remaining_tokens": 0},
    ]
    mapping = make_pm4py_token_replay_stub(expected)

    with TempModulePatcher(mapping):
        results = apply_token_replay(log=[[]], net="n", im="i", fm="f")

    assert results == expected


def test_aggregate_token_replay_results_groups_by_variant():
    mapping = make_pm4py_token_replay_stub([])
    log = [
        [{"concept:name": "A"}, {"concept:name": "B"}],
        [{"concept:name": "A"}, {"concept:name": "C"}],
        [{"concept:name": "A"}, {"concept:name": "B"}],
    ]
    replay_results = [
        {
            "trace_fitness": 1.0,
            "missing_tokens": 0,
            "remaining_tokens": 0,
            "missing_marking": {},
            "remaining_marking": {},
        },
        {
            "trace_fitness": 0.5,
            "missing_tokens": 2,
            "remaining_tokens": 1,
            "missing_marking": {"p1": 1},
            "remaining_marking": {"p2": 1},
        },
        {
            "trace_fitness": 0.8,
            "missing_tokens": 1,
            "remaining_tokens": 0,
            "missing_marking": {"p1": 1},
            "remaining_marking": {},
        },
    ]

    with TempModulePatcher(mapping):
        df = aggregate_token_replay_results(log, replay_results)

    assert set(df.columns) == {
        "variant",
        "frequency",
        "missing_tokens",
        "remaining_tokens",
        "trace_fitness",
        "missing_activities",
        "remaining_activities",
    }

    ab_row = df[df["variant"] == "A -> B"].iloc[0]
    ac_row = df[df["variant"] == "A -> C"].iloc[0]

    assert ab_row["frequency"] == 2
    assert ab_row["missing_tokens"] == 1
    assert ab_row["trace_fitness"] == pytest.approx(0.9)
    assert ab_row["missing_activities"] == ["p1"]

    assert ac_row["frequency"] == 1
    assert ac_row["remaining_tokens"] == 1
    assert ac_row["remaining_activities"] == ["p2"]
