from __future__ import annotations

import pytest

from chatflow_miner.lib.ui.conformance.inputs import (
    _build_synthetic_log_from_variants,
    _normalize_variant_activities,
)


def test_normalize_variant_from_trace_and_manual_entries():
    pm4py = pytest.importorskip("pm4py")
    from pm4py.objects.log.obj import Event, Trace

    trace_variant = Trace([Event({"concept:name": "A"}), Event({"concept:name": "B"})])
    manual_variant = ["C", "D"]

    log = _build_synthetic_log_from_variants([trace_variant, manual_variant])

    assert len(log) == 2
    assert [event[pm4py.util.xes_constants.DEFAULT_NAME_KEY] for event in log[0]] == [
        "A",
        "B",
    ]
    assert [event[pm4py.util.xes_constants.DEFAULT_NAME_KEY] for event in log[1]] == [
        "C",
        "D",
    ]


def test_normalize_variant_activities_handles_missing_concept_name():
    variant = [{"other": "X"}, "Y"]

    normalized = _normalize_variant_activities(variant)

    assert normalized == ["{'other': 'X'}", "Y"]
