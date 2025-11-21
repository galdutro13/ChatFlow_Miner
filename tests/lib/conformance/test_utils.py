from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from chatflow_miner.lib.conformance import utils


def test_ensure_marking_fallback_to_petri_net(monkeypatch):
    """ensure_marking_obj should import Marking from petri_net when petri path missing."""

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "pm4py.objects.petri.petrinet":
            raise ModuleNotFoundError(name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    # Force find_spec to behave normally for existing modules
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda module: None if module == "pm4py.objects.petri.petrinet" else real_find_spec(module),
    )

    net = SimpleNamespace(places={"p1", "p2"})
    marking = utils.ensure_marking_obj(net, {"p1": 1, "p3": 2})

    assert hasattr(marking, "__getitem__")
    assert marking["p1"] == 1
    # place not in net.places is still accepted via name fallback
    assert marking["p3"] == 2
    assert marking.__class__.__name__ == "Marking"


def test_ensure_marking_passthrough_marking(monkeypatch):
    petrinet_mod = importlib.import_module("pm4py.objects.petri_net.utils.petri_utils")
    Marking = getattr(petrinet_mod, "Marking")
    net = SimpleNamespace(places={"p1"})
    existing = Marking()
    existing["p1"] = 1

    assert utils.ensure_marking_obj(net, existing) is existing


@pytest.mark.parametrize("invalid", ["abc", 123, ["list"]])
def test_ensure_marking_invalid_types_raise(invalid):
    net = SimpleNamespace(places=set())
    with pytest.raises(TypeError):
        utils.ensure_marking_obj(net, invalid)
