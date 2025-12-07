from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from chatflow_miner.lib.ui.conformance import inputs


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - attribute access should mirror dict
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def clear(self) -> None:  # pragma: no cover - mirrors dict but keeps attribute access usable
        super().clear()


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> FakeSessionState:
    session = FakeSessionState()
    monkeypatch.setattr(inputs.st, "session_state", session)
    return session


# If any reference_log_dialog_open_* flag remains True after _clear_reference_log returns, the
# stale-flag condition that reopens the dialog on the next run is exposed.
def test_t04_unit_clear_reference_log_resets_state_and_dialog_flags(
    fake_session: FakeSessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_session["reference_log_state"] = {
        "log": {"rows": 10},
        "name": "uploaded.csv",
        "info": {"sep": ";"},
    }
    fake_session["reference_log_dialog_open_discovery"] = True
    fake_session["reference_log_dialog_open_variant"] = True
    fake_session["normative_model"] = {
        "net": "net",
        "initial_marking": "im",
        "final_marking": "fm",
        "source": "upload",
    }

    monkeypatch.setattr(inputs.st, "rerun", lambda *_, **__: None)

    inputs._clear_reference_log()

    assert fake_session["reference_log_state"] == {
        "log": None,
        "name": None,
        "info": None,
    }
    dialog_flags = {
        key: value for key, value in fake_session.items() if key.startswith("reference_log_dialog_open_")
    }
    assert dialog_flags == {key: False for key in dialog_flags}
    assert fake_session["normative_model"] == {
        "net": None,
        "initial_marking": None,
        "final_marking": None,
        "source": None,
    }


# When the dialog runs in a state where the reference log is already loaded, it should
# immediately clear the open flag and avoid trying to upload again.
def test_t05_unit_upload_dialog_close_if_loaded_resets_flag(
    fake_session: FakeSessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_session["reference_log_state"] = {
        "log": {"rows": 2},
        "name": "reference_log.csv",
        "info": {"file_name": "reference_log.csv"},
    }
    fake_session["reference_log_dialog_open_discovery"] = True
    monkeypatch.setattr(inputs.st, "button", lambda *_, **__: False)
    monkeypatch.setattr(inputs.st, "file_uploader", lambda *_, **__: None)
    monkeypatch.setattr(inputs.st, "rerun", lambda *_, **__: None)

    script = """
import streamlit as st
from chatflow_miner.lib.ui.conformance import inputs

inputs._reference_log_dialog(
    key_suffix="discovery", open_state_key="reference_log_dialog_open_discovery"
)
"""
    app = AppTest.from_string(script)
    app.run()

    assert fake_session["reference_log_dialog_open_discovery"] is False
    assert fake_session["reference_log_state"]["log"] == {"rows": 2}
