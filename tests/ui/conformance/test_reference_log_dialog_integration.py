from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from streamlit.testing.v1 import AppTest

from chatflow_miner.lib.ui.conformance import inputs


@pytest.fixture
def dummy_reference_log() -> io.BytesIO:
    buffer = io.BytesIO(
        b"case,activity,start,end\n1,A,2024-01-01,2024-01-01\n2,B,2024-01-02,2024-01-02\n"
    )
    buffer.name = "reference_log.csv"
    return buffer


@pytest.fixture
def discovery_app(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    upload_state: dict[str, io.BytesIO | None] = {"file": None}
    load_calls: dict[str, object] = {}

    real_file_uploader = inputs.st.file_uploader

    def fake_file_uploader(*args, **kwargs):
        if upload_state["file"] is not None:
            return upload_state["file"]
        return real_file_uploader(*args, **kwargs)

    def fake_load_dataset(file_obj, load_info):
        load_calls["file"] = file_obj
        load_calls["load_info"] = load_info
        return {"rows": 3}

    monkeypatch.setattr(inputs.st, "file_uploader", fake_file_uploader)
    monkeypatch.setattr(inputs, "load_dataset", fake_load_dataset)

    script = """
import streamlit as st
from chatflow_miner.lib.ui.conformance import inputs

inputs.render_button_fragment()
inputs._render_discovery_tab()
"""
    app = AppTest.from_string(script)
    return {"app": app, "upload_state": upload_state, "load_calls": load_calls}


@pytest.fixture
def variant_app(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    upload_state: dict[str, io.BytesIO | None] = {"file": None}
    load_calls: dict[str, object] = {}

    real_file_uploader = inputs.st.file_uploader

    def fake_file_uploader(*args, **kwargs):
        if upload_state["file"] is not None:
            return upload_state["file"]
        return real_file_uploader(*args, **kwargs)

    def fake_load_dataset(file_obj, load_info):
        load_calls["file"] = file_obj
        load_calls["load_info"] = load_info
        return {"rows": 5}

    monkeypatch.setattr(inputs.st, "file_uploader", fake_file_uploader)
    monkeypatch.setattr(inputs, "load_dataset", fake_load_dataset)

    script = """
import streamlit as st
from chatflow_miner.lib.ui.conformance import inputs

inputs.render_button_fragment()
inputs._render_variant_tab()
"""
    app = AppTest.from_string(script)
    return {"app": app, "upload_state": upload_state, "load_calls": load_calls}


def _state_get(app: AppTest, key: str, default: object | None = None) -> object | None:
    try:
        return app.session_state[key]
    except KeyError:
        return default


def _open_dialog(app: AppTest, key_suffix: str) -> None:
    app.run()
    app.button(key=f"open_reference_log_dialog_{key_suffix}").click()
    app.run()


def _upload_reference_log(app: AppTest, upload_state: dict[str, io.BytesIO | None], file_obj: io.BytesIO) -> None:
    file_obj.seek(0)
    upload_state["file"] = file_obj
    settled = False
    for _ in range(3):
        app.run()
        state = _state_get(app, "reference_log_state", {})
        if state.get("log") is not None:
            settled = True
            break
    # Render one additional frame after the upload-triggered rerun to ensure the dialog
    # is closed in the stabilised UI state.
    if settled:
        app.run()
    upload_state["file"] = None


def _reference_log_loaded(app: AppTest) -> bool:
    state = _state_get(app, "reference_log_state", {})
    return state.get("log") is not None


def _dialog_open_flag(app: AppTest, suffix: str) -> bool:
    return bool(_state_get(app, f"reference_log_dialog_open_{suffix}", False))


# If the open flag remains True or the upload dialog/file_uploader is still rendered after a
# successful upload, this directly reproduces the bug where reference_log_dialog_open_{suffix}
# is not cleared when the log is loaded.
def test_t01_integration_upload_clears_discovery_dialog_flag(
    discovery_app: dict[str, object], dummy_reference_log: io.BytesIO
) -> None:
    """
    Verify the transition from not loaded & dialog closed -> dialog open -> loaded & dialog closed.
    """
    app: AppTest = discovery_app["app"]  # type: ignore[assignment]
    upload_state: dict[str, io.BytesIO | None] = discovery_app["upload_state"]  # type: ignore[assignment]

    app.run()
    state = _state_get(app, "reference_log_state", {})
    assert _reference_log_loaded(app) is False
    assert state.get("log") is None
    assert state.get("name") is None
    assert state.get("info") is None

    _open_dialog(app, "discovery")

    assert _reference_log_loaded(app) is False
    assert _dialog_open_flag(app, "discovery") is True
    assert len(app.get("file_uploader")) == 1

    _upload_reference_log(app, upload_state, dummy_reference_log)

    assert _reference_log_loaded(app) is True
    assert _dialog_open_flag(app, "discovery") is False
    assert len(app.get("file_uploader")) == 0


# If immediately after removal the discovery upload dialog is visible or the flag is True,
# this confirms the stale True flag behaviour described in the bug report.
def test_t02_integration_remove_does_not_auto_open_discovery_dialog(
    discovery_app: dict[str, object], dummy_reference_log: io.BytesIO
) -> None:
    app: AppTest = discovery_app["app"]  # type: ignore[assignment]
    upload_state: dict[str, io.BytesIO | None] = discovery_app["upload_state"]  # type: ignore[assignment]

    _open_dialog(app, "discovery")
    _upload_reference_log(app, upload_state, dummy_reference_log)

    assert _reference_log_loaded(app) is True

    app.button(key="remove-reference-log").click()
    app.run()

    state = _state_get(app, "reference_log_state", {})
    assert state.get("log") is None
    assert state.get("name") is None
    assert state.get("info") is None
    assert _dialog_open_flag(app, "discovery") is False
    assert len(app.get("file_uploader")) == 0


# If, after removal, the variant upload dialog appears automatically or the flag remains True,
# this manifests the stale-dialog bug for the variant suffix.
def test_t03_integration_remove_does_not_auto_open_variant_dialog(
    variant_app: dict[str, object], dummy_reference_log: io.BytesIO
) -> None:
    app: AppTest = variant_app["app"]  # type: ignore[assignment]
    upload_state: dict[str, io.BytesIO | None] = variant_app["upload_state"]  # type: ignore[assignment]

    _open_dialog(app, "variant")
    assert _dialog_open_flag(app, "variant") is True
    assert len(app.get("file_uploader")) == 1

    _upload_reference_log(app, upload_state, dummy_reference_log)

    assert _reference_log_loaded(app) is True
    assert len(app.get("file_uploader")) == 0

    app.button(key="remove-reference-log").click()
    app.run()

    assert _reference_log_loaded(app) is False
    assert _dialog_open_flag(app, "variant") is False
    assert len(app.get("file_uploader")) == 0


# If the discovery dialog is visible or the open flag remains True after the upload-triggered
# rerun, this demonstrates the stale open-flag behaviour interacting with st.rerun().
def test_t06_edge_upload_rerun_no_dialog_flash(
    discovery_app: dict[str, object], dummy_reference_log: io.BytesIO
) -> None:
    app: AppTest = discovery_app["app"]  # type: ignore[assignment]
    upload_state: dict[str, io.BytesIO | None] = discovery_app["upload_state"]  # type: ignore[assignment]

    _open_dialog(app, "discovery")
    assert _dialog_open_flag(app, "discovery") is True
    assert _reference_log_loaded(app) is False
    assert len(app.get("file_uploader")) >= 1

    dummy_reference_log.seek(0)
    upload_state["file"] = dummy_reference_log

    app.run()

    assert _reference_log_loaded(app) is True
    assert _dialog_open_flag(app, "discovery") is False
    assert len(app.get("file_uploader")) == 0

    app.run()

    assert _reference_log_loaded(app) is True
    assert _dialog_open_flag(app, "discovery") is False
    assert len(app.get("file_uploader")) == 0


@pytest.mark.parametrize("suffix", ["discovery", "variant"])
def test_t07_integration_can_reopen_dialog_after_removal(
    suffix: str, dummy_reference_log: io.BytesIO, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    fixture_name = "discovery_app" if suffix == "discovery" else "variant_app"
    app_context: dict[str, object] = request.getfixturevalue(fixture_name)

    app: AppTest = app_context["app"]  # type: ignore[assignment]
    upload_state: dict[str, io.BytesIO | None] = app_context["upload_state"]  # type: ignore[assignment]

    _open_dialog(app, suffix)
    _upload_reference_log(app, upload_state, dummy_reference_log)

    app.button(key="remove-reference-log").click()
    app.run()

    assert _reference_log_loaded(app) is False
    assert _dialog_open_flag(app, suffix) is False

    app.button(key=f"open_reference_log_dialog_{suffix}").click()
    app.run()

    assert _dialog_open_flag(app, suffix) is True
    assert len(app.get("file_uploader")) == 1

    _upload_reference_log(app, upload_state, dummy_reference_log)

    assert _reference_log_loaded(app) is True
    assert _dialog_open_flag(app, suffix) is False
    assert len(app.get("file_uploader")) == 0
