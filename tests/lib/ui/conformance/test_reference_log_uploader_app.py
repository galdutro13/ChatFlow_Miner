from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from streamlit.testing.v1 import AppTest

from chatflow_miner.lib.ui.conformance import inputs


@pytest.fixture
def dummy_upload() -> io.BytesIO:
    buffer = io.BytesIO(b"case,activity,start,end\n1,A,2024-01-01,2024-01-01")
    buffer.name = "reference_log.csv"
    return buffer


@pytest.fixture
def uploader_script() -> Callable[[str], str]:
    def _script(key_suffix: str = "default") -> str:
        return f'''
import streamlit as st
from chatflow_miner.lib.ui.conformance import inputs

inputs._render_reference_log_uploader(key_suffix={key_suffix!r})
'''

    return _script


@pytest.fixture
def dialog_script() -> Callable[..., str]:
    def _script(*, key_suffix: str = "dialog", with_banner: bool = False) -> str:
        banner = "inputs.render_button_fragment()\n" if with_banner else ""
        return f'''
import streamlit as st
from chatflow_miner.lib.ui.conformance import inputs

{banner}inputs._render_reference_log_dialog_launcher(key_suffix={key_suffix!r})
'''

    return _script


def test_reference_log_uploader_renders_controls(uploader_script: Callable[[str], str]):
    app = AppTest.from_string(uploader_script())

    app.run()

    assert app.info[0].value.startswith(
        "Envie um log de referência (CSV ou XES) para minerar o modelo normativo."
    )
    separators = app.selectbox[0].options
    assert separators == [",", ";", "\t"]
    assert app.selectbox[0].value == ","
    # File uploader elements are not yet specialized in AppTest; ensure they render.
    assert len(app.get("file_uploader")) == 1


def test_reference_log_uploader_tracks_separator_before_upload(
    monkeypatch: pytest.MonkeyPatch, uploader_script: Callable[[str], str], dummy_upload: io.BytesIO
):
    upload_state: dict[str, io.BytesIO | None] = {"file": None}
    calls: dict[str, object] = {}
    rerun_called: dict[str, bool] = {"called": False}

    def fake_file_uploader(*_args, **_kwargs):
        return upload_state["file"]

    def fake_load_dataset(file_obj, load_info):
        calls["file"] = file_obj
        calls["load_info"] = load_info
        return {"rows": 2}

    def fake_rerun(*_args, **_kwargs):
        rerun_called["called"] = True

    monkeypatch.setattr(inputs.st, "file_uploader", fake_file_uploader)
    monkeypatch.setattr(inputs, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(inputs.st, "rerun", fake_rerun)

    app = AppTest.from_string(uploader_script())

    # Initial render uses the default separator.
    app.run()
    assert app.selectbox[0].value == ","

    # User changes the separator before uploading a file.
    app.selectbox[0].select(";")
    app.run()
    assert app.selectbox[0].value == ";"

    # Upload happens on a later rerun, using the chosen separator.
    upload_state["file"] = dummy_upload
    app.run()

    state = app.session_state["reference_log_state"]
    assert state == {
        "log": {"rows": 2},
        "name": "reference_log.csv",
        "info": {"sep": ";", "file_name": "reference_log.csv"},
    }
    assert rerun_called["called"] is True
    assert calls["file"] is dummy_upload
    assert calls["load_info"] == {"sep": ";", "file_name": "reference_log.csv"}


def test_reference_log_uploader_surfaces_errors_without_persisting_state(
    monkeypatch: pytest.MonkeyPatch, uploader_script: Callable[[str], str], dummy_upload: io.BytesIO
):
    errors: list[str] = []

    def fake_file_uploader(*_args, **_kwargs):
        return dummy_upload

    def fake_error(message, *_, **__):
        errors.append(str(message))

    def fake_load_dataset(*_args, **_kwargs):
        inputs.st.error("Erro ao carregar arquivo")
        inputs.st.stop()

    monkeypatch.setattr(inputs.st, "file_uploader", fake_file_uploader)
    monkeypatch.setattr(inputs.st, "error", fake_error)
    monkeypatch.setattr(inputs, "load_dataset", fake_load_dataset)

    app = AppTest.from_string(uploader_script())

    app.run()

    assert errors == ["Erro ao carregar arquivo"]
    assert "reference_log_state" not in app.session_state


def test_reference_log_uploader_maintains_independent_keys(
    uploader_script: Callable[[str], str]
):
    app = AppTest.from_string(
        """
import streamlit as st
from chatflow_miner.lib.ui.conformance import inputs

inputs._render_reference_log_uploader(key_suffix="primary")
inputs._render_reference_log_uploader(key_suffix="dialog")
"""
    )

    app.run()

    primary_sep = app.selectbox(key="reference_log_sep_primary")
    dialog_sep = app.selectbox(key="reference_log_sep_dialog")

    primary_sep.select(";")
    dialog_sep.select("\t")
    app.run()

    assert primary_sep.value == ";"
    assert dialog_sep.value == "\t"
    assert app.session_state["reference_log_sep_primary"] == ";"
    assert app.session_state["reference_log_sep_dialog"] == "\t"


def test_reference_log_dialog_opens_and_renders_controls(dialog_script: Callable[..., str]):
    # Given a launcher that defers rendering until the dialog is opened
    app = AppTest.from_string(dialog_script())

    app.run()
    app.button(key="open_reference_log_dialog_dialog").click()

    # When the dialog is opened on the rerun
    app.run()

    # Then the uploader content appears inside the modal context
    assert app.info[0].value.startswith(
        "Envie um log de referência (CSV ou XES) para minerar o modelo normativo."
    )
    dialog_select = app.selectbox(key="reference_log_sep_dialog")
    assert dialog_select.options == [",", ";", "\t"]
    assert dialog_select.value == ","
    assert len(app.get("file_uploader")) == 1


def test_reference_log_dialog_upload_flow_updates_banner(
    monkeypatch: pytest.MonkeyPatch,
    dialog_script: Callable[..., str],
    dummy_upload: io.BytesIO,
):
    # Given a dialog-backed uploader reachable from the banner area
    upload_state: dict[str, io.BytesIO | None] = {"file": None}
    calls: dict[str, object] = {}
    rerun_called: dict[str, bool] = {"called": False}

    def fake_file_uploader(*_args, **_kwargs):
        return upload_state["file"]

    def fake_load_dataset(file_obj, load_info):
        calls["file"] = file_obj
        calls["load_info"] = load_info
        return {"rows": 4}

    def fake_rerun(*_args, **_kwargs):
        rerun_called["called"] = True

    monkeypatch.setattr(inputs.st, "file_uploader", fake_file_uploader)
    monkeypatch.setattr(inputs, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(inputs.st, "rerun", fake_rerun)

    app = AppTest.from_string(dialog_script(with_banner=True))

    app.run()
    app.button(key="open_reference_log_dialog_dialog").click()
    app.run()

    # When the user tweaks the separator and uploads on a subsequent rerun
    app.selectbox(key="reference_log_sep_dialog").select("\t")
    upload_state["file"] = dummy_upload
    app.run()

    # The rerun requested by the uploader is emulated by another explicit run
    app.run()

    state = app.session_state["reference_log_state"]
    assert state == {
        "log": {"rows": 4},
        "name": "reference_log.csv",
        "info": {"sep": "\t", "file_name": "reference_log.csv"},
    }
    assert rerun_called["called"] is True
    assert calls["file"] is dummy_upload
    assert calls["load_info"] == {"sep": "\t", "file_name": "reference_log.csv"}

    banner = app.caption[0].value
    assert banner == "Usando log de referência: reference_log.csv"
    assert app.session_state["reference_log_dialog_open_dialog"] is False


def test_reference_log_dialog_closure_without_upload(dialog_script: Callable[..., str]):
    # Given the dialog can be opened and closed without selecting a file
    app = AppTest.from_string(dialog_script())

    app.run()
    app.button(key="open_reference_log_dialog_dialog").click()
    app.run()

    app.button(key="close_reference_log_dialog_dialog").click()
    app.run()

    # Then the modal flag is cleared and no reference log is persisted
    assert app.session_state["reference_log_dialog_open_dialog"] is False
    assert app.session_state["reference_log_state"] == {
        "log": None,
        "name": None,
        "info": None,
    }
