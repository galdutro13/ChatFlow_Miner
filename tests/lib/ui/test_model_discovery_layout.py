import streamlit as st

from chatflow_miner.lib.state import initialize_session_state
from chatflow_miner.lib.ui.dashboard.model_discovery import model_discovery


def test_model_discovery_shows_empty_state_when_no_model(monkeypatch):
    st.session_state.clear()
    initialize_session_state()

    class DummyPlaceholder:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, msg, **kwargs):  # noqa: ANN001
            self.messages.append(str(msg))

    dummy = DummyPlaceholder()

    monkeypatch.setattr(model_discovery.__globals__["st"], "empty", lambda: dummy)

    model_discovery(disabled=False)

    assert any("Gerar modelo" in msg for msg in dummy.messages)
