import pandas as pd
import streamlit as st

from chatflow_miner.app import dashboard
from chatflow_miner.lib.state import initialize_session_state, set_log_eventos


def test_initial_toast_prompts_for_log_upload(monkeypatch):
    st.session_state.clear()
    initialize_session_state()

    messages: list[str] = []
    monkeypatch.setattr(dashboard.st, "toast", lambda message, **_: messages.append(str(message)))

    dashboard.maybe_show_discovery_toast()
    dashboard.maybe_show_discovery_toast()

    assert any("novo log de eventos" in msg for msg in messages)
    assert len(messages) == 1


def test_ready_toast_after_new_log_load(monkeypatch):
    st.session_state.clear()
    initialize_session_state()

    messages: list[str] = []
    monkeypatch.setattr(dashboard.st, "toast", lambda message, **_: messages.append(str(message)))

    dashboard.maybe_show_discovery_toast()

    set_log_eventos(pd.DataFrame(), {"file_name": "teste.csv"})

    dashboard.maybe_show_discovery_toast()

    assert any("novo log de eventos" in msg for msg in messages)
    assert any("Pronto para gerar ou visualizar modelos" in msg for msg in messages)
    assert messages[-1] == "Pronto para gerar ou visualizar modelos"
