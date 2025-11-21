import contextlib

import pandas as pd
import streamlit as st

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_EVENT_ID,
    COLUMN_START_TS,
)
from chatflow_miner.lib.event_log.view import EventLogView
from chatflow_miner.lib.state import initialize_session_state
from chatflow_miner.lib.ui.filters import streamlit_fragments


def test_filter_by_variants_persists_selection(monkeypatch):
    st.session_state.clear()
    initialize_session_state()

    df = pd.DataFrame(
        [
            {
                COLUMN_CASE_ID: "C1",
                COLUMN_ACTIVITY: "A",
                COLUMN_START_TS: "2024-01-01T00:00:00",
                COLUMN_EVENT_ID: 0,
            },
            {
                COLUMN_CASE_ID: "C1",
                COLUMN_ACTIVITY: "B",
                COLUMN_START_TS: "2024-01-01T00:01:00",
                COLUMN_EVENT_ID: 1,
            },
            {
                COLUMN_CASE_ID: "C2",
                COLUMN_ACTIVITY: "C",
                COLUMN_START_TS: "2024-01-02T00:00:00",
                COLUMN_EVENT_ID: 0,
            },
        ]
    )

    st.session_state.log_eventos = df
    st.session_state.load_info = {"file_name": "dummy.csv"}

    def fake_spinner(msg: str):
        return contextlib.nullcontext()

    def fake_data_editor(data, **_):
        edited = data.copy()
        edited["selected"] = [True] + [False] * (len(data) - 1)
        return edited

    monkeypatch.setattr(streamlit_fragments, "st", streamlit_fragments.st)
    monkeypatch.setattr(streamlit_fragments.st, "spinner", fake_spinner)
    monkeypatch.setattr(streamlit_fragments.st, "data_editor", fake_data_editor)

    base_view = EventLogView(base_df=df)
    filtered = streamlit_fragments.filter_by_variants(base_view, disabled=False)

    assert st.session_state.selected_variants == ["variant 1"]
    assert filtered is not None
    filtered_df = filtered.compute()
    assert set(filtered_df[COLUMN_CASE_ID]) == {"C1"}
