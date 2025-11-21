import datetime as dt

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


def test_temporal_filter_uses_date_objects(monkeypatch):
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
                COLUMN_CASE_ID: "C2",
                COLUMN_ACTIVITY: "B",
                COLUMN_START_TS: "2024-01-02T00:00:00",
                COLUMN_EVENT_ID: 1,
            },
        ]
    )

    st.session_state.log_eventos = df
    st.session_state.load_info = {"file_name": "dummy.csv"}

    captured = {}

    def fake_slider(label, min_value, max_value, value, **kwargs):  # noqa: ANN001
        captured["label"] = label
        assert isinstance(value[0], dt.date)
        assert isinstance(value[1], dt.date)
        assert isinstance(min_value, dt.date)
        assert isinstance(max_value, dt.date)
        return value

    monkeypatch.setattr(streamlit_fragments.st, "slider", fake_slider)
    monkeypatch.setattr(streamlit_fragments.st, "metric", lambda *args, **kwargs: None)

    base_view = EventLogView(base_df=df)
    filtered_view = streamlit_fragments.temporal_filter(base_view, disabled=False)

    assert filtered_view is not None
    filtered_df = filtered_view.compute()
    assert set(filtered_df[COLUMN_CASE_ID]) == {"C1", "C2"}
    assert captured["label"] == "Intervalo de datas"


def test_temporal_filter_placeholder_uses_dates(monkeypatch):
    st.session_state.clear()
    initialize_session_state()

    captured = {}

    def fake_slider(label, value, **kwargs):  # noqa: ANN001
        captured["label"] = label
        start, end = value
        assert isinstance(start, dt.date)
        assert isinstance(end, dt.date)
        assert kwargs.get("disabled") is True
        return value

    monkeypatch.setattr(streamlit_fragments.st, "slider", fake_slider)

    view = EventLogView(base_df=pd.DataFrame())
    result = streamlit_fragments.temporal_filter(view, disabled=False)

    assert result is None
    assert captured["label"] == "Intervalo de datas"
