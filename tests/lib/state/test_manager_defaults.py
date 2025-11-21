import streamlit as st

from chatflow_miner.lib.state import initialize_session_state


def test_initialize_session_state_sets_processing_and_variant_flags():
    st.session_state.clear()

    initialize_session_state()

    assert "processing_model" in st.session_state
    assert st.session_state.processing_model is False
    assert "selected_variants" in st.session_state
    assert st.session_state.selected_variants == []
    assert st.session_state.log_load_counter == 0
    assert st.session_state.last_toast_log_counter == 0
    assert st.session_state.initial_discovery_toast_shown is False
