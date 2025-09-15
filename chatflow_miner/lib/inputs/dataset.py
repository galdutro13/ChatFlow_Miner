from typing import Any, Dict, Tuple
import pandas as pd
import pm4py
import streamlit as st
from chatflow_miner.lib.utils import load_dataset
from chatflow_miner.lib.state import set_log_eventos, close_input_dialog

@st.dialog("Carregar arquivo de log")
def input_dataset() -> None:
    """
    Carrega um arquivo de log de eventos em formato CSV ou XES e retorna um DataFrame do Pandas.
    """
    load_info = dict()
    uploaded_file = st.file_uploader("Carregar log de eventos (CSV)", type=["csv"])
    load_info["sep"] = st.selectbox("Qual o separador do arquivo CSV?", options=[",", ";", "\t"], index=0)
    if uploaded_file is not None:
        load_info["file_name"] = uploaded_file.name
        log = load_dataset(uploaded_file, load_info)
        set_log_eventos(log=log, load_info=load_info)
        st.button("Enviar", on_click=close_input_dialog)
        st.rerun()
    else:
        st.stop()