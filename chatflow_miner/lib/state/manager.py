from typing import Dict, Tuple, Optional, Union, Sequence, Set
import pandas as pd
import streamlit as st

from chatflow_miner.lib.process_models import ProcessModelRegistry


def initialize_session_state() -> None:
    if "input_dialog" not in st.session_state:
        st.session_state.input_dialog = False
    if "log_eventos" not in st.session_state:
        st.session_state.log_eventos = None
    if "load_info" not in st.session_state:
        st.session_state.load_info = None
    if "process_models" not in st.session_state:
        # process models deve ser um dicionário nome -> modelo
        st.session_state.process_models = ProcessModelRegistry()
        initialize_process_models()


def initialize_process_models() -> None:
    """
    Inicializa o registry de process_models com o item placeholder
    "Criar novo modelo de processo...".
    """
    if "process_models" not in st.session_state:
        st.session_state.process_models = ProcessModelRegistry()
    
    # Adiciona o placeholder apenas se o registry estiver vazio
    if len(st.session_state.process_models) == 0:
        st.session_state.process_models.add("Criar novo modelo de processo...")


def open_input_dialog() -> None:
    st.session_state.input_dialog = True

def close_input_dialog() -> None:
    st.session_state.input_dialog = False

def set_log_eventos(log: pd.DataFrame, load_info: Dict)-> None:
    st.session_state.load_info = load_info
    st.session_state.log_eventos = log

def get_log_eventos(
    which: Optional[Union[str, Sequence[str]]] = None
) -> Optional[Union[Tuple[pd.DataFrame, Dict], pd.DataFrame, Dict]]:
    """
    Retorna `(log_eventos, load_info)` por padrão.
    Se `which` for uma string ou sequência contendo apenas 'load_info' ou 'log_eventos',
    retorna apenas o item solicitado. Se receber ambas literais, retorna o par.
    Retorna None para parâmetros inválidos ou quando o(s) dado(s) não estiver(em) disponível(is).
    """
    log = st.session_state.get("log_eventos")
    info = st.session_state.get("load_info")

    valid: Set[str] = {"log_eventos", "load_info"}

    # Normaliza requested para um conjunto de literais
    if which is None:
        requested = None
    elif isinstance(which, str):
        requested = {which}
    else:
        requested = set(which)

    if requested is None:
        # comportamento original: exige ambos presentes
        if log is not None and info is not None:
            return log, info
        return None

    if not requested.issubset(valid):
        return None

    # Casos específicos
    if requested == {"log_eventos"}:
        if log is not None:
            return log
        return None
    if requested == {"load_info"}:
        if info is not None:
            return info
        return None

    # solicitou ambas (ou solicitação equivalente)
    if log is not None and info is not None:
        return log, info
    return None

def reset_log_eventos() -> None:
    st.session_state.log_eventos = None
    st.session_state.load_info = None
