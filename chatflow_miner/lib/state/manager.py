from collections.abc import Sequence

import pandas as pd
import streamlit as st

from chatflow_miner.lib.process_models import ProcessModelRegistry, ProcessModelView

PLACEHOLDER = "Criar novo modelo de processo..."


def initialize_session_state() -> None:
    """
    Inicializa o estado da sessão Streamlit com valores padrão.

    Esta função deve ser chamada no início de cada aplicação Streamlit para
    garantir que todas as variáveis de estado necessárias estejam definidas.

    Inicializa as seguintes variáveis de estado:
    - input_dialog: Controla a exibição do diálogo de entrada de dados
    - log_eventos: Armazena o DataFrame com os dados do log de eventos
    - load_info: Armazena informações sobre o arquivo carregado
    - process_models: Registry de modelos de processo

    :returns: None
    """
    if "input_dialog" not in st.session_state:
        st.session_state.input_dialog = False
    if "log_eventos" not in st.session_state:
        st.session_state.log_eventos = None
    if "load_info" not in st.session_state:
        st.session_state.load_info = None
    if "process_models" not in st.session_state:
        # Registry de modelos de processo (mapping nome -> ProcessModelView | None)
        st.session_state.process_models = ProcessModelRegistry()
        initialize_process_models()

    # Seleção atual no seletor de modelos
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = PLACEHOLDER

    # Último modelo gerado (não persistido) para exibição no diálogo
    if "latest_generated_model" not in st.session_state:
        st.session_state.latest_generated_model = None

    if "processing_model" not in st.session_state:
        st.session_state.processing_model = False

    if "processing_error" not in st.session_state:
        st.session_state.processing_error = False

    if "selected_variants" not in st.session_state:
        st.session_state.selected_variants = []

    if "log_load_counter" not in st.session_state:
        st.session_state.log_load_counter = 0

    if "last_toast_log_counter" not in st.session_state:
        st.session_state.last_toast_log_counter = 0

    if "initial_discovery_toast_shown" not in st.session_state:
        st.session_state.initial_discovery_toast_shown = False


def get_selected_model() -> str | None:
    """Retorna o nome do modelo de processo selecionado (ou None)."""
    if st.session_state.selected_model != PLACEHOLDER:
        return st.session_state.get("selected_model")
    return None


def set_selected_model(name: str | None) -> None:
    """Define o nome do modelo de processo selecionado (ou None para voltar à criação)."""
    st.session_state.selected_model = name


def initialize_process_models() -> None:
    """
    Inicializa o registry de process_models com o item placeholder
    "Criar novo modelo de processo...".
    """
    if "process_models" not in st.session_state:
        st.session_state.process_models = ProcessModelRegistry()

    # Adiciona o placeholder apenas se o registry estiver vazio
    if len(st.session_state.process_models) == 0:
        st.session_state.process_models.add(PLACEHOLDER)


def open_input_dialog() -> None:
    """
    Abre o diálogo de entrada de dados.

    Define a variável de estado input_dialog como True, o que faz com que
    o componente de entrada de dados seja exibido na interface.

    :returns: None
    """
    st.session_state.input_dialog = True


def close_input_dialog() -> None:
    """
    Fecha o diálogo de entrada de dados.

    Define a variável de estado input_dialog como False, o que faz com que
    o componente de entrada de dados seja ocultado na interface.

    :returns: None
    """
    st.session_state.input_dialog = False


def set_log_eventos(log: pd.DataFrame, load_info: dict) -> None:
    """
    Define os dados do log de eventos e informações de carregamento.

    Armazena no estado da sessão o DataFrame com os dados do log de eventos
    e as informações sobre o arquivo que foi carregado.

    :param log: DataFrame contendo os dados do log de eventos
    :param load_info: Dicionário com informações sobre o arquivo carregado
                     (ex: nome do arquivo, timestamp, etc.)
    :returns: None
    """
    st.session_state.load_info = load_info
    st.session_state.log_eventos = log
    st.session_state.log_load_counter = st.session_state.get("log_load_counter", 0) + 1


def get_log_eventos(
    which: str | Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame | dict | None:
    """
    Retorna `(log_eventos, load_info)` por padrão.
    Se `which` for uma string ou sequência contendo apenas 'load_info' ou 'log_eventos',
    retorna apenas o item solicitado. Se receber ambas literais, retorna o par.
    Retorna None para parâmetros inválidos ou quando o(s) dado(s) não estiver(em) disponível(is).
    """
    log = st.session_state.get("log_eventos")
    info = st.session_state.get("load_info")

    valid: set[str] = {"log_eventos", "load_info"}

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
    """
    Remove os dados do log de eventos do estado da sessão.

    Limpa as variáveis de estado relacionadas aos dados carregados,
    efetivamente "descarregando" o arquivo atual.

    :returns: None
    """
    st.session_state.log_eventos = None
    st.session_state.load_info = None


def get_process_model(name: str) -> ProcessModelView | None:
    """
    Obtém um modelo de processo do registry.

    :param name: Nome do modelo.
    :returns: A ProcessModelView ou None se não encontrada.
    """
    if "process_models" not in st.session_state:
        return None

    return st.session_state.process_models.get(name)
