from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import streamlit as st

from .dfg import DFGModel
from .view import ProcessModelView
from ..filters.view import EventLogView
from ..state.manager import set_selected_model


# -----------------------------
# Naming utilities (pure)
# -----------------------------

def normalize_name(name: str) -> str:
    """Normaliza o nome para comparação: strip + casefold."""
    return name.strip().casefold()


def is_valid_name(name: str) -> bool:
    """Nome é válido se não for vazio após strip."""
    return bool(name and name.strip())


def name_is_unique(name: str, existing: Iterable[str]) -> bool:
    """Compara de forma case-insensitive e ignorando espaços nas extremidades."""
    normalized = normalize_name(name)
    existing_norm = {normalize_name(n) for n in existing}
    return normalized not in existing_norm


# -----------------------------
# Model generation and render
# -----------------------------

def generate_process_model(log_view: EventLogView) -> Dict[str, Any]:
    """Gera dados essenciais do modelo de processo a partir de um EventLogView.

    Retorna apenas dados essenciais (sem objetos pesados/visualizações).
    Estrutura retornada:
      { 'type': 'dfg', 'data': (dfg, start_acts, end_acts) }
    """
    df = log_view.compute()
    model = DFGModel()
    dfg_tuple = model.compute(df)
    return {"type": "dfg", "data": dfg_tuple}


def render_process_graph(model_data: Dict[str, Any]) -> None:
    """Renderiza o grafo do modelo a partir de `model_data` exclusivamente.

    Não acessa globais; apenas `model_data` é usado.
    """
    if model_data.get("type") == "dfg":
        dfg_tuple = model_data["data"]
        gviz = DFGModel().to_graphviz(dfg_tuple, bgcolor="white", rankdir="LR")
        st.graphviz_chart(gviz, width="stretch")
    else:
        st.warning("Tipo de modelo desconhecido para visualização.")


# -----------------------------
# Persistência em session_state
# -----------------------------

@dataclass
class _DFGPrecomputedModel(DFGModel):
    """Modelo DFG que ignora DF e entrega dados pré-computados.

    Usado para armazenar somente dados essenciais no registry sem reter DataFrames.
    """
    precomputed: Tuple[dict, dict, dict]

    def compute(self, df: pd.DataFrame) -> Tuple[dict, dict, dict]:  # type: ignore[override]
        return self.precomputed


def _build_view_from_model_data(model_data: Dict[str, Any]) -> ProcessModelView:
    if model_data.get("type") == "dfg":
        pre = model_data["data"]
        model = _DFGPrecomputedModel(precomputed=pre)
        # DataFrame vazio apenas para satisfazer a assinatura; o modelo ignora.
        empty_df = pd.DataFrame()
        return ProcessModelView(log_view=empty_df, model=model)
    raise ValueError("Tipo de modelo não suportado para persistência.")


def save_model(name: str, model_data: Dict[str, Any]) -> None:
    """Salva o modelo no registry, atualiza seleção e fecha o diálogo.

    Regras:
      - Não salva nomes vazios
      - Não sobrescreve nomes existentes (case-insensitive, strip)
    Em sucesso, seleciona o modelo e fecha o diálogo via st.rerun().
    """
    if not is_valid_name(name):
        raise ValueError("Insira um nome não vazio para salvar o modelo.")

    # Nomes existentes (exclui placeholder se presente)
    existing_names = list(st.session_state.process_models.names)
    placeholder = "Criar novo modelo de processo..."
    existing_names = [n for n in existing_names if n != placeholder]

    if not name_is_unique(name, existing_names):
        raise ValueError("Já existe um modelo com este nome. Escolha outro nome.")

    view = _build_view_from_model_data(model_data)
    st.session_state.process_models.add(name, view)
    set_selected_model(name)

    # Fechar o diálogo
    st.rerun()


# -----------------------------
# UI: diálogo e áreas principais
# -----------------------------

@st.dialog("Modelo de processos gerado com sucesso", dismissible=True)
def show_generated_model_dialog() -> None:
    """Dialogo para exibir a visualização e permitir salvar com um nome."""
    model_data = st.session_state.get("latest_generated_model")
    if model_data is None:
        st.info("Nenhum modelo gerado nesta execução.")
        return

    render_process_graph(model_data)

    with st.form(key="dialog.form.save"):
        name = st.text_input("Nome do modelo", key="dialog.name")
        submitted = st.form_submit_button("Salvar", width="stretch")
        if submitted:
            try:
                with st.spinner("Salvando modelo..."):
                    save_model(name, model_data)
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error("Ocorreu um erro ao salvar o modelo. Verifique o nome e tente novamente.")
                st.exception(exc)


def render_saved_model_ui(selected_name: str) -> None:
    """Renderiza o modelo salvo na área principal, substituindo a UI de filtros."""
    from ..state.manager import get_process_model  # import local para evitar ciclos

    view = get_process_model(selected_name)
    if view is None:
        st.error("Modelo selecionado não encontrado.")
        return

    try:
        gviz = view.to_graphviz(bgcolor="white", rankdir="LR")
        st.graphviz_chart(gviz, width="stretch")
    except Exception as exc:
        st.error("Falha ao renderizar o modelo salvo.")
        st.exception(exc)


