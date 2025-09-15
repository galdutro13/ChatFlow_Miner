from typing import Any, Dict, Tuple

from typing import Set
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS: Set[str] = {"CASE_ID", "ACTIVITY", "START_TIMESTAMP", "END_TIMESTAMP"}

def verify_format(log: pd.DataFrame) -> None:
    """
    Valida o formato do DataFrame. Lança ValueError se faltar coluna.
    (uso recomendado em bibliotecas/serviços)
    """
    missing = REQUIRED_COLUMNS.difference(set(log.columns))
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias: {', '.join(sorted(missing))}")
