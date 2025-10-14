import pandas as pd

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_START_TS,
)

REQUIRED_COLUMNS: set[str] = {
    COLUMN_CASE_ID,
    COLUMN_ACTIVITY,
    COLUMN_START_TS,
    COLUMN_END_TS,
}


def verify_format(log: pd.DataFrame) -> None:
    """
    Valida o formato do DataFrame. Lança ValueError se faltar coluna.
    (uso recomendado em bibliotecas/serviços)
    """
    missing = REQUIRED_COLUMNS.difference(set(log.columns))
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias: {', '.join(sorted(missing))}")
