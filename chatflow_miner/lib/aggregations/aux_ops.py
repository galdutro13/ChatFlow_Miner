"""Auxiliary operations (pre-processing) for case aggregations."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from chatflow_miner.lib.constants import COLUMN_CASE_ID, COLUMN_START_TS, COLUMN_END_TS
from chatflow_miner.lib.aggregations.exceptions import MissingColumnsError


class BaseAuxOp:
    """Operação auxiliar aplicada *antes* da agregação.

    Deve retornar um novo DataFrame (não deve mutar o original).
    """

    required_columns: Sequence[str] = ()

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise MissingColumnsError(f"Colunas ausentes: {missing}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


class NormalizeTimestampsOp(BaseAuxOp):
    """Converte colunas START/END para datetime; cria END = START quando ausente."""

    required_columns = (COLUMN_START_TS,)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2[COLUMN_START_TS] = pd.to_datetime(
            df2[COLUMN_START_TS], errors="coerce", format="mixed"
        )
        if COLUMN_END_TS in df2.columns:
            df2[COLUMN_END_TS] = pd.to_datetime(
                df2[COLUMN_END_TS], errors="coerce", format="mixed"
            )
        else:
            df2[COLUMN_END_TS] = df2[COLUMN_START_TS]
        return df2


class DeriveCaseStartDateOp(BaseAuxOp):
    """Adiciona coluna `CASE_DATE` por case, baseada em `min(START_TIMESTAMP)`."""

    required_columns = (COLUMN_CASE_ID, COLUMN_START_TS)

    def __init__(self, target_col: str = "CASE_DATE") -> None:
        self.target_col = target_col

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(df)
        df2 = df.copy()
        st = pd.to_datetime(df2[COLUMN_START_TS], errors="coerce", format="mixed")
        case_start = st.groupby(df2[COLUMN_CASE_ID]).transform("min")
        df2[self.target_col] = case_start.dt.date
        return df2

