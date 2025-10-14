from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from chatflow_miner.lib.aggregations.exceptions import MissingColumnsError


class BaseCaseAggregator:
    """Interface para agregadores `{case_id: valor}`.

    Ciclo de vida:
      1) `prepare(df)` (opcional): computa estado global.
      2) `compute_case(case_df, state)`: calcula valor para um case.
    """

    required_columns: Sequence[str] = ()

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise MissingColumnsError(f"Colunas ausentes: {missing}")

    def prepare(self, df: pd.DataFrame) -> Any:  # pragma: no cover - padrão
        return None

    def compute_case(
        self, case_df: pd.DataFrame, state: Any
    ) -> Any:  # pragma: no cover
        raise NotImplementedError
