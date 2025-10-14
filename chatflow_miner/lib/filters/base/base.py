from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .exceptions import MissingColumnsError
from .utils import _ensure_bool_series


# -----------------------------------------------------------------------------
# BaseFilter: protocolo / classe base para todos os filtros
# -----------------------------------------------------------------------------
class BaseFilter(ABC):
    """Filtro base que produz uma máscara booleana alinhada ao DataFrame.

    Regras:
      - Implemente :meth:`mask` para devolver uma ``pd.Series[bool]`` com o
        mesmo índice do DataFrame de entrada.
      - Use ``required_columns`` para declarar dependências de colunas.
      - Filtros podem ser combinados com ``&`` (AND), ``|`` (OR) e ``~`` (NOT).

    Notas de implementação:
      - Evite modificar o ``DataFrame`` dentro de ``mask``; trate conversões de
        tipo de forma local (ex.: ``pd.to_numeric(..., errors='coerce')``).
      - Valide colunas com ``_check_columns`` para mensagens de erro consistentes.
    """

    #: Colunas obrigatórias para o filtro
    required_columns: Sequence[str] = ()

    @abstractmethod
    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Calcula a máscara booleana deste filtro.

        Args:
            df: DataFrame do log de eventos.
        Returns:
            Série booleana alinhada ao índice do ``df``.
        Raises:
            MissingColumnsError: se colunas obrigatórias estiverem ausentes.
        """
        raise NotImplementedError

    # --- API auxiliar --------------------------------------------------------
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica o filtro e retorna um *novo* DataFrame filtrado."""
        m = self.mask(df)
        _ensure_bool_series(m, df)
        return df.loc[m].copy()

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise MissingColumnsError(f"Colunas ausentes no DataFrame: {missing}")

    # --- Composição booleana -------------------------------------------------
    def __and__(self, other: BaseFilter) -> AndFilter:
        return AndFilter(self, other)

    def __or__(self, other: BaseFilter) -> OrFilter:
        return OrFilter(self, other)

    def __invert__(self) -> NotFilter:
        return NotFilter(self)


# -----------------------------------------------------------------------------
# Filtros combinadores (AND/OR/NOT)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AndFilter(BaseFilter):
    left: BaseFilter
    right: BaseFilter

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = self.left.mask(df) & self.right.mask(df)
        _ensure_bool_series(m, df)
        return m


@dataclass(frozen=True)
class OrFilter(BaseFilter):
    left: BaseFilter
    right: BaseFilter

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = self.left.mask(df) | self.right.mask(df)
        _ensure_bool_series(m, df)
        return m


@dataclass(frozen=True)
class NotFilter(BaseFilter):
    inner: BaseFilter

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = ~self.inner.mask(df)
        _ensure_bool_series(m, df)
        return m
