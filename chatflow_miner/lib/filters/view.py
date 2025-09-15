from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Any, Union
from collections.abc import Iterable
import pandas as pd

from .base import BaseFilter, _ensure_bool_series
from ..constants import COLUMN_CASE_ID
# -----------------------------------------------------------------------------
# EventLogView: visão lazy sobre o DataFrame
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EventLogView:
    """*Wrapper* imutável e *lazy* para aplicar filtros a um log de eventos.

    Args:
        base_df: DataFrame base (já formatado por ``pm4py.format_dataframe``).
        filters: lista de filtros a aplicar (AND por padrão). Para OR/NOT, use a
                 composição nos próprios filtros (``|``, ``~``).

    Notas:
      - ``filter(...)`` devolve **uma nova** ``EventLogView`` (imutável).
      - ``compute()`` materializa e retorna um *novo* DataFrame.
      - Métodos utilitários chamam ``compute()`` por baixo.
    """

    base_df: pd.DataFrame
    filters: Sequence[BaseFilter] = field(default_factory=tuple)

    def filter(self, flt: Union[BaseFilter, Sequence[BaseFilter]]) -> "EventLogView":
        """Retorna nova view com filtros adicionais (lazy)."""
        new_filters: List[BaseFilter] = list(self.filters)
        if isinstance(flt, Iterable) and not isinstance(flt, BaseFilter):
            new_filters.extend(flt)  # type: ignore[arg-type]
        else:
            new_filters.append(flt)  # type: ignore[arg-type]
        return EventLogView(self.base_df, tuple(new_filters))

    def compute(self) -> pd.DataFrame:
        """Materializa a view aplicando todos os filtros.

        Returns:
            Um novo DataFrame com as linhas aprovadas por **todos** os filtros.
        """
        if not self.filters:
            return self.base_df.copy()

        mask = pd.Series(True, index=self.base_df.index)
        for f in self.filters:
            fm = f.mask(self.base_df)
            _ensure_bool_series(fm, self.base_df)
            mask &= fm
        return self.base_df.loc[mask].copy()

    # Conveniências com materialização
    def head(self, n: int = 5) -> pd.DataFrame:
        return self.compute().head(n)

    def to_csv(self, path: str, **kwargs: Any) -> None:
        self.compute().to_csv(path, index=False, **kwargs)

    def to_xes(self, path: str, case_id_col: str = COLUMN_CASE_ID) -> None:
        """Exporta o resultado para XES usando ``pm4py`` (opcional).

        Esta função importa ``pm4py`` sob demanda para não introduzir dependência
        forte. Lança ``ImportError`` se o pacote não estiver disponível.
        """
        try:
            from pm4py.objects.log.util import dataframe_utils
            from pm4py.objects.conversion.log import converter as log_converter
        except Exception as exc:  # pragma: no cover - caminho de fallback
            raise ImportError(
                "Exportação XES requer 'pm4py' instalado"
            ) from exc

        df = self.compute()
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        params = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: case_id_col}
        event_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)

        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        xes_exporter.apply(event_log, path)