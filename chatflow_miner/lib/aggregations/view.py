"""Lazy view for case aggregations (CaseAggView)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from chatflow_miner.lib.aggregations.aux_ops import BaseAuxOp
from chatflow_miner.lib.aggregations.base import BaseCaseAggregator
from chatflow_miner.lib.aggregations.exceptions import AggregationError
from chatflow_miner.lib.constants import COLUMN_CASE_ID

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseAggView:
    """Wrapper *lazy* para executar uma agregação por caso.

    Use `with_aux(...)` para encadear pré-processamentos e `with_aggregator(...)`
    para definir a estratégia. Nada é computado até `compute()` / `to_dict()`.
    """

    base_df: pd.DataFrame
    aux_ops: tuple[BaseAuxOp, ...] = field(default_factory=tuple)
    aggregator: BaseCaseAggregator | None = None

    def with_aux(self, *ops: BaseAuxOp) -> CaseAggView:
        return CaseAggView(self.base_df, self._extend_aux(ops), self.aggregator)

    def with_aggregator(self, aggregator: BaseCaseAggregator) -> CaseAggView:
        return CaseAggView(self.base_df, self.aux_ops, aggregator)

    def compute(self) -> dict[Any, Any]:
        if self.aggregator is None:
            raise AggregationError(
                "Nenhum agregador definido. Use with_aggregator(...)"
            )

        df = self.base_df
        for op in self.aux_ops:
            df = op.apply(df)

        self.aggregator._check_columns(df)  # type: ignore[attr-defined]

        state = self.aggregator.prepare(df)

        out: dict[Any, Any] = {}
        for case_id, g in df.groupby(COLUMN_CASE_ID, sort=False):
            out[case_id] = self.aggregator.compute_case(g, state)

        logger.debug(
            "CaseAggView.compute: agregador=%s, cases=%d",
            self.aggregator.__class__.__name__,
            len(out),
        )
        return out

    def to_dict(self) -> dict[Any, Any]:
        return self.compute()

    def _extend_aux(self, ops: Iterable[BaseAuxOp]) -> tuple[BaseAuxOp, ...]:
        return tuple(list(self.aux_ops) + list(ops))
