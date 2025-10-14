"""Registry and builder for aggregators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chatflow_miner.lib.aggregations.aggregators import (
    CaseDateAggregator,
    CaseVariantAggregator,
)
from chatflow_miner.lib.aggregations.base import BaseCaseAggregator
from chatflow_miner.lib.aggregations.exceptions import AggregationError

AGGREGATOR_REGISTRY: dict[str, type[BaseCaseAggregator]] = {
    "variant": CaseVariantAggregator,
    "case_date": CaseDateAggregator,
}


def register_aggregator(name: str, cls: type[BaseCaseAggregator]) -> None:
    if not name:
        raise AggregationError("Nome do agregador não pode ser vazio")
    if not issubclass(cls, BaseCaseAggregator):
        raise AggregationError("Classe deve herdar de BaseCaseAggregator")
    AGGREGATOR_REGISTRY[name] = cls


Spec = Mapping[str, Any]


def build_aggregator_from_spec(
    spec: Spec, *, registry: Mapping[str, type[BaseCaseAggregator]] | None = None
) -> BaseCaseAggregator:
    reg = dict(AGGREGATOR_REGISTRY if registry is None else registry)
    t = spec.get("type")
    if not isinstance(t, str):
        raise AggregationError("Spec inválida: campo 'type' ausente/não-string")
    cls = reg.get(t.strip().lower())
    if cls is None:
        raise AggregationError(f"Agregador não suportado: {t}")
    args = spec.get("args", {})
    if not isinstance(args, Mapping):
        raise AggregationError("Spec inválida: 'args' deve ser um dict")
    return cls(**dict(args))  # type: ignore[arg-type]
