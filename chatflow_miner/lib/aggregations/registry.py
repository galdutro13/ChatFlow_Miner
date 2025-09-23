"""Registry and builder for aggregators."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Type

from chatflow_miner.lib.aggregations.base import BaseCaseAggregator
from chatflow_miner.lib.aggregations.aggregators import CaseVariantAggregator, CaseDateAggregator
from chatflow_miner.lib.aggregations.exceptions import AggregationError

AGGREGATOR_REGISTRY: Dict[str, Type[BaseCaseAggregator]] = {
    "variant": CaseVariantAggregator,
    "case_date": CaseDateAggregator,
}


def register_aggregator(name: str, cls: Type[BaseCaseAggregator]) -> None:
    if not name:
        raise AggregationError("Nome do agregador não pode ser vazio")
    if not issubclass(cls, BaseCaseAggregator):
        raise AggregationError("Classe deve herdar de BaseCaseAggregator")
    AGGREGATOR_REGISTRY[name] = cls


Spec = Mapping[str, Any]


def build_aggregator_from_spec(
    spec: Spec, *, registry: Optional[Mapping[str, Type[BaseCaseAggregator]]] = None
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

