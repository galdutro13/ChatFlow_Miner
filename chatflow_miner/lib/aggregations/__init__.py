"""Aggregations (lazy) para logs de eventos (pm4py)

Compatibilidade: este módulo reexporta a API pública agora dividida em
módulos menores dentro de `chatflow_miner.lib.aggregations`.
"""

from __future__ import annotations

# Concrete aggregators
from chatflow_miner.lib.aggregations.aggregators import (
    CaseDateAggregator,
    CaseVariantAggregator,
)

# Aux ops
from chatflow_miner.lib.aggregations.aux_ops import (
    BaseAuxOp,
    DeriveCaseStartDateOp,
    NormalizeTimestampsOp,
)

# Base aggregator interface
from chatflow_miner.lib.aggregations.base import BaseCaseAggregator

# Re-export exceptions
from chatflow_miner.lib.aggregations.exceptions import (
    AggregationError,
    MissingColumnsError,
)

# Data models
from chatflow_miner.lib.aggregations.models import VariantInfo

# Registry & builder
from chatflow_miner.lib.aggregations.registry import (
    AGGREGATOR_REGISTRY,
    build_aggregator_from_spec,
    register_aggregator,
)

# Lazy view
from chatflow_miner.lib.aggregations.view import CaseAggView

__all__ = [
    "AggregationError",
    "MissingColumnsError",
    "BaseAuxOp",
    "NormalizeTimestampsOp",
    "DeriveCaseStartDateOp",
    "VariantInfo",
    "BaseCaseAggregator",
    "CaseVariantAggregator",
    "CaseDateAggregator",
    "CaseAggView",
    "AGGREGATOR_REGISTRY",
    "register_aggregator",
    "build_aggregator_from_spec",
]
