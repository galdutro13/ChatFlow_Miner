"""Aggregations (lazy) para logs de eventos (pm4py)

Compatibilidade: este módulo reexporta a API pública agora dividida em
módulos menores dentro de `chatflow_miner.lib.aggregations`.
"""
from __future__ import annotations

# Re-export exceptions
from chatflow_miner.lib.aggregations.exceptions import (
    AggregationError,
    MissingColumnsError,
)

# Aux ops
from chatflow_miner.lib.aggregations.aux_ops import (
    BaseAuxOp,
    NormalizeTimestampsOp,
    DeriveCaseStartDateOp,
)

# Data models
from chatflow_miner.lib.aggregations.models import VariantInfo

# Base aggregator interface
from chatflow_miner.lib.aggregations.base import BaseCaseAggregator

# Concrete aggregators
from chatflow_miner.lib.aggregations.aggregators import (
    CaseVariantAggregator,
    CaseDateAggregator,
)

# Lazy view
from chatflow_miner.lib.aggregations.view import CaseAggView

# Registry & builder
from chatflow_miner.lib.aggregations.registry import (
    AGGREGATOR_REGISTRY,
    register_aggregator,
    build_aggregator_from_spec,
)

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
