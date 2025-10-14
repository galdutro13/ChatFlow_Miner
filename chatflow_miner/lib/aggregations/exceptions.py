"""Exceptions for aggregations."""

from __future__ import annotations


class AggregationError(Exception):
    """Erro genérico no pipeline de agregação."""


class MissingColumnsError(AggregationError):
    """Lançada quando o DataFrame não contém as colunas exigidas."""
