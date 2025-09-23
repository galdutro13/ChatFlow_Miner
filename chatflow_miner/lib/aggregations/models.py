"""Data models for aggregations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantInfo:
    """Informações sobre uma variante de processo.

    Attributes:
        variant_id: ID estável (hash da string da variante).
        variant: string da variante (ex.: "A>B>C").
        frequency: número de *cases* com essa variante.
        length: quantidade de atividades na variante.
    """

    variant_id: str
    variant: str
    frequency: int
    length: int

