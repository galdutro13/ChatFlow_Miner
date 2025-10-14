"""Pacote de utilitários base para filtros.

Exporta a API pública usada pelos módulos de filtros:
- BaseFilter: classe abstrata base
- Exceções específicas: FilterError, MissingColumnsError, RegistryError
- Combinadores: AndFilter, OrFilter, NotFilter

Este módulo mantém a superfície pública pequena e estável para facilitar
imports como:

    from chatflow_miner.lib.filters.base import BaseFilter, AndFilter

"""

from .base import AndFilter, BaseFilter, NotFilter, OrFilter
from .exceptions import FilterError, MissingColumnsError, RegistryError
from .utils import _ensure_bool_series

__all__ = [
    "BaseFilter",
    "FilterError",
    "MissingColumnsError",
    "RegistryError",
    "AndFilter",
    "OrFilter",
    "NotFilter",
    "_ensure_bool_series",
]
