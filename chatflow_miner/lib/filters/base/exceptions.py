"""Exceções específicas do pacote filters.base.

Mantém as definições de exceção separadas para evitar importações circulares
entre os módulos base e utils.
"""


class FilterError(Exception):
    """Erro genérico ao construir ou aplicar filtros."""


class MissingColumnsError(FilterError):
    """Lançado quando o DataFrame não possui colunas obrigatórias para um filtro."""


class RegistryError(FilterError):
    """Problemas ao registrar ou construir filtros via especificação declarativa."""
