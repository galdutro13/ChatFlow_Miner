from __future__ import annotations

from dataclasses import dataclass, field
import numbers
from typing import Any, Optional
import pandas as pd

from .base import BaseProcessModel
from ..event_log.view import EventLogView

@dataclass
class ProcessModelView:
    """
    Representa uma visão *lazy* de um modelo de processo.

    Combina uma visão de log de eventos (:class:`EventLogView`) com um
    :class:`BaseProcessModel`. O modelo somente será calculado quando
    :meth:`compute` for chamado.
    """
    log_view: Any
    model: BaseProcessModel
    _cached: Optional[Any] = field(default=None, init=False, repr=False)
    _cached_graphviz: dict = field(default_factory=dict, init=False, repr=False)
    _cached_quality: dict[str, float | None] | None = field(
        default=None, init=False, repr=False
    )

    def compute(self) -> Any:
        """
        Materializa o modelo de processo.

        Se o resultado já foi computado anteriormente, utiliza o cache interno.

        :returns: Estrutura de dados retornada pelo modelo.
        """
        if self._cached is not None:
            return self._cached

        # Se log_view for um EventLogView, aplica filtros primeiro
        if isinstance(self.log_view, EventLogView):
            df = self.log_view.compute()
        elif isinstance(self.log_view, pd.DataFrame):
            df = self.log_view
        else:
            raise TypeError(
                "log_view deve ser um EventLogView ou pandas.DataFrame"
            )

        result = self.model.compute(df)
        self._cached = result
        # Limpa cache de visualizações quando o resultado muda
        self._cached_graphviz.clear()
        self._cached_quality = None
        return result

    def to_graphviz(self, **kwargs: Any) -> Any:
        """
        Gera uma visualização do modelo caso suportado.

        Caching: usa a identidade do resultado computado e uma representação
        ordenada dos kwargs para chavear as visualizações.
        """
        result = self.compute()

        # cria chave estável a partir da identidade do resultado e repr dos kwargs
        kwargs_key = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
        cache_key = (id(self._cached), kwargs_key)

        if cache_key in self._cached_graphviz:
            return self._cached_graphviz[cache_key]

        viz = self.model.to_graphviz(result, **kwargs)
        self._cached_graphviz[cache_key] = viz
        return viz

    def quality_metrics(self) -> dict[str, float | None]:
        """Calcula e retorna métricas de qualidade do modelo."""

        if self._cached_quality is not None:
            return self._cached_quality

        if isinstance(self.log_view, EventLogView):
            df = self.log_view.compute()
        elif isinstance(self.log_view, pd.DataFrame):
            df = self.log_view
        else:
            raise TypeError(
                "log_view deve ser um EventLogView ou pandas.DataFrame"
            )

        result = self.compute()

        quality_fn = getattr(self.model, "quality_metrics", None)
        if quality_fn is None:
            raise NotImplementedError(
                f"{type(self.model).__name__} não implementa métricas de qualidade."
            )

        metrics_mapping = quality_fn(df, result)
        metrics_dict = dict(metrics_mapping)

        sanitized: dict[str, float | None] = {}
        for raw_key, value in metrics_dict.items():
            key = str(raw_key)
            if value is None:
                sanitized[key] = None
            elif isinstance(value, numbers.Number):
                sanitized[key] = float(value)
            else:
                raise TypeError(
                    "Os valores das métricas devem ser números ou None."
                )

        self._cached_quality = sanitized
        return sanitized
