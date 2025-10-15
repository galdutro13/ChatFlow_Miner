from __future__ import annotations

import inspect
import logging
import numbers
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pm4py

from ..event_log.view import EventLogView
from .base import BaseProcessModel

LOGGER = logging.getLogger(__name__)


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
    _cached: Any | None = field(default=None, init=False, repr=False)
    _cached_graphviz: dict = field(default_factory=dict, init=False, repr=False)
    _cached_quality: dict[str, float | None] | None = field(
        default=None, init=False, repr=False
    )
    _cached_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _cached_event_log: Any | None = field(default=None, init=False, repr=False)
    _event_log_failed: bool = field(default=False, init=False, repr=False)

    def _materialize_dataframe(self) -> pd.DataFrame:
        """Return the cached DataFrame, materialising it on first use."""

        if self._cached_df is not None:
            return self._cached_df

        if isinstance(self.log_view, EventLogView):
            df = self.log_view.compute()
        elif isinstance(self.log_view, pd.DataFrame):
            df = self.log_view
        else:  # pragma: no cover - defensive guard
            raise TypeError("log_view deve ser um EventLogView ou pandas.DataFrame")

        self._cached_df = df
        return df

    def _materialize_event_log(self, df: pd.DataFrame, *, context: str) -> Any | None:
        """Convert the cached DataFrame to an EventLog once per view."""

        if self._cached_event_log is not None:
            return self._cached_event_log

        if self._event_log_failed or df is None or df.empty:
            return None

        try:
            event_log = pm4py.convert_to_event_log(df)
        except Exception:  # pragma: no cover - logged for observability
            LOGGER.exception(
                "Falha ao converter DataFrame para EventLog durante %s.", context
            )
            self._event_log_failed = True
            return None

        self._cached_event_log = event_log
        return event_log

    def compute(self) -> Any:
        """
        Materializa o modelo de processo.

        Se o resultado já foi computado anteriormente, utiliza o cache interno.

        :returns: Estrutura de dados retornada pelo modelo.
        """
        if self._cached is not None:
            return self._cached

        df = self._materialize_dataframe()
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

        df = self._materialize_dataframe()
        user_supplied_log = kwargs.pop("log", None)
        event_log = (
            user_supplied_log
            if user_supplied_log is not None
            else self._materialize_event_log(df, context="a visualização do modelo")
        )

        viz = self.model.to_graphviz(result, log=event_log, **kwargs)
        self._cached_graphviz[cache_key] = viz
        return viz

    def quality_metrics(self) -> dict[str, float | None]:
        """Calcula e retorna métricas de qualidade do modelo."""

        if self._cached_quality is not None:
            return self._cached_quality

        df = self._materialize_dataframe()
        result = self.compute()

        quality_fn = getattr(self.model, "quality_metrics", None)
        if quality_fn is None:
            raise NotImplementedError(
                f"{type(self.model).__name__} não implementa métricas de qualidade."
            )

        quality_kwargs: dict[str, Any] = {}
        event_log = self._materialize_event_log(
            df, context="o cálculo de métricas de qualidade"
        )
        if event_log is not None:
            try:
                signature = inspect.signature(quality_fn)
            except (TypeError, ValueError):  # pragma: no cover - dynamic callables
                signature = None
            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.name == "event_log" and parameter.kind in (
                        parameter.POSITIONAL_OR_KEYWORD,
                        parameter.KEYWORD_ONLY,
                    ):
                        quality_kwargs["event_log"] = event_log
                        break

        metrics_mapping = quality_fn(df, result, **quality_kwargs)
        metrics_dict = dict(metrics_mapping)

        sanitized: dict[str, float | None] = {}
        for raw_key, value in metrics_dict.items():
            key = str(raw_key)
            if value is None:
                sanitized[key] = None
            elif isinstance(value, numbers.Number):
                sanitized[key] = float(value)
            else:
                raise TypeError("Os valores das métricas devem ser números ou None.")

        self._cached_quality = sanitized
        return sanitized
