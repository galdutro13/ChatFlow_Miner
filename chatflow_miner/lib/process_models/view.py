from __future__ import annotations

from dataclasses import dataclass, field
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