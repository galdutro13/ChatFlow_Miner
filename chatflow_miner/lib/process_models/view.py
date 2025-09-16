from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd

from .base import BaseProcessModel
from ..filters import EventLogView

@dataclass
class ProcessModelView:
    """
    Representa uma visão *lazy* de um modelo de processo.

    Combina uma visão de log de eventos (:class:`EventLogView`) com um
    :class:`BaseProcessModel`. O modelo somente será calculado quando
    :meth:`compute` for chamado.

    :ivar log_view: Instância de :class:`EventLogView` ou DataFrame bruto.
    :ivar model: Instância de :class:`BaseProcessModel` a ser aplicado.
    :ivar _cached: Cache interno com o resultado já computado (se disponível).
    """
    log_view: Any
    model: BaseProcessModel
    _cached: Optional[Any] = field(default=None, init=False, repr=False)

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
        return result

    def to_graphviz(self, **kwargs: Any) -> Any:
        """
        Gera uma visualização do modelo caso suportado.

        Delegado para :meth:`BaseProcessModel.to_graphviz`.

        :param kwargs: Parâmetros adicionais de visualização.
        :returns: Objeto de figura/visualização.
        :raises NotImplementedError: se o modelo não suporta esta operação.
        """
        result = self.compute()
        return self.model.to_graphviz(result, **kwargs)