from __future__ import annotations

from typing import Any
import pandas as pd

class BaseProcessModel:
    """
    Classe base abstrata para modelos de processo.

    Um modelo de processo aceita um DataFrame de eventos (formato pm4py) e
    retorna uma representação de modelo por meio do método :meth:`compute`.

    Subclasses devem sobrescrever o método :meth:`compute`.
    """

    def compute(self, df: pd.DataFrame) -> Any:
        """
        Materializa o modelo a partir de um DataFrame de eventos.

        Este método deve ser implementado pelas subclasses. Recebe um DataFrame
        contendo as colunas necessárias para o modelo específico e retorna uma
        estrutura de dados representando o modelo.

        :param df: DataFrame de eventos formatado.
        :returns: Objeto representando o modelo de processo.
        :raises NotImplementedError: se não implementado em subclasses.
        """
        raise NotImplementedError("Subclasses devem implementar compute()")

    def to_graphviz(
        self, model: Any, log: Any | None = None, **kwargs: Any
    ) -> Any:
        """
        Para modelos que suportam visualização via Graphviz (por exemplo DFG), este
        método pode ser sobrescrito para gerar a figura.

        :param model: Objeto do modelo previamente retornado por :meth:`compute`.
        :param log: Representação do log de eventos utilizada na geração do modelo,
            tipicamente um ``pm4py.objects.log.obj.EventLog``. O valor padrão é
            ``None`` para manter compatibilidade com modelos que não necessitam do
            log durante a visualização.
        :param kwargs: Parâmetros adicionais para a visualização.
        :returns: Objeto de figura ou visualização.
        :raises NotImplementedError: se o modelo não suporta visualização.
        """
        raise NotImplementedError(
            "Este modelo não implementa visualização Graphviz"
        )

    def quality_metrics(
        self, df: pd.DataFrame, model: Any
    ) -> dict[str, float | None]:
        """
        Calcula métricas de qualidade associadas ao modelo de processo gerado.

        Subclasses podem sobrescrever este método para retornar um dicionário com
        métricas específicas do modelo. Cada métrica deve mapear para um valor
        numérico (ou ``None`` quando não aplicável).

        :param df: DataFrame de eventos utilizado na geração do modelo.
        :param model: Modelo previamente retornado por :meth:`compute`.
        :returns: Dicionário contendo métricas de qualidade do modelo.
        :raises NotImplementedError: se não implementado em subclasses.
        """
        raise NotImplementedError(
            "Subclasses devem implementar quality_metrics()"
        )