from typing import Any, Tuple
import pandas as pd
import pm4py
from graphviz import Digraph

from .base import BaseProcessModel

class DFGModel(BaseProcessModel):
    """
    Modelo de Processo baseado em Directly-Follows Graph (DFG).

    Este modelo constrói um grafo diretamente a partir de um DataFrame de eventos
    formatado, representando as relações "directly-follows" entre atividades.

    O método :meth:`compute` retorna uma estrutura de dados representando o DFG,
    que pode ser posteriormente visualizada ou analisada.
    """

    def compute(self, df: pd.DataFrame) -> Tuple[dict, dict, dict]:
        """
        Constrói o Directly-Follows Graph (DFG) a partir do DataFrame de eventos.

        :param df: DataFrame de eventos formatado.
        :returns: Tupla contendo o DFG, atividades iniciais e atividades finais.
        """
        # Implementação específica para construir o DFG
        dfg, start_activity, end_activity = pm4py.discover_dfg(df)
        return dfg, start_activity, end_activity

    def to_graphviz(self, model: Tuple[dict, dict, dict], bgcolor: str = "white", rankdir: str = "LR", max_num_edges: int = 9223372036854775807) -> Digraph:
        """
        Gera uma visualização do DFG usando Graphviz.

        :param model: Tupla contendo o DFG, atividades iniciais e atividades finais.
        :param bgcolor: Cor de fundo da visualização.
        :param rankdir: Direção do grafo ("LR" para esquerda-direita, "TB" para cima-baixo).
        :param max_num_edges: Número máximo de arestas a exibir.
        :returns: Objeto Digraph representando a visualização do DFG.
        """
        # Implementação específica para gerar a visualização do DFG
        from pm4py.visualization.dfg import visualizer as dfg_visualizer
        dfg, start_activities, end_activities = model
        dfg_parameters = dfg_visualizer.Variants.FREQUENCY.value.Parameters
        # Corrige: não passar a builtin `format` (função) — é esperado um string com o formato de imagem.
        # Usar 'svg' ou 'png' ou 'html' conforme suporte do Graphviz/pm4py.
        parameters = {dfg_parameters.FORMAT: "svg", dfg_parameters.START_ACTIVITIES: start_activities,
                      dfg_parameters.END_ACTIVITIES: end_activities, "bgcolor": bgcolor, "rankdir": rankdir,
                      "maxNoOfEdgesInDiagram": max_num_edges}

        gviz = dfg_visualizer.apply(dfg, variant=dfg_visualizer.Variants.FREQUENCY,
                                    parameters=parameters)

        return gviz