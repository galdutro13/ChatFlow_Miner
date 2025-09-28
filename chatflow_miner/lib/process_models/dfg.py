from typing import Any, Dict, Tuple
import math
import logging

import pandas as pd
import pm4py
from graphviz import Digraph

from .base import BaseProcessModel


LOGGER = logging.getLogger(__name__)


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

    def quality_metrics(
        self,
        df: pd.DataFrame,
        model: Tuple[dict, dict, dict],
    ) -> Dict[str, float | None]:
        """Calcula métricas de qualidade para o DFG convertido em rede de Petri.

        Args:
            df: DataFrame do log de eventos (formato pm4py).
            model: Tupla ``(dfg, start_activities, end_activities)`` retornada por
                :meth:`compute`.

        Returns:
            Dicionário com as métricas ``fitness``, ``precision``,
            ``generalization`` e ``simplicity``. Valores ``None`` indicam que a
            métrica não pôde ser calculada.
        """

        metrics: Dict[str, float | None] = {
            "fitness": None,
            "precision": None,
            "generalization": None,
            "simplicity": None,
        }

        dfg_graph, start_activities, end_activities = model

        try:
            from pm4py.objects.conversion.dfg import converter as dfg_converter
            from pm4py.objects.conversion.dfg.variants import (
                to_petri_net_activity_defines_place as dfg_to_petri,
            )

            parameters = {
                dfg_to_petri.Parameters.START_ACTIVITIES: start_activities,
                dfg_to_petri.Parameters.END_ACTIVITIES: end_activities,
            }
            net, initial_marking, final_marking = dfg_converter.apply(
                dfg_graph, parameters=parameters
            )
        except Exception:  # pragma: no cover - logged and handled by returning defaults
            LOGGER.exception("Falha ao converter DFG para rede de Petri para métricas")
            return metrics

        from pm4py.algo.evaluation.replay_fitness import (
            algorithm as replay_fitness_algorithm,
        )
        from pm4py.algo.evaluation.precision import algorithm as precision_algorithm
        from pm4py.algo.evaluation.generalization import (
            algorithm as generalization_algorithm,
        )
        from pm4py.algo.evaluation.simplicity import (
            algorithm as simplicity_algorithm,
        )
        from pm4py.util import constants as pm_constants
        from pm4py.util import xes_constants

        eval_params = {
            pm_constants.PARAMETER_CONSTANT_ACTIVITY_KEY: xes_constants.DEFAULT_NAME_KEY,
        }

        def _safe_number(value: Any) -> float | None:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(num) or math.isinf(num):
                return None
            return num

        if not df.empty:
            try:
                fitness_res = replay_fitness_algorithm.apply(
                    df,
                    net,
                    initial_marking,
                    final_marking,
                    parameters=eval_params,
                )
                fitness_val = fitness_res.get("log_fitness")
                if fitness_val is None:
                    fitness_val = fitness_res.get("average_trace_fitness")
                metrics["fitness"] = _safe_number(fitness_val)
            except Exception:
                LOGGER.exception("Falha ao calcular fitness do modelo DFG")

            try:
                precision_val = precision_algorithm.apply(
                    df,
                    net,
                    initial_marking,
                    final_marking,
                    parameters=eval_params,
                    variant=precision_algorithm.Variants.ETCONFORMANCE_TOKEN,
                )
                metrics["precision"] = _safe_number(precision_val)
            except Exception:
                LOGGER.exception("Falha ao calcular precision do modelo DFG")

            try:
                generalization_val = generalization_algorithm.apply(
                    df,
                    net,
                    initial_marking,
                    final_marking,
                    parameters=eval_params,
                )
                metrics["generalization"] = _safe_number(generalization_val)
            except Exception:
                LOGGER.exception("Falha ao calcular generalization do modelo DFG")

        try:
            simplicity_val = simplicity_algorithm.apply(net)
            metrics["simplicity"] = _safe_number(simplicity_val)
        except Exception:
            LOGGER.exception("Falha ao calcular simplicity do modelo DFG")

        return metrics

