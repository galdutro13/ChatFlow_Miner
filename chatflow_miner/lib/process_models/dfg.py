from typing import Any, Dict, Tuple
import math
import logging

import pandas as pd
import pm4py
from graphviz import Digraph

from .base import BaseProcessModel
from .soundness import evaluate_soundness


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
            df: "pd.DataFrame",
            model: "Tuple[dict, dict, dict]",
    ) -> "Dict[str, float | bool | None]":
        """
        Calcula métricas de qualidade para um DFG convertendo-o para rede de Petri
        e aplicando avaliadores de conformidade de alto desempenho.

        Estratégia de performance e robustez:
        - Converte o DataFrame de eventos para EventLog apenas 1x (evita
          conversões repetidas).
        - Caminho rápido (default):
            * fitness  : token-based replay
            * precision: ETConformance por token-based
        - Fallback alinhado (apenas se necessário):
            * calcula UMA passada de alinhamentos e REUSA:
                - precision via ALIGN_ETCONFORMANCE
                - fitness derivado desses alinhamentos (sem novo replay)
        - generalization e simplicity: como de costume.
        - soundness: avaliada sobre a rede de Petri derivada.
        - Retorna floats, booleanos (para soundness) ou None (se indisponível/erro),
          com logging detalhado.

        Parâmetros de log:
        - Atividade em `xes_constants.DEFAULT_NAME_KEY` (ex.: 'concept:name').

        Parameters
        ----------
        df : pd.DataFrame
            Log de eventos (formato pm4py).
        model : Tuple[dict, dict, dict]
            Tupla ``(dfg, start_activities, end_activities)`` retornada por
            :meth:`compute`.

        Returns
        -------
        Dict[str, float | bool | None]
            Dicionário com métricas: ``fitness``, ``precision``, ``generalization``,
            ``simplicity`` e ``soundness`` (solidez da rede de Petri derivada).
        """
        import math
        import logging

        LOGGER = logging.getLogger(__name__)

        # ---------------------------
        # 1) Converter DFG -> Petri net
        # ---------------------------
        try:
            from pm4py.objects.conversion.dfg import converter as dfg_converter
            from pm4py.objects.conversion.dfg.variants import (
                to_petri_net_activity_defines_place as dfg_to_petri,
            )
            dfg_graph, start_activities, end_activities = model
            conv_params = {
                dfg_to_petri.Parameters.START_ACTIVITIES: start_activities,
                dfg_to_petri.Parameters.END_ACTIVITIES: end_activities,
            }
            net, initial_marking, final_marking = dfg_converter.apply(
                dfg_graph, parameters=conv_params
            )
        except Exception:
            LOGGER.exception("Falha ao converter DFG para rede de Petri para métricas")
            return {
                "fitness": None,
                "precision": None,
                "generalization": None,
                "simplicity": None,
                "soundness": None,
            }

        # ---------------------------
        # 2) Imports dos avaliadores (compatíveis com mudanças de API)
        # ---------------------------
        try:
            from pm4py.algo.evaluation.replay_fitness import algorithm as rf_algorithm
            from pm4py.algo.evaluation.precision import algorithm as prec_algorithm
            from pm4py.algo.evaluation.generalization import algorithm as gen_algorithm
            from pm4py.algo.evaluation.simplicity import algorithm as simp_algorithm
        except Exception:
            # PM4Py em versões antigas/mudanças de pacote
            from pm4py.evaluation import replay_fitness as rf_algorithm  # type: ignore
            from pm4py.evaluation import precision as prec_algorithm  # type: ignore
            from pm4py.evaluation import generalization as gen_algorithm  # type: ignore
            from pm4py.evaluation import simplicity as simp_algorithm  # type: ignore

        # Para reutilizar alinhamentos e derivar fitness a partir deles
        try:
            from pm4py.evaluation.replay_fitness.variants import alignment_based as rf_align_eval  # type: ignore
        except Exception:
            try:
                from pm4py.algo.evaluation.replay_fitness.variants import \
                    alignment_based as rf_align_eval  # type: ignore
            except Exception:
                rf_align_eval = None  # sem reuso de alinhamentos (degradaremos graciosamente)

        from pm4py.util import constants as pm_constants
        from pm4py.util import xes_constants
        import pm4py

        # ---------------------------
        # 3) Utilitários
        # ---------------------------
        def _safe_number(value: "Any") -> "float | None":
            """Converte o valor em float; retorna None para NaN/inf/inválido."""
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(num) or math.isinf(num):
                return None
            return num

        metrics: "Dict[str, float | bool | None]" = {
            "fitness": None,
            "precision": None,
            "generalization": None,
            "simplicity": None,
            "soundness": None,
        }

        try:
            metrics["soundness"] = evaluate_soundness(
                net,
                initial_marking,
                final_marking,
                logger=LOGGER,
            )
        except Exception:  # pragma: no cover - falha defensiva
            LOGGER.exception("Falha inesperada ao calcular soundness do modelo DFG")

        eval_params: "Dict[str, Any]" = {
            pm_constants.PARAMETER_CONSTANT_ACTIVITY_KEY: xes_constants.DEFAULT_NAME_KEY,
        }

        # ---------------------------
        # 4) Simplicity (barata)
        # ---------------------------
        try:
            simp_val = simp_algorithm.apply(net)
            metrics["simplicity"] = _safe_number(simp_val)
        except Exception:
            LOGGER.exception("Falha ao calcular simplicity do modelo DFG")

        # ---------------------------
        # 5) Preparar EventLog 1x
        # ---------------------------
        event_log = None
        if df is not None and not df.empty:
            try:
                event_log = pm4py.convert_to_event_log(df)
            except Exception:
                LOGGER.exception(
                    "Falha ao converter DataFrame para EventLog; seguiremos usando DF onde possível."
                )
                event_log = None
        else:
            # Sem eventos, não há como calcular as métricas baseadas em log
            return metrics

        # ---------------------------
        # 6) Generalization (moderado)
        # ---------------------------
        try:
            gen_val = gen_algorithm.apply(
                event_log if event_log is not None else df,
                net,
                initial_marking,
                final_marking,
                parameters=eval_params,
            )
            metrics["generalization"] = _safe_number(gen_val)
        except Exception:
            LOGGER.exception("Falha ao calcular generalization do modelo DFG")

        # ---------------------------
        # 7) Caminho rápido (token-based) para fitness e precision
        # ---------------------------
        try:
            fitness_res = rf_algorithm.apply(
                event_log if event_log is not None else df,
                net,
                initial_marking,
                final_marking,
                parameters=eval_params,
                variant=getattr(rf_algorithm, "Variants").TOKEN_BASED,
            )
            fitness_val = fitness_res.get("log_fitness") or fitness_res.get("average_trace_fitness")
            metrics["fitness"] = _safe_number(fitness_val)
        except Exception:
            LOGGER.exception(
                "Falha ao calcular fitness (token-based) para o modelo DFG. "
                "Se necessário, tentaremos derivar via alinhamentos."
            )

        precision_ok = False
        try:
            prec_val = prec_algorithm.apply(
                event_log if event_log is not None else df,
                net,
                initial_marking,
                final_marking,
                parameters=eval_params,
                variant=getattr(prec_algorithm, "Variants").ETCONFORMANCE_TOKEN,
            )
            metrics["precision"] = _safe_number(prec_val)
            precision_ok = metrics["precision"] is not None
        except Exception:
            LOGGER.exception(
                "Falha ao calcular precision (ETConformance token-based) para o modelo DFG."
            )
            precision_ok = False

        # ---------------------------
        # 8) Fallback alinhado (1x) se precision falhou/None
        #    Reuso dos alinhamentos para também derivar fitness, se necessário.
        # ---------------------------
        if not precision_ok:
            try:
                alignments = pm4py.conformance.conformance_diagnostics_alignments(
                    event_log if event_log is not None else pm4py.convert_to_event_log(df),
                    net,
                    initial_marking,
                    final_marking,
                    multi_processing=True,
                )

                # Precision via ALIGN_ETCONFORMANCE, injetando alinhamentos quando aceito
                try:
                    metrics["precision"] = _safe_number(
                        prec_algorithm.apply(
                            event_log if event_log is not None else df,
                            net,
                            initial_marking,
                            final_marking,
                            parameters={**eval_params, "aligned_traces": alignments},
                            variant=getattr(prec_algorithm, "Variants").ALIGN_ETCONFORMANCE,
                        )
                    )
                except Exception:
                    # fallback: deixa a variante calcular seus próprios alinhamentos
                    metrics["precision"] = _safe_number(
                        prec_algorithm.apply(
                            event_log if event_log is not None else df,
                            net,
                            initial_marking,
                            final_marking,
                            parameters=eval_params,
                            variant=getattr(prec_algorithm, "Variants").ALIGN_ETCONFORMANCE,
                        )
                    )

                # Se fitness ainda não foi preenchido, derivá-lo dos mesmos alinhamentos
                if metrics["fitness"] is None and rf_align_eval is not None:
                    try:
                        fitness_from_al = rf_align_eval.evaluate(alignments, parameters=eval_params)
                        fitness_val = fitness_from_al.get("log_fitness") or fitness_from_al.get("average_trace_fitness")
                        metrics["fitness"] = _safe_number(fitness_val)
                    except Exception:
                        LOGGER.exception("Falha ao avaliar fitness a partir de alinhamentos pré-computados (DFG).")

            except Exception:
                LOGGER.exception(
                    "Fallback alinhado falhou (cálculo de alinhamentos/precision) para o modelo DFG."
                )

        return metrics


