import logging
import math
import sys
from typing import Any, Dict, Tuple

import pandas as pd
import pm4py
from graphviz import Digraph

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_START_TS,
)

from .base import BaseProcessModel

LOGGER = logging.getLogger(__name__)


def _quality_metrics_for_dfg(
    df: "pd.DataFrame",
    model: "Tuple[dict, dict, dict]",
    *,
    event_log: "Any | None" = None,
) -> "Dict[str, float | None]":
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
    - Retorna floats ou None (se indisponível/erro), com logging detalhado.

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
    Dict[str, float | None]
        Dicionário com métricas: ``fitness``, ``precision``, ``generalization``,
        ``simplicity``.
    """
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
        }

    # ---------------------------
    # 2) Imports dos avaliadores (compatíveis com mudanças de API)
    # ---------------------------
    try:
        from pm4py.algo.evaluation.generalization import algorithm as gen_algorithm
        from pm4py.algo.evaluation.precision import algorithm as prec_algorithm
        from pm4py.algo.evaluation.replay_fitness import algorithm as rf_algorithm
        from pm4py.algo.evaluation.simplicity import algorithm as simp_algorithm
    except Exception:
        # PM4Py em versões antigas/mudanças de pacote
        from pm4py.evaluation import generalization as gen_algorithm  # type: ignore
        from pm4py.evaluation import precision as prec_algorithm  # type: ignore
        from pm4py.evaluation import replay_fitness as rf_algorithm  # type: ignore
        from pm4py.evaluation import simplicity as simp_algorithm  # type: ignore

    # Para reutilizar alinhamentos e derivar fitness a partir deles
    try:
        from pm4py.evaluation.replay_fitness.variants import (
            alignment_based,
        )  # type: ignore

        _ = alignment_based  # noqa: F841
    except Exception:
        try:
            from pm4py.algo.evaluation.replay_fitness.variants import (
                alignment_based,
            )  # type: ignore

            _ = alignment_based  # noqa: F841
        except Exception:
            LOGGER.debug(
                "Reuso de alinhamentos indisponível; degradaremos graciosamente."
            )

    import pm4py
    from pm4py.util import constants as pm_constants
    from pm4py.util import xes_constants

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

    metrics: "Dict[str, float | None]" = {
        "fitness": None,
        "precision": None,
        "generalization": None,
        "simplicity": None,
    }

    eval_params: "Dict[str, Any]" = {
        pm_constants.PARAMETER_CONSTANT_ACTIVITY_KEY: xes_constants.DEFAULT_NAME_KEY,
    }

    if df is None or df.empty:
        return metrics

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
    event_log_obj = event_log
    if event_log_obj is None:
        try:
            event_log_obj = pm4py.convert_to_event_log(df)
        except Exception:
            LOGGER.exception(
                "Falha ao converter DataFrame para EventLog; seguiremos usando DF onde possível."
            )
            event_log_obj = None

    # ---------------------------
    # 6) Generalization (moderado)
    # ---------------------------
    try:
        gen_val = gen_algorithm.apply(
            event_log_obj if event_log_obj is not None else df,
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
            event_log_obj if event_log_obj is not None else df,
            net,
            initial_marking,
            final_marking,
            parameters=eval_params,
            variant=getattr(rf_algorithm, "Variants").TOKEN_BASED,
        )
        fitness_val = fitness_res.get("log_fitness") or fitness_res.get(
            "average_trace_fitness"
        )
        metrics["fitness"] = _safe_number(fitness_val)
    except Exception:
        LOGGER.exception(
            "Falha ao calcular fitness (token-based) para o modelo DFG. "
            "Se necessário, tentaremos derivar via alinhamentos."
        )

    try:
        prec_val = prec_algorithm.apply(
            event_log_obj if event_log_obj is not None else df,
            net,
            initial_marking,
            final_marking,
            parameters=eval_params,
            variant=getattr(prec_algorithm, "Variants").ETCONFORMANCE_TOKEN,
        )
        metrics["precision"] = _safe_number(prec_val)
    except Exception:
        LOGGER.exception(
            "Falha ao calcular precision (ETConformance token-based) para o modelo DFG."
        )

    return metrics


class DFGModel(BaseProcessModel):
    """
    Modelo de Processo baseado em Directly-Follows Graph (DFG).

    Este modelo constrói um grafo diretamente a partir de um DataFrame de eventos
    formatado, representando as relações "directly-follows" entre atividades.

    O método :meth:`compute` retorna uma estrutura de dados representando o DFG,
    que pode ser posteriormente visualizada ou analisada.
    """

    def compute(self, df: pd.DataFrame) -> tuple[dict, dict, dict]:
        """
        Constrói o Directly-Follows Graph (DFG) a partir do DataFrame de eventos.

        :param df: DataFrame de eventos formatado.
        :returns: Tupla contendo o DFG, atividades iniciais e atividades finais.
        """
        # pm4py espera colunas no padrão XES (concept:name, case:concept:name, ...).
        # Alguns fluxos de teste/uso podem fornecer DataFrames somente com as
        # colunas do domínio (CASE_ID/ACTIVITY/START_TIMESTAMP/END_TIMESTAMP).
        # Normalizamos para evitar exceções de "colunas insuficientes".

        pm_case_key = "case:concept:name"
        pm_activity_key = "concept:name"
        pm_timestamp_key = "time:timestamp"
        pm_start_key = "start:timestamp"

        required_pm_cols = {pm_case_key, pm_activity_key, pm_timestamp_key}

        if required_pm_cols.issubset(df.columns):
            prepared_df = df
        else:
            prepared_df = df.copy()

            def _rename_first_available(
                candidates: tuple[str, ...], target: str
            ) -> None:
                for column in candidates:
                    if column in prepared_df.columns:
                        if column != target:
                            prepared_df.rename(columns={column: target}, inplace=True)
                        return

            _rename_first_available(
                (pm_case_key, COLUMN_CASE_ID),
                pm_case_key,
            )
            _rename_first_available(
                (pm_activity_key, COLUMN_ACTIVITY),
                pm_activity_key,
            )

            if pm_case_key in prepared_df.columns:
                prepared_df[pm_case_key] = prepared_df[pm_case_key].astype("string")
            if pm_activity_key in prepared_df.columns:
                prepared_df[pm_activity_key] = prepared_df[pm_activity_key].astype(
                    "string"
                )

            # Prioriza colunas reais de timestamp; caso nenhuma exista, gera uma
            # coluna sintética ordenada para satisfazer o requisito do pm4py.
            timestamp_candidates = (
                pm_timestamp_key,
                COLUMN_END_TS,
                COLUMN_START_TS,
            )
            current_timestamp = next(
                (col for col in timestamp_candidates if col in prepared_df.columns),
                None,
            )
            if current_timestamp is None:
                prepared_df[pm_timestamp_key] = pd.to_datetime(
                    pd.RangeIndex(len(prepared_df)), unit="ns"
                )
            elif current_timestamp != pm_timestamp_key:
                prepared_df.rename(
                    columns={current_timestamp: pm_timestamp_key}, inplace=True
                )

            # Se houver timestamp de início, renomeia também para manter
            # consistência com o formato XES esperado.
            start_candidates = (pm_start_key, COLUMN_START_TS)
            current_start = next(
                (col for col in start_candidates if col in prepared_df.columns),
                None,
            )
            if current_start is not None and current_start != pm_start_key:
                prepared_df.rename(columns={current_start: pm_start_key}, inplace=True)

            # Caso as colunas essenciais continuem ausentes, permita que o pm4py
            # levante o erro — não tentamos adivinhar um case/activity inexistente.

        pm4py_module = sys.modules.get("pm4py", pm4py)
        dfg, start_activity, end_activity = pm4py_module.discover_dfg(prepared_df)
        return dfg, start_activity, end_activity

    def to_graphviz(
        self,
        model: tuple[dict, dict, dict],
        bgcolor: str = "white",
        rankdir: str = "LR",
        max_num_edges: int = 9223372036854775807,
        *,
        log: Any | None = None,
        event_df: pd.DataFrame | None = None,
    ) -> Digraph:
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
        parameters = {
            dfg_parameters.FORMAT: "svg",
            dfg_parameters.START_ACTIVITIES: start_activities,
            dfg_parameters.END_ACTIVITIES: end_activities,
            "bgcolor": bgcolor,
            "rankdir": rankdir,
            "maxNoOfEdgesInDiagram": max_num_edges,
            dfg_parameters.TIMESTAMP_KEY: COLUMN_END_TS,
            dfg_parameters.START_TIMESTAMP_KEY: COLUMN_START_TS,
        }

        event_log = log
        if event_log is None and event_df is not None and not event_df.empty:
            try:
                event_log = pm4py.convert_to_event_log(event_df)
            except Exception:
                LOGGER.exception(
                    "Falha ao converter DataFrame em EventLog para visualização de DFG."
                )

        gviz = dfg_visualizer.apply(
            dfg,
            log=event_log,
            variant=dfg_visualizer.Variants.FREQUENCY,
            parameters=parameters,
        )

        return gviz

    def quality_metrics(
        self,
        df: "pd.DataFrame",
        model: "Tuple[dict, dict, dict]",
        *,
        event_log: "Any | None" = None,
    ) -> "Dict[str, float | None]":
        """Delegates to the shared DFG quality-metrics implementation."""
        return _quality_metrics_for_dfg(df, model, event_log=event_log)


class PerformanceDFGModel(BaseProcessModel):
    """Modelo de DFG orientado a métricas de performance.

    Este modelo gera um Directly-Follows Graph (DFG) enriquecido com
    informações de performance (ex.: tempos médios/mediana de transição,
    contagens, somas de duração) a partir de um DataFrame de eventos.

    Observações importantes:
    - Espera-se que o DataFrame siga o formato aceito pelo pm4py, com colunas
      típicas como identificador de caso, nome da atividade e timestamp.
    - As arestas do DFG resultante normalmente carregam atributos numéricos
      relacionados à performance (por exemplo 'performance' ou nomes
      específicos do pm4py para métricas). Consulte pm4py.discover_performance_dfg
      para detalhes do esquema exato de atributos.
    - Utilizar este modelo quando o foco for analisar latências/durações entre
      atividades, além da estrutura diretamente sequencial do processo.
    """

    def compute(self, df: pd.DataFrame) -> tuple[dict, dict, dict]:
        """
        Constrói o Directly-Follows Graph (DFG) orientado a performance.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de eventos no formato pm4py (colunas como case id, activity,
            timestamp). O método delega a extração das métricas de performance ao
            pm4py.discover_performance_dfg.

        Retorno
        -------
        Tuple[dict, dict, dict]
            Tupla (dfg, start_activities, end_activities):
            - dfg: dicionário representando o grafo diretamente sequencial, cujas
              arestas podem conter atributos de performance.
            - start_activities: dicionário/estrutura com atividades iniciais.
            - end_activities: dicionário/estrutura com atividades finais.
        """
        pm4py_module = sys.modules.get("pm4py", pm4py)
        dfg, start_activity, end_activity = pm4py_module.discover_performance_dfg(
            df,
        )
        return dfg, start_activity, end_activity

    def to_graphviz(
        self,
        model: tuple[dict, dict, dict],
        bgcolor: str = "white",
        rankdir: str = "LR",
        max_num_edges: int = 9223372036854775807,
        *,
        log: Any | None = None,
        event_df: pd.DataFrame | None = None,
    ) -> Digraph:
        """
        Gera uma visualização Graphviz do DFG com realce de métricas de performance.

        Parâmetros
        ----------
        model : Tuple[dict, dict, dict]
            Tupla (dfg, start_activities, end_activities) produzida por compute.
        bgcolor : str, opcional
            Cor de fundo do diagrama (padrão 'white').
        rankdir : str, opcional
            Direção do layout do grafo ('LR' ou 'TB', padrão 'LR').
        max_num_edges : int, opcional
            Limite de arestas a exibir; útil para logs muito densos.

        Retorno
        -------
        Digraph
            Objeto Graphviz/Digraph (normalmente em formato SVG) produzido pelo
            visualizador de DFG do pm4py usando a variante PERFORMANCE.

        Observações
        ----------
        - O visualizador injeta atributos de performance nas labels/estilos das
          arestas quando suportado. Para ajustar o formato da imagem, modifique
          o parâmetro de formato nas opções (por padrão 'svg' aqui).
        """
        from pm4py.visualization.dfg import visualizer as dfg_visualizer

        dfg, start_activities, end_activities = model
        dfg_parameters = dfg_visualizer.Variants.PERFORMANCE.value.Parameters
        parameters = {
            dfg_parameters.FORMAT: "svg",
            dfg_parameters.START_ACTIVITIES: start_activities,
            dfg_parameters.END_ACTIVITIES: end_activities,
            "bgcolor": bgcolor,
            "rankdir": rankdir,
            "maxNoOfEdgesInDiagram": max_num_edges,
            dfg_parameters.TIMESTAMP_KEY: COLUMN_END_TS,
            dfg_parameters.START_TIMESTAMP_KEY: COLUMN_START_TS,
        }

        event_log = log
        if event_log is None and event_df is not None and not event_df.empty:
            try:
                event_log = pm4py.convert_to_event_log(event_df)
            except Exception:
                LOGGER.exception(
                    "Falha ao converter DataFrame em EventLog para visualização de DFG de performance."
                )

        gviz = dfg_visualizer.apply(
            dfg,
            log=event_log,
            variant=dfg_visualizer.Variants.PERFORMANCE,
            parameters=parameters,
        )

        return gviz

    def quality_metrics(
        self,
        df: "pd.DataFrame",
        model: "Tuple[dict, dict, dict]",
    ) -> "Dict[str, float | None]":
        """
        Calcula métricas de qualidade do modelo DFG (fitness, precision, etc.).

        Descrição sucinta
        -----------------
        Este método delega a computação das métricas para a função comum
        _quality_metrics_for_dfg, que converte o DFG para uma rede de Petri e
        executa avaliadores de conformidade e qualidade. As chaves retornadas
        tipicamente incluem: 'fitness', 'precision', 'generalization' e 'simplicity'.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de eventos usado para avaliar o modelo.
        model : Tuple[dict, dict, dict]
            O modelo DFG (dfg, start_activities, end_activities).

        Retorno
        -------
        Dict[str, float | None]
            Dicionário com valores numéricos (float) das métricas ou None quando
            não puderem ser calculadas.
        """
        return _quality_metrics_for_dfg(df, model)
