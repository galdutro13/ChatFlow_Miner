from __future__ import annotations

import logging
import math
from typing import Any, Dict, Tuple

import pandas as pd

from .base import BaseProcessModel

LOGGER = logging.getLogger(__name__)


class PetriNetModel(BaseProcessModel):
    """Modelo de processo baseado em descoberta de redes de Petri.

    Constrói redes de Petri a partir de logs de eventos formatados pela pm4py.

    Este modelo utiliza algoritmos de descoberta (como o Inductive Miner) para
    derivar uma rede de Petri a partir de um DataFrame de eventos já formatado.

    O método :meth:`compute` retorna a tupla ``(net, initial_marking,
    final_marking)`` que pode ser utilizada para visualização ou avaliação do
    modelo.
    """

    def compute(self, df: pd.DataFrame) -> tuple[Any, Any, Any]:
        """Descobre a rede de Petri utilizando o Inductive Miner."""
        import pm4py

        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
            df, noise_threshold=0.2
        )
        return net, initial_marking, final_marking

    def to_graphviz(
        self,
        model: tuple[Any, Any, Any],
        *,
        bgcolor: str = "white",
        rankdir: str = "LR",
        format: str = "svg",
        log: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Gera uma visualização Graphviz da rede de Petri."""
        from pm4py.visualization.petri_net import visualizer as pn_visualizer

        net, initial_marking, final_marking = model
        parameters_cls = pn_visualizer.Variants.WO_DECORATION.value.Parameters

        # Aceita ``log`` para manter compatibilidade com a nova assinatura da
        # classe base, embora a visualização atual não o utilize diretamente.
        _ = log

        fmt = kwargs.pop("format", format)
        bg = kwargs.pop("bgcolor", bgcolor)
        rd = kwargs.pop("rankdir", rankdir)

        parameters: dict[str, Any] = {
            parameters_cls.FORMAT: fmt,
            "bgcolor": bg,
        }

        rankdir_param = getattr(parameters_cls, "RANKDIR", None)
        if rankdir_param is not None:
            parameters[rankdir_param] = rd
        else:
            parameters["rankdir"] = rd
        parameters.update(kwargs)

        gviz = pn_visualizer.apply(
            net,
            initial_marking,
            final_marking,
            parameters=parameters,
            variant=pn_visualizer.Variants.WO_DECORATION,
        )
        return gviz

    def quality_metrics(
        self,
        df: pd.DataFrame,
        model: Tuple[Any, Any, Any],
    ) -> Dict[str, float | None]:
        """
        Calcula métricas de qualidade para a rede de Petri descoberta, evitando
        recomputações caras por alinhamento e priorizando variantes de alta
        performance.

        Estratégia:
        - Converte o DataFrame para EventLog apenas 1x.
        - Caminho rápido (default):
            * fitness: token-based replay (rápido e estável)
            * precision: ETConformance por token-based (rápido)
        - Se precision token-based falhar, faz 1 passada de alinhamentos e REUSA
          o resultado para:
            * precision: ALIGN_ETCONFORMANCE
            * fitness: derivado dos alinhamentos (sem novo replay)
        - generalization e simplicity permanecem inalteradas.
        - Retorna floats ou None (se indisponível/erro), com logging detalhado.

        Parâmetros esperados no log:
        - Atividade em `xes_constants.DEFAULT_NAME_KEY` (ex.: 'concept:name').

        Returns
        -------
        Dict[str, float | None]
            Dicionário com métricas: fitness, precision, generalization, simplicity.
        """
        # Imports locais para robustez a mudanças de API entre versões
        try:
            from pm4py.algo.evaluation.generalization import algorithm as gen_algorithm
            from pm4py.algo.evaluation.precision import algorithm as prec_algorithm
            from pm4py.algo.evaluation.replay_fitness import algorithm as rf_algorithm
            from pm4py.algo.evaluation.simplicity import algorithm as simp_algorithm
        except Exception:  # fallback para caminhos antigos/novos
            from pm4py.evaluation import generalization as gen_algorithm  # type: ignore
            from pm4py.evaluation import precision as prec_algorithm  # type: ignore
            from pm4py.evaluation import replay_fitness as rf_algorithm  # type: ignore
            from pm4py.evaluation import simplicity as simp_algorithm  # type: ignore

        # Para avaliar fitness a partir de alinhamentos já calculados (reuso)
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

        def _safe_number(value: Any) -> float | None:
            """Converte para float robustamente, retornando None se NaN/inf ou inválido."""
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(num) or math.isinf(num):
                return None
            return num

        metrics: Dict[str, float | None] = {
            "fitness": None,
            "precision": None,
            "generalization": None,
            "simplicity": None,
        }

        net, initial_marking, final_marking = model

        # Converte o DF para EventLog apenas uma vez (evita custo repetido)
        event_log = None
        if df is not None and not df.empty:
            try:
                event_log = pm4py.convert_to_event_log(df)
            except Exception:
                # Se a conversão falhar, ainda tentaremos com DF onde possível.
                LOGGER.exception(
                    "Falha ao converter DataFrame para EventLog; prosseguindo com DF quando suportado."
                )
                event_log = None

        # Parâmetros compartilhados
        eval_params: Dict[str, Any] = {
            pm_constants.PARAMETER_CONSTANT_ACTIVITY_KEY: xes_constants.DEFAULT_NAME_KEY,
        }

        # ---------------------------
        # Simplicity (barata)
        # ---------------------------
        try:
            simp_val = simp_algorithm.apply(net)
            metrics["simplicity"] = _safe_number(simp_val)
        except Exception:
            LOGGER.exception("Falha ao calcular simplicity da rede de Petri")

        # Nada mais a fazer se não há eventos
        if event_log is None and (df is None or df.empty):
            return metrics

        # ---------------------------
        # Generalization (moderado)
        # ---------------------------
        try:
            # PM4Py aceita EventLog ou DF; preferimos EventLog se disponível
            gen_val = gen_algorithm.apply(
                event_log if event_log is not None else df,
                net,
                initial_marking,
                final_marking,
                parameters=eval_params,
            )
            metrics["generalization"] = _safe_number(gen_val)
        except Exception:
            LOGGER.exception("Falha ao calcular generalization da rede de Petri")

        # ---------------------------
        # Caminho RÁPIDO por padrão:
        #   - Fitness: TOKEN_BASED
        #   - Precision: ETCONFORMANCE_TOKEN
        # ---------------------------
        # Fitness (token-based)
        try:
            fitness_res = rf_algorithm.apply(
                event_log if event_log is not None else df,
                net,
                initial_marking,
                final_marking,
                parameters=eval_params,
                variant=getattr(
                    rf_algorithm, "Variants"
                ).TOKEN_BASED,  # força caminho rápido
            )
            # PM4Py retorna dict com 'log_fitness' ou 'average_trace_fitness'
            fitness_val = fitness_res.get("log_fitness") or fitness_res.get(
                "average_trace_fitness"
            )
            metrics["fitness"] = _safe_number(fitness_val)
        except Exception:
            LOGGER.exception(
                "Falha ao calcular fitness (token-based). Tentaremos reaproveitar alinhamentos se existirem."
            )

        # Precision (token-based ETConformance)
        try:
            metrics["precision"] = _safe_number(
                prec_algorithm.apply(
                    event_log if event_log is not None else df,
                    net,
                    initial_marking,
                    final_marking,
                    parameters=eval_params,
                    variant=getattr(
                        prec_algorithm, "Variants"
                    ).ETCONFORMANCE_TOKEN,  # rápido
                )
            )
        except Exception:
            LOGGER.exception(
                "Falha ao calcular precision (ETConformance por token-based)."
            )

        return metrics
