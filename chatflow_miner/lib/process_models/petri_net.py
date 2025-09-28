from __future__ import annotations

from typing import Any, Dict, Tuple
import logging
import math

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

    def compute(self, df: pd.DataFrame) -> Tuple[Any, Any, Any]:
        """Descobre a rede de Petri utilizando o Inductive Miner."""
        import pm4py

        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(df)
        return net, initial_marking, final_marking

    def to_graphviz(
        self,
        model: Tuple[Any, Any, Any],
        *,
        bgcolor: str = "white",
        rankdir: str = "LR",
        format: str = "svg",
        **kwargs: Any,
    ) -> Any:
        """Gera uma visualização Graphviz da rede de Petri."""
        from pm4py.visualization.petri_net import visualizer as pn_visualizer

        net, initial_marking, final_marking = model
        parameters_cls = pn_visualizer.Variants.WO_DECORATION.value.Parameters

        fmt = kwargs.pop("format", format)
        bg = kwargs.pop("bgcolor", bgcolor)
        rd = kwargs.pop("rankdir", rankdir)

        parameters: Dict[str, Any] = {
            parameters_cls.FORMAT: fmt,
            "bgcolor": bg,
            "rankdir": rd,
        }
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
        """Calcula métricas de qualidade para a rede de Petri descoberta."""

        metrics: Dict[str, float | None] = {
            "fitness": None,
            "precision": None,
            "generalization": None,
            "simplicity": None,
        }

        net, initial_marking, final_marking = model

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
            except Exception:  # pragma: no cover - logado abaixo
                LOGGER.exception("Falha ao calcular fitness da rede de Petri")

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
            except Exception:  # pragma: no cover - logado abaixo
                LOGGER.exception("Falha ao calcular precision da rede de Petri")

            try:
                generalization_val = generalization_algorithm.apply(
                    df,
                    net,
                    initial_marking,
                    final_marking,
                    parameters=eval_params,
                )
                metrics["generalization"] = _safe_number(generalization_val)
            except Exception:  # pragma: no cover - logado abaixo
                LOGGER.exception(
                    "Falha ao calcular generalization da rede de Petri",
                )

        try:
            simplicity_val = simplicity_algorithm.apply(net)
            metrics["simplicity"] = _safe_number(simplicity_val)
        except Exception:  # pragma: no cover - logado abaixo
            LOGGER.exception("Falha ao calcular simplicity da rede de Petri")

        return metrics
