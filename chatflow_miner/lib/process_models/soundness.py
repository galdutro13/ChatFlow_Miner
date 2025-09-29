"""Utilidades para avaliar a *soundness* (solidez) de redes de Petri."""

from __future__ import annotations

from copy import deepcopy
import logging
import multiprocessing
import pickle
from typing import Any


LOGGER = logging.getLogger(__name__)


def _woflan_worker(payload: bytes, queue: Any) -> None:
    """Executa o Woflan em subprocesso e devolve resultado booleano ou erro."""

    try:
        net, initial_marking, final_marking, parameters = pickle.loads(payload)
        try:
            from pm4py.algo.analysis.woflan import algorithm as woflan_algorithm
        except Exception:  # pragma: no cover - caminho defensivo para versões antigas
            from pm4py.algo.analysis import woflan as woflan_module  # type: ignore

            woflan_algorithm = woflan_module.algorithm  # type: ignore[attr-defined]

        result = woflan_algorithm.apply(
            net,
            initial_marking,
            final_marking,
            parameters=parameters,
        )
        queue.put(("ok", bool(result)))
    except Exception as exc:  # pragma: no cover - executado no subprocesso
        queue.put(("error", repr(exc)))


def _run_woflan_with_timeout(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    parameters: dict[str, Any],
    *,
    timeout_seconds: float,
    logger: logging.Logger,
) -> bool | None:
    """Executa o Woflan em subprocesso com timeout rígido."""

    try:
        payload = pickle.dumps((net, initial_marking, final_marking, parameters))
    except Exception:
        logger.exception("Falha ao serializar rede de Petri para avaliação de soundness")
        return None

    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:  # pragma: no cover - fallback para ambientes sem 'spawn'
        ctx = multiprocessing.get_context()

    queue = ctx.Queue()
    process = ctx.Process(target=_woflan_worker, args=(payload, queue))
    process.daemon = True
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        logger.warning(
            "Tempo limite excedido ao avaliar soundness via Woflan; retornando None."
        )
        queue.close()
        queue.join_thread()
        process.close()
        return None

    try:
        status, value = queue.get_nowait()
    except Exception:
        logger.warning(
            "Woflan não retornou resultado para soundness; assumindo valor indefinido."
        )
        queue.close()
        queue.join_thread()
        process.close()
        return None

    queue.close()
    queue.join_thread()
    process.close()

    if status == "ok":
        return bool(value)

    logger.warning(
        "Woflan falhou ao avaliar soundness: %s", value
    )
    return None


def evaluate_soundness(
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    *,
    timeout_seconds: float = 10.0,
    logger: logging.Logger | None = None,
) -> bool | None:
    """Avalia a soundness (solidez) de uma rede de Petri.

    Estratégia:
    - Garante que a rede é uma WF-net; caso contrário, soundness não é avaliada.
    - Executa uma verificação rápida (``check_easy_soundness_of_wfnet``) e
      retorna imediatamente quando possível.
    - Antes do Woflan, aplica ``apply_simple_reduction`` em uma cópia profunda
      da rede, reduzindo fragmentos silenciosos sem modificar o modelo original.
    - Executa o Woflan em subprocesso com timeout rígido para evitar explosão de
      espaço de estados; falhas retornam ``None`` com logging informativo.
    """

    logger = logger or LOGGER

    try:
        from pm4py.objects.petri_net.utils import check_soundness as check_soundness_utils
    except Exception:
        logger.exception("Falha ao importar utilitários de soundness da pm4py")
        return None

    try:
        is_wfnet = check_soundness_utils.check_wfnet(net)
    except Exception:
        logger.exception("Falha ao verificar se a rede é uma WF-net")
        return None

    if not is_wfnet:
        logger.info("Rede de Petri não é WF-net; soundness indefinida para a dashboard.")
        return None

    try:
        easy_result = check_soundness_utils.check_easy_soundness_of_wfnet(net)
    except Exception:
        logger.debug("Verificação rápida de soundness falhou", exc_info=True)
        easy_result = None

    if easy_result is True:
        return True
    if easy_result is False:
        logger.debug("Verificação rápida indicou rede não sound.")
        return False

    net_for_woflan = net
    initial_for_woflan = initial_marking
    final_for_woflan = final_marking

    try:
        net_for_woflan = deepcopy(net)
        initial_for_woflan = deepcopy(initial_marking)
        final_for_woflan = deepcopy(final_marking)
        try:
            from pm4py.objects.petri_net.utils import reduction

            reduction.apply_simple_reduction(net_for_woflan)
        except Exception:
            logger.debug(
                "Redução estrutural simples não pôde ser aplicada antes do Woflan.",
                exc_info=True,
            )
    except Exception:
        logger.debug("Não foi possível clonar a rede antes do Woflan", exc_info=True)
        net_for_woflan = net
        initial_for_woflan = initial_marking
        final_for_woflan = final_marking

    try:
        from pm4py.algo.analysis.woflan import algorithm as woflan_algorithm
    except Exception:
        logger.exception("Falha ao importar Woflan para avaliação de soundness")
        return None

    parameters = {
        woflan_algorithm.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
        woflan_algorithm.Parameters.RETURN_DIAGNOSTICS: False,
        woflan_algorithm.Parameters.PRINT_DIAGNOSTICS: False,
    }

    return _run_woflan_with_timeout(
        net_for_woflan,
        initial_for_woflan,
        final_for_woflan,
        parameters,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )

