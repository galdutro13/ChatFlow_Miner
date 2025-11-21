from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, List, Tuple

import pandas as pd

from .utils import _import_module


def apply_alignments(log: Any, net: Any, im: Any, fm: Any) -> list[dict[str, Any]]:
    alignments = _import_module("pm4py.algo.conformance.alignments.algorithm")
    return alignments.apply(log, net, im, fm)


def _extract_variants(log: Iterable[Any], activity_key: str) -> List[Tuple[Any, ...]]:
    variants: list[tuple[Any, ...]] = []
    for trace in log:
        variant = tuple(event.get(activity_key) for event in trace)
        variants.append(variant)
    return variants


def _move_type_from_step(step: Any) -> tuple[str | None, Any, Any]:
    move_type: str | None = None
    log_move = None
    model_move = None

    if isinstance(step, dict):
        move_type = step.get("move_type")
        pair = step.get("pair") or step.get("move")
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            log_move, model_move = pair[0], pair[1]
    elif isinstance(step, (list, tuple)):
        if len(step) >= 3:
            log_move, model_move, move_type = step[0], step[1], step[2]
        elif len(step) == 2:
            log_move, model_move = step[0], step[1]
            move_type = step[1] if isinstance(step[1], str) else None
        elif len(step) == 1:
            move_type = step[0] if isinstance(step[0], str) else None

    return move_type, log_move, model_move


def _classify_move(step: Any) -> str | None:
    move_type, log_move, model_move = _move_type_from_step(step)
    normalized = (move_type or "").lower()

    if "log" in normalized:
        return "log"
    if "model" in normalized:
        return "model"
    if "sync" in normalized or "synchronous" in normalized:
        return "sync"

    if log_move is not None and model_move is not None:
        if log_move == model_move:
            return "sync"
        if model_move == ">>":
            return "log"
        if log_move == ">>":
            return "model"
    return None


def aggregate_alignment_results(
    log: Iterable[Any], alignment_results: list[dict[str, Any]]
) -> pd.DataFrame:
    xes_constants = _import_module("pm4py.util.xes_constants")
    activity_key = getattr(xes_constants, "DEFAULT_NAME_KEY", "concept:name")

    variants = _extract_variants(log, activity_key)
    if len(variants) != len(alignment_results):
        raise ValueError("log e resultados de alinhamento devem ter o mesmo tamanho")

    aggregated: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "fitness_values": [],
            "cost_values": [],
            "log_moves": 0,
            "model_moves": 0,
            "sync_moves": 0,
            "frequency": 0,
        }
    )

    for variant, result in zip(variants, alignment_results):
        bucket = aggregated[variant]
        bucket["frequency"] += 1
        fitness = result.get("fitness") or result.get("trace_fitness")
        cost = result.get("cost")
        if fitness is not None:
            bucket["fitness_values"].append(float(fitness))
        if cost is not None:
            bucket["cost_values"].append(float(cost))

        for step in result.get("alignment", []) or []:
            move_kind = _classify_move(step)
            if move_kind == "log":
                bucket["log_moves"] += 1
            elif move_kind == "model":
                bucket["model_moves"] += 1
            elif move_kind == "sync":
                bucket["sync_moves"] += 1

    records: list[dict[str, Any]] = []
    for variant, data in aggregated.items():
        fitness_list = data["fitness_values"]
        cost_list = data["cost_values"]
        avg_fitness = sum(fitness_list) / len(fitness_list) if fitness_list else 0.0
        avg_cost = sum(cost_list) / len(cost_list) if cost_list else 0.0
        variant_label = " -> ".join(str(act) for act in variant)
        records.append(
            {
                "variant": variant_label,
                "frequency": data["frequency"],
                "fitness": avg_fitness,
                "cost": avg_cost,
                "n_log_moves": data["log_moves"],
                "n_model_moves": data["model_moves"],
                "n_sync_moves": data["sync_moves"],
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(by=["fitness", "cost"], ascending=[True, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df
