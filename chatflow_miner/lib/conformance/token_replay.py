from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, List, Tuple

import pandas as pd

from .utils import _import_module


def apply_token_replay(log: Any, net: Any, im: Any, fm: Any) -> list[dict[str, Any]]:
    token_replay = _import_module("pm4py.algo.conformance.tokenreplay.algorithm")
    return token_replay.apply(log, net, im, fm)


def _extract_variants(log: Iterable[Any], activity_key: str) -> List[Tuple[Any, ...]]:
    variants: list[tuple[Any, ...]] = []
    for trace in log:
        variant = tuple(event.get(activity_key) for event in trace)
        variants.append(variant)
    return variants


def _collect_places(marking: Any) -> list[str]:
    if marking is None:
        return []

    if hasattr(marking, "items"):
        places = [str(place) for place, count in marking.items() if count]
        return sorted(set(places))
    return []


def aggregate_token_replay_results(
    log: Iterable[Any], replay_results: list[dict[str, Any]]
) -> pd.DataFrame:
    xes_constants = _import_module("pm4py.util.xes_constants")
    activity_key = getattr(xes_constants, "DEFAULT_NAME_KEY", "concept:name")

    variants = _extract_variants(log, activity_key)
    if len(variants) != len(replay_results):
        raise ValueError("log e resultados de replay devem ter o mesmo tamanho")

    aggregated: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "frequency": 0,
            "missing_tokens": 0,
            "remaining_tokens": 0,
            "fitness_values": [],
            "missing_activities": set(),
            "remaining_activities": set(),
        }
    )

    for variant, result in zip(variants, replay_results):
        bucket = aggregated[variant]
        bucket["frequency"] += 1
        bucket["missing_tokens"] += int(result.get("missing_tokens", 0))
        bucket["remaining_tokens"] += int(result.get("remaining_tokens", 0))
        fitness = result.get("trace_fitness")
        if fitness is not None:
            bucket["fitness_values"].append(float(fitness))

        missing_places = _collect_places(result.get("missing_marking"))
        remaining_places = _collect_places(result.get("remaining_marking"))
        bucket["missing_activities"].update(missing_places)
        bucket["remaining_activities"].update(remaining_places)

    records: list[dict[str, Any]] = []
    for variant, data in aggregated.items():
        fitness_list = data["fitness_values"]
        avg_fitness = sum(fitness_list) / len(fitness_list) if fitness_list else 0.0
        variant_label = " -> ".join(str(act) for act in variant)
        records.append(
            {
                "variant": variant_label,
                "frequency": data["frequency"],
                "missing_tokens": data["missing_tokens"],
                "remaining_tokens": data["remaining_tokens"],
                "trace_fitness": avg_fitness,
                "missing_activities": sorted(data["missing_activities"]),
                "remaining_activities": sorted(data["remaining_activities"]),
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(by=["trace_fitness", "frequency"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df
