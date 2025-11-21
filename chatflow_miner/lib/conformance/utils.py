from __future__ import annotations

import importlib
import importlib.util as importlib_util
import sys
from typing import Any, Iterable


class MissingDependencyError(ImportError):
    """Erro específico para dependências ausentes do pm4py."""


def _import_module(module_name: str) -> Any:
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib_util.find_spec(module_name)
    if spec is None:
        raise MissingDependencyError(
            "O módulo pm4py é necessário para análise de conformidade. "
            f"Módulo ausente: {module_name}"
        )
    return importlib.import_module(module_name)


def _import_first_available(module_names: Iterable[str]) -> Any:
    """Return first importable module in ``module_names``.

    Raises ``MissingDependencyError`` when none are available.
    """

    errors: list[str] = []
    for name in module_names:
        try:
            return _import_module(name)
        except MissingDependencyError as exc:  # pragma: no cover - error captured in errors
            errors.append(str(exc))
        except ModuleNotFoundError as exc:
            errors.append(str(exc))

    raise MissingDependencyError("; ".join(errors))


def ensure_marking_obj(net: Any, marking: Any) -> Any:
    """Converte dicionários de marcação em ``Marking`` do pm4py."""

    petrinet_mod = _import_first_available(
        (
            "pm4py.objects.petri.petrinet",  # pm4py <=2.7
            "pm4py.objects.petri_net.utils.petri_utils",  # pm4py >=2.7.14 or alt installs
        )
    )
    Marking = getattr(petrinet_mod, "Marking")

    if isinstance(marking, Marking):
        return marking

    if isinstance(marking, dict):
        result = Marking()
        for place, count in marking.items():
            if place in net.places or getattr(place, "name", None) in net.places:
                result[place] = count
            else:
                result[getattr(place, "name", place)] = count
        return result

    raise TypeError("marking deve ser um dict ou instância de Marking")


def export_pnml(net: Any, im: Any, fm: Any, output_path: str) -> str:
    """Exporta uma rede de Petri em formato PNML."""
    pnml_exporter = _import_module("pm4py.objects.petri.exporter.exporter")
    pnml_exporter.apply(net, im, fm, output_path)
    return output_path


def export_bpmn(model: Any, output_path: str) -> str:
    """Exporta um modelo BPMN."""
    bpmn_exporter = _import_module("pm4py.objects.bpmn.exporter.exporter")
    bpmn_exporter.apply(model, output_path)
    return output_path
