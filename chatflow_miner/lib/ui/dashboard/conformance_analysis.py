from __future__ import annotations

from typing import Any

import streamlit as st

from chatflow_miner.lib.conformance.alignments import (
    aggregate_alignment_results,
    apply_alignments,
)
from chatflow_miner.lib.conformance.token_replay import (
    aggregate_token_replay_results,
    apply_token_replay,
)
from chatflow_miner.lib.conformance.utils import MissingDependencyError, _import_module
from chatflow_miner.lib.state import get_log_eventos
from chatflow_miner.lib.ui.conformance.inputs import render_normative_model_selector


def _get_event_log(log_df: Any) -> Any:
    pm4py = _import_module("pm4py")
    return pm4py.convert_to_event_log(log_df)


def _render_model_summary(normative_model: dict[str, Any]) -> None:
    net = normative_model.get("net")
    if net is None:
        st.caption("Nenhum modelo normativo carregado.")
        return

    places = len(getattr(net, "places", []) or [])
    transitions = len(getattr(net, "transitions", []) or [])
    st.info(f"Rede de Petri: {places} lugares, {transitions} transições", icon="ℹ️")


@st.fragment
def _render_token_replay_tab(log_df: Any, normative_model: dict[str, Any]) -> None:
    net = normative_model.get("net")
    im = normative_model.get("initial_marking")
    fm = normative_model.get("final_marking")

    if st.button("Executar Replay", key="run_token_replay", disabled=net is None):
        if net is None or im is None or fm is None:
            st.warning("Carregue um modelo normativo antes de executar o replay")
            return
        if log_df is None:
            st.warning("Carregue um log de eventos para executar o replay")
            return

        with st.spinner("Calculando tokens..."):
            try:
                event_log = _get_event_log(log_df)
                replay_results = apply_token_replay(event_log, net, im, fm)
                aggregated = aggregate_token_replay_results(event_log, replay_results)
            except MissingDependencyError as exc:
                st.error(str(exc))
                return

        mean_fitness = (
            float(aggregated["trace_fitness"].mean()) if not aggregated.empty else 0.0
        )
        st.metric("Fitness Média (Trace Level)", f"{mean_fitness:.3f}")

        display_df = aggregated.copy()
        display_df["is_fit"] = (
            (display_df["missing_tokens"] == 0) & (display_df["remaining_tokens"] == 0)
        )

        st.dataframe(
            display_df,
            hide_index=True,
            column_config={
                "is_fit": st.column_config.CheckboxColumn(
                    "Traço conforme", disabled=True
                )
            },
            use_container_width=True,
        )


@st.fragment
def _render_alignment_tab(log_df: Any, normative_model: dict[str, Any]) -> None:
    net = normative_model.get("net")
    im = normative_model.get("initial_marking")
    fm = normative_model.get("final_marking")

    if st.button("Executar Alinhamento", key="run_alignments", disabled=net is None):
        if net is None or im is None or fm is None:
            st.warning("Carregue um modelo normativo antes de executar alinhamentos")
            return
        if log_df is None:
            st.warning("Carregue um log de eventos para executar alinhamentos")
            return

        status = st.status("Iniciando alinhamento", expanded=False)
        try:
            event_log = _get_event_log(log_df)
            status.update(label="Calculando alinhamentos...", state="running")
            alignment_results = apply_alignments(event_log, net, im, fm)
            aggregated = aggregate_alignment_results(event_log, alignment_results)
        except MissingDependencyError as exc:
            status.update(label="Dependência ausente", state="error")
            st.error(str(exc))
            return
        status.update(label="Alinhamentos concluídos", state="complete")

        mean_cost = float(aggregated["cost"].mean()) if not aggregated.empty else 0.0
        st.metric("Custo Médio de Alinhamento", f"{mean_cost:.3f}")

        max_cost = aggregated["cost"].max() if not aggregated.empty else 1.0
        st.dataframe(
            aggregated,
            hide_index=True,
            column_config={
                "cost": st.column_config.ProgressColumn(
                    "Custo", min_value=0.0, max_value=float(max_cost)
                ),
            },
            use_container_width=True,
        )

def render_conformance_analysis() -> None:
    log_df = get_log_eventos(which="log_eventos")

    with st.container(border=True):
        render_normative_model_selector(log_df)
        st.divider()

        _render_model_summary(st.session_state.get("normative_model", {}))

    token_tab, align_tab = st.tabs([
        "Token-Based Replay (Rápido)",
        "Alignments (Preciso)",
    ])

    with token_tab:
        _render_token_replay_tab(log_df, st.session_state.get("normative_model", {}))
    with align_tab:
        _render_alignment_tab(log_df, st.session_state.get("normative_model", {}))
