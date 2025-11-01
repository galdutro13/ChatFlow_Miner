"""Concrete case aggregators (variant, case_date)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from chatflow_miner.lib.aggregations.base import BaseCaseAggregator
from chatflow_miner.lib.aggregations.models import VariantInfo
from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_AGENT,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_EVENT_ID,
    COLUMN_START_TS,
)

logger = logging.getLogger(__name__)


class CaseVariantAggregator(BaseCaseAggregator):
    """Computa a **variante** de cada case e devolve `{case_id: VariantInfo}`.

    Regras:
      - Ordena eventos por `START_TIMESTAMP` (fallback por `EVENT_ID`).
      - Opcionalmente ignora eventos com `AGENTE == 'syst'`.
      - Concatena `ACTIVITY` com `joiner` (padrão: ">").
      - Gera `variant_id` como SHA1 da string.
      - Frequência por variante é calculada **globalmente** em `prepare`.
    """

    required_columns = (
        COLUMN_CASE_ID,
        COLUMN_ACTIVITY,
        COLUMN_START_TS,
        COLUMN_EVENT_ID,
    )

    def __init__(self, *, ignore_syst: bool = False, joiner: str = ">") -> None:
        self.ignore_syst = bool(ignore_syst)
        self.joiner = joiner

    def prepare(
        self, df: pd.DataFrame
    ) -> tuple[dict[Any, str], dict[str, VariantInfo]]:
        self._check_columns(df)
        df2 = df.copy()

        df2[COLUMN_START_TS] = pd.to_datetime(
            df2[COLUMN_START_TS], errors="coerce", format="mixed"
        )
        try:
            ev_order = pd.to_numeric(df2[COLUMN_EVENT_ID], errors="coerce")
        except Exception:
            ev_order = df2[COLUMN_EVENT_ID]
        df2["__EV_ORDER__"] = ev_order

        if self.ignore_syst and COLUMN_AGENT in df2.columns:
            mask = df2[COLUMN_AGENT].astype(str).str.lower() != "syst"
            df2 = df2.loc[mask]

        df2 = df2.sort_values(
            [COLUMN_CASE_ID, COLUMN_START_TS, "__EV_ORDER__"], kind="mergesort"
        )
        seq = df2.groupby(COLUMN_CASE_ID)[COLUMN_ACTIVITY].agg(list)
        case_to_variant: dict[Any, str] = seq.apply(
            lambda xs: self.joiner.join(map(str, xs)) if len(xs) else ""
        ).to_dict()

        freq = pd.Series(case_to_variant).value_counts()

        var_to_info: dict[str, VariantInfo] = {}
        # Assign human-friendly sequential variant IDs in descending frequency order
        # e.g. 'variant 1' for the most frequent variant, 'variant 2' for the next, etc.
        for idx, (variant_str, frequency) in enumerate(freq.items(), start=1):
            # ensure variant_str is str
            variant_s = str(variant_str)
            variant_id = f"variant {idx}"
            var_to_info[variant_s] = VariantInfo(
                variant_id=variant_id,
                variant=variant_s,
                frequency=int(frequency),
                length=len(variant_s.split(self.joiner)) if variant_s else 0,
            )

        return case_to_variant, var_to_info

    def compute_case(
        self,
        case_df: pd.DataFrame,
        state: tuple[dict[Any, str], dict[str, VariantInfo]],
    ) -> VariantInfo:
        case_to_variant, var_to_info = state
        case_id = case_df.iloc[0][COLUMN_CASE_ID]
        vstr = case_to_variant.get(case_id, "")
        return var_to_info[vstr]


class CaseDateAggregator(BaseCaseAggregator):
    """Agregador que retorna a data do caso (YYYY-MM-DD) usando o menor
    valor de `COLUMN_START_TS` encontrado em cada case.

    Detalhes
    -------
    - required_columns: (COLUMN_CASE_ID, COLUMN_START_TS)
    - `prepare(df)`: não mantém estado global (retorna None).
    - `compute_case(case_df, state)`: converte a coluna `COLUMN_START_TS` em
      datetime, calcula o mínimo e retorna a string ISO da data (`YYYY-MM-DD`)
      ou `None` quando não houver timestamp válido.

    Retorno
    ------
    - str | None: data em formato ISO (`YYYY-MM-DD`) ou `None` quando ausente.

    Exemplo de uso
    --------------
    Crie uma instância de `CaseDateAggregator` e aplique via `CaseAggView` após
    normalizar timestamps com `NormalizeTimestampsOp`.

    Exemplo (conceitual):
        agg = CaseDateAggregator()
        view = CaseAggView(df).with_aux(NormalizeTimestampsOp()).with_aggregator(agg)
        result = view.compute()
    """

    required_columns = (COLUMN_CASE_ID, COLUMN_START_TS)

    def prepare(self, df: pd.DataFrame) -> None:
        return None

    def compute_case(self, case_df: pd.DataFrame, state: Any) -> Any:
        st = pd.to_datetime(case_df[COLUMN_START_TS], errors="coerce", format="mixed")
        d = st.min()
        return None if pd.isna(d) else d.date().isoformat()


class CaseDurationAggregator(BaseCaseAggregator):
    """Calcula a duração total de cada caso como ``pd.Timedelta``.

    Pré-requisitos
    ---------------
    - ``NormalizeTimestampsOp`` deve ser aplicado antes da agregação para
      garantir que as colunas de timestamp estejam normalizadas.
    - Em caso de ``END_TIMESTAMP`` ausente ou inválido, o evento é tratado
      como se terminasse no ``START_TIMESTAMP`` correspondente.
    """

    required_columns = (COLUMN_CASE_ID, COLUMN_START_TS, COLUMN_END_TS)

    def prepare(self, df: pd.DataFrame) -> None:
        self._check_columns(df)
        return None

    def compute_case(self, case_df: pd.DataFrame, state: Any) -> pd.Timedelta:
        start_ts = pd.to_datetime(
            case_df[COLUMN_START_TS], errors="coerce", format="mixed"
        )
        end_ts = pd.to_datetime(
            case_df[COLUMN_END_TS], errors="coerce", format="mixed"
        )

        if start_ts.empty:
            return pd.Timedelta(0)

        start_min = start_ts.min()
        if pd.isna(start_min):
            return pd.Timedelta(0)

        # Substitui END_TIMESTAMP ausente/NaT pelo START_TIMESTAMP do evento
        end_filled = end_ts.fillna(start_ts)
        end_max = end_filled.max()
        if pd.isna(end_max):
            end_max = start_min

        duration = end_max - start_min
        if isinstance(duration, pd.Timedelta):
            return duration
        return pd.Timedelta(duration)
