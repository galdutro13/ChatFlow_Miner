from typing import Any

import pandas as pd

from .base import BaseFilter
from ..constants import *

# -----------------------------------------------------------------------------
# Filtros concretos (exemplos úteis para mineração de processos)
# -----------------------------------------------------------------------------
class AgentFilter(BaseFilter):
    """Mantém eventos cujo ``AGENTE`` corresponde ao escolhido e, opcionalmente,
    **sempre** mantém eventos do sistema (``syst``).

    Esta classe atende ao requisito: filtrar por ``AGENTE=ai`` ou ``AGENTE=human``
    incluindo registros com ``AGENTE == 'syst'`` quando ``include_syst=True``.

    Args:
        agent: "ai" ou "human" (case-insensitive). Qualquer outro valor lança erro.
        include_syst: Se ``True``, mantém sempre eventos com agente ``"syst"``.

    Raises:
        ValueError: se ``agent`` não for "ai" nem "human".
    """

    required_columns = (COLUMN_AGENT,)

    def __init__(self, agent: str, include_syst: bool = True) -> None:
        a = (agent or "").strip().lower()
        if a not in {"ai", "human"}:
            raise ValueError("agent deve ser 'ai' ou 'human'")
        self.agent = a
        self.include_syst = bool(include_syst)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        self._check_columns(df)
        serie = df[COLUMN_AGENT].astype(str).str.lower()
        m = serie == self.agent
        if self.include_syst:
            m = m | (serie == "syst")
        return m


class CaseHasActivityFilter(BaseFilter):
    """Mantém **todos** os eventos dos *cases* que possuem a atividade informada.

    Args:
        activity_name: nome da atividade alvo (match exato).
    """

    required_columns = (COLUMN_CASE_ID, COLUMN_ACTIVITY)

    def __init__(self, activity_name: str) -> None:
        self.activity_name = activity_name

    def mask(self, df: pd.DataFrame) -> pd.Series:
        self._check_columns(df)
        # Identifica CASE_IDs que possuem a atividade alvo
        cases_ok = (
            df.loc[df[COLUMN_ACTIVITY] == self.activity_name, COLUMN_CASE_ID]
              .dropna()
              .unique()
        )
        return df[COLUMN_CASE_ID].isin(cases_ok)


class TimeWindowFilter(BaseFilter):
    """Mantém eventos que intersectam (ou estão contidos em) uma janela temporal.

    A janela é aplicada contra ``START_TIMESTAMP`` e ``END_TIMESTAMP`` (se existir).

    Args:
        start: limite inferior (timestamp-like aceito pelo pandas).
        end: limite superior (timestamp-like aceito pelo pandas).
        mode: "touches" (padrão) mantém eventos que tocam a janela; "inside" mantém
              apenas os totalmente contidos.
    """

    required_columns = (COLUMN_START_TS,)

    def __init__(self, start: Any = None, end: Any = None, mode: str = "touches") -> None:
        self.start = pd.to_datetime(start) if start is not None else None
        self.end = pd.to_datetime(end) if end is not None else None
        self.mode = mode

    def mask(self, df: pd.DataFrame) -> pd.Series:
        self._check_columns(df)
        st = pd.to_datetime(df[COLUMN_START_TS], errors="coerce")
        if COLUMN_END_TS in df.columns:
            en = pd.to_datetime(df[COLUMN_END_TS], errors="coerce")
        else:
            en = st

        m = pd.Series(True, index=df.index)
        if self.start is not None and self.end is not None:
            if self.mode == "inside":
                m &= (st >= self.start) & (en <= self.end)
            else:  # touches (qualquer interseção)
                m &= (en >= self.start) & (st <= self.end)
        elif self.start is not None:
            if self.mode == "inside":
                m &= st >= self.start
            else:
                m &= en >= self.start
        elif self.end is not None:
            if self.mode == "inside":
                m &= en <= self.end
            else:
                m &= st <= self.end
        return m.fillna(False)