from typing import Any

import pandas as pd
import pm4py
import streamlit as st

from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_START_TS,
)
from chatflow_miner.lib.utils.verify import verify_format


def load_dataset(file: str, load_options: dict[Any, Any]) -> pd.DataFrame:
    """
    Carrega um arquivo de log de eventos em formato CSV ou XES e retorna um DataFrame do Pandas.
    """
    try:
        df = pd.read_csv(file, sep=load_options["sep"])
        cols = df.columns.tolist()
        if "duration_seconds" in cols:
            df.drop(columns="duration_seconds", inplace=True)

        verify_format(df)

        df[COLUMN_START_TS] = pd.to_datetime(df[COLUMN_START_TS])
        df[COLUMN_END_TS] = pd.to_datetime(df[COLUMN_END_TS])

        log = pm4py.format_dataframe(
            df,
            case_id=COLUMN_CASE_ID,
            activity_key=COLUMN_ACTIVITY,
            # timestamp_key=COLUMN_START_TS)
            timestamp_key=COLUMN_END_TS,
            start_timestamp_key=COLUMN_START_TS,
        )
        return log

    except ValueError as exp:
        st.error(exp)
        st.stop()
    except Exception:
        st.error("Erro ao carregar arquivo de log")
        st.stop()
