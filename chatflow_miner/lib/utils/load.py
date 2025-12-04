from pathlib import Path
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


def load_dataset(file: Any, load_options: dict[Any, Any]) -> pd.DataFrame:
    """
    Carrega um arquivo de log de eventos em formato CSV ou XES e retorna um DataFrame do Pandas.
    """
    try:
        ext = Path(getattr(file, "name", "")).suffix.lower()
        if ext == ".xes":
            df = pm4py.convert_to_dataframe(pm4py.read_xes(file))
            df = df.rename(
                columns={
                    "case:concept:name": COLUMN_CASE_ID,
                    "concept:name": COLUMN_ACTIVITY,
                    "time:timestamp": COLUMN_END_TS,
                    "start_timestamp": COLUMN_START_TS,
                }
            )
            if COLUMN_START_TS not in df.columns and COLUMN_END_TS in df.columns:
                df[COLUMN_START_TS] = df[COLUMN_END_TS]
        else:
            df = pd.read_csv(file, sep=load_options.get("sep", ","))
            df = df.drop(columns="duration_seconds", errors="ignore").copy()

        verify_format(df)

        for column in (COLUMN_CASE_ID, COLUMN_ACTIVITY):
            df[column] = df[column].astype("string").str.strip()
            empty_mask = df[column].fillna("").eq("")
            df.loc[empty_mask, column] = pd.NA

        for column in (COLUMN_START_TS, COLUMN_END_TS):
            df[column] = df[column].astype("string").str.strip()
            empty_mask = df[column].fillna("").eq("")
            df.loc[empty_mask, column] = pd.NA
            df[column] = pd.to_datetime(df[column])

        required_columns = [
            COLUMN_CASE_ID,
            COLUMN_ACTIVITY,
            COLUMN_START_TS,
            COLUMN_END_TS,
        ]
        missing_mask = df[required_columns].isna().any(axis=1)
        if missing_mask.any():
            df = df.loc[~missing_mask].copy()

        log = pm4py.format_dataframe(
            df,
            case_id=COLUMN_CASE_ID,
            activity_key=COLUMN_ACTIVITY,
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
