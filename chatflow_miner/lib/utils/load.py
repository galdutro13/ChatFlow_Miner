from typing import Any, Dict, Tuple

import pandas as pd
import pm4py
import streamlit as st

from chatflow_miner.lib.utils.verify import verify_format

def load_dataset(file: str, load_options: Dict[Any, Any]) -> pd.DataFrame:
    """
    Carrega um arquivo de log de eventos em formato CSV ou XES e retorna um DataFrame do Pandas.
    """
    try:
        df = pd.read_csv(file, sep=load_options["sep"])
        cols = df.columns.tolist()
        if "duration_seconds" in cols:
            df.drop(columns="duration_seconds", inplace=True)

        verify_format(df)

        df['START_TIMESTAMP'] = pd.to_datetime(df['START_TIMESTAMP'])
        df['END_TIMESTAMP'] = pd.to_datetime(df['END_TIMESTAMP'])

        log = pm4py.format_dataframe(df,
                                     case_id='CASE_ID',
                                     activity_key='ACTIVITY',
                                     timestamp_key='END_TIMESTAMP',
                                     start_timestamp_key='START_TIMESTAMP')
        return log

    except ValueError as exp:
        st.error(exp)
        st.stop()
    except:
        st.error("Erro ao carregar arquivo de log")
        st.stop()