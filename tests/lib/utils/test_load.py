import io
import textwrap
import warnings
from pathlib import Path

import pandas as pd

from chatflow_miner.lib.utils import load_dataset

_WARNING_MESSAGE = "Some rows of the Pandas data frame have been removed"


def test_load_dataset_filters_out_rows_with_missing_required_fields():
    csv_text = textwrap.dedent(
        """
        CASE_ID;EVENT_ID;ACTIVITY;START_TIMESTAMP;END_TIMESTAMP;AGENTE
        case-1;0;start;2024-01-01 00:00:00;2024-01-01 00:01:00;bot
        case-1;1;;2024-01-01 00:01:00;2024-01-01 00:02:00;bot
        case-1;2;   ;2024-01-01 00:02:00;2024-01-01 00:03:00;human
        case-2;0;start;;2024-01-02 01:01:00;bot
        case-2;1;end;2024-01-02 01:01:00;2024-01-02 01:02:00;human
        """
    ).strip()

    buffer = io.StringIO(csv_text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = load_dataset(buffer, {"sep": ";"})

    assert len(log) == 2
    assert log["ACTIVITY"].tolist() == ["start", "end"]
    assert not any(_WARNING_MESSAGE in str(w.message) for w in caught)


def test_load_dataset_preserves_rows_for_valid_logs():
    dataset_path = Path(__file__).resolve().parents[3] / "event_log_example.csv"
    expected_rows = pd.read_csv(dataset_path, sep=";").shape[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = load_dataset(dataset_path, {"sep": ";"})

    assert len(log) == expected_rows
    assert not any(_WARNING_MESSAGE in str(w.message) for w in caught)
