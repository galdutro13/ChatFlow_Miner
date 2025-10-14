import pandas as pd
import pytest

from chatflow_miner.lib.constants import COLUMN_CASE_ID
from chatflow_miner.lib.filters.builtins import CaseFilter


def make_df(case_ids):
    return pd.DataFrame(
        {
            COLUMN_CASE_ID: case_ids,
            "value": range(len(case_ids)),
        }
    )


def test_case_filter_keeps_matching_cases():
    df = make_df(["A", "A", "B", "C", "D"])
    filt = CaseFilter(["A", "C"])  # should keep cases A and C (all their events)

    mask = filt.mask(df)
    assert mask.dtype == bool
    # rows 0,1 are A and row 3 is C
    assert mask.tolist() == [True, True, False, True, False]


def test_case_filter_with_missing_and_types():
    # include NaN and integer ids
    df = make_df(["x", None, 42, "y", 42])
    filt = CaseFilter(["x", 42])

    mask = filt.mask(df)
    # expect rows with 'x' and 42 to be True; None should be False
    assert mask.tolist() == [True, False, True, False, True]


def test_case_filter_empty_list_results_in_none_kept():
    df = make_df(["a", "b", "c"])
    filt = CaseFilter([])
    mask = filt.mask(df)
    assert mask.tolist() == [False, False, False]


def test_case_filter_raises_when_missing_column():
    # construct a df without the required column
    df = pd.DataFrame({"other": [1, 2, 3]})
    filt = CaseFilter(["a"])
    with pytest.raises(Exception):
        filt.mask(df)
