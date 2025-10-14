import pandas as pd

from chatflow_miner.lib.aggregations import (
    CaseAggView,
    CaseDateAggregator,
    CaseVariantAggregator,
    NormalizeTimestampsOp,
    VariantInfo,
    build_aggregator_from_spec,
)
from chatflow_miner.lib.constants import (
    COLUMN_ACTIVITY,
    COLUMN_AGENT,
    COLUMN_CASE_ID,
    COLUMN_END_TS,
    COLUMN_EVENT_ID,
    COLUMN_START_TS,
)


def _df():
    return pd.DataFrame(
        [
            {
                COLUMN_CASE_ID: "A",
                COLUMN_EVENT_ID: 0,
                COLUMN_ACTIVITY: "START-DIALOGUE-SYTEM",
                COLUMN_START_TS: "2025-07-11 04:53:10.509884",
                COLUMN_END_TS: "2025-07-11 04:53:10.509884",
                COLUMN_AGENT: "syst",
            },
            {
                COLUMN_CASE_ID: "A",
                COLUMN_EVENT_ID: 1,
                COLUMN_ACTIVITY: "SAUDAÇÃO INICIAL DO CHATBOT",
                COLUMN_START_TS: "2025-07-11 04:53:10.509884",
                COLUMN_END_TS: "2025-07-11 04:53:11.509884",
                COLUMN_AGENT: "ai",
            },
            {
                COLUMN_CASE_ID: "A",
                COLUMN_EVENT_ID: 2,
                COLUMN_ACTIVITY: "FINALIZAÇÃO COM RECLAMAÇÃO",
                COLUMN_START_TS: "2025-07-11 04:53:11.509884",
                COLUMN_END_TS: "2025-07-11 04:57:00.851884",
                COLUMN_AGENT: "human",
            },
            {
                COLUMN_CASE_ID: "B",
                COLUMN_EVENT_ID: 0,
                COLUMN_ACTIVITY: "SAUDAÇÃO INICIAL DO CHATBOT",
                COLUMN_START_TS: "2025-07-11 05:00:00",
                COLUMN_END_TS: "2025-07-11 05:00:00.5",
                COLUMN_AGENT: "ai",
            },
            {
                COLUMN_CASE_ID: "B",
                COLUMN_EVENT_ID: 1,
                COLUMN_ACTIVITY: "FINALIZAÇÃO COM RECLAMAÇÃO",
                COLUMN_START_TS: "2025-07-11 05:01:00",
                COLUMN_END_TS: "2025-07-11 05:01:01",
                COLUMN_AGENT: "human",
            },
        ]
    )


def test_variant_ignore_syst():
    df = _df()
    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseVariantAggregator(ignore_syst=True, joiner=">"))
    )
    result = view.compute()
    assert set(result.keys()) == {"A", "B"}
    a: VariantInfo = result["A"]
    b: VariantInfo = result["B"]
    assert a.variant == "SAUDAÇÃO INICIAL DO CHATBOT>FINALIZAÇÃO COM RECLAMAÇÃO"
    assert b.variant == a.variant
    assert a.variant_id == b.variant_id
    assert a.frequency == 2 and b.frequency == 2
    assert a.length == 2


def test_variant_with_syst_changes_sequence():
    df = _df()
    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseVariantAggregator(ignore_syst=False, joiner=">"))
    )
    result = view.compute()
    a: VariantInfo = result["A"]
    assert a.variant.startswith("START-DIALOGUE-SYTEM>")
    assert a.frequency == 1


def test_case_date_aggregator():
    df = _df()
    view = (
        CaseAggView(df)
        .with_aux(NormalizeTimestampsOp())
        .with_aggregator(CaseDateAggregator())
    )
    result = view.compute()
    assert result["A"] == "2025-07-11"
    assert result["B"] == "2025-07-11"


def test_build_from_spec():
    df = _df()
    spec = {"type": "variant", "args": {"ignore_syst": True, "joiner": ">"}}
    agg = build_aggregator_from_spec(spec)
    view = CaseAggView(df).with_aux(NormalizeTimestampsOp()).with_aggregator(agg)
    result = view.compute()
    assert isinstance(result["A"], VariantInfo)
