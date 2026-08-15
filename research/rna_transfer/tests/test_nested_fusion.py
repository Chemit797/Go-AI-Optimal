import pandas as pd

from src.evaluate_nested_m6_m9_fusion import _select


def test_family_guardrail_uses_unfiltered_m6_baseline():
    rows = []
    for fold in range(4):
        rows.extend(
            [
                {
                    "candidate": "blend_w0",
                    "family": "blend",
                    "fold": fold,
                    "fc_pcc": 0.30,
                    "context_residual_pcc": 0.10,
                    "high_effect_pcc": 0.60,
                    "high_effect_f1": 0.20,
                },
                {
                    "candidate": "high_specialist_w1_t1_g0.25",
                    "family": "high_specialist",
                    "fold": fold,
                    "fc_pcc": 0.40,
                    "context_residual_pcc": 0.09,
                    "high_effect_pcc": 0.61,
                    "high_effect_f1": 0.21,
                },
            ]
        )
    winner = _select(
        pd.DataFrame(rows),
        outer_fold=0,
        selector="guarded",
        family="high_specialist",
    )
    assert winner["candidate"] == "high_specialist_w1_t1_g0.25"
