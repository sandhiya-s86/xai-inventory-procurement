"""The explanation must be an account of the arithmetic, not a story about it.

These tests fail if the decomposition shown to the user stops adding up to
the quantity the system actually recommends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xaierp.explain.decision_explanation import explain_decision, render_explanation
from xaierp.policy.reorder import decide_reorder
from xaierp.utils.config import load_config, resolve


def _decision(**kw):
    base = dict(
        expected_demand_leadtime=420.0, safety_stock=95.0,
        inventory_position=160.0, lead_time_days=8,
        moq=48, pack_size=12, supplier_capacity=5000, unit_cost=1.75,
        mean_daily_demand=30.0,
    )
    base.update(kw)
    return decide_reorder(**base)


def test_explanation_reconciles_with_the_decision():
    d = _decision()
    exp = explain_decision(
        sku="12345", description="TEST ITEM", date="2011-09-01",
        decision=d, forecast_7d=300.0, reorder_probability=0.82, shap_top=None,
    )
    assert exp["reconciles"] is True
    assert sum(exp["decomposition"].values()) == pytest.approx(
        exp["decomposition_total"], abs=0.01
    )
    assert exp["recommended_qty"] == d.final_qty


def test_explanation_refuses_to_render_a_broken_decomposition():
    """A tampered decision must raise rather than print a wrong explanation."""
    d = _decision()
    d.safety_component += 1000.0   # break the identity without changing final_qty
    assert not d.reconciles()
    with pytest.raises(AssertionError):
        explain_decision(
            sku="12345", description="TEST", date="2011-09-01",
            decision=d, forecast_7d=300.0, reorder_probability=0.5, shap_top=None,
        )


def test_risk_band_follows_the_probability():
    for prob, expected in [(0.95, "HIGH"), (0.70, "HIGH"),
                           (0.55, "MEDIUM"), (0.40, "MEDIUM"),
                           (0.10, "LOW")]:
        exp = explain_decision(
            sku="1", description="x", date="2011-01-01", decision=_decision(),
            forecast_7d=10.0, reorder_probability=prob, shap_top=None,
        )
        assert exp["stockout_risk"] == expected, f"p={prob}"


def test_moq_and_pack_are_mentioned_only_when_they_bite():
    # MOQ far above the requirement -> it must be named in the reasons.
    d = _decision(expected_demand_leadtime=20, safety_stock=0,
                  inventory_position=0, moq=144, pack_size=1)
    exp = explain_decision("1", "x", "2011-01-01", d, 20.0, 0.9, None)
    assert any("minimum order quantity" in r for r in exp["quantity_reasons"])

    # Requirement already a clean multiple -> no pack-rounding sentence.
    d2 = _decision(expected_demand_leadtime=240, safety_stock=0,
                   inventory_position=0, moq=12, pack_size=12)
    exp2 = explain_decision("1", "x", "2011-01-01", d2, 240.0, 0.9, None)
    assert not any("whole packs" in r for r in exp2["quantity_reasons"])


def test_capacity_shortfall_is_surfaced_as_a_warning():
    d = _decision(expected_demand_leadtime=9000, safety_stock=500,
                  inventory_position=0, supplier_capacity=1000)
    exp = explain_decision("1", "x", "2011-01-01", d, 9000.0, 0.99, None)
    assert any("WARNING" in r for r in exp["quantity_reasons"])


def test_shap_reasons_quote_real_feature_values():
    shap_top = {
        "positive": [
            {"feature": "rolling_mean_28", "shap_value": 41.5, "feature_value": 62.3}
        ],
        "negative": [
            {"feature": "zero_rate_28", "shap_value": -12.0, "feature_value": 0.25}
        ],
    }
    exp = explain_decision("1", "x", "2011-01-01", _decision(), 300.0, 0.8, shap_top)
    joined = " ".join(exp["demand_reasons"])
    assert "62.3" in joined, "the real feature value must appear in the text"
    assert "41.5" in joined, "the real attribution must appear in the text"
    assert "average daily demand over the last 28 days" in joined


def test_rendered_text_shows_the_total():
    exp = explain_decision("12345", "TEST ITEM", "2011-09-01",
                           _decision(), 300.0, 0.82, None)
    text = render_explanation(exp)
    assert "FINAL ORDER QUANTITY" in text
    assert "Why this order quantity?" in text
    assert str(exp["recommended_qty"]) in text.replace(",", "")


def test_generated_case_studies_all_reconcile():
    """The artefact actually shipped to the report must reconcile."""
    cfg = load_config()
    path = resolve(cfg["paths"]["reports"]) / "generated" / "case_studies.json"
    if not path.exists():
        pytest.skip("case studies not generated -- run scripts/run_explainability.py")

    cases = json.loads(path.read_text(encoding="utf-8"))
    assert cases, "no case studies were generated"
    for c in cases:
        assert c["reconciles"] is True, f"case {c['sku']} does not reconcile"
        assert sum(c["decomposition"].values()) == pytest.approx(
            c["decomposition_total"], abs=0.01
        ), f"case {c['sku']} decomposition does not sum to the recommendation"
        assert c["recommended_qty"] >= 0
