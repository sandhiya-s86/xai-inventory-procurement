"""Reorder quantity rules: never negative, MOQ honoured, packs respected."""

from __future__ import annotations

import math

import pytest

from xaierp.policy.reorder import (
    decide_reorder,
    safety_stock_from_quantiles,
    safety_stock_from_sigma,
)


def test_never_orders_a_negative_quantity():
    d = decide_reorder(
        expected_demand_leadtime=10, safety_stock=5,
        inventory_position=500, lead_time_days=7,
    )
    assert d.final_qty == 0
    assert d.reorder is False


def test_orders_when_stock_is_short():
    d = decide_reorder(
        expected_demand_leadtime=300, safety_stock=50,
        inventory_position=100, lead_time_days=7, moq=0, pack_size=1,
    )
    assert d.final_qty == 250
    assert d.reorder is True


def test_moq_raises_a_small_order():
    """The spec's worked example: need 185, MOQ 50, pack 50 -> 200."""
    d = decide_reorder(
        expected_demand_leadtime=185, safety_stock=0,
        inventory_position=0, lead_time_days=5, moq=50, pack_size=50,
    )
    assert d.final_qty == 200
    assert d.moq_adjustment == 0          # 185 already exceeds the MOQ of 50
    assert d.pack_adjustment == 15        # rounded 185 -> 200


def test_moq_applies_only_when_something_is_needed():
    d = decide_reorder(
        expected_demand_leadtime=10, safety_stock=0,
        inventory_position=1000, lead_time_days=5, moq=144, pack_size=12,
    )
    assert d.final_qty == 0, "MOQ must not force an order that is not needed"


def test_moq_floor_is_enforced():
    d = decide_reorder(
        expected_demand_leadtime=20, safety_stock=0,
        inventory_position=10, lead_time_days=5, moq=144, pack_size=1,
    )
    assert d.final_qty == 144
    assert d.moq_adjustment == pytest.approx(134)


@pytest.mark.parametrize("pack", [1, 6, 12, 24])
def test_order_is_always_a_whole_number_of_packs(pack):
    d = decide_reorder(
        expected_demand_leadtime=137, safety_stock=23,
        inventory_position=11, lead_time_days=9, moq=12, pack_size=pack,
    )
    assert d.final_qty % pack == 0
    assert d.final_qty >= 137 + 23 - 11


def test_capacity_caps_the_order_and_reports_the_gap():
    d = decide_reorder(
        expected_demand_leadtime=5000, safety_stock=500,
        inventory_position=0, lead_time_days=14, moq=12, pack_size=12,
        supplier_capacity=1000,
    )
    # 1000 is not a multiple of 12, and a part pack cannot be bought, so the
    # cap rounds down to 996.
    assert d.final_qty == 996
    assert d.final_qty % 12 == 0
    assert d.final_qty <= 1000
    assert d.capacity_adjustment < 0
    assert d.coverage_gap == pytest.approx(4504)


def test_capacity_cap_preserves_the_pack_multiple():
    for cap, pack in [(1000, 12), (997, 6), (55, 24), (123, 7)]:
        d = decide_reorder(
            expected_demand_leadtime=100000, safety_stock=0,
            inventory_position=0, lead_time_days=10, moq=1, pack_size=pack,
            supplier_capacity=cap,
        )
        assert d.final_qty % pack == 0, f"cap={cap} pack={pack} broke the pack rule"
        assert d.final_qty <= cap
        assert d.reconciles()


def test_components_always_sum_to_the_final_quantity():
    cases = [
        dict(expected_demand_leadtime=300, safety_stock=50, inventory_position=100,
             lead_time_days=7, moq=48, pack_size=12),
        dict(expected_demand_leadtime=10, safety_stock=2, inventory_position=900,
             lead_time_days=3, moq=144, pack_size=24),
        dict(expected_demand_leadtime=5000, safety_stock=900, inventory_position=20,
             lead_time_days=21, moq=12, pack_size=6, supplier_capacity=800),
        dict(expected_demand_leadtime=0, safety_stock=0, inventory_position=0,
             lead_time_days=1, moq=0, pack_size=1),
        dict(expected_demand_leadtime=137.4, safety_stock=22.6, inventory_position=11.1,
             lead_time_days=9, moq=12, pack_size=7),
    ]
    for kw in cases:
        d = decide_reorder(**kw)
        total = sum(d.components().values())
        assert abs(total - d.final_qty) < 1e-6, (
            f"components sum to {total}, final_qty is {d.final_qty}, inputs {kw}"
        )
        assert d.reconciles()


def test_order_value_matches_quantity_times_cost():
    d = decide_reorder(
        expected_demand_leadtime=200, safety_stock=0, inventory_position=0,
        lead_time_days=7, moq=0, pack_size=1, unit_cost=2.5,
    )
    assert d.order_value == pytest.approx(d.final_qty * 2.5)


def test_classical_safety_stock_grows_with_lead_time_and_volatility():
    a = safety_stock_from_sigma(10, 4, 0.95)
    b = safety_stock_from_sigma(10, 16, 0.95)
    c = safety_stock_from_sigma(20, 4, 0.95)
    assert b > a, "longer lead time must need more buffer"
    assert c > a, "more volatile demand must need more buffer"
    assert b == pytest.approx(a * 2, rel=1e-6), "should scale with sqrt(L)"
    assert safety_stock_from_sigma(0, 7, 0.95) == 0.0


def test_higher_service_level_requires_more_buffer():
    assert safety_stock_from_sigma(10, 7, 0.99) > safety_stock_from_sigma(10, 7, 0.90)


def test_quantile_safety_stock_is_non_negative():
    assert safety_stock_from_quantiles(50, 120) == 0.0
    assert safety_stock_from_quantiles(120, 50) == 70.0
    assert safety_stock_from_quantiles(120, 50, scale=2.0) == 140.0


def test_real_recommendations_obey_every_constraint(recommendations):
    df = recommendations
    assert (df["rec_qty"] >= 0).all(), "a negative order quantity was recommended"

    ordered = df.loc[df["rec_qty"] > 0]
    # Pack multiples.
    assert ((ordered["rec_qty"] % ordered["pack_size"]) == 0).all()
    # MOQ floor, except where supplier capacity legitimately cuts the order.
    capped = ordered["rec_capacity_adjustment"] < -1e-9
    assert (ordered.loc[~capped, "rec_qty"] >= ordered.loc[~capped, "moq"]).all()
    # Capacity ceiling.
    assert (ordered["rec_qty"] <= ordered["supplier_capacity"] + 1e-6).all()
