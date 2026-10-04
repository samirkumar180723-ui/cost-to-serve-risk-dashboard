"""
Synthetic Logistics Order Data Generator
------------------------------------------------------------------------
Generates order-level e-commerce logistics data across 10 shipping
corridors, with:
  - A built-in "worst corridor" (high cost-to-serve, high delay rate)
    for the risk model and heatmap to discover
  - A free-shipping promotion rolled out in ONE corridor at a known date,
    to support Difference-in-Differences causal analysis
  - Realistic, explicitly-stated parameter choices (documented inline)
    so every downstream number is traceable to an assumption here.

All data is SYNTHETIC. It is calibrated to be internally consistent and
plausible, not sourced from any real company.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_ORDERS = 20_000
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2026-09-30")
PROMO_START_DATE = pd.Timestamp("2026-01-01")  # free-shipping promo begins
TREATED_CORRIDOR = "Corridor_07"                # the one corridor that gets the promo
WORST_CORRIDOR = "Corridor_09"                  # deliberately bad corridor (separate from treated)

CORRIDORS = [f"Corridor_{i:02d}" for i in range(1, 11)]

# Base parameters per corridor: (base_shipping_cost, base_distance_km, base_delay_rate)
# Corridor_09 is deliberately the worst: high cost, high distance, high delay rate.
CORRIDOR_PARAMS = {}
for c in CORRIDORS:
    if c == WORST_CORRIDOR:
        CORRIDOR_PARAMS[c] = {"cost_base": 420, "distance_base": 850, "delay_rate_base": 0.25}
    else:
        CORRIDOR_PARAMS[c] = {
            "cost_base": rng.uniform(120, 220),
            "distance_base": rng.uniform(150, 500),
            "delay_rate_base": rng.uniform(0.05, 0.12),
        }


def generate_orders():
    order_dates = pd.to_datetime(
        rng.integers(START_DATE.value // 10**9, END_DATE.value // 10**9, size=N_ORDERS),
        unit="s",
    )
    corridors = rng.choice(CORRIDORS, size=N_ORDERS)

    order_value = rng.gamma(shape=3.0, scale=600, size=N_ORDERS)  # INR, right-skewed like real order values
    order_value = np.clip(order_value, 150, 15000)

    distance_km = np.array([
        CORRIDOR_PARAMS[c]["distance_base"] + rng.normal(0, 40) for c in corridors
    ])
    distance_km = np.clip(distance_km, 20, None)

    shipping_cost = np.array([
        CORRIDOR_PARAMS[c]["cost_base"] for c in corridors
    ]) + distance_km * 0.15 + rng.normal(0, 15, size=N_ORDERS)
    shipping_cost = np.clip(shipping_cost, 30, None)

    is_treated_corridor = (corridors == TREATED_CORRIDOR)
    is_post_promo = (order_dates >= PROMO_START_DATE)
    free_shipping_applied = is_treated_corridor & is_post_promo

    # DiD ground truth: the promo causes a TRUE +180 INR bump in order value
    # in the treated corridor after the promo starts (customers buy more to
    # "make it worth it" / larger basket size). This is what the causal
    # analysis should recover.
    TRUE_PROMO_EFFECT_ON_ORDER_VALUE = 180
    order_value = order_value + free_shipping_applied * TRUE_PROMO_EFFECT_ON_ORDER_VALUE

    # Delay/cancellation risk, higher in the worst corridor and for longer distances
    base_delay_rate = np.array([CORRIDOR_PARAMS[c]["delay_rate_base"] for c in corridors])
    # Scaled down from an earlier version that saturated at the clip ceiling for
    # the worst corridor (850km -> 0.55 effect, pre-clip prob of 0.8) -- rescaled
    # so distance contributes a realistic, non-saturating effect.
    distance_effect = (distance_km - 300) / 3000  # longer distance -> more risk
    weekday = order_dates.dayofweek
    is_weekend = (weekday >= 5).astype(float)

    # Genuine NONLINEAR INTERACTION (not just an additive effect): weekend
    # orders on long-distance routes are disproportionately riskier, reflecting
    # limited weekend logistics capacity for long-haul deliveries specifically.
    # This gives a tree-based model (XGBoost) real interaction structure to
    # learn that a purely additive/linear model (Logistic Regression) cannot
    # represent without manually engineering the interaction term.
    long_distance_flag = (distance_km > 500).astype(float)
    weekend_longhaul_interaction = is_weekend * long_distance_flag * 0.18

    delay_prob = np.clip(
        base_delay_rate + distance_effect + weekend_longhaul_interaction, 0.02, 0.55
    )
    is_delayed_or_cancelled = rng.binomial(1, delay_prob)

    df = pd.DataFrame({
        "order_id": np.arange(1, N_ORDERS + 1),
        "order_date": order_dates,
        "corridor": corridors,
        "order_value": order_value.round(2),
        "distance_km": distance_km.round(1),
        "shipping_cost": shipping_cost.round(2),
        "free_shipping_applied": free_shipping_applied.astype(int),
        "is_treated_corridor": is_treated_corridor.astype(int),
        "is_post_promo": is_post_promo.astype(int),
        "is_delayed_or_cancelled": is_delayed_or_cancelled,
        "day_of_week": weekday,
        "month": order_dates.month,
    })
    df["cost_to_serve_pct_of_value"] = (df["shipping_cost"] / df["order_value"] * 100).round(2)

    return df.sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    df = generate_orders()
    df.to_csv("/home/claude/cost_to_serve/data/orders.csv", index=False)

    print(f"Generated {len(df):,} synthetic orders.")
    print(f"\nOverall delay/cancellation rate: {df['is_delayed_or_cancelled'].mean():.1%}")
    print(f"\nWorst corridor ({WORST_CORRIDOR}) stats:")
    worst = df[df["corridor"] == WORST_CORRIDOR]
    print(f"  Avg cost-to-serve %% of order value: {worst['cost_to_serve_pct_of_value'].mean():.1f}%")
    print(f"  Delay/cancellation rate:            {worst['is_delayed_or_cancelled'].mean():.1%}")
    print(f"\nTreated corridor ({TREATED_CORRIDOR}) promo coverage:")
    print(df[df['corridor'] == TREATED_CORRIDOR]['free_shipping_applied'].value_counts())
