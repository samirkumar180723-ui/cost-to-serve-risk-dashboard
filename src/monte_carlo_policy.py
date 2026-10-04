"""
Monte Carlo Simulation: Shipping-Subsidy Policy Margin Impact
------------------------------------------------------------------------
Uses the DiD-estimated promo effect (from causal_analysis.py) as an
INPUT to this simulation, rather than re-assuming a number -- this
chains the three analyses together into one coherent pipeline:
  1. Causal analysis estimates the TRUE effect of the subsidy (with
     uncertainty, via its confidence interval)
  2. This simulation propagates that uncertainty, plus cost and demand
     uncertainty, through to a full margin-impact distribution
  3. The risk model's worst-corridor finding informs WHERE a policy
     fix should target first (handled in the Streamlit app)
"""

import numpy as np

rng = np.random.default_rng(99)

N_SIMULATIONS = 20_000
MONTHLY_ORDERS_IN_TREATED_CORRIDOR = 3_000
MONTHS = 12
SUBSIDY_COST_PER_ORDER = 60  # INR, direct cost of offering free shipping


def run_simulation(did_effect_mean, did_effect_ci_width):
    # Propagate the DiD estimate's own uncertainty (from its 95% CI) into
    # the simulation, rather than treating it as a single fixed number.
    promo_effect_samples = rng.normal(did_effect_mean, did_effect_ci_width / 3.92, N_SIMULATIONS)
    # (CI width / 3.92 converts a 95% CI back to a std dev, since 95% CI ~ +/-1.96*SE)

    # Additional uncertainty: how much does subsidy removal reduce order
    # volume (customers who were only ordering because of free shipping)?
    volume_sensitivity = rng.triangular(0.02, 0.08, 0.18, N_SIMULATIONS)  # % volume lost if subsidy removed

    margin_pct = rng.triangular(0.12, 0.18, 0.25, N_SIMULATIONS)  # gross margin on order value

    # Scenario: REMOVE the subsidy. Trade-off:
    #   (+) save the direct subsidy cost per order
    #   (-) lose the extra order-value lift (promo_effect) AND lose some
    #       orders entirely (volume_sensitivity)
    orders_per_month = MONTHLY_ORDERS_IN_TREATED_CORRIDOR
    retained_orders = orders_per_month * (1 - volume_sensitivity)

    margin_with_subsidy = (
        orders_per_month * margin_pct * (1800 + promo_effect_samples)  # avg order value + promo lift
        - orders_per_month * SUBSIDY_COST_PER_ORDER
    )
    margin_without_subsidy = retained_orders * margin_pct * 1800  # loses the promo lift on remaining orders

    monthly_margin_delta = (margin_without_subsidy - margin_with_subsidy)
    annual_margin_delta = monthly_margin_delta * MONTHS

    return annual_margin_delta


def report(did_effect_mean=193.48, did_effect_ci_width=(289.31 - 97.65)):
    results = run_simulation(did_effect_mean, did_effect_ci_width)

    p5, p25, p50, p75, p95 = np.percentile(results, [5, 25, 50, 75, 95])
    prob_removal_helps = (results > 0).mean() * 100

    print("=" * 72)
    print("MONTE CARLO SIMULATION: SHIPPING-SUBSIDY REMOVAL POLICY IMPACT")
    print("=" * 72)
    print(f"\n{N_SIMULATIONS:,} simulated scenarios | Annual margin impact of REMOVING the subsidy (INR)\n")

    print(f"  5th percentile:   {p5:>14,.0f}")
    print(f"  25th percentile:  {p25:>14,.0f}")
    print(f"  Median:           {p50:>14,.0f}")
    print(f"  75th percentile:  {p75:>14,.0f}")
    print(f"  95th percentile:  {p95:>14,.0f}\n")

    print(f"Probability that REMOVING the subsidy improves annual margin: {prob_removal_helps:.1f}%\n")

    print("-" * 72)
    print("INTERPRETATION")
    print("-" * 72)
    if prob_removal_helps < 40:
        verdict = ("Removing the subsidy is likely to HURT margin once the order-value lift\n"
                   "and volume loss are both accounted for -- the subsidy appears to be paying\n"
                   "for itself. This combines the causal estimate (Step 1) with cost/demand\n"
                   "uncertainty (Step 2) into one risk-aware recommendation.")
    elif prob_removal_helps > 60:
        verdict = ("Removing the subsidy is likely to IMPROVE margin -- the direct subsidy\n"
                   "cost appears to outweigh the order-value lift it generates.")
    else:
        verdict = ("The outcome is genuinely uncertain (close to a coin-flip) -- recommend\n"
                   "a controlled pilot (remove subsidy in one comparable corridor) rather than\n"
                   "a blanket policy change, given how much the outcome depends on volume-\n"
                   "sensitivity assumptions that are not yet well pinned down.")
    print(verdict + "\n")


if __name__ == "__main__":
    report()
