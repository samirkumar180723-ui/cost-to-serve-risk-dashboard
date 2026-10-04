"""
Cost-to-Serve & RTO Risk Analyzer -- Streamlit Dashboard
------------------------------------------------------------------------
Combines three analyses into one interactive tool:
  1. Cost-to-serve heatmap by shipping corridor
  2. Live delay/cancellation risk predictor (XGBoost)
  3. Interactive shipping-subsidy policy simulator (Monte Carlo)

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from risk_model import train_and_evaluate, build_preprocessor, FEATURE_COLS_NUMERIC, FEATURE_COLS_CATEGORICAL, FLAG_COLS
from causal_analysis import run_did_analysis
from monte_carlo_policy import run_simulation

st.set_page_config(page_title="Cost-to-Serve & RTO Risk Analyzer", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "orders.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["order_date"])


@st.cache_resource
def get_trained_model(df):
    return train_and_evaluate(df)


df = load_data()

st.title("📦 Cost-to-Serve & RTO Risk Analyzer")
st.caption(
    "All data in this dashboard is SYNTHETIC, generated with documented, "
    "internally-consistent assumptions for case-practice purposes -- not from "
    "a real company. Methodology (time-based split, SMOTE on train-only, "
    "DiD causal validation) is designed to be defensible, not just impressive."
)

tab1, tab2, tab3 = st.tabs(["🗺️ Cost-to-Serve Heatmap", "⚠️ Live Risk Predictor", "💰 Subsidy Policy Simulator"])

# ------------------------------------------------------------------
# TAB 1: HEATMAP
# ------------------------------------------------------------------
with tab1:
    st.subheader("Cost-to-Serve & Delay Rate by Corridor")

    corridor_summary = df.groupby("corridor").agg(
        avg_cost_to_serve_pct=("cost_to_serve_pct_of_value", "mean"),
        delay_rate=("is_delayed_or_cancelled", "mean"),
        order_count=("order_id", "count"),
        avg_order_value=("order_value", "mean"),
    ).reset_index().sort_values("avg_cost_to_serve_pct", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cost-to-Serve (% of order value)**")
        st.bar_chart(corridor_summary.set_index("corridor")["avg_cost_to_serve_pct"])
    with col2:
        st.markdown("**Delay / Cancellation Rate**")
        st.bar_chart(corridor_summary.set_index("corridor")["delay_rate"])

    worst = corridor_summary.iloc[0]
    st.error(
        f"🚩 **Worst corridor: {worst['corridor']}** -- "
        f"{worst['avg_cost_to_serve_pct']:.1f}% cost-to-serve ratio and "
        f"{worst['delay_rate']:.1%} delay/cancellation rate. "
        f"Recommend prioritizing fulfillment fixes here over any blanket policy change."
    )

    st.dataframe(corridor_summary.style.format({
        "avg_cost_to_serve_pct": "{:.1f}%", "delay_rate": "{:.1%}",
        "avg_order_value": "₹{:,.0f}",
    }), use_container_width=True)

# ------------------------------------------------------------------
# TAB 2: LIVE RISK PREDICTOR
# ------------------------------------------------------------------
with tab2:
    st.subheader("Predict Delay/Cancellation Risk at Order Placement")

    with st.spinner("Training model (cached after first run)..."):
        results = get_trained_model(df)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Logistic Regression AUC", f"{results['log_reg_auc']:.3f}")
        st.metric("XGBoost AUC", f"{results['xgb_auc']:.3f}")
        st.caption(
            "Note: XGBoost does not materially beat the simpler baseline here "
            "(confirmed via 5-fold CV) -- a legitimate finding, not a bug. "
            "Risk drivers in this data are close to linear/additive."
        )

    with col2:
        st.markdown("**Try a live prediction:**")
        corridor_input = st.selectbox("Corridor", sorted(df["corridor"].unique()))
        order_value_input = st.slider("Order Value (INR)", 150, 10000, 1800)
        distance_input = st.slider("Distance (km)", 20, 1000, 300)
        weekend_input = st.checkbox("Weekend order?")
        promo_input = st.checkbox("Free shipping applied?")

        input_row = pd.DataFrame([{
            "order_value": order_value_input,
            "distance_km": distance_input,
            "day_of_week": 5 if weekend_input else 2,
            "month": 6,
            "corridor": corridor_input,
            "free_shipping_applied": int(promo_input),
        }])

        X_input = results["preprocessor"].transform(
            input_row[FEATURE_COLS_NUMERIC + FEATURE_COLS_CATEGORICAL + FLAG_COLS]
        )
        risk_prob = results["xgb_model"].predict_proba(X_input)[0, 1]

        risk_threshold = st.slider("Decision threshold", 0.0, 1.0, 0.3, 0.05)
        st.metric("Predicted Delay/Cancellation Risk", f"{risk_prob:.1%}")
        if risk_prob >= risk_threshold:
            st.warning("⚠️ Above threshold -- flag for proactive intervention")
        else:
            st.success("✅ Below threshold -- standard fulfillment")

# ------------------------------------------------------------------
# TAB 3: POLICY SIMULATOR
# ------------------------------------------------------------------
with tab3:
    st.subheader("Shipping-Subsidy Removal: Margin Impact Simulator")

    model, did_coef, did_pval, conf_int = run_did_analysis(df)
    st.info(
        f"**Causal estimate (Difference-in-Differences):** the free-shipping promo "
        f"increases average order value by **₹{did_coef:.0f}** "
        f"(95% CI: ₹{conf_int[0]:.0f} to ₹{conf_int[1]:.0f}, p={did_pval:.4f})"
    )

    st.markdown("**Adjust assumptions and re-run the simulation:**")
    col1, col2 = st.columns(2)
    with col1:
        monthly_orders = st.slider("Monthly orders in corridor", 500, 10000, 3000, 500)
    with col2:
        n_sims = st.select_slider("Number of simulations", [1000, 5000, 10000, 20000], value=10000)

    if st.button("Run Simulation"):
        with st.spinner("Running Monte Carlo simulation..."):
            ci_width = conf_int[1] - conf_int[0]
            results_sim = run_simulation(did_coef, ci_width)

        p5, p50, p95 = np.percentile(results_sim, [5, 50, 95])
        prob_helps = (results_sim > 0).mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("5th pct. (downside)", f"₹{p5:,.0f}")
        col2.metric("Median", f"₹{p50:,.0f}")
        col3.metric("95th pct. (upside)", f"₹{p95:,.0f}")

        st.metric("Probability removing subsidy IMPROVES margin", f"{prob_helps:.1f}%")

        if prob_helps < 40:
            st.success("✅ Recommendation: KEEP the subsidy -- it appears to be paying for itself.")
        elif prob_helps > 60:
            st.warning("⚠️ Recommendation: Consider REMOVING the subsidy.")
        else:
            st.info("ℹ️ Outcome is genuinely uncertain -- recommend a controlled pilot before a full rollout.")

st.divider()
st.caption(
    "Built by Samir Kumar | Methodology: time-based train/test split, SMOTE on "
    "training data only, Difference-in-Differences causal validation, Monte Carlo "
    "uncertainty propagation. All findings validated against known ground truth "
    "in the underlying synthetic data generator."
)
