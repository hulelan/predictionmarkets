"""Streamlit dashboard for prediction market baseline analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediction Markets — Baseline Analysis", layout="wide")

# ─── Data Loading ────────────────────────────────────────────────────────────


@st.cache_data
def load_data():
    """Load markets and merge with all baseline predictions."""
    markets = pd.DataFrame(json.load(open("data/processed/markets.json")))
    markets["resolution_date"] = pd.to_datetime(markets["resolution_date"])

    pred_dir = Path("data/predictions")
    model_preds = {}
    for f in sorted(pred_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in f.read_text().strip().split("\n") if line]
        if rows:
            df = pd.DataFrame(rows)
            model_name = df["model"].iloc[0]
            model_preds[model_name] = df

    return markets, model_preds


markets, model_preds = load_data()
model_names = list(model_preds.keys())

# Build merged dataframe: one row per market with all model predictions
merged = markets[["id", "question", "market_probability", "category", "platform", "resolution_date", "volume", "url"]].copy()
for name in model_names:
    df = model_preds[name][["market_id", "probability", "confidence", "reasoning"]].copy()
    short = name.split("/")[-1]
    df = df.rename(columns={
        "probability": f"pred_{short}",
        "confidence": f"conf_{short}",
        "reasoning": f"reason_{short}",
    })
    merged = merged.merge(df, left_on="id", right_on="market_id", how="left").drop(columns=["market_id"])

pred_cols = [c for c in merged.columns if c.startswith("pred_")]
short_names = [c.replace("pred_", "") for c in pred_cols]

# Derived columns
merged["model_mean"] = merged[pred_cols].mean(axis=1)
merged["model_std"] = merged[pred_cols].std(axis=1)
merged["model_spread"] = merged[pred_cols].max(axis=1) - merged[pred_cols].min(axis=1)
merged["mean_vs_market"] = merged["model_mean"] - merged["market_probability"]

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("Filters")
cats = ["All"] + sorted(merged["category"].unique().tolist())
sel_cat = st.sidebar.selectbox("Category", cats)
if sel_cat != "All":
    merged = merged[merged["category"] == sel_cat]

platforms = ["All"] + sorted(merged["platform"].unique().tolist())
sel_plat = st.sidebar.selectbox("Platform", platforms)
if sel_plat != "All":
    merged = merged[merged["platform"] == sel_plat]

min_spread = st.sidebar.slider("Min model disagreement (spread)", 0.0, 1.0, 0.0, 0.01)
merged = merged[merged["model_spread"] >= min_spread]

st.sidebar.markdown("---")
st.sidebar.metric("Markets shown", len(merged))

# ─── Tab Layout ──────────────────────────────────────────────────────────────

tab_overview, tab_scatter, tab_disagree, tab_explore, tab_calibration = st.tabs(
    ["Overview", "Model vs Market", "Disagreement", "Market Explorer", "Calibration (post-resolution)"]
)

# ─── Tab 1: Overview ─────────────────────────────────────────────────────────

with tab_overview:
    st.header("Baseline Summary")

    cols = st.columns(len(model_names) + 1)
    with cols[0]:
        st.metric("Markets", len(merged))

    for i, name in enumerate(model_names):
        short = name.split("/")[-1]
        col_name = f"pred_{short}"
        valid = merged[col_name].dropna()
        failed = merged[col_name].isna().sum()
        with cols[i + 1]:
            st.metric(name, f"mean {valid.mean():.3f}", delta=f"{failed} failed" if failed else None, delta_color="inverse")

    st.subheader("Prediction Distributions")
    hist_data = {}
    for name in model_names:
        short = name.split("/")[-1]
        hist_data[name] = merged[f"pred_{short}"].dropna()
    hist_data["Market Price"] = merged["market_probability"]
    chart_df = pd.DataFrame(hist_data)
    st.bar_chart(chart_df.melt(var_name="Source", value_name="Probability").groupby(["Source", pd.cut(chart_df.melt(var_name="Source", value_name="Probability")["Probability"], bins=20)]).size().unstack(fill_value=0), height=300)

    # Simpler histogram approach
    st.subheader("Distribution Comparison")
    for name in model_names:
        short = name.split("/")[-1]
        vals = merged[f"pred_{short}"].dropna()
        counts, edges = np.histogram(vals, bins=20, range=(0, 1))
        bin_labels = [f"{edges[i]:.2f}" for i in range(len(counts))]
        st.caption(name)
        st.bar_chart(pd.DataFrame({"count": counts}, index=bin_labels), height=150)

    mkt_counts, mkt_edges = np.histogram(merged["market_probability"], bins=20, range=(0, 1))
    st.caption("Market Price")
    st.bar_chart(pd.DataFrame({"count": mkt_counts}, index=[f"{mkt_edges[i]:.2f}" for i in range(len(mkt_counts))]), height=150)

    # Category breakdown
    st.subheader("By Category")
    cat_stats = []
    for cat in merged["category"].unique():
        row = {"category": cat, "count": len(merged[merged["category"] == cat])}
        for name in model_names:
            short = name.split("/")[-1]
            row[f"mean_{short}"] = merged.loc[merged["category"] == cat, f"pred_{short}"].mean()
        row["market_mean"] = merged.loc[merged["category"] == cat, "market_probability"].mean()
        cat_stats.append(row)
    cat_df = pd.DataFrame(cat_stats).sort_values("count", ascending=False)
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

# ─── Tab 2: Model vs Market ──────────────────────────────────────────────────

with tab_scatter:
    st.header("Model Predictions vs Market Price")
    st.caption("Points on the diagonal = model agrees with market. Systematic deviations = exploitable bias.")

    sel_model = st.selectbox("Model", model_names)
    short = sel_model.split("/")[-1]
    col_name = f"pred_{short}"

    scatter_df = merged[["market_probability", col_name, "question", "category"]].dropna()
    scatter_df = scatter_df.rename(columns={col_name: "model_prediction"})

    st.scatter_chart(
        scatter_df,
        x="market_probability",
        y="model_prediction",
        color="category",
        height=500,
    )

    # Bias analysis
    diff = scatter_df["model_prediction"] - scatter_df["market_probability"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean bias", f"{diff.mean():+.4f}")
    c2.metric("Median bias", f"{diff.median():+.4f}")
    c3.metric("MAE vs market", f"{diff.abs().mean():.4f}")
    c4.metric("Correlation", f"{scatter_df['market_probability'].corr(scatter_df['model_prediction']):.4f}")

    # Where does this model deviate most from market?
    st.subheader(f"Largest deviations: {sel_model} vs Market")
    scatter_df["deviation"] = diff.abs()
    scatter_df["signed_deviation"] = diff
    top_devs = scatter_df.nlargest(15, "deviation")[["question", "market_probability", "model_prediction", "signed_deviation", "category"]]
    st.dataframe(top_devs, use_container_width=True, hide_index=True)

# ─── Tab 3: Disagreement ─────────────────────────────────────────────────────

with tab_disagree:
    st.header("Where Models Disagree Most")
    st.caption("High disagreement = the models have different information or reasoning. These are the best targets for an agent with web search.")

    disagree_df = merged[["question", "market_probability", "model_mean", "model_std", "model_spread", "category"] + pred_cols].copy()
    disagree_df = disagree_df.sort_values("model_spread", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean spread", f"{merged['model_spread'].mean():.4f}")
    c2.metric("Median spread", f"{merged['model_spread'].median():.4f}")
    c3.metric("Markets with spread > 0.1", f"{(merged['model_spread'] > 0.1).sum()}")

    st.subheader("Highest Disagreement Markets")
    display_cols = ["question", "market_probability", "model_spread", "model_mean"] + pred_cols + ["category"]
    st.dataframe(
        disagree_df[display_cols].head(30),
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_probability": st.column_config.NumberColumn(format="%.3f"),
            "model_spread": st.column_config.NumberColumn(format="%.3f"),
            "model_mean": st.column_config.NumberColumn(format="%.3f"),
            **{c: st.column_config.NumberColumn(format="%.3f") for c in pred_cols},
        },
    )

    st.subheader("Disagreement by Category")
    cat_disagree = merged.groupby("category").agg(
        count=("model_spread", "size"),
        mean_spread=("model_spread", "mean"),
        max_spread=("model_spread", "max"),
    ).sort_values("mean_spread", ascending=False).reset_index()
    st.dataframe(cat_disagree, use_container_width=True, hide_index=True)

# ─── Tab 4: Market Explorer ──────────────────────────────────────────────────

with tab_explore:
    st.header("Market Explorer")

    search = st.text_input("Search questions", "")
    if search:
        mask = merged["question"].str.contains(search, case=False, na=False)
        explore_df = merged[mask]
    else:
        explore_df = merged

    sort_col = st.selectbox("Sort by", ["model_spread", "volume", "market_probability", "mean_vs_market"] + pred_cols, index=0)
    sort_dir = st.radio("Direction", ["Descending", "Ascending"], horizontal=True)

    explore_df = explore_df.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

    display = explore_df[["question", "market_probability", "model_mean", "model_spread", "mean_vs_market", "category", "platform"] + pred_cols].head(50)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_probability": st.column_config.NumberColumn("Market", format="%.3f"),
            "model_mean": st.column_config.NumberColumn("Model Mean", format="%.3f"),
            "model_spread": st.column_config.NumberColumn("Spread", format="%.3f"),
            "mean_vs_market": st.column_config.NumberColumn("Mean-Market", format="%+.3f"),
            **{c: st.column_config.NumberColumn(format="%.3f") for c in pred_cols},
        },
    )

    # Detail view
    st.subheader("Market Detail")
    if len(explore_df) > 0:
        sel_idx = st.selectbox("Select market", explore_df.index, format_func=lambda i: explore_df.loc[i, "question"][:100])
        row = explore_df.loc[sel_idx]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Market Price", f"{row['market_probability']:.1%}")
            for name in model_names:
                short = name.split("/")[-1]
                val = row[f"pred_{short}"]
                diff = val - row["market_probability"]
                st.metric(name, f"{val:.1%}", delta=f"{diff:+.1%} vs market")
        with c2:
            for name in model_names:
                short = name.split("/")[-1]
                reason_col = f"reason_{short}"
                if reason_col in row and pd.notna(row.get(reason_col)):
                    st.caption(name)
                    st.write(row[reason_col])

# ─── Tab 5: Calibration ──────────────────────────────────────────────────────

with tab_calibration:
    st.header("Calibration Analysis")
    st.info("This tab will populate once markets resolve and resolution data is available. Upload a resolution file or check back after March 8-9.")

    st.subheader("Pre-resolution: Model vs Market Calibration")
    st.caption("How well do models track the market price? (Not ground truth, but a proxy.)")

    for name in model_names:
        short = name.split("/")[-1]
        col_name = f"pred_{short}"
        valid = merged[["market_probability", col_name]].dropna()

        # Bin by market probability
        valid["bin"] = pd.cut(valid["market_probability"], bins=10, labels=False)
        cal = valid.groupby("bin").agg(
            market_mean=("market_probability", "mean"),
            model_mean=(col_name, "mean"),
            count=("market_probability", "size"),
        ).reset_index()

        st.caption(f"{name} — binned model mean vs market mean")
        cal_chart = cal[["market_mean", "model_mean"]].set_index(cal["market_mean"])
        st.line_chart(cal_chart, height=250)

        brier_proxy = ((valid[col_name] - valid["market_probability"]) ** 2).mean()
        st.caption(f"MSE vs market: {brier_proxy:.6f}")
