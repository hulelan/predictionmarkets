"""Streamlit dashboard for prediction market forecasting performance."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediction Markets — Agent Performance", layout="wide")

# ─── Data Loading ────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)  # reload every 60s for live updates
def load_data():
    """Load markets and all predictions (agent + baselines)."""
    markets = pd.DataFrame(json.load(open("data/processed/markets.json")))
    markets["resolution_date"] = pd.to_datetime(markets["resolution_date"])

    pred_dir = Path("data/predictions")
    all_preds = {}
    for f in sorted(pred_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in f.read_text().strip().split("\n") if line]
        if rows:
            all_preds[f.stem] = pd.DataFrame(rows)

    return markets, all_preds


markets, all_preds = load_data()

# Identify agent results
judge_df = all_preds.get("ensemble_agent")
has_judge = judge_df is not None and len(judge_df) > 0

# Identify baseline + search baseline predictions
baseline_keys = [k for k in all_preds if k.startswith(("openai_", "anthropic_", "google_")) and "run_" not in k]
search_keys = [k for k in all_preds if k.startswith("search_")]
agent_keys = [k for k in all_preds if k.startswith("tool_agent_")]

# Build main dataframe
df = markets[["id", "question", "market_probability", "category", "platform",
              "resolution_date", "volume", "url"]].copy()

# Add resolved status
if "resolved" in markets.columns:
    df["resolved"] = markets["resolved"].fillna(False).astype(bool)
    df["outcome"] = markets.get("outcome")
else:
    df["resolved"] = False
    df["outcome"] = None

if "last_refreshed" in markets.columns:
    df["last_refreshed"] = markets["last_refreshed"]

# Merge agent judge predictions
if has_judge:
    judge_merge = judge_df[["market_id", "probability", "confidence", "reasoning"]].copy()
    judge_merge = judge_merge.rename(columns={
        "probability": "agent_pred",
        "confidence": "agent_conf",
        "reasoning": "agent_reasoning",
    })
    df = df.merge(judge_merge, left_on="id", right_on="market_id", how="left").drop(columns=["market_id"])

    # Add run details if available
    if "run_probs" in judge_df.columns:
        run_probs = judge_df.set_index("market_id")["run_probs"]
        df["run_probs"] = df["id"].map(run_probs)
        df["run_spread"] = df["run_probs"].apply(
            lambda x: max(x) - min(x) if isinstance(x, list) and len(x) > 1 else 0
        )

# Merge search baselines
for key in search_keys:
    pred_df = all_preds[key][["market_id", "probability"]].copy()
    short = key.replace("search_", "s_")
    pred_df = pred_df.rename(columns={"probability": f"pred_{short}"})
    df = df.merge(pred_df, left_on="id", right_on="market_id", how="left").drop(columns=["market_id"])

# Merge raw baselines
for key in baseline_keys:
    pred_df = all_preds[key][["market_id", "probability"]].copy()
    short = key.split("_")[0][:6]  # e.g. "openai", "anthro", "google"
    pred_df = pred_df.rename(columns={"probability": f"base_{short}"})
    df = df.merge(pred_df, left_on="id", right_on="market_id", how="left").drop(columns=["market_id"])

# Derived columns
if has_judge:
    df["agent_vs_market"] = df["agent_pred"] - df["market_probability"]
    df["agent_abs_dev"] = df["agent_vs_market"].abs()

# Brier scores for resolved markets
resolved_df = df[df["resolved"] == True].copy()
has_resolutions = len(resolved_df) > 0
if has_resolutions:
    resolved_df["outcome_binary"] = resolved_df["outcome"].map({"Yes": 1, "No": 0})
    resolved_df = resolved_df.dropna(subset=["outcome_binary"])
    has_resolutions = len(resolved_df) > 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("Filters")

cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
sel_cat = st.sidebar.selectbox("Category", cats)
if sel_cat != "All":
    df = df[df["category"] == sel_cat]

platforms = ["All"] + sorted(df["platform"].unique().tolist())
sel_plat = st.sidebar.selectbox("Platform", platforms)
if sel_plat != "All":
    df = df[df["platform"] == sel_plat]

show_resolved = st.sidebar.radio("Resolution status", ["All", "Resolved", "Unresolved"])
if show_resolved == "Resolved":
    df = df[df["resolved"] == True]
elif show_resolved == "Unresolved":
    df = df[df["resolved"] != True]

st.sidebar.markdown("---")
st.sidebar.metric("Markets shown", len(df))
n_resolved = df["resolved"].sum() if "resolved" in df.columns else 0
st.sidebar.metric("Resolved", int(n_resolved))

if "last_refreshed" in df.columns:
    latest = df["last_refreshed"].dropna()
    if len(latest) > 0:
        st.sidebar.caption(f"Last refresh: {str(latest.iloc[0])[:19]}")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Performance",
    "Agent Predictions",
    "Scoring",
    "Market Explorer",
    "Baselines",
])

# ─── Tab 1: Performance Overview ─────────────────────────────────────────────

with tabs[0]:
    st.header("Agent Performance")

    if not has_judge:
        st.warning("No agent predictions found. Run the agent pipeline first.")
    else:
        # Key metrics
        valid = df.dropna(subset=["agent_pred"])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Markets predicted", len(valid))
        c2.metric("Resolved", int(n_resolved))
        c3.metric("Mean deviation from market", f"{valid['agent_vs_market'].mean():+.3f}")
        c4.metric("Median |deviation|", f"{valid['agent_abs_dev'].median():.3f}")
        agrees = (valid["agent_abs_dev"] < 0.03).sum()
        c5.metric("Within 3% of market", f"{agrees}/{len(valid)}")

        if has_resolutions:
            st.subheader("Brier Scores (resolved markets)")
            r = resolved_df.dropna(subset=["outcome_binary"])
            if len(r) > 0:
                scores = {}
                # Agent
                agent_r = r.dropna(subset=["agent_pred"])
                if len(agent_r) > 0:
                    scores["Agent (Judge)"] = {
                        "brier": ((agent_r["agent_pred"] - agent_r["outcome_binary"]) ** 2).mean(),
                        "n": len(agent_r),
                    }
                # Market
                scores["Market Price"] = {
                    "brier": ((r["market_probability"] - r["outcome_binary"]) ** 2).mean(),
                    "n": len(r),
                }
                # Search baselines
                for key in search_keys:
                    short = key.replace("search_", "s_")
                    col = f"pred_{short}"
                    if col in r.columns:
                        br = r.dropna(subset=[col])
                        if len(br) > 0:
                            scores[f"Search {key.split('_', 1)[1]}"] = {
                                "brier": ((br[col] - br["outcome_binary"]) ** 2).mean(),
                                "n": len(br),
                            }

                score_df = pd.DataFrame(scores).T
                score_df.columns = ["Brier Score", "N"]
                score_df = score_df.sort_values("Brier Score")
                st.dataframe(score_df, use_container_width=True)
        else:
            st.info("No markets resolved yet. Brier scores will appear here once markets resolve.")

        # Distribution of agent predictions vs market
        st.subheader("Agent vs Market Distribution")
        chart_data = pd.DataFrame({
            "Agent Prediction": valid["agent_pred"],
            "Market Price": valid["market_probability"],
        })
        st.scatter_chart(
            chart_data,
            x="Market Price",
            y="Agent Prediction",
            height=400,
        )

        # Where agent deviates most
        st.subheader("Largest Agent Deviations from Market")
        dev_df = valid.nlargest(20, "agent_abs_dev")[
            ["question", "market_probability", "agent_pred", "agent_vs_market",
             "agent_conf", "platform", "category"]
        ].copy()
        dev_df.columns = ["Question", "Market", "Agent", "Delta", "Confidence", "Platform", "Category"]
        st.dataframe(
            dev_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Market": st.column_config.NumberColumn(format="%.3f"),
                "Agent": st.column_config.NumberColumn(format="%.3f"),
                "Delta": st.column_config.NumberColumn(format="%+.3f"),
            },
        )

# ─── Tab 2: Agent Predictions Detail ─────────────────────────────────────────

with tabs[1]:
    st.header("Agent Predictions")

    if not has_judge:
        st.warning("No agent predictions found.")
    else:
        valid = df.dropna(subset=["agent_pred"])

        # Summary by confidence
        conf_summary = valid.groupby("agent_conf").agg(
            count=("agent_pred", "size"),
            mean_abs_dev=("agent_abs_dev", "mean"),
        ).reset_index()
        st.dataframe(conf_summary, use_container_width=True, hide_index=True)

        # Full table
        st.subheader("All Predictions")
        search = st.text_input("Search", "", key="agent_search")
        show_df = valid
        if search:
            show_df = show_df[show_df["question"].str.contains(search, case=False, na=False)]

        sort_col = st.selectbox("Sort by", ["agent_abs_dev", "agent_vs_market", "market_probability", "volume"], key="agent_sort")
        show_df = show_df.sort_values(sort_col, ascending=False)

        display = show_df[["question", "market_probability", "agent_pred", "agent_vs_market",
                           "agent_conf", "category", "platform", "resolved"]].head(100)
        display.columns = ["Question", "Market", "Agent", "Delta", "Confidence", "Category", "Platform", "Resolved"]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Market": st.column_config.NumberColumn(format="%.3f"),
                "Agent": st.column_config.NumberColumn(format="%.3f"),
                "Delta": st.column_config.NumberColumn(format="%+.3f"),
            },
        )

        # Detail view
        st.subheader("Prediction Detail")
        if len(show_df) > 0:
            sel_idx = st.selectbox(
                "Select market",
                show_df.index,
                format_func=lambda i: show_df.loc[i, "question"][:100],
                key="agent_detail",
            )
            row = show_df.loc[sel_idx]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Market Price", f"{row['market_probability']:.1%}")
                st.metric("Agent Prediction", f"{row['agent_pred']:.1%}",
                          delta=f"{row['agent_vs_market']:+.1%} vs market")
                st.metric("Confidence", row["agent_conf"])
                if row.get("resolved"):
                    st.metric("Outcome", row.get("outcome", "Unknown"))
                if isinstance(row.get("run_probs"), list):
                    st.caption("Individual agent runs:")
                    for i, p in enumerate(row["run_probs"], 1):
                        st.text(f"  Run {i}: {p:.3f}")
            with c2:
                st.caption("Judge Reasoning")
                st.write(row.get("agent_reasoning", "No reasoning available"))

# ─── Tab 3: Scoring ──────────────────────────────────────────────────────────

with tabs[2]:
    st.header("Scoring & Calibration")

    if not has_resolutions:
        st.info("Waiting for markets to resolve. This tab will populate automatically as outcomes come in.")
        st.caption(f"Current status: {int(n_resolved)} / {len(df)} markets resolved")
    else:
        r = resolved_df.dropna(subset=["outcome_binary"])

        st.subheader(f"Resolved: {len(r)} markets")

        # Outcome distribution
        c1, c2 = st.columns(2)
        yes_count = (r["outcome_binary"] == 1).sum()
        no_count = (r["outcome_binary"] == 0).sum()
        c1.metric("Resolved Yes", int(yes_count))
        c2.metric("Resolved No", int(no_count))

        # Calibration chart for agent
        if has_judge and "agent_pred" in r.columns:
            agent_r = r.dropna(subset=["agent_pred"])
            if len(agent_r) >= 5:
                st.subheader("Agent Calibration")
                agent_r["bin"] = pd.cut(agent_r["agent_pred"], bins=10, labels=False)
                cal = agent_r.groupby("bin").agg(
                    predicted=("agent_pred", "mean"),
                    actual=("outcome_binary", "mean"),
                    count=("outcome_binary", "size"),
                ).reset_index()
                st.line_chart(
                    cal[["predicted", "actual"]].set_index(cal["predicted"]),
                    height=300,
                )

        # Per-market results
        st.subheader("Resolved Market Results")
        results_display = r[["question", "market_probability", "outcome"]].copy()
        if has_judge and "agent_pred" in r.columns:
            results_display["agent_pred"] = r["agent_pred"]
            results_display["agent_error"] = (r["agent_pred"] - r["outcome_binary"]).abs()
            results_display["market_error"] = (r["market_probability"] - r["outcome_binary"]).abs()
            results_display["agent_better"] = results_display["agent_error"] < results_display["market_error"]

        st.dataframe(results_display, use_container_width=True, hide_index=True)

# ─── Tab 4: Market Explorer ──────────────────────────────────────────────────

with tabs[3]:
    st.header("Market Explorer")

    search = st.text_input("Search questions", "", key="explore_search")
    explore_df = df
    if search:
        explore_df = explore_df[explore_df["question"].str.contains(search, case=False, na=False)]

    cols_to_show = ["question", "market_probability", "platform", "category", "resolved"]
    if has_judge:
        cols_to_show.insert(2, "agent_pred")
        cols_to_show.insert(3, "agent_vs_market")

    sort_options = [c for c in cols_to_show if c not in ("question", "resolved")]
    sort_col = st.selectbox("Sort by", sort_options, key="explore_sort")
    sort_dir = st.radio("Direction", ["Descending", "Ascending"], horizontal=True, key="explore_dir")
    explore_df = explore_df.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

    st.dataframe(
        explore_df[cols_to_show].head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_probability": st.column_config.NumberColumn("Market", format="%.3f"),
            "agent_pred": st.column_config.NumberColumn("Agent", format="%.3f"),
            "agent_vs_market": st.column_config.NumberColumn("Delta", format="%+.3f"),
        },
    )

    # Detail with link
    if len(explore_df) > 0:
        sel_idx = st.selectbox(
            "Select market for detail",
            explore_df.index,
            format_func=lambda i: explore_df.loc[i, "question"][:100],
            key="explore_detail",
        )
        row = explore_df.loc[sel_idx]
        if pd.notna(row.get("url")):
            st.markdown(f"[Open on {row['platform']}]({row['url']})")

# ─── Tab 5: Baselines ────────────────────────────────────────────────────────

with tabs[4]:
    st.header("Baseline Comparison")

    search_pred_cols = [c for c in df.columns if c.startswith("pred_s_")]
    base_pred_cols = [c for c in df.columns if c.startswith("base_")]
    all_pred_cols = search_pred_cols + base_pred_cols

    if not all_pred_cols:
        st.info("No baseline predictions found. Run baselines first.")
    else:
        # Summary stats
        st.subheader("Baseline Summary")
        summary_rows = []
        for col in all_pred_cols:
            valid = df[col].dropna()
            label = col.replace("pred_s_", "search/").replace("base_", "raw/")
            mae = (valid - df.loc[valid.index, "market_probability"]).abs().mean()
            summary_rows.append({
                "Model": label,
                "N": len(valid),
                "Mean Pred": valid.mean(),
                "MAE vs Market": mae,
            })
        if has_judge:
            agent_valid = df["agent_pred"].dropna()
            mae = (agent_valid - df.loc[agent_valid.index, "market_probability"]).abs().mean()
            summary_rows.insert(0, {
                "Model": "Agent (Judge)",
                "N": len(agent_valid),
                "Mean Pred": agent_valid.mean(),
                "MAE vs Market": mae,
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if has_resolutions:
            st.subheader("Brier Scores (resolved)")
            r = resolved_df.dropna(subset=["outcome_binary"])
            brier_rows = []
            for col in all_pred_cols:
                valid = r.dropna(subset=[col])
                if len(valid) > 0:
                    brier = ((valid[col] - valid["outcome_binary"]) ** 2).mean()
                    label = col.replace("pred_s_", "search/").replace("base_", "raw/")
                    brier_rows.append({"Model": label, "Brier": brier, "N": len(valid)})
            # Market baseline
            brier_rows.append({
                "Model": "Market Price",
                "Brier": ((r["market_probability"] - r["outcome_binary"]) ** 2).mean(),
                "N": len(r),
            })
            if has_judge:
                agent_r = r.dropna(subset=["agent_pred"])
                if len(agent_r) > 0:
                    brier_rows.insert(0, {
                        "Model": "Agent (Judge)",
                        "Brier": ((agent_r["agent_pred"] - agent_r["outcome_binary"]) ** 2).mean(),
                        "N": len(agent_r),
                    })
            brier_df = pd.DataFrame(brier_rows).sort_values("Brier")
            st.dataframe(brier_df, use_container_width=True, hide_index=True)
