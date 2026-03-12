"""Audit app: compare all predictions per question + color-coded aggregation table."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediction Audit", layout="wide")

# ─── Data Loading ────────────────────────────────────────────────────────────

PRED_FILES = {
    "gpt-5.4": "openai_gpt-5.4.jsonl",
    "claude-opus": "anthropic_claude-opus-4-6.jsonl",
    "gemini-pro": "google_gemini-2.5-pro.jsonl",
    "agent_claude_t03": "tool_agent_claude_t03.jsonl",
    "agent_gpt_t03": "tool_agent_gpt_t03.jsonl",
    "agent_gemini_t03": "tool_agent_gemini_t03.jsonl",
    "agent_claude_t07": "tool_agent_claude_t07.jsonl",
    "agent_gpt_t07": "tool_agent_gpt_t07.jsonl",
    "ensemble": "ensemble_agent.jsonl",
}


@st.cache_data
def load_all():
    markets = json.load(open("data/processed/markets.json"))
    market_lookup = {m["id"]: m for m in markets}

    # All market IDs in baseline ordering
    all_ids = []
    for line in open("data/predictions/openai_gpt-5.4.jsonl").read().strip().split("\n"):
        if line:
            all_ids.append(json.loads(line)["market_id"])

    # Load all predictions
    all_preds = {}
    for label, fname in PRED_FILES.items():
        path = Path("data/predictions") / fname
        if not path.exists():
            continue
        preds = {}
        for line in path.read_text().strip().split("\n"):
            if line:
                row = json.loads(line)
                preds[row["market_id"]] = row
        all_preds[label] = preds

    # Load resolutions (all available)
    resolutions = {}
    for res_file in Path("data/predictions").glob("resolutions*.json"):
        resolutions.update(json.load(open(res_file)))

    return all_ids, market_lookup, all_preds, resolutions


all_ids, market_lookup, all_preds, resolutions = load_all()
model_names = list(all_preds.keys())

# Agent models that had web search (could see post-hoc results)
AGENT_MODELS = [k for k in model_names if "agent" in k]
POST_HOC_THRESHOLD = 0.05  # if any agent predicted <= 0.05 or >= 0.95, likely saw the answer


def is_post_hoc(mid: str) -> bool:
    """Check if any agent gave an extreme prediction suggesting it saw the result."""
    for label in AGENT_MODELS:
        pred = all_preds.get(label, {}).get(mid)
        if pred and pred.get("probability") is not None:
            p = float(pred["probability"])
            if p <= POST_HOC_THRESHOLD or p >= (1 - POST_HOC_THRESHOLD):
                return True
    return False


# Build filtered ID list (genuine forecasts only)
genuine_ids = [mid for mid in all_ids if not is_post_hoc(mid)]
n_total = len(all_ids)
n_genuine = len(genuine_ids)
n_filtered = n_total - n_genuine

n_markets = len(all_ids)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_detail, tab_table = st.tabs(["Per-Question Detail", "Aggregation Table"])

# ─── Tab 1: Per-Question Detail ─────────────────────────────────────────────

with tab_detail:
    st.header("Per-Question Predictions")

    idx = st.number_input("Market #", min_value=1, max_value=n_markets, value=1, step=1)
    mid = all_ids[idx - 1]
    m = market_lookup.get(mid, {})

    resolved = resolutions.get(mid)
    outcome_str = {1: "YES", 0: "NO"}.get(resolved, "Unresolved")
    outcome_color = {1: "green", 0: "red"}.get(resolved, "gray")

    st.markdown(f"### {m.get('question', mid)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Price", f"{m.get('market_probability', 0):.1%}")
    c2.markdown(f"**Outcome:** :{outcome_color}[**{outcome_str}**]")
    c3.metric("Platform", m.get("platform", "?"))
    c4.metric("Category", m.get("category", "?"))

    st.markdown(f"**Resolution date:** {m.get('resolution_date', '?')}")
    if is_post_hoc(mid):
        st.warning("An agent gave an extreme prediction on this market (likely post-hoc)")
    st.markdown("---")

    # Show all model predictions
    for label in model_names:
        pred = all_preds[label].get(mid)
        if not pred:
            continue

        prob = pred.get("probability")
        conf = pred.get("confidence", "?")
        reasoning = pred.get("reasoning", "")
        searches = pred.get("search_calls", None)

        if prob is not None:
            prob_f = float(prob)
            # Color based on distance from outcome
            if resolved is not None:
                err = abs(prob_f - resolved)
                if err < 0.1:
                    dot = "🟢"
                elif err < 0.3:
                    dot = "🟡"
                else:
                    dot = "🔴"
            else:
                dot = "⚪"

            search_str = f" | {searches} searches" if searches else ""
            st.markdown(f"**{dot} {label}** — prob: `{prob_f:.3f}` | conf: `{conf}`{search_str}")
            if reasoning:
                with st.expander("Reasoning"):
                    st.write(reasoning)
        else:
            st.markdown(f"**⚪ {label}** — no prediction")

    # Navigation
    col_prev, col_next = st.columns(2)
    with col_prev:
        if idx > 1:
            st.button("Previous", key="prev")
    with col_next:
        if idx < n_markets:
            st.button("Next", key="next")

# ─── Tab 2: Aggregation Table ───────────────────────────────────────────────

with tab_table:
    st.header("All Predictions — Color-Coded Table")

    # Filter options
    fc1, fc2 = st.columns(2)
    filter_resolved = fc1.checkbox("Only show resolved markets", value=True)
    hide_post_hoc = fc2.checkbox("Hide post-hoc (agent saw answer)", value=True)

    rows = []
    for i, mid in enumerate(all_ids):
        m = market_lookup.get(mid, {})
        resolved = resolutions.get(mid)

        if filter_resolved and resolved is None:
            continue
        if hide_post_hoc and is_post_hoc(mid):
            continue

        row = {
            "#": i + 1,
            "Question": m.get("question", mid)[:80],
            "Category": m.get("category", "?"),
            "Result": {1: "YES", 0: "NO"}.get(resolved, "?"),
            "Market": m.get("market_probability", None),
        }

        for label in model_names:
            pred = all_preds[label].get(mid)
            if pred and pred.get("probability") is not None:
                row[label] = float(pred["probability"])
            else:
                row[label] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        st.warning("No resolved markets found.")
    else:
        # Color-coding function: green = close to outcome, red = far from outcome
        def color_pred(val, outcome):
            if pd.isna(val) or outcome is None:
                return ""
            if outcome == "YES":
                target = 1.0
            elif outcome == "NO":
                target = 0.0
            else:
                return ""
            err = abs(val - target)
            # Gradient: 0 error = bright green, 0.5+ error = bright red
            if err <= 0.1:
                return "background-color: #1a9641; color: white"
            elif err <= 0.2:
                return "background-color: #a6d96a; color: black"
            elif err <= 0.3:
                return "background-color: #ffffbf; color: black"
            elif err <= 0.4:
                return "background-color: #fdae61; color: black"
            else:
                return "background-color: #d7191c; color: white"

        pred_cols = ["Market"] + model_names

        def style_row(row):
            outcome = row["Result"]
            styles = [""] * len(row)
            for i, col in enumerate(row.index):
                if col in pred_cols and pd.notna(row[col]):
                    styles[i] = color_pred(row[col], outcome)
                elif col == "Result":
                    if outcome == "YES":
                        styles[i] = "background-color: #1a9641; color: white; font-weight: bold"
                    elif outcome == "NO":
                        styles[i] = "background-color: #d7191c; color: white; font-weight: bold"
            return styles

        styled = df.style.apply(style_row, axis=1).format(
            {col: "{:.3f}" for col in pred_cols if col in df.columns},
            na_rep="—",
        )

        st.dataframe(styled, use_container_width=True, height=800)

        # Legend
        st.markdown("""
        **Color legend** (distance from true outcome):
        - 🟩 **< 0.1** — Excellent
        - 🟨 **0.1 – 0.2** — Good
        - 🟨 **0.2 – 0.3** — Fair
        - 🟧 **0.3 – 0.4** — Poor
        - 🟥 **> 0.4** — Bad
        """)

        # Summary stats at bottom
        st.markdown("---")
        st.subheader("Brier Scores (resolved markets only)")

        resolved_df = df[df["Result"].isin(["YES", "NO"])].copy()
        resolved_df["outcome_num"] = (resolved_df["Result"] == "YES").astype(float)

        brier_rows = []
        for col in pred_cols:
            valid = resolved_df[[col, "outcome_num"]].dropna()
            if len(valid) > 0:
                bs = ((valid[col] - valid["outcome_num"]) ** 2).mean()
                brier_rows.append({"Model": col, "Brier Score": f"{bs:.4f}", "N": len(valid)})

        brier_df = pd.DataFrame(brier_rows).sort_values("Brier Score")
        st.dataframe(brier_df, use_container_width=True, hide_index=True)
