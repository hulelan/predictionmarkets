# Master Plan: Prediction Market Forecasting Challenge

**Last updated**: March 7, 2026 (Saturday evening)

---

## The Assignment

> Sample ~1000 of the most active prediction markets that are currently open. Evaluate GPT-5.2, Claude, and Gemini as baselines. Build an agent that outperforms all three. Submit results before markets resolve so scores are collectively verifiable. Markets must have the most volume and odds < 95%.

---

## Current Status

### Done
- [x] Project structure, Poetry environment, all dependencies installed
- [x] **Polymarket** integration — Gamma API, paginated, no auth
- [x] **Kalshi** integration — trade API v2, excludes MVE parlays, paginated
- [x] **Manifold** integration — search API sorted by 24h volume
- [x] Dynamic date window (always tomorrow + day after, regardless of when you run it)
- [x] Volume-sorted filtering: all 1000 markets have real trading volume (min $135)
- [x] LLM evaluators refactored to use **OpenRouter** (single API key for all 3 models)
- [x] Custom agent code (ensemble + extremization + web search + meta-reasoning)
- [x] Scoring pipeline (Brier score, log score, calibration plots, resolution checker)
- [x] **1000 markets fetched and saved** to `data/processed/markets.json`

### Not Done
- [ ] **Set up `.env`** — need `OPENROUTER_API_KEY` (and optionally `TAVILY_API_KEY`)
- [ ] **Run baselines** — smoke-test on 5 markets, then full run on 1000
- [ ] **Run custom agent** — blocked on baselines
- [ ] **Generate & submit predictions** — `results/submission.json`
- [ ] **Score after resolution** — run resolution checker Monday/Tuesday

---

## Market Sources

| Platform | Type | Markets | Median Vol | Total Vol | Notes |
|----------|------|---------|-----------|-----------|-------|
| **Kalshi** | Real money, CFTC-regulated | 602 | $644 | $1.8M | Largest market count. College basketball, NBA, NHL, UFC, tennis, crypto. |
| **Polymarket** | Real money, crypto-settled | 389 | $998 | $12.7M | Highest dollar volume. Soccer, UFC, elections, cricket, weather. |
| **Manifold** | Play money (Mana) | 9 | $2,464 | $39k | Small but adds diversity (AI, geopolitics, personal markets). |
| **Total** | | **1000** | **$754** | **$14.5M** | All vol > $135. Resolving March 8-9. |

### Not Integrated (and why)
- **Metaculus**: API returned 403. Long-term questions anyway (months/years).
- **PredictIt**: No volume data in API, politics-only, non-commercial license.

---

## Top Markets by Volume

| Vol | Prob | Platform | Question |
|-----|------|----------|----------|
| $3.5M | 52.5% | Polymarket | Will FC Barcelona win on 2026-03-08? |
| $779k | 36.5% | Polymarket | UFC 326: Oliveira vs. Holloway |
| $646k | 47.5% | Polymarket | Canadiens vs. Kings |
| $576k | 56.5% | Polymarket | Islanders vs. Sharks |
| $517k | 69.5% | Polymarket | Will India win the 2026 ICC T20 World Cup? |
| $432k | 30.8% | Polymarket | Will New Zealand win T20 World Cup? |
| $422k | 60.5% | Polymarket | UFC 326: Rodrigues vs. Ferreira |
| $391k | 39.1% | Polymarket | Will The Greens win Baden-Württemberg? |

---

## How It Works (Dynamic Dates)

`scripts/fetch_markets.py` automatically calculates:
- **tomorrow** and **day after tomorrow** from the current date
- Fetches from all 3 platforms in parallel
- Filters: volume > $100, probability ∈ [2%, 98%]
- Sorts by volume descending, takes top 1000
- Deduplicates cross-platform (same question on Poly + Kalshi)

Run it any day and it picks up the right markets.

---

## What Needs to Happen Next

### 1. Set up OpenRouter API key
```bash
cp .env.example .env
# Add your OPENROUTER_API_KEY (one key routes to GPT-5.2, Claude, Gemini)
# Optionally add TAVILY_API_KEY for agent web search
```

### 2. Smoke-test baselines (5 markets)
```bash
poetry run python scripts/run_baselines.py --limit 5
```

### 3. Full baseline run (1000 markets, ~20 min, ~$17 via OpenRouter)
```bash
poetry run python scripts/run_baselines.py
```

### 4. Run custom agent (~30 min, ~$30)
```bash
poetry run python scripts/run_agent.py
```

### 5. Generate submission
```bash
poetry run python scripts/generate_report.py
# Output: results/submission.json
```

### 6. Score after resolution (Monday/Tuesday)
```bash
poetry run python scripts/score_predictions.py
# Output: results/scores.json, results/calibration.png
```

---

## Cost Estimate (via OpenRouter)

| Component | Est. Cost |
|-----------|-----------|
| GPT-5.2 baseline × 1000 markets | ~$3.60 |
| Claude Opus 4.6 baseline × 1000 | ~$10.00 |
| Gemini 2.5 Pro baseline × 1000 | ~$3.60 |
| Custom agent (meta-reasoning on disagreement cases) | ~$30 |
| Tavily web search (free tier) | $0 |
| **Total** | **~$47** |

---

## Custom Agent Strategy

Doesn't call LLMs from scratch — combines the 3 baselines intelligently:

1. **Consensus path** (baselines agree within 10%, match market): Light extremization → done. Cheap.
2. **Disagreement path** (baseline spread > 20%): Web search for recent news → Claude meta-reasons over all evidence → informed override.
3. **Default path**: Volume-weighted blend (high-vol markets trust market price more, low-vol trust LLMs more) → extremize.

**Extremization**: Pushes probabilities away from 50% via logit scaling. Corrects the well-documented underconfidence of averaged forecasts. Factor ~1.3.

---

## Scoring

- **Brier score** (primary): `mean((prediction - outcome)^2)`. Perfect = 0, coin flip = 0.25.
- **Log score** (secondary): Rewards confident correct predictions, heavily penalizes confident wrong ones.
- **Calibration plot**: Predicted probability vs. observed frequency across 10 bins.
- **Breakdown**: By model, by category, by platform, by volume tier.

---

## Key Files

```
scripts/fetch_markets.py          # Fetch 1000 markets (dynamic dates, 3 platforms)
scripts/run_baselines.py          # Run GPT/Claude/Gemini via OpenRouter
scripts/run_agent.py              # Run custom agent
scripts/score_predictions.py      # Score after resolution
scripts/generate_report.py        # Generate submission JSON

src/data/polymarket.py            # Polymarket Gamma API client
src/data/kalshi.py                # Kalshi trade API v2 client
src/data/manifold.py              # Manifold search API client (sorted by 24h vol)
src/data/fetcher.py               # Orchestrator: all platforms → normalize → filter → top 1000
src/data/normalizer.py            # NormalizedMarket Pydantic schema

src/models/openrouter_model.py    # OpenRouter evaluator (routes to any model)
src/models/evaluator.py           # Baseline runner config
src/models/prompts.py             # Shared forecasting prompt

src/agent/agent.py                # Custom agent (ensemble + web search + meta-reasoning)
src/agent/ensemble.py             # Weighted ensemble + extremization math

src/scoring/brier.py              # Brier & log score
src/scoring/calibration.py        # Calibration curve plots
src/scoring/resolution.py         # Resolution checker (all 3 platforms)
```

---

## GPT-5.3 Note

The assignment says GPT-5.3, but GPT-5.3-Codex is not available via API (ChatGPT/CLI only). We use GPT-5.2 via OpenRouter as the OpenAI baseline. If 5.3 gets API access, we switch.
