# Active: Improve Market Selection & Run Baselines

**Date**: March 7, 2026
**Priority**: High — must submit predictions before markets resolve

---

## Problem: Current Market Sample is Not "Most Liquid"

Our 1000-market sample has serious quality issues:
- **368 markets (37%) have zero volume** — no price discovery, predictions meaningless
- **Median volume = $45** — most markets are illiquid noise
- **Only 133 markets have vol > $10k** — the real signal
- Manifold fetched by close-date instead of liquidity
- Kalshi not included at all

## Plan

### Step 1: Fix data pipeline to prioritize liquidity

**a) Manifold — sort by volume, not close-date**
- Change sort from `close-date` to `24-hour-vol` or `liquidity`
- Endpoint: `GET /v0/search-markets?sort=24-hour-vol&filter=open&contractType=BINARY`
- Then filter locally for close dates in our window
- This should surface the MrBeast puzzle ($477k combined), Iran regime, Russia-Ukraine markets

**b) Add Kalshi**
- Endpoint: `GET https://api.elections.kalshi.com/trade-api/v2/markets?status=open&limit=200`
- Fields: `ticker`, `title`, `yes_bid`, `no_bid`, `volume`, `close_time`
- **Issue**: Research shows no Kalshi markets expire before March 21 — earliest is March 21
- **Decision**: Include Kalshi markets that expire March 18-31 for a longer scoring tail, or skip Kalshi if we want fast results
- Use `mve_filter=exclude` to skip multivariate parlay bundles

**c) Enforce minimum volume**
- Filter: `volume > 100` minimum for Polymarket, `volume > 50` for Manifold
- This removes the zero-volume noise while keeping enough markets

**d) Rebalance the sample**
- Target: top 1000 by volume across ALL platforms combined
- If we can't reach 1000 with liquid markets, accept fewer (e.g., 500-700 quality > 1000 junk)

### Step 2: Re-fetch markets
- Run updated `scripts/fetch_markets.py`
- Verify volume distribution is healthier

### Step 3: Set up API keys
User needs to create `.env` with:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
TAVILY_API_KEY=tvly-...  (optional)
```

### Step 4: Run baselines
```bash
# Test on 5 markets
poetry run python scripts/run_baselines.py --limit 5

# Full run
poetry run python scripts/run_baselines.py
```
- GPT-5.2: ~$3.60 for 1000 markets
- Claude Opus 4.6: ~$10.00 (rate limit bottleneck, ~7 min)
- Gemini 2.5 Pro: ~$3.60

### Step 5: Run custom agent
```bash
poetry run python scripts/run_agent.py
```
- Ensemble of 3 baselines + extremization
- Web search on disagreement cases (if Tavily key provided)
- Est. cost: ~$30

### Step 6: Generate submission
```bash
poetry run python scripts/generate_report.py
```
Output: `results/submission.json`

### Step 7: Score after resolution (March 9-18)
```bash
poetry run python scripts/score_predictions.py
```
Output: `results/scores.json`, `results/calibration.png`

---

## Open Questions

1. **Kalshi**: Include despite no markets expiring before March 21? Or skip for faster scoring?
2. **Volume threshold**: What minimum volume makes a market "worth predicting"? $100? $1000?
3. **Quality vs quantity**: Better to have 500 liquid markets or 1000 with 37% zero-volume?
