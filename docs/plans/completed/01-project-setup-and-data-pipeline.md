# Completed: Project Setup & Data Pipeline

**Date**: March 7, 2026

## What was done

### 1. Project Structure
Created full directory hierarchy:
- `src/data/` — Market fetching, normalization, filtering
- `src/models/` — LLM evaluator classes (GPT-5.2, Claude Opus 4.6, Gemini 2.5 Pro)
- `src/agent/` — Custom ensemble agent with extremization + web search + meta-reasoning
- `src/scoring/` — Brier score, log score, calibration plots, resolution checker
- `src/utils/` — Disk cache, structured logging
- `scripts/` — 5 CLI scripts (fetch, baselines, agent, score, report)
- `configs/settings.yaml` — All configuration
- `docs/plans/active|completed/`, `docs/references/`

### 2. Dependencies
Poetry environment with: `httpx`, `openai`, `anthropic`, `google-genai`, `pandas`, `matplotlib`, `seaborn`, `pydantic`, `tavily-python`, `tenacity`, `tqdm`, `rich`

### 3. Data Pipeline (working, needs improvement)
- **Polymarket fetcher** (`src/data/polymarket.py`): Gamma API, paginated, rate-limited
- **Manifold fetcher** (`src/data/manifold.py`): Search API, sorted by close-date
- **Normalizer** (`src/data/normalizer.py`): `NormalizedMarket` Pydantic schema
- **Filters** (`src/data/filters.py`): Probability ∈ [2%, 98%], deduplication
- **Orchestrator** (`src/data/fetcher.py`): Fetch → normalize → filter → dedupe → top 1000

### 4. Initial Fetch Results
- **1000 markets** saved to `data/processed/markets.json` and `.csv`
- 909 Polymarket + 91 Manifold
- Resolution dates: March 6-18, 2026
- Top markets: Academy Awards, NBA games, geopolitics

## Known Issues

**The data quality is poor for "most liquid" markets:**
- 368/1000 markets (37%) have **zero volume**
- Median volume is only $45
- Only 133 markets have volume > $10k
- Only 39 markets have volume > $100k

**Root causes:**
1. Manifold is sorted by close-date, not volume — pulls low-liquidity markets
2. No minimum volume filter applied (min_volume=0)
3. Kalshi not included
4. Many Polymarket sports sub-markets have near-zero volume

These issues are tracked in the active plan.
