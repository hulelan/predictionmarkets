# Plan: Multi-Run Ensembling

**Priority**: Highest — cheapest high-impact improvement per AIA Forecaster ablations
**Estimated Brier improvement**: ~0.01-0.02 (variance reduction alone)

---

## The Problem

Every baseline model runs once per market at temperature 0.3. The AIA Forecaster paper (arxiv 2511.07678) shows individual LLM forecast runs are *extremely* noisy — their Figure 3 demonstrates that single-run evaluations "drastically understate" actual variability. Averaging 5-10 independent runs is the single cheapest way to improve Brier scores.

Our current agent treats each baseline prediction as a point truth. It isn't — it's one sample from a distribution. We're building on sand.

## The Goal

Run each forecast N times independently and aggregate to a stable estimate. The agent should work with *distributions* of baseline predictions, not point estimates.

## Key Design Decisions

### How many runs?

AIA shows sharp improvement from 1->5 runs with diminishing returns beyond 10. But we have 1000 markets x 3 models. The cost curve:

| Runs | Total API calls | Est. cost (all 3 models) | Brier gain |
|------|----------------|--------------------------|------------|
| 1 (current) | 3,000 | ~$17 | baseline |
| 3 | 9,000 | ~$51 | significant |
| 5 | 15,000 | ~$85 | near-optimal |
| 10 | 30,000 | ~$170 | diminishing |

**Suggestion**: Default to N=3 for cost reasons, allow N=5 via flag. Or: use a single cheap model (Gemini Flash) for ensemble diversity at N=5, keep expensive models at N=1.

### Aggregation method

- **Trimmed mean** (drop highest and lowest, average rest) — robust to outlier runs. AIA uses this. Requires N>=3.
- **Simple mean** — works fine per AIA ("simple averaging substantially outperforms elaborate aggregation schemes"). Good for N=3.
- **Median** — most robust to outliers but throws away information.

Start with simple mean for N=3, trimmed mean for N>=5.

### Temperature

Current: 0.3 (low). For ensembling to work, runs need to be *different*. Options:
- Raise to 0.7 for ensemble runs (AIA's approach)
- Keep 0.3 but rely on inherent LLM nondeterminism
- Vary temperature across runs (0.3, 0.5, 0.7) for diversity

The research suggests 0.7 is the sweet spot for diverse-but-not-crazy forecasts.

## What Changes

### 1. Ensemble runner (new module or extension of evaluator)

The core loop becomes: for each market, run the model N times at temp 0.7, collect N probabilities, aggregate. This replaces the single-shot `predict()` call.

Implementation considerations:
- Can reuse `OpenRouterEvaluator` — just call `predict()` N times per market
- Concurrency: still bounded by semaphore, but now N*1000 tasks instead of 1000
- Caching: need to store all N runs, not just the aggregate (for analysis)
- Cache key: `{model_name}_run_{i}` or store as array in a single cache entry

### 2. Cache format evolution

Current: one JSONL file per model, one line per market.
Needed: either N JSONL files per model (`gpt-5.2_run_0.jsonl`, `gpt-5.2_run_1.jsonl`, ...) or a single file with a `runs` array per market.

The multi-file approach is simpler and backwards-compatible with existing cache loading. The single-file approach is cleaner for analysis.

### 3. Agent receives distribution, not point

`CustomAgent.predict()` currently takes `baselines: dict[str, float]`. It should receive something richer — either `dict[str, list[float]]` (all runs) or `dict[str, RunStats]` where RunStats has mean, std, min, max. The agent can then use spread-across-runs as a confidence signal (high variance = the model is unsure, weight it less).

### 4. Baseline runner script changes

`run_baselines.py` needs a `--runs N` flag. The evaluator batch loop runs N passes. Progress bar shows total (N * num_markets).

## Cost Control Options

If budget is tight:
- Use N=5 only for high-spread markets (where baselines disagree >0.15), N=1 for consensus markets
- Use cheap model (Gemini 2.0 Flash via OpenRouter) for ensemble diversity runs, keep expensive models at N=1
- Run N=3 on first pass, add more runs only where std across runs is high

## What This Enables Downstream

- The agent can detect when a model is *uncertain* (high run-to-run variance) vs *wrong* (consistently off from market)
- Supervisor agent (Plan 06) gets richer signal for disagreement resolution
- Calibration analysis can report per-model reliability bands

## Success Metric

Compare single-run vs N-run ensemble Brier scores on held-out resolved markets. Expect 0.01-0.02 improvement. If no improvement, the baselines were already stable (unlikely per AIA findings, but possible for short-horizon sports markets where the question is less ambiguous).
