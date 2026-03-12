# Plan: Calibrated Extremization (Platt Scaling)

**Priority**: Medium — mathematically grounded, cheap to implement, ~0.007 Brier improvement
**Estimated Brier improvement**: 0.005-0.01

---

## The Problem

Our current extremization uses a fixed factor of 1.3 applied uniformly. The AIA Forecaster paper derives two key results:

1. LLMs systematically hedge toward 0.5, even on high-certainty events. This is a well-documented bias.
2. Platt scaling (sigmoid transformation with d=sqrt(3)) is mathematically equivalent to log-odds extremization of geometric means. The optimal factor is ~1.73, not 1.3.

We're leaving Brier points on the table by being too conservative.

## The Goal

Replace the fixed 1.3 extremization factor with a properly calibrated transformation, ideally learned from data but starting with the theoretically-derived sqrt(3) factor.

## Key Design Decisions

### Factor: 1.3 vs sqrt(3) vs learned

| Approach | Factor | Pros | Cons |
|----------|--------|------|------|
| Current | 1.3 | Conservative, safe | Undercorrects hedging bias |
| AIA theoretical | sqrt(3) ~= 1.73 | Derived from Platt scaling math | May overcorrect for our domain |
| Learned from data | varies | Optimal for our specific markets | Needs resolved training data |

**Recommendation**: Switch to 1.73 as default. Once we have resolved markets from the first batch, fit the optimal factor on that data for future runs.

### Where to apply extremization

Current: applied to the final blended probability.
AIA insight: extremization is most effective on ensemble averages (which hedge more than individuals) and on mid-range probabilities (0.2-0.8). It's ineffective or harmful when the initial estimate is on the wrong side of 0.5.

Proposed:
- Apply to ensemble mean (after averaging baselines), not to individual baselines
- Apply *after* blending with market price
- Don't extremize probabilities already near 0/1 (current code handles this with the 0.01/0.99 guard — keep it)
- Consider *not* extremizing when the ensemble has very high internal agreement at a moderate probability (e.g., all 3 models say 0.45 — maybe it really is 0.45, not hedging)

### Platt scaling vs simple log-odds

Simple log-odds extremization (current approach):
```
new_p = sigmoid(factor * logit(p))
```

Full Platt scaling:
```
new_p = sigmoid(a * logit(p) + b)
```

Platt scaling adds a bias term `b` that can shift the overall probability up or down, not just stretch it. This corrects for systematic over/under-prediction. But fitting `a` and `b` requires calibration data (resolved markets with known outcomes).

**Phase 1**: Just increase the factor to 1.73. Zero additional complexity.
**Phase 2**: After scoring the first batch, fit Platt parameters (a, b) on the resolved data. Apply to future runs.

### Per-model extremization

Different models hedge differently. GPT might be more confident than Claude on average. Rather than one global factor, we could learn per-model factors:
- Run each model on historical resolved markets
- Measure each model's calibration curve
- Derive per-model correction factors

This is Phase 2/3 work — requires resolved data we don't have yet.

## What Changes

### Phase 1 (immediate, 5 minutes of work)

1. Change the default `extremize_factor` in `CustomAgent.__init__` from 1.3 to 1.73
2. Change the consensus path factor from 1.2 to 1.5 (less aggressive for consensus, where we're already confident)
3. Add the factor as a config parameter that can be tuned

### Phase 2 (after first batch resolves)

1. Add a calibration fitting script that:
   - Loads predictions and outcomes from the first batch
   - Fits optimal Platt parameters (a, b) using logistic regression on logit-transformed predictions vs outcomes
   - Saves parameters to a config file
2. Update `extremize()` to optionally use Platt parameters instead of simple factor

### Phase 3 (ongoing optimization)

1. Per-model calibration parameters
2. Domain-specific factors (sports markets may need different correction than political markets)
3. Time-to-resolution adjustment (markets closing in 2 hours may need less extremization than those closing in 2 days)

## Important Caveat from AIA

> "Corrections prove ineffective when initial judgments fall on the wrong side of 0.5, emphasizing that upstream evidence and reasoning are the primary mechanisms for improving forecast quality."

Extremization amplifies the signal. If the signal is wrong (model says 0.55 but truth is 0.30), extremizing makes it worse. This is why Plans 03 (ensembling) and 04 (agentic search) are higher priority — they improve the signal. This plan amplifies whatever signal we have.

## Success Metric

On resolved markets: compare Brier scores with factor=1.3 vs factor=1.73 vs fitted Platt. The AIA paper saw ~0.007 improvement. If our models are less hedgy (possible for sports markets with clear favorites), the gain may be smaller.
