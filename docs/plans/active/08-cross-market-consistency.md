# Plan: Cross-Market Consistency

**Priority**: Low-Medium — niche edge, but exploits a unique structural advantage
**Estimated Brier improvement**: 0.005 on affected markets (small subset)

---

## The Problem

We forecast 1000 markets independently. But some markets are logically related:
- "Will India win the T20 World Cup?" and "Will New Zealand win the T20 World Cup?" — probabilities of all contenders must sum to ~1.0 (minus ties/draws)
- "Will Oliveira beat Holloway?" and "Will Holloway beat Oliveira?" — must sum to ~1.0
- Correlated markets: if Barcelona wins, multiple Barcelona-related markets resolve together
- Conditional relationships: "Will X happen by March 9?" implies "Will X happen by March 15?"

Our baselines can produce inconsistent forecasts across related markets. If GPT says India has 70% to win the T20 World Cup and also gives New Zealand 50%, those can't both be right. Detecting and correcting these inconsistencies is free alpha.

The Semantic Trading paper (arxiv 2512.02436) uses clustering and relationship discovery to find these pairs, achieving 60-70% accuracy on relationship prediction and ~20% returns on exploiting mispricings between them.

## The Goal

Detect logically related markets in our sample and enforce consistency constraints across their forecasts. This catches a specific failure mode of independent forecasting and corrects it without additional API calls.

## Key Design Decisions

### How to find related markets

1. **Complementary markets**: Same event, opposite outcomes. Detect via question similarity + same resolution date. "Will Team A win?" and "Will Team B win?" in the same game.
2. **Mutually exclusive markets**: Multiple outcomes of the same event (tournament winners, election candidates). Probabilities must sum to ≤1.0.
3. **Implication relationships**: "X by March 9" implies "X by March 15". Can detect via shared keywords + different dates.
4. **Correlation clusters**: Markets likely to move together (all NBA games on the same night might correlate with general sports betting trends, but this is weaker signal).

Focus on types 1 and 2 — they have hard logical constraints that are easy to verify and correct.

### Detection method

- **Embedding similarity**: Embed all 1000 market questions, cluster by cosine similarity, manually review clusters. Good for finding related markets but expensive if using LLM embeddings.
- **Keyword + date matching**: Group markets by event keywords (team names, tournament names) and resolution date. Fast, no API cost, catches most sports-related pairs.
- **Platform-provided grouping**: Polymarket groups related outcomes under a single "event" (same `condition_id` or `group_slug`). Kalshi has event tickers. Use this metadata when available.

Start with platform-provided grouping (it's free and accurate), supplement with keyword matching for cross-platform pairs.

### Consistency enforcement

Once we find a group of mutually exclusive markets (e.g., "India wins T20 WC" / "New Zealand wins T20 WC" / "England wins T20 WC"):

1. Get our forecasted probabilities for each outcome
2. Check if they sum to >1.0 (overcomplete) or <<1.0 (undercomplete, implying a large "other" category)
3. If inconsistent: normalize proportionally so they sum to 1.0 (or to 1.0 minus a reasonable "other" estimate)
4. For binary complement pairs ("Team A wins" / "Team B wins" in a game): enforce p(A) + p(B) = 1.0

This is a post-processing step — runs after all individual forecasts are complete, requires no additional API calls.

### When not to enforce

- Markets that look related but have different resolution criteria (e.g., "Will India win the T20 WC?" vs "Will India reach the T20 WC semifinals?" — these are related but not mutually exclusive)
- Markets on different platforms with slightly different resolution rules
- When the "other" category is legitimate (tournament winner markets where we only track a few contenders)

Be conservative: only enforce constraints when the logical relationship is clear.

## What Changes

### 1. Market grouping module

A new module (or function in analyzer.py) that:
- Groups markets by platform-provided event IDs where available
- Runs keyword-based matching for cross-platform grouping
- Returns groups of related markets with the type of relationship (complement, mutually_exclusive, implication)

### 2. Consistency post-processor

After all forecasts are generated, before submission:
- For each group, check logical constraints
- If violated: adjust probabilities proportionally
- Log all adjustments for review
- Flag large adjustments (>0.05) for supervisor review (Plan 06)

### 3. Integration point

This runs as a final step in `run_agent.py`, after all individual market predictions are complete. It's a pure post-processing step that reads and modifies the prediction cache.

## Cost Estimate

Zero additional API cost. This is pure logic applied to existing predictions. Implementation time is the only cost.

## Scope Limitation

For this first run, focus only on:
- Binary complement pairs (Team A vs Team B in the same game) — there are probably 50-100 of these in our sample
- Tournament winner groups (T20 WC, UFC fights with multiple related markets)

Don't try to discover subtle correlations or implication chains. Keep it simple.

## Success Metric

Count how many market groups have inconsistent forecasts before correction. Measure the average adjustment size. After resolution, check whether consistency-enforced predictions have better Brier scores than the uncorrected ones. Even a small improvement on 50-100 markets is worth it since the cost is zero.
