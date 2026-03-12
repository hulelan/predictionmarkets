# Plan: Supervisor Agent

**Priority**: Medium — improves on naive averaging, but depends on Plans 03/04
**Estimated Brier improvement**: ~0.005-0.01 over simple mean aggregation

---

## The Problem

Our current agent has a "meta-reasoning" path that fires only on high-disagreement cases (spread > 0.2). When it fires, it does a web search and asks Claude to reconcile the baselines. When it doesn't fire, we do a volume-weighted blend — essentially a weighted average with no reasoning.

The AIA Forecaster shows that a supervisor agent that *investigates disagreements* (rather than just averaging) beats naive aggregation: 0.1125 vs 0.1140 Brier. The key difference: the supervisor identifies *why* agents disagree and does targeted searches to resolve the ambiguity, rather than splitting the difference.

But the AIA paper also shows the improvement is modest — simple mean is surprisingly hard to beat with fancy aggregation. The supervisor's value comes specifically from its ability to do follow-up research, not from better weighting math.

## The Goal

Evolve the current meta-reasoning path into a proper supervisor agent that:
1. Reviews all baseline predictions (not just high-disagreement ones)
2. Identifies the *source* of disagreement (knowledge gap, ambiguity, stale info)
3. Does targeted searches to resolve specific disagreements
4. Produces a final probability with transparent reasoning

## Key Design Decisions

### When does the supervisor engage?

Running a full supervisor pass on all 1000 markets is expensive. The value is in resolving genuine ambiguity, not rubber-stamping consensus.

Proposed triggers (any one is sufficient):
- **Baseline disagreement**: spread > 0.15 (lowered from current 0.20)
- **Market-vs-model divergence**: |market_price - ensemble_mean| > 0.15
- **High uncertainty**: if multi-run ensembling (Plan 03) shows high within-model variance
- **Novel/ambiguous question**: model reasoning includes hedging language ("uncertain", "depends on")

For consensus markets (all sources agree within 0.10), skip the supervisor entirely. Extremize and submit.

### Supervisor architecture

The supervisor is NOT just another LLM call that sees all the numbers and picks one. It's a structured reasoning agent:

```
Input: market question, baselines (with individual runs if available),
       market price, any search context from Plan 04

Step 1: DIAGNOSE
  "Model A says 0.70, Model B says 0.40, market says 0.55.
   Model A and the market roughly agree. Model B is the outlier.
   Possible reasons: Model B has stale training data about [X],
   or Model B is correctly pricing in risk that others miss."

Step 2: INVESTIGATE
  Generate 1-2 targeted search queries to resolve the specific
  disagreement identified in Step 1.

Step 3: ADJUDICATE
  "Based on [search results], Model B appears to be wrong about [X]
   because [evidence]. The correct probability is closer to Model A's
   estimate, adjusted for [Y]."

Output: probability, reasoning, which_model_was_closest, evidence_used
```

### Model choice for supervisor

The supervisor needs to be the best reasoning model available — it's doing the hardest cognitive work. Use Claude Opus or GPT-5.2 (whichever performs best on initial baselines). This is the one place where model quality matters most.

But: the supervisor only fires on ~25-40% of markets (disagreement/divergence cases), so cost is bounded.

### Relationship to existing meta-reasoning

The current `_meta_reason()` method in `agent.py` is a proto-supervisor. It already:
- Takes baselines + market price + web context
- Asks Claude to analyze disagreement
- Returns a reconciled probability

The upgrade path is:
1. Give it the structured diagnose/investigate/adjudicate prompt
2. Let it generate its own search queries (instead of receiving pre-searched context)
3. Fire it on more cases (lower threshold + market divergence trigger)
4. Return richer output (which model was closest, evidence trail)

This is an evolution of existing code, not a rewrite.

### Interaction with agentic search (Plan 04)

Two options:
- **Search-then-supervise**: Agentic search runs first (Plan 04), supervisor gets the search results as context
- **Supervisor-drives-search**: Supervisor decides what to search for based on the disagreement it identifies

AIA uses the second approach (supervisor does its own searches). This is better because the searches are targeted at resolving specific disagreements rather than general information gathering. But it means the supervisor needs search tool access.

**Recommendation**: Both. Plan 04's agentic search provides baseline context for all markets. The supervisor does *additional* targeted searches only when it identifies a specific information gap. This avoids redundant searching on consensus markets while giving the supervisor the tools it needs on hard cases.

## What Changes

### 1. Upgrade the meta-reasoning prompt

The current `META_REASONING_PROMPT` asks the model to "analyze why the models disagree" but doesn't structure the reasoning into diagnose/investigate/adjudicate phases. The new prompt should:
- Explicitly ask for a diagnosis of *why* disagreement exists
- Present each model's reasoning (not just its number)
- Ask for a confidence-weighted final estimate
- Request the model to flag if it needs more information (trigger for follow-up search)

### 2. Expand trigger conditions

Lower `disagreement_threshold` from 0.2 to 0.15. Add market-vs-ensemble divergence trigger. These are config parameters in `CustomAgent.__init__`.

### 3. Give supervisor search capability

The supervisor should be able to call `AgenticSearcher.search()` (from Plan 04) with its own generated queries. This means passing the searcher instance into `_meta_reason()` and letting it do 1-2 additional targeted queries.

### 4. Structured output

The supervisor returns not just a probability but:
- `probability`: final estimate
- `reasoning`: structured explanation
- `closest_baseline`: which model was most accurate (useful for learning model weights)
- `evidence`: key facts that informed the decision
- `search_queries_used`: for debugging and analysis

Over time, tracking `closest_baseline` lets us learn which model to trust more on which types of questions.

## Cost Estimate

- Supervisor fires on ~300-400 markets (30-40% of 1000)
- Each supervisor call: ~500 tokens in, ~300 tokens out = ~$0.02 with Opus
- Additional targeted searches: 1-2 per market = ~$0.01 per market
- Total: ~$9-12 for supervisor + ~$3-4 for searches = ~$12-16

This is within the existing ~$30 agent budget.

## What Not To Do

- Don't build a complex multi-agent debate system. AIA shows simple mean is hard to beat. The supervisor's value is in its *search capability*, not in sophisticated aggregation math.
- Don't run the supervisor on consensus markets. It's a waste of money and can actually *hurt* by second-guessing correct consensus.
- Don't try to learn model weights from within a single run. We don't have enough resolved data yet. Use uniform weights until we do.

## Success Metric

Compare agent with and without supervisor on resolved markets. Expect ~0.005-0.01 Brier improvement, concentrated on high-disagreement markets. Track which baselines the supervisor agrees with most — this informs future model weighting.
