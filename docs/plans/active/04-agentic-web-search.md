# Plan: Agentic Web Search

**Priority**: Highest — largest absolute impact per AIA ablations (0.085 vs 0.116 Brier)
**Estimated Brier improvement**: ~0.02-0.04

---

## The Problem

Our current web search is a single non-agentic Tavily call: strip the question down to keywords, search once, return 3 results truncated to 200 chars. This fires only on disagreement cases (spread > 0.2). The AIA Forecaster ablations show this is exactly the kind of weak search pipeline that prior work found "didn't help" — and then concluded search was useless.

The truth is the opposite: *agentic* search (iterative, adaptive, model-directed) is the single highest-impact component. The difference is 3.6x in live market experiments. The key insight: the model should decide what to search for, evaluate what it finds, and search again if needed.

## The Goal

Replace the current "fire-and-forget" search with an iterative search loop where the LLM generates queries, evaluates results, and decides whether to dig deeper. Apply this to all markets where the agent doesn't have high confidence, not just disagreement cases.

## Key Design Decisions

### When to search

Current: only when baseline spread > 0.2 (~maybe 10-20% of markets).
Proposed tiers:
- **Skip search**: Consensus markets (all baselines + market agree within 0.05, extreme probability >0.9 or <0.1). These are "done" — searching won't help.
- **Light search**: Default path. One round of search, 3-5 results. ~60-70% of markets.
- **Deep search**: Disagreement cases OR markets where light search reveals surprising info. Multiple rounds. ~15-25% of markets.

### Search loop architecture

The agentic search follows a ReAct-style loop:

```
1. LLM generates 2-3 search queries based on the market question
2. Execute searches in parallel
3. LLM evaluates results: "Do I have enough information to forecast?"
   - If yes: produce forecast with citations
   - If no: generate refined follow-up queries, go to step 2
4. Cap at 3 rounds maximum (cost control)
```

This is fundamentally different from our current approach where we do regex-based query construction (`_build_search_query`). The model should generate queries because it understands what information would actually update its beliefs.

### Search provider

Current: Tavily (free tier, 1000 searches/month).
Options:
- **Tavily Pro**: $50/month, 5000 searches. Probably sufficient for 1000 markets x 2-3 queries each.
- **Perplexity API (via OpenRouter)**: Returns synthesized answers with citations. More expensive per query but might need fewer rounds.
- **Multiple providers**: Use Tavily for initial queries, fall back to Perplexity for deep search rounds. The Metaculus bot template supports AskNews, OpenRouter, Perplexity — could swap between them.
- **Direct web fetch**: For specific URLs found in search results, fetch and extract the full article. The ksadov bot uses a separate Gemini Flash call to clean HTML — smart cost optimization.

Start with Tavily (already integrated) but make the search interface pluggable so we can swap providers.

### Query generation model

Using Claude Opus for query generation is wasteful. Use a cheap, fast model (Gemini Flash or GPT-4o-mini via OpenRouter) for:
- Generating search queries from market questions
- Evaluating search result relevance (keep/discard)
- Summarizing articles

Reserve the expensive model for the final reasoning step only.

### Result processing

Current: truncate to 200 chars per result. This throws away most of the signal.
Better:
- Fetch full article content for top results (WebFetch or Tavily extract mode)
- Use cheap model to summarize each article into 2-3 key facts relevant to the market question
- Concatenate summaries as context for the reasoning model
- Track source dates — recent news is exponentially more valuable for short-horizon markets

## What Changes

### 1. New `AgenticSearcher` class (replaces or extends `WebSearcher`)

Core interface:
```python
async def search(self, question: str, description: str,
                 resolution_date: datetime, depth: str = "light") -> SearchContext
```

Returns a `SearchContext` dataclass with:
- `summaries: list[str]` — key facts found
- `sources: list[dict]` — URLs, titles, dates
- `queries_used: list[str]` — for debugging/analysis
- `rounds: int` — how many search rounds were needed
- `confidence: str` — did the search find decisive information?

### 2. Query generation via LLM

Instead of regex stripping, ask a cheap model: "Given this prediction market question, generate 3 diverse web search queries that would help you forecast the outcome. Focus on recent news, scheduled events, and relevant statistics."

For sports markets (majority of our dataset): queries should target injury reports, recent form, head-to-head records, betting odds from other sources.

### 3. Integration with agent pipeline

The agent currently has 3 strategies (consensus/disagreement/default). With agentic search, the flow becomes:

1. Check if consensus → skip search, extremize, done
2. For everything else → run agentic search (light or deep based on spread)
3. Feed search context + baselines + market price into reasoning model
4. Extremize and submit

### 4. Cost controls

- Cap at 3 search rounds per market (hard limit)
- Use cheap model for query gen + article summarization (~$0.001/market)
- Use search only on non-consensus markets (~700 of 1000)
- Total search cost estimate: 700 markets x 3 queries x $0.01/search = ~$21 in Tavily + ~$3 in cheap model calls

## Sports Market Considerations

~75% of our markets are sports (NBA, NHL, UFC, soccer, cricket, tennis). For these:
- Search for: injury reports, lineup announcements, recent form/results, weather (outdoor sports), venue records
- Time sensitivity is extreme — a key player injury 2 hours before game time changes everything
- Betting odds from other sportsbooks are highly informative (these markets are essentially sports betting)
- Consider adding a specialized sports data source (odds API, ESPN headlines) as an alternative to general web search

## What This Enables

- The agent has *evidence* for its predictions, not just "three models said different things"
- Can detect when a market's price is stale (news broke after last trade)
- Provides transparent reasoning trails in the submission
- Foundation for the supervisor agent (Plan 06) to do targeted follow-up searches

## Success Metric

A/B test: run agent with and without agentic search on the same markets. Compare Brier scores. AIA saw 0.031 improvement; we should see at least 0.015 given our shorter-horizon, more-informed markets (sports odds are already efficient).
