# Plan: Sports Domain Specialization

**Priority**: High for this specific dataset — ~75% of markets are sports
**Estimated Brier improvement**: 0.01-0.03 (domain-specific edge over generic LLM)

---

## The Problem

Our market sample is ~75% sports: NBA, NHL, UFC, soccer, cricket, tennis. Generic LLM forecasting prompts treat "Will the Islanders beat the Sharks?" the same as "Will the Greens win Baden-Wurttemberg?" But these are fundamentally different forecasting problems:

- **Sports**: Near-term (hours to days), objective resolution, rich structured data available (odds, stats, injuries, form), highly efficient markets (sportsbooks have been doing this for decades)
- **Politics/events**: Longer-term, subjective resolution criteria possible, unstructured information, less efficient markets

An LLM reasoning from first principles about an NBA game is bringing a philosophy degree to a statistics fight. The information that moves sports markets is *structured data* — injury reports, lineup changes, recent results, venue effects — not general reasoning.

## The Goal

Build a sports-specific forecasting pathway that leverages structured data sources and sports-domain knowledge to outperform generic LLM prompts on the ~750 sports markets in our sample.

## Key Design Decisions

### Market classification

Before we can specialize, we need to classify markets by domain. Options:
- **Keyword-based**: Check for team names, league names (NBA, NHL, UFC, etc.), player names. Fast, no API cost.
- **Category field**: Some platforms provide category metadata (Kalshi has event categories, Polymarket has tags). Use when available.
- **LLM classification**: Ask a cheap model to categorize. Expensive for 1000 markets.

Start with keyword-based classification. The Kalshi and Polymarket category fields can supplement. We don't need 100% accuracy — getting 90% of sports markets tagged is enough.

### Sports-specific data sources

The highest-value information for sports markets:

1. **Odds from other sportsbooks**: If Polymarket says Islanders at 56% and DraftKings/FanDuel consensus is 52%, DraftKings is almost certainly more accurate (deeper liquidity, professional oddsmakers). Cross-referencing sportsbook odds is essentially free alpha against prediction market prices.
   - The Odds API (free tier: 500 requests/month) provides lines from multiple sportsbooks
   - Alternatively, scrape odds from ESPN, Action Network, or OddsChecker

2. **Injury reports / lineup news**: A star player being ruled out can swing odds 5-15%. This information is available on ESPN, team Twitter/X accounts, official injury reports (NBA/NHL mandate pre-game injury reports).
   - This is where agentic search (Plan 04) pays off most — "Is [star player] playing tonight?" is a specific, answerable question

3. **Recent form**: Team/fighter records over last 5-10 games. Available from sports stats APIs.

4. **Head-to-head records**: Some matchups have strong historical patterns.

5. **Venue/travel effects**: Home court advantage, back-to-back games, travel distance.

### The sportsbook-odds shortcut

Here's the uncomfortable truth: for sports markets resolving in <48 hours, the most effective strategy might be to simply retrieve sportsbook consensus odds and use those as our prediction, lightly adjusted. Professional sportsbooks process vastly more information than any LLM can in a single prompt.

This isn't "cheating" — it's recognizing that the sportsbook odds ARE the most informed prior available. Our edge, if any, comes from:
- **Timing**: If our data fetch is more recent than the prediction market's last trade, we may have newer odds
- **Market inefficiency**: Prediction markets (especially Polymarket) may lag sportsbook odds by minutes to hours
- **Line movement**: If odds are moving in one direction, the prediction market may not have caught up

### How to integrate with the agent

Option A: **Sportsbook odds as another baseline**
- Fetch odds, convert to implied probability (remove vig)
- Treat as a 4th baseline alongside GPT/Claude/Gemini
- The ensemble/supervisor handles it like any other input

Option B: **Sportsbook odds as the primary signal**
- For sports markets, start with sportsbook consensus as the prior
- Use LLM baselines only as a sanity check / divergence signal
- If LLM disagrees with sportsbook by >10%, investigate (maybe LLM knows something about a breaking story)

Option A is cleaner and works with existing architecture. Option B is probably more accurate but requires restructuring the agent flow.

**Recommendation**: Start with Option A. If sportsbook odds consistently dominate LLM baselines (likely), promote to Option B in a later iteration.

### Sports-specific prompting

For markets classified as sports, use a domain-specific prompt that:
- Asks the model to consider specific sports factors (injuries, form, matchup, venue)
- Provides structured context (odds, recent results) instead of asking the model to reason from general knowledge
- Doesn't waste tokens on "consider base rates for similar events" (irrelevant for a specific NBA game)

## What Changes

### 1. Market classifier

Add a `classify_domain(market: NormalizedMarket) -> str` function that returns "sports", "politics", "crypto", "entertainment", "other" based on keywords + platform category metadata. Attach domain to the `NormalizedMarket` (or compute on the fly).

### 2. Odds data source (optional but high-value)

A new module `src/data/odds_api.py` or similar that:
- Takes a market question + resolution date
- Matches it to an event on a sportsbook odds API
- Returns consensus implied probability (after vig removal)
- Caches results (odds don't change faster than our fetch interval)

The matching problem (linking "Will FC Barcelona win on 2026-03-08?" to the specific event in the odds API) is the hardest part. May need fuzzy matching or LLM-assisted matching.

### 3. Sports-specific search queries

When Plan 04's agentic search fires on a sports market, the query generator should know to search for:
- `"[Team A] vs [Team B] injury report March 2026"`
- `"[Player name] status [date]"`
- `"[League] odds [Team A] [Team B]"`

Rather than generic news queries.

### 4. Sport-aware agent routing

In `CustomAgent.predict()`, check domain. For sports markets:
- If sportsbook odds available: blend with high weight toward sportsbook
- Use sports-specific prompt for any LLM reasoning steps
- Adjust extremization (sports markets may need less extremization since odds are already well-calibrated)

## Cost Estimate

- The Odds API free tier (500 req/month) may suffice for ~750 sports markets if we batch by event
- Additional search cost for injury reports: ~$5 via Tavily
- No additional LLM cost — just different prompting

## Risk: Market Matching

The hardest part of the sportsbook odds approach is matching our prediction market questions to the correct sportsbook events. "Will FC Barcelona win on 2026-03-08?" needs to be matched to a specific La Liga fixture. This could fail for:
- Ambiguous questions ("Will Team X win?" — win what?)
- Non-standard events (UFC specific fights, cricket formats)
- Markets that don't have sportsbook equivalents (some Manifold markets)

Mitigation: match conservatively, only use sportsbook odds when confident in the match. Fall back to generic agent for unmatched markets.

## Success Metric

Compare Brier scores on sports markets: generic agent vs sports-specialized agent. The benchmark is sportsbook implied probability — if we can't beat sportsbook odds on sports markets, we should just use them directly (which is still a win, since it means our agent is well-calibrated).
