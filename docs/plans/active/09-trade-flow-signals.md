# Plan 09: Trade Flow Signals & Predictive Analysis

**Status**: Active
**Created**: 2026-03-08

---

## Objective

Pull trade-level data from Kalshi, Manifold, and Polymarket to explore whether trade flow signals (whale bets, bet concentration, price momentum) are predictive of market outcomes. Use these signals to improve our custom agent's probability estimates.

---

## Signal Hypotheses

### 1. Whale Trade Direction
**Theory**: Large trades from sophisticated bettors contain private information.
- **Kalshi**: Trade history API gives individual trades with `count` (size in contracts), `taker_side` (yes/no), and `created_time`
- **Manifold**: Bets API gives `amount`, `outcome` (YES/NO), and `userId`
- **Metric**: Net whale flow = sum of large trades (> 90th percentile) weighted by direction
- **Signal**: If net whale flow is strongly YES → probability should be higher than market price

### 2. Bet Concentration (Herfindahl-style)
**Theory**: Markets where a few large bettors dominate are less efficient; concentrated positions may indicate informed trading.
- **Metric**: HHI of bet sizes (sum of squared shares of total volume)
- **Signal**: High HHI + whale direction agrees with our forecast → increase confidence

### 3. Recent Price Momentum
**Theory**: Markets moving strongly in one direction may have information not yet fully priced in.
- **Metric**: Price change over last 24h / 6h / 1h
- **Signal**: Strong momentum in our forecast direction → slight extremization boost

### 4. Volume Surge
**Theory**: Sudden volume spikes often precede information events.
- **Metric**: Recent volume vs. average daily volume
- **Signal**: High volume surge + price movement → more weight on current price

---

## Data Sources

### Kalshi Trade History
```
GET https://api.elections.kalshi.com/trade-api/v2/markets/trades?ticker={ticker}&limit=1000
```
Returns: `[{count, taker_side, yes_price, no_price, created_time}, ...]`

### Manifold Bets
```
GET https://api.manifold.markets/v0/bets?contractId={id}&limit=1000
```
Returns: `[{amount, outcome, probBefore, probAfter, userId, createdTime}, ...]`

### Polymarket CLOB
```
GET https://clob.polymarket.com/trades?asset_id={token_id}
```
Returns trade history with price and size. Requires `clobTokenIds` from market data.

---

## Implementation Plan

### Step 1: Trade Data Fetchers (`src/data/trade_flow.py`)
- `fetch_kalshi_trades(ticker) -> list[dict]`
- `fetch_manifold_bets(contract_id) -> list[dict]`
- `fetch_polymarket_trades(token_id) -> list[dict]`
- Unified `TradeRecord` dataclass: `{timestamp, size, direction, price, platform}`

### Step 2: Signal Computation (`src/agent/trade_signals.py`)
- `compute_whale_flow(trades, percentile=90) -> float` (-1 to +1, net directional)
- `compute_concentration(trades) -> float` (0 to 1, HHI)
- `compute_momentum(trades, window_hours=24) -> float` (price change)
- `compute_volume_surge(trades, window_hours=6) -> float` (ratio vs avg)
- `compute_all_signals(trades) -> dict` (all signals for one market)

### Step 3: Integration with Agent
- After ensemble but before extremization in `run_agent.py`:
  - Fetch trade data for each market
  - Compute signals
  - Use whale flow to nudge probability (small adjustment, capped at ±0.05)
  - Use momentum to modulate extremization factor

### Step 4: Backtesting (Optional, time permitting)
- Pull recently settled markets (last 2 weeks) with their trade histories
- Compute signals at T-1 day (day before resolution)
- Check correlation between signals and outcomes
- Validate that whale flow is actually predictive before using it live

---

## Risk & Constraints

- Trade history APIs may be rate-limited or return partial data
- Polymarket CLOB API previously returned 403 — may need to skip
- Time pressure: markets resolve tomorrow, so focus on Kalshi + Manifold trades first
- Signal adjustments should be small (±0.05 max) to avoid hurting calibration
- This is exploratory — if signals aren't clearly predictive, don't use them

---

## Cost

- Zero LLM cost — pure API data fetching and math
- Only API rate limiting is a constraint
