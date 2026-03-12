"""Tool-augmented forecasting agent.

The reasoning model (Claude/GPT via OpenRouter) gets a web_search tool
backed by Perplexity Sonar. It decides what to search, when, and how
many times — then produces a final probability.

Replaces the old AgenticSearcher + meta-reasoning two-step pipeline
with a single multi-turn tool-use conversation. One OpenRouter key
covers both the reasoning model and Perplexity.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from src.data.normalizer import NormalizedMarket
from src.data.trade_flow import TradeRecord, fetch_trades_for_market
from src.models.base import clamp_probability, extract_json
from src.models.prompts import TOOL_AGENT_SYSTEM_PROMPT
from src.utils.logger import log

# Tool definitions for the reasoning model
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this to look up "
            "recent news, injury reports, event results, odds, statistics, "
            "or any factual question that would help you forecast the market. "
            "You can call this multiple times with different queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include dates when relevant.",
                },
            },
            "required": ["query"],
        },
    },
}

PRICE_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_price_history",
        "description": (
            "Get the price/probability history chart for this market. "
            "Returns an ASCII chart showing how the market price moved over time, "
            "with timestamps and annotations for big moves. Use this to understand "
            "WHAT the market has already reacted to — if a price jump coincides "
            "with a news event you found, that event is already priced in. "
            "Call this BEFORE or AFTER web_search to compare news timing with price moves."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

AGENT_TOOLS = [WEB_SEARCH_TOOL, PRICE_HISTORY_TOOL]


def render_ascii_chart(trades: list[TradeRecord], width: int = 60, height: int = 15) -> str:
    """Render trade data as an ASCII price chart with timeline.

    Returns something like:
    100%|
     90%|                                          ████████████
     80%|                              ████████████
     70%|                  ████████████
     60%|      ████████████
     50%|██████
     40%|
        +----+----+----+----+----+----+----+----+----+----+
        Mar1      Mar3      Mar5      Mar7      Mar9

    Plus a summary of significant moves with timestamps.
    """
    if not trades:
        return "No trade data available for this market."

    trades = sorted(trades, key=lambda t: t.timestamp)

    # Sample if too many
    if len(trades) > width:
        step = len(trades) / width
        sampled = [trades[int(i * step)] for i in range(width)]
        sampled[0] = trades[0]
        sampled[-1] = trades[-1]
    else:
        sampled = trades

    prices = [t.price for t in sampled]
    min_p = max(0, min(prices) - 0.05)
    max_p = min(1, max(prices) + 0.05)
    p_range = max_p - min_p if max_p > min_p else 0.1

    # Build the chart grid
    lines = []
    for row in range(height, -1, -1):
        p_level = min_p + (row / height) * p_range
        label = f"{p_level:>4.0%}|"
        bar = ""
        for p in prices:
            normalized = (p - min_p) / p_range
            fill_level = row / height
            if normalized >= fill_level:
                bar += "█"
            else:
                bar += " "
        lines.append(label + bar)

    # X-axis
    axis = "    +" + "-" * len(prices)
    lines.append(axis)

    # Time labels
    first_ts = sampled[0].timestamp
    last_ts = sampled[-1].timestamp
    label_line = f"    {first_ts.strftime('%b %d %H:%M')}"
    padding = len(prices) - len(first_ts.strftime("%b %d %H:%M")) - len(last_ts.strftime("%b %d %H:%M"))
    if padding > 0:
        label_line += " " * padding + last_ts.strftime("%b %d %H:%M")
    lines.append(label_line)

    chart = "\n".join(lines)

    # Add significant moves summary
    moves = []
    prev_price = sampled[0].price
    for t in sampled[1:]:
        delta = t.price - prev_price
        if abs(delta) >= 0.05:
            direction = "↑" if delta > 0 else "↓"
            moves.append(
                f"  {direction} {t.timestamp.strftime('%b %d %H:%M UTC')}: "
                f"{prev_price:.0%} → {t.price:.0%} ({delta:+.0%})"
            )
        prev_price = t.price

    summary_parts = [
        f"Total trades: {len(trades)}",
        f"Time span: {first_ts.strftime('%b %d %H:%M')} → {last_ts.strftime('%b %d %H:%M')} UTC",
        f"Price range: {min(prices):.0%} → {max(prices):.0%}",
        f"Current: {prices[-1]:.0%}",
    ]

    result = "PRICE CHART:\n" + chart + "\n\n"
    result += "SUMMARY: " + " | ".join(summary_parts) + "\n"
    if moves:
        result += "\nSIGNIFICANT MOVES (≥5%):\n" + "\n".join(moves) + "\n"
    else:
        result += "\nNo significant moves (≥5%) in this period.\n"

    return result


class ToolAgent:
    """Forecasting agent that uses tool-calling to search the web via Perplexity.

    The reasoning model decides what to search for. Perplexity Sonar
    executes the searches with built-in web access. Everything goes
    through OpenRouter with a single API key.

    Supports market-type-aware routing: sports markets get a shorter prompt
    and fewer search calls (market price is almost always right), while
    non-sports markets get the full superforecaster methodology.
    """

    def __init__(
        self,
        reasoning_model: str = "anthropic/claude-opus-4-6",
        search_model: str = "perplexity/sonar",
        temperature: float = 0.3,
        max_search_calls: int = 5,
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for ToolAgent")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.reasoning_model = reasoning_model
        self.search_model = search_model
        self.temperature = temperature
        self.max_search_calls = max_search_calls

    async def predict(
        self,
        market: NormalizedMarket,
        baselines: dict[str, float] | None = None,
    ) -> dict:
        """Generate a prediction using tool-augmented reasoning.

        The reasoning model gets the market context and can call web_search
        and get_price_history as many times as it wants (up to max_search_calls).
        """
        user_prompt = self._build_prompt(market, baselines or {})

        messages = [
            {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        search_calls_made = 0
        price_history_fetched = False

        # Tool-use loop: let the model search and view charts until ready
        max_tool_rounds = self.max_search_calls + 2  # extra room for price_history
        for _ in range(max_tool_rounds):
            response = await self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=16384,
            )

            choice = response.choices[0]

            # If the model is done (no tool calls), extract the answer
            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                return self._parse_response(choice.message.content or "", market, search_calls_made)

            # Process tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                if tool_call.function.name == "get_price_history" and not price_history_fetched:
                    log.debug(f"ToolAgent price_history [{market.id}]")
                    chart = await self._get_price_chart(market)
                    price_history_fetched = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": chart,
                    })
                elif tool_call.function.name == "get_price_history" and price_history_fetched:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Price history already retrieved above.",
                    })
                elif tool_call.function.name == "web_search" and search_calls_made < self.max_search_calls:
                    query = json.loads(tool_call.function.arguments).get("query", "")
                    log.debug(f"ToolAgent search [{market.id}]: {query[:80]}")

                    search_result = await self._execute_search(query)
                    search_calls_made += 1

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_result,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Search limit reached.",
                    })

        # Force a final answer with no tools available
        log.debug(f"Forcing final answer for {market.id} (searches={search_calls_made})")
        response = await self.client.chat.completions.create(
            model=self.reasoning_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=16384,
            # No tools param — model MUST produce text
        )
        return self._parse_response(
            response.choices[0].message.content or "", market, search_calls_made
        )

    def _parse_response(self, text: str, market: NormalizedMarket, search_calls: int) -> dict:
        """Parse the model's final text response into a prediction dict."""
        ts = datetime.utcnow().isoformat() + "Z"
        try:
            result = extract_json(text)
            result["probability"] = clamp_probability(result["probability"])
            result["strategy"] = "tool_agent"
            result["search_calls"] = search_calls
            result["timestamp"] = ts
            return result
        except (ValueError, KeyError) as e:
            log.warning(f"ToolAgent parse failed for {market.id}: {e}")
            return {
                "probability": market.market_probability,
                "confidence": "low",
                "strategy": "tool_agent_fallback",
                "reasoning": f"Failed to parse response: {text[:200]}",
                "search_calls": search_calls,
                "timestamp": ts,
            }

    async def _execute_search(self, query: str) -> str:
        """Call Perplexity Sonar via OpenRouter to search the web."""
        try:
            response = await self.client.chat.completions.create(
                model=self.search_model,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content or "No results found."
        except Exception as e:
            log.warning(f"Perplexity search failed for '{query[:60]}': {e}")
            return f"Search failed: {e}"

    async def _get_price_chart(self, market: NormalizedMarket) -> str:
        """Fetch trade history and render as ASCII chart + significant moves."""
        try:
            market_dict = {
                "platform": market.platform.value,
                "id": market.id,
                "raw_data": market.raw_data,
            }
            trades = await fetch_trades_for_market(market_dict)
        except Exception as e:
            log.debug(f"Trade fetch failed for {market.id}: {e}")
            return "Price history unavailable for this market."

        if not trades:
            return "No trade data available for this market (platform may not expose trade history)."

        return render_ascii_chart(trades)

    def _build_prompt(
        self,
        market: NormalizedMarket,
        baselines: dict[str, float],
    ) -> str:
        """Build the user prompt with market context and baselines."""
        baseline_lines = "\n".join(
            f"- {model}: {prob:.1%}" for model, prob in baselines.items()
        )

        description = market.description[:500] if market.description else "Not provided"
        resolution_date = market.resolution_date.strftime("%Y-%m-%d")

        # Extract snapshot context from raw platform data
        price_context = self._extract_price_context(market)

        parts = [
            f"MARKET QUESTION: {market.question}",
            f"RESOLUTION CRITERIA: {description}",
            f"RESOLUTION DATE: {resolution_date}",
            f"TODAY: {date.today().isoformat()}",
            f"PLATFORM: {market.platform.value}",
            "",
            f"CURRENT MARKET PRICE: {market.market_probability:.1%}",
        ]

        if price_context:
            parts.append(f"\nMARKET SNAPSHOT:\n{price_context}")

        if baseline_lines:
            parts.append(f"\nBASELINE MODEL PREDICTIONS:\n{baseline_lines}")

        parts.append(
            "\nYou have two tools: web_search and get_price_history. "
            "Use get_price_history to see how the market price moved over time — "
            "this reveals what the market has ALREADY reacted to. "
            "Use web_search for current news. Compare news timestamps to price moves "
            "to determine what's already priced in. "
            "Then provide your final probability as JSON."
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_price_context(market: NormalizedMarket) -> str:
        """Extract timestamped price movement from raw platform data."""
        raw = market.raw_data
        lines = []

        if market.platform.value == "polymarket":
            # Polymarket provides price change windows
            d1 = raw.get("oneDayPriceChange")
            w1 = raw.get("oneWeekPriceChange")
            m1 = raw.get("oneMonthPriceChange")
            if d1 is not None:
                lines.append(f"- 24h price change: {d1:+.1%}")
            if w1 is not None:
                lines.append(f"- 7-day price change: {w1:+.1%}")
            if m1 is not None:
                lines.append(f"- 30-day price change: {m1:+.1%}")

            v24 = raw.get("volume24hr")
            v1w = raw.get("volume1wk")
            if v24:
                lines.append(f"- 24h volume: ${v24:,.0f}")
            if v1w:
                lines.append(f"- 7-day volume: ${v1w:,.0f}")

            last_trade = raw.get("lastTradePrice")
            if last_trade is not None:
                lines.append(f"- Last trade price: {last_trade:.1%}")

            created = raw.get("createdAt", "")
            updated = raw.get("updatedAt", "")
            if created:
                lines.append(f"- Market created: {created[:19]}")
            if updated:
                lines.append(f"- Last updated: {updated[:19]}")

        elif market.platform.value == "kalshi":
            # Kalshi provides bid/ask + previous
            yes_bid = raw.get("yes_bid")
            yes_ask = raw.get("yes_ask")
            prev_bid = raw.get("previous_yes_bid")
            prev_ask = raw.get("previous_yes_ask")

            if yes_bid is not None and yes_ask is not None:
                lines.append(f"- Current bid/ask: {yes_bid}¢ / {yes_ask}¢")
            if prev_bid is not None and prev_ask is not None:
                if prev_bid > 0 or prev_ask > 0:
                    lines.append(f"- Previous bid/ask: {prev_bid}¢ / {prev_ask}¢")
                    if yes_bid and prev_bid:
                        mid_now = (yes_bid + yes_ask) / 2
                        mid_prev = (prev_bid + prev_ask) / 2
                        if mid_prev > 0:
                            lines.append(f"- Mid price moved: {mid_prev:.0f}¢ → {mid_now:.0f}¢")

            v24 = raw.get("volume_24h")
            if v24:
                lines.append(f"- 24h volume: {v24:,} contracts")

            created = raw.get("created_time", "")
            updated = raw.get("updated_time", "")
            if created:
                lines.append(f"- Market created: {created[:19]}")
            if updated:
                lines.append(f"- Last updated: {updated[:19]}")

        elif market.platform.value == "manifold":
            # Manifold provides timestamps in milliseconds
            created = raw.get("createdTime")
            last_bet = raw.get("lastBetTime")
            v24 = raw.get("volume24Hours")

            if created:
                from datetime import datetime as _dt
                lines.append(f"- Market created: {_dt.fromtimestamp(created/1000).isoformat()[:19]}")
            if last_bet:
                from datetime import datetime as _dt
                lines.append(f"- Last bet: {_dt.fromtimestamp(last_bet/1000).isoformat()[:19]}")
            if v24:
                lines.append(f"- 24h volume: {v24:,.0f} mana")

        return "\n".join(lines) if lines else ""
