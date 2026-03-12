#!/usr/bin/env python3
"""Explore trade flow signals on current markets.

Fetches trade data for the top N markets by volume, computes signals,
and reports which markets have actionable trade flow patterns.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.trade_flow import fetch_trades_for_market
from src.agent.trade_signals import compute_all_signals, signal_adjustment
from src.utils.logger import log


async def main():
    # Load current markets
    markets_path = Path("data/processed/markets.json")
    if not markets_path.exists():
        print("No markets.json found. Run fetch_markets.py first.")
        return

    with open(markets_path) as f:
        markets = json.load(f)

    # Take top 20 by volume for exploration
    markets.sort(key=lambda m: m.get("volume", 0), reverse=True)
    sample = markets[:20]

    print(f"Exploring trade signals for top {len(sample)} markets by volume\n")
    print(f"{'Platform':<12} {'Vol':>10} {'Prob':>6} {'Trades':>7} {'Whale':>7} {'Mom24h':>7} {'Surge':>6} {'Question'}")
    print("-" * 120)

    results = []

    for market in sample:
        platform = market.get("platform", "?")
        vol = market.get("volume", 0)
        prob = market.get("market_probability", 0.5)
        question = market.get("question", "?")[:50]

        trades = await fetch_trades_for_market(market)
        signals = compute_all_signals(trades)

        adjusted, explanation = signal_adjustment(prob, signals)

        result = {
            "id": market["id"],
            "platform": platform,
            "question": market.get("question", ""),
            "volume": vol,
            "market_probability": prob,
            "signals": signals,
            "adjusted_probability": adjusted,
            "adjustment_explanation": explanation,
        }
        results.append(result)

        print(
            f"{platform:<12} "
            f"${vol:>9,.0f} "
            f"{prob:>5.1%} "
            f"{signals['n_trades']:>7} "
            f"{signals['whale_flow']:>+6.2f} "
            f"{signals['momentum_24h']:>+6.3f} "
            f"{signals['volume_surge']:>5.1f}x "
            f"{question}"
        )

        if abs(adjusted - prob) > 0.001:
            print(f"  -> Adjusted: {prob:.1%} -> {adjusted:.1%} ({explanation})")

        # Rate limiting
        await asyncio.sleep(0.3)

    # Summary
    has_data = [r for r in results if r["signals"]["has_data"]]
    has_adjustment = [r for r in results if abs(r["adjusted_probability"] - r["market_probability"]) > 0.001]

    print(f"\n--- Summary ---")
    print(f"Markets with trade data: {len(has_data)}/{len(results)}")
    print(f"Markets with signal adjustments: {len(has_adjustment)}")

    if has_data:
        avg_trades = sum(r["signals"]["n_trades"] for r in has_data) / len(has_data)
        print(f"Average trades per market: {avg_trades:.0f}")

    # Save detailed results
    out_path = Path("data/processed/trade_signals_sample.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
