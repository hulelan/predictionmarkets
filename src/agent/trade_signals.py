"""Compute predictive signals from trade-level data."""

import statistics
from datetime import datetime, timedelta, timezone

from src.data.trade_flow import TradeRecord


def compute_whale_flow(
    trades: list[TradeRecord],
    percentile: float = 90,
) -> float:
    """Net directional flow from large trades.

    Returns a value from -1.0 (all whale flow is NO) to +1.0 (all YES).
    0.0 means balanced or no trades.
    """
    if not trades:
        return 0.0

    sizes = [t.size for t in trades]
    if len(sizes) < 5:
        return 0.0

    sorted_sizes = sorted(sizes)
    threshold_idx = int(len(sorted_sizes) * percentile / 100)
    threshold = sorted_sizes[min(threshold_idx, len(sorted_sizes) - 1)]

    whale_yes = sum(t.size for t in trades if t.size >= threshold and t.direction == "yes")
    whale_no = sum(t.size for t in trades if t.size >= threshold and t.direction == "no")
    total_whale = whale_yes + whale_no

    if total_whale == 0:
        return 0.0

    return (whale_yes - whale_no) / total_whale


def compute_concentration(trades: list[TradeRecord]) -> float:
    """Herfindahl-Hirschman Index of trade sizes.

    Returns 0.0 (perfectly dispersed) to 1.0 (single trade dominates).
    """
    if len(trades) < 2:
        return 0.0

    total = sum(t.size for t in trades)
    if total == 0:
        return 0.0

    shares = [t.size / total for t in trades]
    hhi = sum(s * s for s in shares)
    return hhi


def compute_momentum(
    trades: list[TradeRecord],
    window_hours: float = 24,
) -> float:
    """Price change over the last N hours.

    Returns the difference: recent_avg_price - older_avg_price.
    Positive means price moving toward YES.
    """
    if len(trades) < 2:
        return 0.0

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)

    recent = [t for t in trades if t.timestamp >= cutoff]
    older = [t for t in trades if t.timestamp < cutoff]

    if not recent or not older:
        # Fall back: split by median timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        mid = len(sorted_trades) // 2
        older = sorted_trades[:mid]
        recent = sorted_trades[mid:]

    if not recent or not older:
        return 0.0

    recent_avg = statistics.mean(t.price for t in recent)
    older_avg = statistics.mean(t.price for t in older)

    return recent_avg - older_avg


def compute_volume_surge(
    trades: list[TradeRecord],
    window_hours: float = 6,
) -> float:
    """Ratio of recent volume to average hourly volume.

    Returns > 1.0 if volume is surging, < 1.0 if below average.
    """
    if len(trades) < 2:
        return 1.0

    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    first_ts = sorted_trades[0].timestamp
    last_ts = sorted_trades[-1].timestamp
    total_hours = max((last_ts - first_ts).total_seconds() / 3600, 1.0)

    total_volume = sum(t.size for t in trades)
    avg_hourly = total_volume / total_hours

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    recent_volume = sum(t.size for t in trades if t.timestamp >= cutoff)
    recent_hourly = recent_volume / window_hours

    if avg_hourly == 0:
        return 1.0

    return recent_hourly / avg_hourly


def compute_all_signals(trades: list[TradeRecord]) -> dict:
    """Compute all trade flow signals for a market.

    Returns dict with signal values and metadata.
    """
    if not trades:
        return {
            "whale_flow": 0.0,
            "concentration": 0.0,
            "momentum_24h": 0.0,
            "momentum_6h": 0.0,
            "volume_surge": 1.0,
            "n_trades": 0,
            "has_data": False,
        }

    return {
        "whale_flow": compute_whale_flow(trades),
        "concentration": compute_concentration(trades),
        "momentum_24h": compute_momentum(trades, window_hours=24),
        "momentum_6h": compute_momentum(trades, window_hours=6),
        "volume_surge": compute_volume_surge(trades, window_hours=6),
        "n_trades": len(trades),
        "has_data": True,
    }


def signal_adjustment(
    base_probability: float,
    signals: dict,
    max_adjustment: float = 0.05,
) -> tuple[float, str]:
    """Apply trade signal adjustments to a base probability.

    Conservative approach: only adjust when signals are strong and agree.
    Returns (adjusted_probability, explanation).

    Max adjustment is ±0.05 (5 percentage points).
    """
    if not signals.get("has_data") or signals["n_trades"] < 10:
        return base_probability, "insufficient trade data"

    whale = signals["whale_flow"]
    momentum = signals["momentum_24h"]

    # Only adjust if whale flow and momentum agree in direction
    # Both positive = evidence for YES; both negative = evidence for NO
    if whale * momentum <= 0:
        return base_probability, "mixed signals (whale/momentum disagree)"

    # Strength: geometric mean of |whale_flow| and |momentum * 5|
    # (momentum is typically small, scale it up)
    whale_strength = abs(whale)
    momentum_strength = min(abs(momentum) * 5, 1.0)

    if whale_strength < 0.3:
        return base_probability, f"weak whale signal ({whale:.2f})"

    # Agreement strength: both must be at least moderate
    combined_strength = (whale_strength * momentum_strength) ** 0.5

    # Direction: positive = more YES, negative = more NO
    direction = 1.0 if whale > 0 else -1.0

    # Scale adjustment by combined strength, cap at max_adjustment
    adjustment = direction * combined_strength * max_adjustment
    adjustment = max(-max_adjustment, min(max_adjustment, adjustment))

    adjusted = max(0.01, min(0.99, base_probability + adjustment))

    explanation = (
        f"trade signal adjustment: whale={whale:+.2f}, "
        f"momentum={momentum:+.3f}, adj={adjustment:+.3f}"
    )
    return adjusted, explanation
