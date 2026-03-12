"""Sports-specific forecasting utilities.

Provides a domain-specific prompt, sport-aware search query generation,
and odds/vig-removal helpers for sports prediction markets.
"""

from __future__ import annotations

import re

from src.utils.logger import log


SPORTS_FORECASTING_PROMPT = """You are a sports forecasting expert analyzing a specific matchup.

MATCHUP: {question}
RESOLUTION: {description}
EVENT DATE: {resolution_date}

CURRENT ODDS:
- Prediction market: {market_probability:.1%}
{sportsbook_line}

BASELINE MODEL PREDICTIONS:
{baselines}

{search_section}

Consider these sport-specific factors:
- Recent form and head-to-head record
- Injuries, suspensions, or lineup changes
- Home/away advantage and venue factors
- Rest days and schedule fatigue

Provide your probability estimate. If sportsbook odds are available, weight them heavily \
-- professional oddsmakers have deeper information than language models on sports events.

Respond as JSON: {{"probability": <float 0.01-0.99>, "confidence": "<low|medium|high>", "reasoning": "<brief explanation>"}}"""


def format_sports_prompt(
    question: str,
    description: str,
    resolution_date: str,
    market_probability: float,
    baselines: dict[str, float],
    sportsbook_odds: float | None = None,
    search_context: str | None = None,
) -> str:
    """Build a fully formatted sports forecasting prompt.

    Args:
        question: The market question.
        description: Resolution criteria / description.
        resolution_date: Human-readable resolution date string.
        market_probability: Current prediction market probability.
        baselines: Dict of model_name -> probability.
        sportsbook_odds: Fair implied probability from sportsbooks, or None.
        search_context: Recent news/injury context from web search, or None.

    Returns:
        Formatted prompt string ready for an LLM call.
    """
    # Format sportsbook line
    if sportsbook_odds is not None:
        sportsbook_line = f"- Sportsbook consensus: {sportsbook_odds:.1%}"
    else:
        sportsbook_line = ""

    # Format baselines
    baseline_lines = []
    for model, prob in baselines.items():
        baseline_lines.append(f"- {model}: {prob:.1%}")
    baselines_str = "\n".join(baseline_lines) if baseline_lines else "No baselines available."

    # Format search section
    if search_context:
        search_section = f"RECENT NEWS/INJURY REPORTS:\n{search_context}"
    else:
        search_section = ""

    return SPORTS_FORECASTING_PROMPT.format(
        question=question,
        description=description[:500] if description else "Not provided",
        resolution_date=resolution_date,
        market_probability=market_probability,
        sportsbook_line=sportsbook_line,
        baselines=baselines_str,
        search_section=search_section,
    )


def generate_sports_search_queries(question: str, sport_type: str) -> list[str]:
    """Generate sport-specific search queries for web search.

    Instead of generic news queries, targets:
    - Injury reports and lineup news
    - Recent form / results
    - Betting odds from sportsbooks

    Args:
        question: The market question (e.g., "Will the Lakers beat the Celtics?").
        sport_type: Sport identifier (e.g., "nba", "nhl", "ufc").

    Returns:
        List of 2-4 targeted search queries.
    """
    teams = _extract_names_from_question(question)
    queries: list[str] = []

    if len(teams) >= 2:
        team_a, team_b = teams[0], teams[1]

        # Matchup / odds query
        queries.append(f"{team_a} vs {team_b} odds betting lines March 2026")

        # Injury / status queries
        if sport_type in ("nba", "nhl", "mlb", "nfl", "soccer"):
            queries.append(f"{team_a} {team_b} injury report lineup March 2026")
            queries.append(f"{team_a} vs {team_b} recent results head to head 2026")
        elif sport_type == "ufc":
            queries.append(f"{team_a} {team_b} fight status injury update 2026")
            queries.append(f"{team_a} vs {team_b} MMA odds prediction 2026")
        elif sport_type == "cricket":
            queries.append(f"{team_a} {team_b} cricket squad injury update 2026")
            queries.append(f"{team_a} vs {team_b} cricket head to head 2026")
        elif sport_type == "tennis":
            queries.append(f"{team_a} {team_b} tennis injury fitness update 2026")
            queries.append(f"{team_a} vs {team_b} tennis head to head record")
        else:
            queries.append(f"{team_a} {team_b} latest news March 2026")

    elif len(teams) == 1:
        # Single team/player identified
        name = teams[0]
        queries.append(f"{name} latest news March 2026")
        if sport_type in ("nba", "nhl", "mlb", "nfl", "soccer"):
            queries.append(f"{name} injury report lineup status March 2026")
        elif sport_type == "ufc":
            queries.append(f"{name} UFC fight status odds 2026")
        elif sport_type == "tennis":
            queries.append(f"{name} tennis results fitness 2026")
        elif sport_type == "cricket":
            queries.append(f"{name} cricket squad selection 2026")

    else:
        # Couldn't extract names -- fall back to question-based search
        clean_q = _clean_question(question)
        queries.append(f"{clean_q} odds March 2026")
        queries.append(f"{clean_q} latest news 2026")

    # Always add a league-specific query if we know the sport
    league_queries = {
        "nba": "NBA scores results today March 2026",
        "nhl": "NHL scores results today March 2026",
        "ufc": "UFC upcoming fights card March 2026",
        "soccer": "football soccer results scores March 2026",
        "cricket": "cricket scores results March 2026",
        "tennis": "ATP WTA tennis results March 2026",
        "mlb": "MLB scores results today March 2026",
        "nfl": "NFL news March 2026",
        "f1": "Formula 1 results standings March 2026",
        "boxing": "boxing upcoming fights March 2026",
        "golf": "PGA golf tournament results March 2026",
    }
    league_q = league_queries.get(sport_type)
    if league_q and len(queries) < 4:
        queries.append(league_q)

    return queries[:4]


def remove_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Convert sportsbook implied probabilities to fair probabilities.

    Sportsbook odds include vig (overround), so implied probabilities
    typically sum to > 1.0. This normalizes them to sum to exactly 1.0.

    Example:
        If implied probs are 0.55 and 0.52 (sum=1.07),
        fair probs are 0.55/1.07 = 0.514 and 0.52/1.07 = 0.486.

    Args:
        odds_a: Implied probability for outcome A (from sportsbook odds).
        odds_b: Implied probability for outcome B (from sportsbook odds).

    Returns:
        Tuple of (fair_prob_a, fair_prob_b) that sum to 1.0.
    """
    total = odds_a + odds_b
    if total <= 0:
        log.warning(f"Invalid odds sum: {total} (odds_a={odds_a}, odds_b={odds_b})")
        return 0.5, 0.5

    return odds_a / total, odds_b / total


def american_to_implied(american_odds: int) -> float:
    """Convert American odds to implied probability.

    Positive odds (e.g., +150): probability = 100 / (odds + 100)
    Negative odds (e.g., -200): probability = |odds| / (|odds| + 100)

    Args:
        american_odds: American-style odds (e.g., -150, +200).

    Returns:
        Implied probability (0 to 1, includes vig).
    """
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def decimal_to_implied(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal-style odds (e.g., 2.50 means +150 American).

    Returns:
        Implied probability (0 to 1, includes vig).
    """
    if decimal_odds <= 0:
        return 0.5
    return 1.0 / decimal_odds


def _extract_names_from_question(question: str) -> list[str]:
    """Extract team/player names from a market question.

    Uses common patterns like "X vs Y", "X beat Y", "X or Y", "X - Y".
    Returns a list of 0-2 names.
    """
    names: list[str] = []

    # Pattern: "X vs Y", "X vs. Y", "X v Y"
    vs_match = re.search(
        r"(?:Will\s+(?:the\s+)?|Can\s+(?:the\s+)?)?(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)(?:\?|$|\s+(?:on|in|at|for)\b)",
        question,
        re.IGNORECASE,
    )
    if vs_match:
        names = [_clean_name(vs_match.group(1)), _clean_name(vs_match.group(2))]
        return [n for n in names if n]

    # Pattern: "Will X beat/defeat Y"
    beat_match = re.search(
        r"(?:Will|Can|Does)\s+(?:the\s+)?(.+?)\s+(?:beat|defeat|win against|overcome)\s+(?:the\s+)?(.+?)(?:\?|$)",
        question,
        re.IGNORECASE,
    )
    if beat_match:
        names = [_clean_name(beat_match.group(1)), _clean_name(beat_match.group(2))]
        return [n for n in names if n]

    # Pattern: "X - Y" (European style)
    dash_match = re.search(
        r"(?:Will\s+)?(?:the\s+)?(.+?)\s+-\s+(.+?)(?:\?|$|\s+(?:on|in|at)\b)",
        question,
        re.IGNORECASE,
    )
    if dash_match:
        a = _clean_name(dash_match.group(1))
        b = _clean_name(dash_match.group(2))
        if len(a.split()) <= 5 and len(b.split()) <= 5:
            names = [a, b]
            return [n for n in names if n]

    # Pattern: "Will X win" (single team)
    win_match = re.search(
        r"(?:Will|Can|Does)\s+(?:the\s+)?(.+?)\s+win\b",
        question,
        re.IGNORECASE,
    )
    if win_match:
        name = _clean_name(win_match.group(1))
        if name and len(name.split()) <= 5:
            return [name]

    return names


def _clean_name(name: str) -> str:
    """Clean an extracted team/player name."""
    name = name.strip()
    name = re.sub(r"^(?:the|The|THE)\s+", "", name)
    name = name.rstrip("?.!,;:")
    return name.strip()


def _clean_question(question: str) -> str:
    """Strip prediction market phrasing from a question for search."""
    q = question.rstrip("?").strip()
    for prefix in ["Will ", "Will the ", "Is ", "Does ", "Can "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
    return q
