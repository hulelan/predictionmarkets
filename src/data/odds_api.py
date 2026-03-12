"""Sportsbook odds provider using The Odds API.

Fetches consensus odds from multiple sportsbooks and converts to fair
implied probabilities (vig removed). Gracefully returns None when no
API key is configured or when a matching event cannot be found.

API docs: https://the-odds-api.com/
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import httpx

from src.agent.sports import remove_vig
from src.utils.logger import log


# Mapping from our internal sport keys to The Odds API sport keys.
# See: https://the-odds-api.com/liveapi/guides/v4/#get-sports
SPORT_KEY_MAP: dict[str, str] = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    "ufc": "mma_mixed_martial_arts",
    "soccer": "soccer",  # Needs league-specific key; we try multiple below
    "tennis": "tennis",  # Placeholder; API has tournament-specific keys
    "cricket": "cricket",  # Limited API support
    "f1": "motorsport_formula1",
    "boxing": "boxing_boxing",
    "golf": "golf",
}

# For soccer, we try the most popular leagues in order.
SOCCER_LEAGUE_KEYS: list[str] = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_usa_mls",
    "soccer_brazil_campeonato",
    "soccer_mexico_ligamx",
]


class OddsProvider:
    """Fetch sportsbook odds from The Odds API.

    Usage:
        provider = OddsProvider()
        if provider.available:
            odds = await provider.get_event_odds("nba", "Lakers", "Celtics", date(2026, 3, 10))
    """

    BASE_URL = "https://api.the-odds-api.com/v4/sports"

    def __init__(self):
        self.api_key = os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            log.info("ODDS_API_KEY not set -- sportsbook odds disabled")

    @property
    def available(self) -> bool:
        """Whether the provider has a configured API key."""
        return self.api_key is not None

    async def get_event_odds(
        self,
        sport: str,
        team_a: str,
        team_b: str,
        event_date: date,
    ) -> dict | None:
        """Fetch consensus odds for a specific event.

        Tries to match the event by team name substring matching and
        date filtering. Only returns odds when confident in the match.

        Args:
            sport: Internal sport key (e.g., "nba", "nhl", "ufc").
            team_a: First team or competitor name.
            team_b: Second team or competitor name.
            event_date: Expected date of the event.

        Returns:
            Dict with keys "team_a_prob", "team_b_prob", "source" or None
            if the event cannot be matched or odds are unavailable.
        """
        if not self.available:
            return None

        sport_keys = self._resolve_sport_keys(sport)
        if not sport_keys:
            log.debug(f"No API sport key mapping for: {sport}")
            return None

        for sport_key in sport_keys:
            result = await self._fetch_and_match(
                sport_key, team_a, team_b, event_date
            )
            if result is not None:
                return result

        return None

    async def get_implied_probability(
        self,
        sport: str,
        team_a: str,
        team_b: str,
        event_date: date,
    ) -> float | None:
        """Get the fair implied probability for team_a winning (vig removed).

        Convenience wrapper around get_event_odds that returns just team_a's
        fair probability.

        Args:
            sport: Internal sport key.
            team_a: First team/competitor name.
            team_b: Second team/competitor name.
            event_date: Expected date of the event.

        Returns:
            Fair probability for team_a (0 to 1), or None.
        """
        odds = await self.get_event_odds(sport, team_a, team_b, event_date)
        if odds is None:
            return None
        return odds["team_a_prob"]

    def _resolve_sport_keys(self, sport: str) -> list[str]:
        """Map internal sport key to one or more Odds API sport keys."""
        if sport == "soccer":
            return SOCCER_LEAGUE_KEYS
        mapped = SPORT_KEY_MAP.get(sport)
        if mapped:
            return [mapped]
        return []

    async def _fetch_and_match(
        self,
        sport_key: str,
        team_a: str,
        team_b: str,
        event_date: date,
    ) -> dict | None:
        """Fetch events for a sport and try to match the target event."""
        url = f"{self.BASE_URL}/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us,eu,uk",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                events = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                log.warning("Odds API: invalid API key")
            elif e.response.status_code == 429:
                log.warning("Odds API: rate limited")
            else:
                log.warning(f"Odds API HTTP error: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            log.warning(f"Odds API request failed: {e}")
            return None

        if not isinstance(events, list):
            return None

        # Try to match the event
        matched_event = self._match_event(events, team_a, team_b, event_date)
        if matched_event is None:
            return None

        # Extract consensus odds from bookmakers
        return self._extract_consensus_odds(matched_event, team_a)

    def _match_event(
        self,
        events: list[dict],
        team_a: str,
        team_b: str,
        event_date: date,
    ) -> dict | None:
        """Find the best matching event from API results.

        Uses team name substring matching and date filtering.
        Only returns a match when confident.
        """
        team_a_lower = team_a.lower()
        team_b_lower = team_b.lower()

        candidates: list[tuple[int, dict]] = []

        for event in events:
            home = event.get("home_team", "").lower()
            away = event.get("away_team", "").lower()

            # Check if both team names match (substring matching)
            a_matches = (
                team_a_lower in home
                or team_a_lower in away
                or home in team_a_lower
                or away in team_a_lower
            )
            b_matches = (
                team_b_lower in home
                or team_b_lower in away
                or home in team_b_lower
                or away in team_b_lower
            )

            if not (a_matches and b_matches):
                continue

            # Check date proximity
            commence_str = event.get("commence_time", "")
            if not commence_str:
                continue

            try:
                commence = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                event_day = commence.date()
            except (ValueError, TypeError):
                continue

            day_diff = abs((event_day - event_date).days)
            if day_diff > 2:
                # Too far from expected date -- skip
                continue

            # Score: prefer exact date match, then 1-day off, then 2-day off
            score = 10 - day_diff
            # Bonus for exact substring match (not just partial)
            if team_a_lower == home or team_a_lower == away:
                score += 5
            if team_b_lower == home or team_b_lower == away:
                score += 5

            candidates.append((score, event))

        if not candidates:
            return None

        # Return the best match
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _extract_consensus_odds(
        self,
        event: dict,
        team_a: str,
    ) -> dict | None:
        """Extract consensus (average) odds across bookmakers.

        Averages the h2h odds from all available bookmakers, then
        removes the vig to get fair probabilities.
        """
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return None

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # Determine which API team corresponds to our team_a
        team_a_lower = team_a.lower()
        if team_a_lower in home_team.lower() or home_team.lower() in team_a_lower:
            target_team = home_team
            other_team = away_team
        else:
            target_team = away_team
            other_team = home_team

        target_odds_list: list[float] = []
        other_odds_list: list[float] = []

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                target_price: float | None = None
                other_price: float | None = None
                for outcome in outcomes:
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if name == target_team:
                        target_price = float(price)
                    elif name == other_team:
                        other_price = float(price)

                if target_price is not None and other_price is not None:
                    target_odds_list.append(target_price)
                    other_odds_list.append(other_price)

        if not target_odds_list:
            return None

        # Average decimal odds across bookmakers
        avg_target_decimal = sum(target_odds_list) / len(target_odds_list)
        avg_other_decimal = sum(other_odds_list) / len(other_odds_list)

        # Convert to implied probabilities
        target_implied = 1.0 / avg_target_decimal
        other_implied = 1.0 / avg_other_decimal

        # Remove vig
        fair_target, fair_other = remove_vig(target_implied, other_implied)

        source_names = [b.get("title", b.get("key", "unknown")) for b in bookmakers[:5]]
        source_str = f"Average of {len(bookmakers)} books ({', '.join(source_names)})"

        return {
            "team_a_prob": fair_target,
            "team_b_prob": fair_other,
            "source": source_str,
            "bookmaker_count": len(bookmakers),
            "raw_decimal_odds": {
                "team_a": round(avg_target_decimal, 3),
                "team_b": round(avg_other_decimal, 3),
            },
        }
