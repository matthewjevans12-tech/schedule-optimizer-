from __future__ import annotations

PRODUCT_ENGINE_VERSION = "7.1.0"

import csv
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
try:
    import streamlit as st
except Exception:
    class _StreamlitStub:
        @staticmethod
        def cache_data(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    st = _StreamlitStub()
from bs4 import BeautifulSoup

try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except Exception:
    sort_items = None
    SORTABLES_AVAILABLE = False

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    cp_model = None
    ORTOOLS_AVAILABLE = False



@dataclass(frozen=True)
class Team:
    name: str
    subdivision: str  # FBS or FCS
    conference: str
    is_a4: bool = False
    parity_managed: bool = True


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    moveable: bool = True
    locked: bool = False
    notes: str = ""

    # Production scheduling semantics.
    neutral: bool = False
    campus_home_team: str = ""
    game_status: str = "CONTRACTED"      # CONTRACTED, PENDING, HOLD, CONCEPT
    moveability: str = "MOVABLE"         # MOVABLE, FLEXIBLE, LOCKED, UNKNOWN
    game_type: str = "NONCONFERENCE"     # CONFERENCE, FCS_GUARANTEE, FBS, A4, RIVALRY, NONCONFERENCE
    guarantee: Optional[float] = None
    contract_link: str = ""
    earliest_week: Optional[int] = None
    latest_week: Optional[int] = None
    source: str = ""
    last_verified: str = ""
    confidence: str = "INFERRED"         # AUTHORITATIVE, VERIFIED, INFERRED
    date_text: str = ""

    def involves(self, team: str) -> bool:
        return team in (self.home_team, self.away_team)

    def opponents(self) -> Tuple[str, str]:
        return self.home_team, self.away_team

    def site_for(self, team: str) -> str:
        """Campus/travel site, independent of designated home-team bookkeeping."""
        if not self.involves(team):
            return "NONE"
        if self.neutral:
            return "NEUTRAL"
        campus_home = self.campus_home_team or self.home_team
        return "HOME" if team == campus_home else "AWAY"

    def is_conference_game(self, teams: Dict[str, "Team"]) -> bool:
        if str(self.game_type).upper() == "CONFERENCE":
            return True
        home = teams.get(self.home_team)
        away = teams.get(self.away_team)
        return bool(
            home and away
            and home.subdivision == away.subdivision == "FBS"
            and home.conference
            and home.conference == away.conference
        )


@dataclass(frozen=True)
class Slot:
    team: str
    season: int
    week: int
    status: str  # OPEN, FLEX, BLOCKED, NEED_FCS, NEED_FBS, NEED_A4
    location: str = "ANY"  # HOME, AWAY, ANY


@dataclass(frozen=True)
class Need:
    team: str
    season: int
    week: int
    need_type: str  # FCS_BUY, FBS, A4
    location: str  # HOME, AWAY, ANY
    min_guarantee: Optional[int] = None
    max_guarantee: Optional[int] = None
    notes: str = ""


@dataclass(frozen=True)
class Move:
    game_id: str
    home_team: str
    away_team: str
    from_week: int
    to_week: int


@dataclass
class Solution:
    title: str
    moves: List[Move]
    score: float
    parity_before: Dict[str, str] = field(default_factory=dict)
    parity_after: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Intent:
    action: str
    season: Optional[int] = None
    target_week: Optional[int] = None
    conference: Optional[str] = None
    # Multi-scope fields let the LLM express requests such as
    # "fix every conference in Weeks 1, 2 and 3" without losing information.
    target_weeks: List[int] = field(default_factory=list)
    conferences: List[str] = field(default_factory=list)
    all_conferences: bool = False
    team_a: Optional[str] = None
    team_b: Optional[str] = None
    preserve_fbs_conference_parity: bool = True
    max_additional_moves: int = 4
    opponent_class: str = "ANY"
    location: str = "ANY"
    max_guarantee: Optional[int] = None
    summary: str = ""

    # Human scheduling context.
    # Must/Cannot fields become hard solver constraints.
    # Prefer fields only break ties after feasibility and minimum move count.
    constraint_teams: List[str] = field(default_factory=list)
    max_consecutive_away: Optional[int] = None
    max_consecutive_home: Optional[int] = None
    sequence_start_week: int = 0
    sequence_end_week: int = 13
    a4_move_policy: str = "NORMAL"  # NORMAL, PREFER_NOT, NEVER
    prefer_fcs_moves: bool = False
    avoid_game_ids: List[str] = field(default_factory=list)
    coach_context: str = ""

    # Advanced optimization policy. The UI normally leaves this automatic.
    # All strategies still satisfy hard constraints and minimize changed games first.
    optimization_strategy: str = "FEWEST_CHANGE"  # FEWEST_CHANGE, PROTECT_MARQUEE, COACH_FIT

    # Generic rules generated by the UI constraint builder.
    # Rules are dictionaries so new rule types can be added without changing
    # the stored profile schema. "MUST" and "CANNOT" are hard constraints;
    # "PREFER" contributes only after minimum game count is fixed.
    rules: List[Dict[str, object]] = field(default_factory=list)





class ScheduleStore:
    def __init__(self, teams: List[Team], games: List[Game], slots: List[Slot], needs: List[Need]):
        self.teams = {t.name: t for t in teams}
        self.games = {g.game_id: g for g in games}
        self.slots = {(s.team, s.season, s.week): s for s in slots}
        self.needs = needs

    def copy_games(self) -> Dict[str, Game]:
        return dict(self.games)

    def find_game(self, team_a: str, team_b: str, season: Optional[int] = None) -> Optional[Game]:
        wanted = {team_a, team_b}
        for game in self.games.values():
            if season is not None and game.season != season:
                continue
            if {game.home_team, game.away_team} == wanted:
                return game
        return None

    def game_for_team_week(self, games: Dict[str, Game], team: str, season: int, week: int, exclude_game_id: Optional[str] = None) -> Optional[Game]:
        for game in games.values():
            if game.game_id == exclude_game_id:
                continue
            if game.season == season and game.week == week and game.involves(team):
                return game
        return None

    def slot(self, team: str, season: int, week: int) -> Optional[Slot]:
        return self.slots.get((team, season, week))

    def slot_allows_game(self, team: str, season: int, week: int) -> bool:
        slot = self.slot(team, season, week)
        if slot is None:
            return False  # explicit availability is required in this MVP
        return slot.status in {"OPEN", "FLEX", "NEED_FCS", "NEED_FBS", "NEED_A4"}

    def conference_members(self, conference: str) -> List[Team]:
        return [t for t in self.teams.values() if t.subdivision == "FBS" and t.conference == conference and t.parity_managed]

    def fbs_conferences(self) -> List[str]:
        return sorted({t.conference for t in self.teams.values() if t.subdivision == "FBS" and t.conference and t.parity_managed})





class NonConferenceOptimizer:
    """Deterministic non-conference scheduling engine.

    The LLM may interpret a user's request, but this class is the authority on
    availability, legal moves, parity, and solution ranking.
    """

    def __init__(self, store: ScheduleStore):
        self.store = store

    def solve(self, intent: Intent) -> List[Solution]:
        action = intent.action.upper()
        if action == "MOVE_GAME":
            return self.solve_move_game(intent)
        if action == "MAKE_CONFERENCE_EVEN":
            return self.solve_make_conference_even(intent)
        if action in {"FIND_BUY_GAME", "FIND_FCS_BUY_GAME"}:
            return self.find_buy_games(intent)
        if action == "FIND_A4_GAME":
            return self.find_a4_games(intent)
        return []

    def conference_parity(self, games: Dict[str, Game], season: int, week: int) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for conference in self.store.fbs_conferences():
            members = self.store.conference_members(conference)
            member_names = {t.name for t in members}
            nonconf_teams: Set[str] = set()
            for game in games.values():
                if game.season != season or game.week != week:
                    continue
                home = self.store.teams.get(game.home_team)
                away = self.store.teams.get(game.away_team)
                if home and away and home.conference == away.conference and home.subdivision == away.subdivision == "FBS":
                    continue
                if game.home_team in member_names:
                    nonconf_teams.add(game.home_team)
                if game.away_team in member_names:
                    nonconf_teams.add(game.away_team)
            available = len(members) - len(nonconf_teams)
            result[conference] = f"{'EVEN' if available % 2 == 0 else 'ODD'} ({available} available; {len(nonconf_teams)} non-conf)"
        return result

    def parity_violation_count(self, games: Dict[str, Game], season: int, weeks: Iterable[int]) -> int:
        count = 0
        for week in weeks:
            parity = self.conference_parity(games, season, week)
            count += sum(1 for value in parity.values() if value.startswith("ODD"))
        return count

    def parity_issue_details(
        self,
        games: Dict[str, Game],
        season: int,
        weeks: Iterable[int] = range(0, 14),
        conferences: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, object]]:
        """Return every odd conference/week state with enough context to act on it."""
        allowed = set(conferences) if conferences is not None else None
        issues: List[Dict[str, object]] = []
        for week in sorted({int(w) for w in weeks}):
            parity = self.conference_parity(games, season, week)
            for conference, value in sorted(parity.items()):
                if allowed is not None and conference not in allowed:
                    continue
                if not value.startswith("ODD"):
                    continue
                members = self.store.conference_members(conference)
                member_names = {t.name for t in members}
                nonconf_teams: Set[str] = set()
                game_labels: List[str] = []
                for game in games.values():
                    if game.season != season or game.week != week:
                        continue
                    home = self.store.teams.get(game.home_team)
                    away = self.store.teams.get(game.away_team)
                    if home and away and home.conference == away.conference and home.subdivision == away.subdivision == "FBS":
                        continue
                    involved = False
                    if game.home_team in member_names:
                        nonconf_teams.add(game.home_team)
                        involved = True
                    if game.away_team in member_names:
                        nonconf_teams.add(game.away_team)
                        involved = True
                    if involved:
                        game_labels.append(f"{game.away_team} @ {game.home_team}")
                available = len(members) - len(nonconf_teams)
                issues.append({
                    "conference": conference,
                    "week": week,
                    "available": available,
                    "conference_size": len(members),
                    "nonconf_count": len(nonconf_teams),
                    "nonconf_teams": sorted(nonconf_teams),
                    "games": sorted(set(game_labels)),
                    "next_action": f"Change one {conference} non-conference team appearance into or out of Week {week} to flip parity.",
                })
        return issues

    def _candidate_weeks(self, game: Game, preferred_week: Optional[int] = None) -> List[int]:
        weeks = list(range(0, 15))
        weeks = [w for w in weeks if w != game.week]
        if preferred_week is not None and preferred_week in weeks:
            weeks.remove(preferred_week)
            weeks.insert(0, preferred_week)
        weeks.sort(key=lambda w: (0 if preferred_week == w else 1, abs(w - game.week), w))
        return weeks

    def _can_use_week(self, games: Dict[str, Game], game: Game, week: int) -> bool:
        if not self.store.slot_allows_game(game.home_team, game.season, week):
            return False
        if not self.store.slot_allows_game(game.away_team, game.season, week):
            return False
        return True

    def _conflicts(self, games: Dict[str, Game], game: Game, week: int) -> List[Game]:
        seen: Dict[str, Game] = {}
        for team in (game.home_team, game.away_team):
            conflict = self.store.game_for_team_week(games, team, game.season, week, exclude_game_id=game.game_id)
            if conflict:
                seen[conflict.game_id] = conflict
        return list(seen.values())

    def _relocate_for_target(
        self,
        games: Dict[str, Game],
        game_id: str,
        target_week: int,
        moves: List[Move],
        max_moves: int,
        visited: Set[Tuple[str, int]],
    ) -> List[Tuple[Dict[str, Game], List[Move]]]:
        if len(moves) >= max_moves:
            return []
        key = (game_id, target_week)
        if key in visited:
            return []
        visited = set(visited)
        visited.add(key)

        game = games[game_id]
        if game.locked or not game.moveable:
            return []
        if not self._can_use_week(games, game, target_week):
            return []

        conflicts = self._conflicts(games, game, target_week)
        if not conflicts:
            new_games = dict(games)
            new_games[game_id] = replace(game, week=target_week)
            new_moves = moves + [Move(game_id, game.home_team, game.away_team, game.week, target_week)]
            return [(new_games, new_moves)]

        # Resolve one conflict at a time. Each conflict can itself create a cascade.
        conflict = conflicts[0]
        if conflict.locked or not conflict.moveable:
            return []

        outcomes: List[Tuple[Dict[str, Game], List[Move]]] = []
        for alt_week in self._candidate_weeks(conflict):
            if alt_week == target_week:
                continue
            relocated = self._relocate_for_target(
                games, conflict.game_id, alt_week, moves, max_moves, visited
            )
            for g2, m2 in relocated:
                # The conflict has moved. Re-attempt this same placement; remove
                # the current key from the cycle guard so the solver can finish
                # the original move without allowing an infinite loop.
                retry_visited = set(visited)
                retry_visited.discard(key)
                outcomes.extend(self._relocate_for_target(g2, game_id, target_week, m2, max_moves, retry_visited))
                if len(outcomes) >= 30:
                    return outcomes
        return outcomes

    def solve_move_game(self, intent: Intent) -> List[Solution]:
        if not all([intent.team_a, intent.team_b, intent.season is not None, intent.target_week is not None]):
            return []
        game = self.store.find_game(intent.team_a, intent.team_b, intent.season)
        if not game:
            return []

        base_games = self.store.copy_games()
        weeks_touched = set([game.week, intent.target_week])
        parity_before = {}
        for week in sorted(weeks_touched):
            for conf, status in self.conference_parity(base_games, intent.season, week).items():
                parity_before[f"{conf} W{week}"] = status

        raw = self._relocate_for_target(
            base_games,
            game.game_id,
            intent.target_week,
            [],
            max_moves=max(1, intent.max_additional_moves + 1),
            visited=set(),
        )

        solutions: List[Solution] = []
        signatures = set()
        for games, moves in raw:
            if not moves:
                continue
            sig = tuple(sorted((m.game_id, m.from_week, m.to_week) for m in moves))
            if sig in signatures:
                continue
            signatures.add(sig)
            touched = {m.from_week for m in moves} | {m.to_week for m in moves}
            before_violations = self.parity_violation_count(base_games, intent.season, touched)
            after_violations = self.parity_violation_count(games, intent.season, touched)

            if intent.preserve_fbs_conference_parity and after_violations > before_violations:
                continue

            parity_after: Dict[str, str] = {}
            for week in sorted(touched):
                for conf, status in self.conference_parity(games, intent.season, week).items():
                    parity_after[f"{conf} W{week}"] = status

            added_moves = max(0, len(moves) - 1)
            distance = sum(abs(m.to_week - m.from_week) for m in moves)
            parity_delta = before_violations - after_violations
            score = max(0, min(100, 100 - added_moves * 15 - distance * 1.5 - after_violations * 8 + parity_delta * 8))
            warnings = []
            if after_violations:
                warnings.append(f"{after_violations} FBS conference/week parity issue(s) remain in affected weeks.")
            explanation = self._explain_moves(moves, before_violations, after_violations)
            solutions.append(Solution(
                title=f"{len(moves)}-move solution",
                moves=moves,
                score=round(score, 1),
                parity_before=parity_before,
                parity_after=parity_after,
                warnings=warnings,
                explanation=explanation,
            ))

        solutions.sort(key=lambda s: (-s.score, len(s.moves)))
        return solutions[:5]

    def _direct_parity_move_solution(self, base: Dict[str, Game], game: Game, to_week: int, intent: Intent) -> Optional[Solution]:
        """Evaluate one direct game move without any cascade search.

        Broad conference-parity requests should be fast. We therefore try direct
        one-game fixes first and only invoke the recursive cascade solver when no
        direct fix exists.
        """
        if game.locked or not game.moveable or to_week == game.week:
            return None
        if not self._can_use_week(base, game, to_week):
            return None
        if self._conflicts(base, game, to_week):
            return None

        new_games = dict(base)
        new_games[game.game_id] = replace(game, week=to_week)
        touched = {game.week, to_week}

        target_status = self.conference_parity(new_games, intent.season, intent.target_week).get(intent.conference, "")
        if not target_status.startswith("EVEN"):
            return None

        before_violations = self.parity_violation_count(base, intent.season, touched)
        after_violations = self.parity_violation_count(new_games, intent.season, touched)
        if intent.preserve_fbs_conference_parity and after_violations > before_violations:
            return None

        parity_before: Dict[str, str] = {}
        parity_after: Dict[str, str] = {}
        for week in sorted(touched):
            for conf, status in self.conference_parity(base, intent.season, week).items():
                parity_before[f"{conf} W{week}"] = status
            for conf, status in self.conference_parity(new_games, intent.season, week).items():
                parity_after[f"{conf} W{week}"] = status

        move = Move(game.game_id, game.home_team, game.away_team, game.week, to_week)
        distance = abs(to_week - game.week)
        parity_delta = before_violations - after_violations
        score = max(0, min(100, 100 - distance * 1.5 - after_violations * 8 + parity_delta * 8))
        warnings = []
        if after_violations:
            warnings.append(f"{after_violations} FBS conference/week parity issue(s) remain in affected weeks.")
        direction = "into" if to_week == intent.target_week else "out of"
        explanation = (
            f"Move {game.away_team} at {game.home_team} from Week {game.week} to Week {to_week}. "
            f"This moves a {intent.conference} non-conference commitment {direction} Week {intent.target_week} "
            f"and changes {intent.conference} Week {intent.target_week} to {target_status}."
        )
        return Solution(
            title="1-move parity fix",
            moves=[move],
            score=round(score, 1),
            parity_before=parity_before,
            parity_after=parity_after,
            warnings=warnings,
            explanation=explanation,
        )

    def solve_make_conference_even(self, intent: Intent) -> List[Solution]:
        if not intent.conference or intent.season is None or intent.target_week is None:
            return []

        base = self.store.copy_games()
        current = self.conference_parity(base, intent.season, intent.target_week).get(intent.conference)
        if current and current.startswith("EVEN"):
            return [Solution(
                title="Already even",
                moves=[],
                score=100,
                explanation=f"{intent.conference} is already even in Week {intent.target_week}: {current}.",
            )]

        members = {t.name for t in self.store.conference_members(intent.conference)}

        # Only a non-conference game involving exactly one member of the target
        # conference can toggle that conference's weekly parity.
        candidates: List[Game] = []
        for g in base.values():
            if g.season != intent.season or g.locked or not g.moveable:
                continue
            member_count = int(g.home_team in members) + int(g.away_team in members)
            if member_count == 1:
                candidates.append(g)

        # PHASE 1: Fast direct fixes. This is the normal answer for a broad
        # "make the conference even" request and avoids a national brute-force search.
        direct: List[Solution] = []

        # A) Move a target-conference nonconference game INTO the odd week.
        for game in sorted(candidates, key=lambda g: (abs(g.week - intent.target_week), g.week, g.home_team, g.away_team)):
            if game.week == intent.target_week:
                continue
            sol = self._direct_parity_move_solution(base, game, intent.target_week, intent)
            if sol:
                direct.append(sol)

        # B) Move a target-week nonconference game OUT to the closest feasible week.
        for game in [g for g in candidates if g.week == intent.target_week]:
            found_for_game = 0
            for alt_week in self._candidate_weeks(game):
                sol = self._direct_parity_move_solution(base, game, alt_week, intent)
                if sol:
                    direct.append(sol)
                    found_for_game += 1
                    if found_for_game >= 3:
                        break

        if direct:
            uniq: Dict[Tuple, Solution] = {}
            for sol in direct:
                sig = tuple((m.game_id, m.from_week, m.to_week) for m in sol.moves)
                if sig not in uniq or sol.score > uniq[sig].score:
                    uniq[sig] = sol
            return sorted(uniq.values(), key=lambda s: (-s.score, len(s.moves)))[:5]

        # PHASE 2: No direct fix exists. Search only the closest candidates and
        # allow a small cascade. This keeps public-data mode responsive.
        cascades: List[Solution] = []
        nearest = sorted(candidates, key=lambda g: (0 if g.week == intent.target_week else 1, abs(g.week - intent.target_week)))[:16]

        for game in nearest:
            target_weeks = [intent.target_week] if game.week != intent.target_week else self._candidate_weeks(game)[:5]
            for target in target_weeks:
                temp_intent = Intent(
                    action="MOVE_GAME",
                    season=intent.season,
                    target_week=target,
                    conference=intent.conference,
                    team_a=game.home_team,
                    team_b=game.away_team,
                    preserve_fbs_conference_parity=True,
                    max_additional_moves=min(2, intent.max_additional_moves),
                    summary=intent.summary,
                )
                for sol in self.solve_move_game(temp_intent):
                    status = sol.parity_after.get(f"{intent.conference} W{intent.target_week}", "")
                    if status.startswith("EVEN"):
                        cascades.append(sol)
                if len(cascades) >= 10:
                    break
            if len(cascades) >= 10:
                break

        uniq: Dict[Tuple, Solution] = {}
        for sol in cascades:
            sig = tuple(sorted((m.game_id, m.from_week, m.to_week) for m in sol.moves))
            if sig not in uniq or sol.score > uniq[sig].score:
                uniq[sig] = sol
        return sorted(uniq.values(), key=lambda s: (-s.score, len(s.moves)))[:5]

    @staticmethod
    def market_week_prior(week: int) -> int:
        """Soft prior for nonconference market liquidity.

        Most nonconference inventory is concentrated early in the season.
        This is deliberately a preference, never a hard constraint.
        Internal Weeks 0-3 correspond to user-facing Weeks 1-4.
        """
        week = int(week)
        if 0 <= week <= 3:
            return 24
        if 4 <= week <= 5:
            return 10
        if 6 <= week <= 8:
            return 4
        return 0

    def find_buy_games(self, intent: Intent) -> List[Solution]:
        """Find buy/guarantee-game matches, preferring explicit school needs.

        Weeks 1-4 receive a soft market-liquidity bonus when the user does not
        require a specific week. Later weeks remain valid.
        """
        if not intent.team_a or intent.season is None:
            return []
        requester = self.store.teams.get(intent.team_a)
        if not requester:
            return []

        base_games = self.store.copy_games()
        weeks = [int(intent.target_week)] if intent.target_week is not None else list(range(14))
        results: List[Solution] = []

        explicit_needs = [
            n for n in self.store.needs
            if int(n.season) == int(intent.season)
        ]

        wanted_subdivision = "FCS" if requester.subdivision == "FBS" else "FBS"

        for week in weeks:
            if self.store.game_for_team_week(base_games, requester.name, int(intent.season), week):
                continue
            if not self.store.slot_allows_game(requester.name, int(intent.season), week):
                continue

            for candidate in self.store.teams.values():
                if candidate.name == requester.name or candidate.subdivision != wanted_subdivision:
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, int(intent.season), week):
                    continue
                if not self.store.slot_allows_game(candidate.name, int(intent.season), week):
                    continue

                candidate_needs = [
                    n for n in explicit_needs
                    if n.team == candidate.name and int(n.week) == week
                    and str(n.need_type).upper() in {"FCS_BUY", "BUY", "FBS", "FBS_BUY"}
                ]
                requester_needs = [
                    n for n in explicit_needs
                    if n.team == requester.name and int(n.week) == week
                    and str(n.need_type).upper() in {"FCS_BUY", "BUY", "FBS", "FBS_BUY"}
                ]

                explicit = bool(candidate_needs or requester_needs)
                need = candidate_needs[0] if candidate_needs else (requester_needs[0] if requester_needs else None)

                if requester.subdivision == "FBS":
                    home_team, away_team = requester.name, candidate.name
                    # Buy-game host is normally HOME. Respect a hard user location request.
                    if str(intent.location).upper() == "AWAY":
                        continue
                    if need and need.location not in {"AWAY", "ANY", "HOME"}:
                        continue
                    if intent.max_guarantee is not None and need and need.min_guarantee is not None:
                        if int(need.min_guarantee) > int(intent.max_guarantee):
                            continue
                else:
                    home_team, away_team = candidate.name, requester.name
                    if str(intent.location).upper() == "HOME":
                        continue

                liquidity = self.market_week_prior(week)
                score = 68 + liquidity + (40 if explicit else 0)
                if candidate.is_a4 and requester.subdivision == "FCS":
                    score += 3

                guarantee_text = ""
                if need:
                    if need.min_guarantee is not None:
                        guarantee_text = f" Minimum guarantee: ${int(need.min_guarantee):,}."
                    elif need.max_guarantee is not None:
                        guarantee_text = f" Maximum guarantee: ${int(need.max_guarantee):,}."

                market_text = "high-liquidity early-season week" if week <= 3 else "later-season opportunity"
                intent_text = (
                    "An explicit compatible school need is recorded."
                    if explicit
                    else "This is an availability candidate; confirm actual interest."
                )
                results.append(Solution(
                    title=f"Week {week + 1} · {away_team} @ {home_team}",
                    moves=[],
                    score=float(score),
                    explanation=(
                        f"{home_team} and {away_team} have no known game in Week {week + 1}. "
                        f"{intent_text} Week {week + 1} is a {market_text}.{guarantee_text}"
                    ),
                    metadata={
                        "match_type": "BUY_GAME",
                        "requester": requester.name,
                        "candidate": candidate.name,
                        "week": week,
                        "home_team": home_team,
                        "away_team": away_team,
                        "explicit_need": explicit,
                        "market_liquidity": "HIGH" if week <= 3 else "NORMAL",
                        "min_guarantee": need.min_guarantee if need else None,
                        "max_guarantee": need.max_guarantee if need else None,
                    },
                ))

        results.sort(
            key=lambda s: (
                -float(s.score),
                int((s.metadata or {}).get("week", 99)),
                s.title,
            )
        )
        return results[:20]

    def find_a4_games(self, intent: Intent) -> List[Solution]:
        """Find A4-vs-A4 opportunities across one week or the entire season."""
        if not intent.team_a or intent.season is None:
            return []
        team = self.store.teams.get(intent.team_a)
        if not team or not team.is_a4:
            return []

        base_games = self.store.copy_games()
        weeks = [int(intent.target_week)] if intent.target_week is not None else list(range(14))
        results: List[Solution] = []
        explicit_needs = [n for n in self.store.needs if int(n.season) == int(intent.season)]

        for week in weeks:
            if self.store.game_for_team_week(base_games, team.name, int(intent.season), week):
                continue
            if not self.store.slot_allows_game(team.name, int(intent.season), week):
                continue

            requester_needs = [
                n for n in explicit_needs
                if n.team == team.name and int(n.week) == week and str(n.need_type).upper() == "A4"
            ]

            for candidate in self.store.teams.values():
                if (
                    not candidate.is_a4
                    or candidate.name == team.name
                    or candidate.conference == team.conference
                ):
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, int(intent.season), week):
                    continue
                if not self.store.slot_allows_game(candidate.name, int(intent.season), week):
                    continue

                candidate_needs = [
                    n for n in explicit_needs
                    if n.team == candidate.name and int(n.week) == week and str(n.need_type).upper() == "A4"
                ]
                explicit = bool(requester_needs and candidate_needs)

                # Determine a sensible site from explicit need/location preferences.
                requested_location = str(intent.location or "ANY").upper()
                if requested_location == "AWAY":
                    home_team, away_team = candidate.name, team.name
                else:
                    home_team, away_team = team.name, candidate.name

                # If the candidate explicitly says HOME and requester says HOME,
                # that is not a compatible pairing.
                req_loc = requester_needs[0].location if requester_needs else requested_location
                cand_loc = candidate_needs[0].location if candidate_needs else "ANY"
                if req_loc == "HOME" and cand_loc == "HOME":
                    continue
                if req_loc == "AWAY" and cand_loc == "AWAY":
                    continue
                if req_loc == "AWAY":
                    home_team, away_team = candidate.name, team.name
                elif cand_loc == "AWAY":
                    home_team, away_team = team.name, candidate.name
                elif cand_loc == "HOME":
                    home_team, away_team = candidate.name, team.name

                liquidity = self.market_week_prior(week)
                score = 70 + liquidity + (40 if explicit else 0)
                results.append(Solution(
                    title=f"Week {week + 1} · {away_team} @ {home_team}",
                    moves=[],
                    score=float(score),
                    explanation=(
                        f"{team.name} and {candidate.name} are A4 programs in different conferences "
                        f"with no known game in Week {week + 1}. "
                        + (
                            "Both schools have compatible A4 needs recorded."
                            if explicit
                            else "This is an availability candidate; confirm mutual scheduling intent."
                        )
                    ),
                    metadata={
                        "match_type": "A4",
                        "requester": team.name,
                        "candidate": candidate.name,
                        "week": week,
                        "home_team": home_team,
                        "away_team": away_team,
                        "explicit_need": explicit,
                        "market_liquidity": "HIGH" if week <= 3 else "NORMAL",
                    },
                ))

        results.sort(
            key=lambda s: (
                -float(s.score),
                int((s.metadata or {}).get("week", 99)),
                s.title,
            )
        )
        return results[:20]

    def _explain_moves(self, moves: List[Move], before: int, after: int) -> str:
        chain = " → ".join(f"{m.home_team}-{m.away_team} W{m.from_week}→W{m.to_week}" for m in moves)
        if after < before:
            parity = f"The move reduces affected FBS parity issues from {before} to {after}."
        elif after == before:
            parity = f"The move does not increase FBS parity issues ({after} remain in the affected weeks)."
        else:
            parity = f"The move increases parity issues from {before} to {after}."
        return f"Move chain: {chain}. {parity}"





class AdvancedNonConferenceOptimizer(NonConferenceOptimizer):
    """CP-SAT optimization layer for the College Football Non-Conference Scheduling Optimizer.

    The LLM only translates natural language into Intent. This class is the
    scheduling authority. When OR-Tools is available it solves the relevant
    scheduling neighborhood as a constraint-programming problem; the legacy
    deterministic routines remain a safe fallback for development environments
    that do not have OR-Tools installed.
    """

    PARITY_PENALTY = 25_000
    MOVE_PENALTY = 1_400
    DISTANCE_PENALTY = 60
    BALANCE_PENALTY = 5_000

    def __init__(self, store: ScheduleStore, time_limit_seconds: float = 3.0):
        super().__init__(store)
        self.time_limit_seconds = float(time_limit_seconds)
        self.last_solver_status = "Fallback"
        self.last_solver_seconds = 0.0

    @property
    def engine_name(self) -> str:
        return "Lexicographic CP-SAT" if ORTOOLS_AVAILABLE else "Deterministic fallback"

    def solve(self, intent: Intent) -> List[Solution]:
        action = (intent.action or "").upper()
        if action == "OPTIMIZE_NATIONAL":
            return self.optimize_national(intent)
        if action == "BALANCE_FCS_GAMES":
            return self.balance_fcs_games(intent)
        if action == "BALANCE_CONTROLLED_GAMES":
            return self.balance_controlled_games(intent)
        if action == "OPTIMIZE_MARKET":
            return self.optimize_market(intent)
        if action == "MOVE_GAME":
            return self.solve_move_game(intent)
        if action == "MAKE_CONFERENCE_EVEN":
            return self.solve_make_conference_even(intent)
        if action in {"FIND_BUY_GAME", "FIND_FCS_BUY_GAME"}:
            return self.find_buy_games(intent)
        if action == "FIND_A4_GAME":
            return self.find_a4_games(intent)
        return []

    def _candidate_weeks_for_cp(self, game: Game, target_week: Optional[int] = None, wide: bool = False) -> List[int]:
        if game.locked or not game.moveable:
            return [game.week]
        radius = 13 if wide else 5
        weeks = []
        for week in range(0, 14):
            if week == game.week:
                weeks.append(week)
                continue
            if game.earliest_week is not None and week < int(game.earliest_week):
                continue
            if game.latest_week is not None and week > int(game.latest_week):
                continue
            if abs(week - game.week) > radius and week != target_week:
                continue
            if self.store.slot_allows_game(game.home_team, game.season, week) and self.store.slot_allows_game(game.away_team, game.season, week):
                weeks.append(week)
        if target_week is not None and target_week not in weeks:
            if self.store.slot_allows_game(game.home_team, game.season, target_week) and self.store.slot_allows_game(game.away_team, game.season, target_week):
                weeks.append(target_week)
        return sorted(set(weeks))

    def _base_bad_parity(self, season: int) -> Dict[Tuple[str, int], bool]:
        base = self.store.copy_games()
        result = {}
        for week in range(0, 14):
            for conf, value in self.conference_parity(base, season, week).items():
                result[(conf, week)] = value.startswith("ODD")
        return result

    def _intent_scope(self, intent: Intent) -> Tuple[List[str], List[int]]:
        """Return the conference/week scope the user explicitly asked to optimize.

        Empty scope means the full national season. Singular legacy fields are
        folded into the list fields so old UI controls keep working.
        """
        if intent.all_conferences:
            conferences = self.store.fbs_conferences()
        elif intent.conferences:
            valid = set(self.store.fbs_conferences())
            conferences = [c for c in intent.conferences if c in valid]
        elif intent.conference:
            conferences = [intent.conference]
        else:
            conferences = self.store.fbs_conferences()

        weeks = [int(w) for w in intent.target_weeks if 0 <= int(w) <= 13]
        if not weeks and intent.target_week is not None and 0 <= int(intent.target_week) <= 13:
            weeks = [int(intent.target_week)]
        if not weeks:
            weeks = list(range(0, 14))
        return sorted(set(conferences)), sorted(set(weeks))

    def _scoped_bad_count(self, games: Dict[str, Game], season: int, conferences: List[str], weeks: List[int]) -> int:
        count = 0
        for week in weeks:
            parity = self.conference_parity(games, season, week)
            for conf in conferences:
                if parity.get(conf, "").startswith("ODD"):
                    count += 1
        return count

    def _game_conference_coeff(self, game: Game, conference: str) -> int:
        members = {t.name for t in self.store.conference_members(conference)}
        if not members:
            return 0
        home = self.store.teams.get(game.home_team)
        away = self.store.teams.get(game.away_team)
        # A same-conference FBS matchup is not counted as non-conference inventory.
        if home and away and home.subdivision == away.subdivision == "FBS" and home.conference == away.conference == conference:
            return 0
        return int(game.home_team in members) + int(game.away_team in members)

    def _is_fbs_fcs(self, game: Game) -> bool:
        home = self.store.teams.get(game.home_team)
        away = self.store.teams.get(game.away_team)
        if not home or not away:
            return False
        return {home.subdivision, away.subdivision} == {"FBS", "FCS"}

    def _is_a4_matchup(self, game: Game) -> bool:
        home = self.store.teams.get(game.home_team)
        away = self.store.teams.get(game.away_team)
        if not home or not away:
            return False
        return bool(
            home.subdivision == away.subdivision == "FBS"
            and home.is_a4
            and away.is_a4
        )

    def _repair_neighborhood_game_ids(
        self,
        target_game: Game,
        season_games: List[Game],
        depth: Optional[int],
    ) -> Set[str]:
        """Return a local game neighborhood around the requested transaction.

        Depth 1 = games directly sharing a team with the requested game.
        Each additional depth expands through the teams touched by those games.
        None = full-season search.
        """
        if depth is None:
            return {g.game_id for g in season_games}
        depth = max(0, int(depth))
        active_teams: Set[str] = {target_game.home_team, target_game.away_team}
        active_games: Set[str] = {target_game.game_id}
        for _ in range(depth + 1):
            newly_touched: Set[str] = set()
            for game in season_games:
                if game.game_id in active_games:
                    continue
                if game.home_team in active_teams or game.away_team in active_teams:
                    active_games.add(game.game_id)
                    newly_touched.add(game.home_team)
                    newly_touched.add(game.away_team)
            if not newly_touched:
                break
            active_teams.update(newly_touched)
        return active_games

    def _game_disruption_weight(self, game: Game) -> int:
        """Scheduling-cost prior used only after minimum move count is fixed."""
        home = self.store.teams.get(game.home_team)
        away = self.store.teams.get(game.away_team)
        if self._is_fbs_fcs(game):
            return 1
        if self._is_a4_matchup(game):
            return 30
        if home and away and home.subdivision == away.subdivision == "FBS":
            return 10
        return 4

    @staticmethod
    def _clear_cp_objective(model: object) -> None:
        """Clear an OR-Tools objective across supported CP-SAT versions."""
        try:
            model.ClearObjective()
        except Exception:
            try:
                model.Proto().ClearField("objective")
            except Exception:
                pass

    @staticmethod
    def _clear_cp_hint(model: object) -> None:
        try:
            model.Proto().ClearField("solution_hint")
        except Exception:
            pass

    def _add_solution_hint(
        self,
        model: object,
        x: Dict[Tuple[str, int], object],
        values: Dict[Tuple[str, int], int],
    ) -> None:
        """Warm-start the next lexicographic stage from the prior stage."""
        if not values:
            return
        self._clear_cp_hint(model)
        for key, var in x.items():
            if key in values:
                try:
                    model.AddHint(var, int(values[key]))
                except Exception:
                    return

    def _cp_optimize(self, intent: Intent, mode: str, repair_depth: Optional[int] = None, strategy: str = "FEWEST_CHANGE") -> List[Solution]:
        if not ORTOOLS_AVAILABLE or intent.season is None:
            return []

        season = int(intent.season)
        season_games = [g for g in self.store.games.values() if g.season == season and 0 <= g.week <= 13]
        if not season_games:
            return []

        target_game = None
        if mode == "move":
            if not intent.team_a or not intent.team_b or intent.target_week is None:
                return []
            target_game = self.store.find_game(intent.team_a, intent.team_b, season)
            if target_game is None:
                return []

        # Large-neighborhood repair search: for a direct repair, only games
        # connected to the requested transaction are allowed to move initially.
        # The caller progressively expands the neighborhood only when needed.
        wide = mode in {"national", "fcs_balance", "controlled_balance"}
        active_repair_games: Optional[Set[str]] = None
        if mode == "move" and target_game is not None:
            active_repair_games = self._repair_neighborhood_game_ids(
                target_game, season_games, repair_depth
            )

        candidate_weeks: Dict[str, List[int]] = {}
        for game in season_games:
            target = int(intent.target_week) if target_game and game.game_id == target_game.game_id and intent.target_week is not None else None

            if (
                mode == "move"
                and active_repair_games is not None
                and game.game_id not in active_repair_games
            ):
                # Freeze unrelated inventory. This prevents a local transaction
                # from "repairing" Navy/Notre Dame or another distant game.
                candidate_weeks[game.game_id] = [game.week]
            elif (
                str(intent.a4_move_policy or "NORMAL").upper() == "NEVER"
                and self._is_a4_matchup(game)
                and not (target_game and game.game_id == target_game.game_id)
            ):
                candidate_weeks[game.game_id] = [game.week]
            else:
                candidate_weeks[game.game_id] = self._candidate_weeks_for_cp(
                    game, target_week=target, wide=wide
                )
                if game.week not in candidate_weeks[game.game_id]:
                    candidate_weeks[game.game_id].append(game.week)
                candidate_weeks[game.game_id] = sorted(set(candidate_weeks[game.game_id]))

        model = cp_model.CpModel()
        x: Dict[Tuple[str, int], object] = {}
        for game in season_games:
            vars_for_game = []
            for week in candidate_weeks[game.game_id]:
                var = model.NewBoolVar(f"g_{game.game_id}_w{week}")
                x[(game.game_id, week)] = var
                vars_for_game.append(var)
            model.AddExactlyOne(vars_for_game)

        # A school may have at most one known non-conference game in a week.
        team_week_vars: Dict[Tuple[str, int], List[object]] = {}
        for game in season_games:
            for week in candidate_weeks[game.game_id]:
                for team in (game.home_team, game.away_team):
                    team_week_vars.setdefault((team, week), []).append(x[(game.game_id, week)])
        for vars_list in team_week_vars.values():
            if len(vars_list) > 1:
                model.Add(sum(vars_list) <= 1)

        # Human scheduling rules: maximum consecutive away/home games.
        # The public MVP only sees the schedule context loaded into this store.
        # A production data feed should include conference games as well so
        # travel-streak rules evaluate the full schedule.
        constrained_teams = [t for t in intent.constraint_teams if t in self.store.teams]
        seq_start = max(0, min(13, int(intent.sequence_start_week)))
        seq_end = max(seq_start, min(13, int(intent.sequence_end_week)))

        def add_streak_limit(team: str, max_streak: Optional[int], away: bool) -> None:
            if max_streak is None:
                return
            max_streak = int(max_streak)
            if max_streak < 1:
                return
            window_len = max_streak + 1
            if seq_end - seq_start + 1 < window_len:
                return
            for start in range(seq_start, seq_end - window_len + 2):
                terms = []
                for week in range(start, start + window_len):
                    for game in season_games:
                        correct_site = (
                            game.site_for(team) == ("AWAY" if away else "HOME")
                        )
                        if correct_site and (game.game_id, week) in x:
                            terms.append(x[(game.game_id, week)])
                if terms:
                    model.Add(sum(terms) <= max_streak)

        for team in constrained_teams:
            add_streak_limit(team, intent.max_consecutive_away, away=True)
            add_streak_limit(team, intent.max_consecutive_home, away=False)

        # Generic Must / Cannot rules.
        # All user-facing weeks are 1-14; rules arrive here as internal 0-13.
        hard_rules = [
            r for r in (intent.rules or [])
            if str(r.get("hardness", "")).upper() in {"MUST", "CANNOT"}
            and bool(r.get("active", True))
        ]

        def _rule_team(r: Dict[str, object]) -> str:
            return str(r.get("team") or "")

        def _rule_range(r: Dict[str, object]) -> Tuple[int, int]:
            a = max(0, min(13, int(r.get("start_week", 0))))
            b = max(a, min(13, int(r.get("end_week", 13))))
            return a, b

        def _team_vars(team: str, week: int, site: Optional[str] = None, a4_only: bool = False) -> List[object]:
            terms: List[object] = []
            for g in season_games:
                if not g.involves(team) or (g.game_id, week) not in x:
                    continue
                if site and g.site_for(team) != site:
                    continue
                if a4_only and not self._is_a4_matchup(g):
                    continue
                terms.append(x[(g.game_id, week)])
            return terms

        for rule in hard_rules:
            rtype = str(rule.get("rule_type") or "").upper()
            team = _rule_team(rule)
            start_w, end_w = _rule_range(rule)
            value = int(rule.get("value", 1) or 1)
            game_id = str(rule.get("game_id") or "")

            if rtype in {"MAX_CONSECUTIVE_AWAY", "MAX_CONSECUTIVE_HOME"} and team:
                site = "AWAY" if rtype.endswith("AWAY") else "HOME"
                window = value + 1
                if window > 1:
                    for start in range(start_w, end_w - window + 2):
                        terms: List[object] = []
                        for week in range(start, start + window):
                            terms.extend(_team_vars(team, week, site=site))
                        if terms:
                            model.Add(sum(terms) <= value)

            elif rtype == "MIN_CAMPUS_HOME_IN_RANGE" and team:
                terms: List[object] = []
                for week in range(start_w, end_w + 1):
                    terms.extend(_team_vars(team, week, site="HOME"))
                if terms:
                    model.Add(sum(terms) >= value)
                elif value > 0:
                    return []

            elif rtype == "MAX_WEEKS_WITHOUT_CAMPUS_HOME" and team:
                # Every window of value+1 weeks must contain at least one campus-home game.
                window = value + 1
                for start in range(start_w, end_w - window + 2):
                    terms: List[object] = []
                    for week in range(start, start + window):
                        terms.extend(_team_vars(team, week, site="HOME"))
                    if terms:
                        model.Add(sum(terms) >= 1)
                    else:
                        return []

            elif rtype == "MUST_CAMPUS_HOME_WEEK" and team:
                week = start_w
                terms = _team_vars(team, week, site="HOME")
                if terms:
                    model.Add(sum(terms) >= 1)
                else:
                    return []

            elif rtype == "CANNOT_AWAY_WEEK" and team:
                week = start_w
                terms = _team_vars(team, week, site="AWAY")
                if terms:
                    model.Add(sum(terms) == 0)

            elif rtype == "PROTECT_BYE_WEEK" and team:
                week = start_w
                terms = _team_vars(team, week)
                if terms:
                    model.Add(sum(terms) == 0)

            elif rtype == "MAX_CONSECUTIVE_A4" and team:
                window = value + 1
                for start in range(start_w, end_w - window + 2):
                    terms: List[object] = []
                    for week in range(start, start + window):
                        terms.extend(_team_vars(team, week, a4_only=True))
                    if terms:
                        model.Add(sum(terms) <= value)

            elif rtype == "LOCK_GAME" and game_id:
                g = self.store.games.get(game_id)
                if g and (game_id, g.week) in x:
                    model.Add(x[(game_id, g.week)] == 1)

            elif rtype == "GAME_WEEK_WINDOW" and game_id:
                for week in candidate_weeks.get(game_id, []):
                    if week < start_w or week > end_w:
                        model.Add(x[(game_id, week)] == 0)

        # Specific requested move is a hard constraint.
        if target_game is not None:
            tw = int(intent.target_week)
            if (target_game.game_id, tw) not in x:
                return []
            model.Add(x[(target_game.game_id, tw)] == 1)

        base_bad = self._base_bad_parity(season)
        scope_conferences, scope_weeks = self._intent_scope(intent)
        scope_keys = {(c, w) for c in scope_conferences for w in scope_weeks}
        parity_bad: Dict[Tuple[str, int], object] = {}
        for conf in self.store.fbs_conferences():
            members = self.store.conference_members(conf)
            if not members:
                continue
            desired_remainder = len(members) % 2
            for week in range(0, 14):
                terms = []
                max_count = 0
                for game in season_games:
                    coeff = self._game_conference_coeff(game, conf)
                    if coeff and (game.game_id, week) in x:
                        terms.append(coeff * x[(game.game_id, week)])
                        max_count += coeff
                count = model.NewIntVar(0, max(len(members), max_count), f"nc_{conf}_{week}")
                model.Add(count == (sum(terms) if terms else 0))
                rem = model.NewIntVar(0, 1, f"rem_{conf}_{week}")
                model.AddModuloEquality(rem, count, 2)
                bad = model.NewBoolVar(f"bad_{conf}_{week}")
                if desired_remainder == 0:
                    model.Add(bad == rem)
                else:
                    model.Add(bad + rem == 1)
                parity_bad[(conf, week)] = bad

                # Do not turn a currently healthy conference/week into a new
                # parity problem unless the caller explicitly permits it.
                if intent.preserve_fbs_conference_parity and not base_bad.get((conf, week), False):
                    model.Add(bad == 0)

        if mode == "parity":
            if not intent.conference or intent.target_week is None:
                return []
            key = (intent.conference, int(intent.target_week))
            if key not in parity_bad:
                return []
            model.Add(parity_bad[key] == 0)

        # An explicitly scoped national parity request is a REQUIREMENT, not a
        # preference. Example: “make all conferences even in Weeks 0, 1 and 2.”
        # Every requested conference/week is therefore a hard constraint. The
        # objective below only decides which feasible solution moves the fewest
        # games and keeps those moves closest to their original dates.
        hard_national_parity_scope = (
            mode == "national"
            and bool(intent.target_weeks or intent.target_week is not None)
            and bool(intent.all_conferences or intent.conferences or intent.conference)
        )
        if hard_national_parity_scope:
            for key in sorted(scope_keys):
                if key in parity_bad:
                    model.Add(parity_bad[key] == 0)

        changed_vars = []
        changed_by_game: Dict[str, object] = {}
        distance_terms = []
        for game in season_games:
            current = x.get((game.game_id, game.week))
            changed = model.NewBoolVar(f"changed_{game.game_id}")
            if current is not None:
                model.Add(changed + current == 1)
            else:
                model.Add(changed == 1)
            changed_vars.append(changed)
            changed_by_game[game.game_id] = changed
            for week in candidate_weeks[game.game_id]:
                distance_terms.append(abs(week - game.week) * x[(game.game_id, week)])

        # Keep the repair neighborhood tight for interactive responsiveness.
        if mode == "move":
            model.Add(sum(changed_vars) <= max(1, int(intent.max_additional_moves) + 1))
        elif mode == "parity":
            model.Add(sum(changed_vars) <= max(2, int(intent.max_additional_moves) + 2))
        elif mode in {"fcs_balance", "controlled_balance"}:
            model.Add(sum(changed_vars) <= 18)
        elif mode == "national":
            # Explicit multi-conference parity requests may legitimately require
            # a larger repair chain. Keep a guardrail, but do not prematurely
            # force the solver to leave requested parity issues unresolved.
            model.Add(sum(changed_vars) <= (60 if hard_national_parity_scope else 30))

        # -----------------------------------------------------------------
        # Exact lexicographic optimization for the core repair workflows.
        #
        # Hard constraints above define what MUST happen.
        # Then we optimize in separate solves so a lower-priority preference
        # can never buy an extra game move:
        #   1. minimum games moved
        #   2. minimum displacement OR minimum disruption (strategy dependent)
        #   3. minimum disruption / displacement
        #   4. minimum human-preference penalty
        #   5. minimum unrelated parity damage for scoped conference repair
        # -----------------------------------------------------------------
        move_expr = sum(changed_vars)
        distance_expr = sum(distance_terms)

        disruption_terms = []
        for game in season_games:
            weight = self._game_disruption_weight(game)
            disruption_terms.append(weight * changed_by_game[game.game_id])
        disruption_expr = sum(disruption_terms)

        preference_terms = []
        if bool(intent.prefer_fcs_moves):
            for game in season_games:
                if not self._is_fbs_fcs(game):
                    preference_terms.append(8 * changed_by_game[game.game_id])

        if str(intent.a4_move_policy or "NORMAL").upper() == "PREFER_NOT":
            for game in season_games:
                if self._is_a4_matchup(game):
                    preference_terms.append(20 * changed_by_game[game.game_id])

        avoid_ids = set(intent.avoid_game_ids or [])
        for game_id in avoid_ids:
            if game_id in changed_by_game:
                preference_terms.append(30 * changed_by_game[game_id])

        # Generic preference rules are tie-breakers only.
        for rule in (intent.rules or []):
            if str(rule.get("hardness", "")).upper() != "PREFER" or not bool(rule.get("active", True)):
                continue
            rtype = str(rule.get("rule_type") or "").upper()
            game_id = str(rule.get("game_id") or "")
            weight = max(1, int(rule.get("weight", 10) or 10))
            team = str(rule.get("team") or "")
            week = max(0, min(13, int(rule.get("start_week", 0))))

            if rtype == "AVOID_MOVE_GAME" and game_id in changed_by_game:
                preference_terms.append(weight * changed_by_game[game_id])
            elif rtype == "PREFER_KEEP_BYE" and team:
                for g in season_games:
                    if g.involves(team) and (g.game_id, week) in x:
                        preference_terms.append(weight * x[(g.game_id, week)])
            elif rtype == "PREFER_CAMPUS_HOME_WEEK" and team:
                away_or_neutral = []
                for g in season_games:
                    if g.involves(team) and (g.game_id, week) in x and g.site_for(team) != "HOME":
                        away_or_neutral.append(x[(g.game_id, week)])
                if away_or_neutral:
                    preference_terms.append(weight * sum(away_or_neutral))

        preference_expr = sum(preference_terms) if preference_terms else 0
        outside_parity_expr = sum(
            v for key, v in parity_bad.items()
            if key not in scope_keys
        ) if parity_bad else 0

        # Balance modes remain a single specialized objective. The repair and
        # hard-scope conference modes use exact staged optimization below.
        balance_terms = []
        if mode == "fcs_balance":
            fcs_games = [g for g in season_games if self._is_fbs_fcs(g)]
            total = len(fcs_games)
            target = round(total / 14) if total else 0
            for week in range(0, 14):
                terms = [x[(g.game_id, week)] for g in fcs_games if (g.game_id, week) in x]
                count = model.NewIntVar(0, max(1, total), f"fcs_count_{week}")
                model.Add(count == (sum(terms) if terms else 0))
                dev = model.NewIntVar(0, max(1, total), f"fcs_dev_{week}")
                model.AddAbsEquality(dev, count - target)
                balance_terms.append(self.BALANCE_PENALTY * dev)

        if mode == "controlled_balance" and intent.conference:
            conf = intent.conference
            members = self.store.conference_members(conf)
            total_controlled = sum(self._game_conference_coeff(g, conf) for g in season_games)
            target = round(total_controlled / 14) if total_controlled else 0
            for week in range(0, 14):
                terms = []
                for game in season_games:
                    coeff = self._game_conference_coeff(game, conf)
                    if coeff and (game.game_id, week) in x:
                        terms.append(coeff * x[(game.game_id, week)])
                count = model.NewIntVar(0, max(len(members), total_controlled), f"controlled_{conf}_{week}")
                model.Add(count == (sum(terms) if terms else 0))
                dev = model.NewIntVar(0, max(len(members), total_controlled, 1), f"controlled_dev_{conf}_{week}")
                model.AddAbsEquality(dev, count - target)
                balance_terms.append(self.BALANCE_PENALTY * dev)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = 7
        solver.parameters.log_search_progress = False

        total_started = time.perf_counter()
        stage_records: List[Dict[str, object]] = []
        hint_values: Dict[Tuple[str, int], int] = {}

        def solve_stage(
            label: str,
            expr: object,
            lock_result: bool = True,
            time_fraction: float = 0.25,
        ) -> Tuple[int, Optional[int]]:
            self._clear_cp_objective(model)
            model.Minimize(expr)
            if hint_values:
                self._add_solution_hint(model, x, hint_values)
            solver.parameters.max_time_in_seconds = max(
                0.45, float(self.time_limit_seconds) * float(time_fraction)
            )
            stage_started = time.perf_counter()
            stage_status = solver.Solve(model)
            stage_seconds = time.perf_counter() - stage_started
            if stage_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                stage_records.append({
                    "stage": label,
                    "status": solver.StatusName(stage_status),
                    "seconds": round(stage_seconds, 3),
                    "value": None,
                    "proven": False,
                })
                return stage_status, None

            try:
                stage_value = int(round(solver.ObjectiveValue()))
            except Exception:
                stage_value = None

            stage_records.append({
                "stage": label,
                "status": solver.StatusName(stage_status),
                "seconds": round(stage_seconds, 3),
                "value": stage_value,
                "proven": stage_status == cp_model.OPTIMAL,
            })

            hint_values.clear()
            for key, var in x.items():
                hint_values[key] = int(solver.Value(var))

            if lock_result and stage_value is not None:
                model.Add(expr == stage_value)
            return stage_status, stage_value

        core_lexicographic = (
            mode in {"move", "parity"}
            or hard_national_parity_scope
        )

        if core_lexicographic:
            strategy = str(strategy or intent.optimization_strategy or "FEWEST_CHANGE").upper()
            stages: List[Tuple[str, object, float]] = [
                ("Minimum games moved", move_expr, 0.42),
            ]

            if strategy == "PROTECT_MARQUEE":
                stages.extend([
                    ("Minimum game disruption", disruption_expr, 0.22),
                    ("Minimum date displacement", distance_expr, 0.18),
                    ("Best human preference fit", preference_expr, 0.10),
                ])
            elif strategy == "COACH_FIT":
                stages.extend([
                    ("Best human preference fit", preference_expr, 0.22),
                    ("Minimum game disruption", disruption_expr, 0.18),
                    ("Minimum date displacement", distance_expr, 0.18),
                ])
            else:
                stages.extend([
                    ("Minimum date displacement", distance_expr, 0.22),
                    ("Minimum game disruption", disruption_expr, 0.18),
                    ("Best human preference fit", preference_expr, 0.10),
                ])

            if hard_national_parity_scope:
                stages.append(("Minimum outside-scope parity damage", outside_parity_expr, 0.08))

            status = cp_model.UNKNOWN
            for stage_label, stage_expr, fraction in stages:
                # A zero constant adds no information; skip the solve.
                if isinstance(stage_expr, int) and stage_expr == 0:
                    stage_records.append({
                        "stage": stage_label,
                        "status": "SKIPPED",
                        "seconds": 0.0,
                        "value": 0,
                        "proven": True,
                    })
                    continue
                status, _ = solve_stage(
                    stage_label,
                    stage_expr,
                    lock_result=True,
                    time_fraction=fraction,
                )
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    break
        else:
            # Legacy analytical modes can use a combined objective because they
            # are not administrator "must happen" repair transactions.
            objective_terms = []
            scoped_bad_vars = [v for k, v in parity_bad.items() if k in scope_keys]
            if mode == "national" and scoped_bad_vars:
                objective_terms.append((self.PARITY_PENALTY * 5) * sum(scoped_bad_vars))
            objective_terms.append(self.PARITY_PENALTY * sum(parity_bad.values()))
            objective_terms.append(self.MOVE_PENALTY * move_expr)
            objective_terms.append(self.DISTANCE_PENALTY * distance_expr)
            objective_terms.extend(balance_terms)
            objective_terms.append(preference_expr)

            self._clear_cp_objective(model)
            model.Minimize(sum(objective_terms))
            solver.parameters.max_time_in_seconds = self.time_limit_seconds
            stage_started = time.perf_counter()
            status = solver.Solve(model)
            stage_seconds = time.perf_counter() - stage_started
            stage_records.append({
                "stage": "Combined analytical objective",
                "status": solver.StatusName(status),
                "seconds": round(stage_seconds, 3),
                "value": int(round(solver.ObjectiveValue())) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
                "proven": status == cp_model.OPTIMAL,
            })

        self.last_solver_seconds = time.perf_counter() - total_started
        self.last_solver_status = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if hard_national_parity_scope:
                current_issues = self.parity_issue_details(
                    self.store.copy_games(), season, scope_weeks, scope_conferences
                )
                scope_label = f"{', '.join(scope_conferences)} · Weeks {', '.join(str(w) for w in scope_weeks)}"
                return [Solution(
                    title="Requested parity target is infeasible",
                    moves=[],
                    score=0.0,
                    warnings=[
                        "The solver could not make every requested conference/week even under the currently loaded availability, moveability, and one-game-per-team-per-week constraints."
                    ],
                    explanation=(
                        f"No feasible schedule was found for {scope_label}. Nothing was partially applied. "
                        "Review the remaining odd conference/week states below or relax one constraint and try again."
                    ),
                    metadata={
                        "mode": "national",
                        "solver_status": self.last_solver_status,
                        "solver_seconds": round(self.last_solver_seconds, 3),
                        "hard_scope": True,
                        "scope_before_bad": len(current_issues),
                        "scope_after_bad": len(current_issues),
                        "scope_conferences": scope_conferences,
                        "scope_weeks": scope_weeks,
                        "unresolved_issues": current_issues,
                        "infeasible": True,
                    },
                )]
            return []

        after_games = self.store.copy_games()
        moves: List[Move] = []
        for game in season_games:
            assigned = game.week
            for week in candidate_weeks[game.game_id]:
                if solver.Value(x[(game.game_id, week)]) == 1:
                    assigned = week
                    break
            if assigned != game.week:
                moves.append(Move(game.game_id, game.home_team, game.away_team, game.week, assigned))
                after_games[game.game_id] = replace(game, week=assigned)

        before_bad_count = sum(1 for v in base_bad.values() if v)
        after_bad_count = self.parity_violation_count(after_games, season, range(0, 14))
        scope_before_bad = self._scoped_bad_count(self.store.copy_games(), season, scope_conferences, scope_weeks)
        scope_after_bad = self._scoped_bad_count(after_games, season, scope_conferences, scope_weeks)
        explicit_weeks = set(scope_weeks if (intent.target_weeks or intent.target_week is not None) else [])
        touched = sorted(({m.from_week for m in moves} | {m.to_week for m in moves} | explicit_weeks))
        parity_before: Dict[str, str] = {}
        parity_after: Dict[str, str] = {}
        for week in touched:
            for conf, value in self.conference_parity(self.store.copy_games(), season, week).items():
                parity_before[f"{conf} W{week}"] = value
            for conf, value in self.conference_parity(after_games, season, week).items():
                parity_after[f"{conf} W{week}"] = value

        mode_label = {
            "move": "Requested move optimized",
            "parity": "Conference parity optimized",
            "national": "National schedule optimized",
            "fcs_balance": "FCS weekly distribution optimized",
            "controlled_balance": "Controlled-game distribution optimized",
        }.get(mode, "Schedule optimized")
        distance = sum(abs(m.to_week - m.from_week) for m in moves)
        warnings = []

        if mode == "move" and target_game is not None:
            requested_distance = abs(int(intent.target_week) - int(target_game.week))
            additional_moves = max(0, len(moves) - 1)
            cascade_distance = max(0, distance - requested_distance)
            score = max(0.0, min(100.0, 100.0 - 15.0 * additional_moves - 0.8 * cascade_distance))
            target_label = f"{target_game.away_team} @ {target_game.home_team}"
            if additional_moves == 0:
                explanation = (
                    f"Minimal-change solution with {self.engine_name}: move {target_label} from Week {target_game.week} "
                    f"to Week {int(intent.target_week)}. Both teams are available, so no other game needs to move. "
                    f"Unrelated national parity issues were intentionally left untouched. "
                    f"Solver status: {self.last_solver_status} in {self.last_solver_seconds:.2f}s."
                )
            else:
                explanation = (
                    f"Minimal-change solution with {self.engine_name}: move {target_label} from Week {target_game.week} "
                    f"to Week {int(intent.target_week)} and make {additional_moves} additional move(s) required to resolve "
                    f"a direct team/week or newly-created parity conflict. Unrelated schedule issues were left untouched. "
                    f"Solver status: {self.last_solver_status} in {self.last_solver_seconds:.2f}s."
                )
            lex_proven = bool(stage_records) and all(
                bool(r.get("proven", False))
                for r in stage_records
                if str(r.get("status")) != "SKIPPED"
            )
            if not lex_proven:
                warnings.append(
                    "The solver found the best lexicographic path within the interactive time budget; "
                    "one or more stages were not mathematically proven optimal."
                )
            result_title = "Minimal-change solution"
        else:
            score = max(0.0, min(100.0, 100.0 - 5.5 * len(moves) - 0.8 * distance - 4.0 * after_bad_count + 4.0 * max(0, before_bad_count - after_bad_count)))
            if after_bad_count:
                warnings.append(f"{after_bad_count} FBS conference/week parity issue(s) remain nationally after this optimization.")
            if status == cp_model.FEASIBLE:
                warnings.append("The solver found a high-quality feasible solution within the time limit; it did not prove global optimality.")
            if mode == "national":
                conf_scope = "all FBS conferences" if intent.all_conferences or (not intent.conference and not intent.conferences) else ", ".join(scope_conferences)
                week_scope = ", ".join(f"W{w}" for w in scope_weeks)
                scope_sentence = (
                    f"Within the requested scope ({conf_scope}; {week_scope}), odd conference/week slots changed "
                    f"from {scope_before_bad} to {scope_after_bad}. "
                )
            else:
                scope_sentence = ""
            explanation = (
                f"{mode_label} with {self.engine_name}. The solver evaluated the feasible game/week graph, "
                f"moved {len(moves)} game(s), and changed national parity issues from {before_bad_count} to {after_bad_count}. "
                f"{scope_sentence}Solver status: {self.last_solver_status} in {self.last_solver_seconds:.2f}s."
            )
            if mode == "national" and scope_after_bad > 0:
                warnings.append(
                    f"{scope_after_bad} requested conference/week parity issue(s) remain. This is the best solution found "
                    f"within the current move limits and public-data assumptions."
                )
            result_title = "Recommended optimization"
        return [Solution(
            title=result_title,
            moves=sorted(moves, key=lambda m: (m.from_week, m.home_team, m.away_team)),
            score=round(score, 1),
            parity_before=parity_before,
            parity_after=parity_after,
            warnings=warnings,
            explanation=explanation,
            metadata={
                "mode": mode,
                "solver_status": self.last_solver_status,
                "solver_seconds": round(self.last_solver_seconds, 3),
                "moves": len(moves),
                "additional_moves": max(0, len(moves) - 1) if mode == "move" else None,
                "before_bad_count": before_bad_count,
                "after_bad_count": after_bad_count,
                "scope_before_bad": scope_before_bad,
                "scope_after_bad": scope_after_bad,
                "scope_conferences": scope_conferences,
                "scope_weeks": scope_weeks,
                "status_is_optimal": status == cp_model.OPTIMAL,
                "hard_scope": bool(hard_national_parity_scope),
                "season": season,
                "unresolved_issues": self.parity_issue_details(after_games, season, range(0, 14)),
                "constraint_teams": list(intent.constraint_teams),
                "max_consecutive_away": intent.max_consecutive_away,
                "max_consecutive_home": intent.max_consecutive_home,
                "sequence_start_week": intent.sequence_start_week,
                "sequence_end_week": intent.sequence_end_week,
                "a4_move_policy": intent.a4_move_policy,
                "prefer_fcs_moves": bool(intent.prefer_fcs_moves),
                "avoid_game_ids": list(intent.avoid_game_ids),
                "coach_context": intent.coach_context,
                "optimization_strategy": str(strategy or intent.optimization_strategy or "FEWEST_CHANGE").upper(),
                "repair_depth": repair_depth,
                "lexicographic_stages": stage_records,
                "lexicographic_proven": bool(stage_records) and all(
                    bool(r.get("proven", False))
                    for r in stage_records
                    if str(r.get("status")) != "SKIPPED"
                ),
                "disruption_cost": sum(
                    self._game_disruption_weight(
                        next(g for g in season_games if g.game_id == m.game_id)
                    )
                    for m in moves
                ) if moves else 0,
            },
        )]

    def solve_move_game(self, intent: Intent) -> List[Solution]:
        """Return one fast primary answer. Alternatives are explicitly on-demand."""
        if not ORTOOLS_AVAILABLE:
            return super().solve_move_game(intent)

        for depth in [0, 1, 2, 3, None]:
            result = self._cp_optimize(
                intent,
                "move",
                repair_depth=depth,
                strategy="FEWEST_CHANGE",
            )
            if result:
                primary = result[0]
                primary.title = "Best path"
                primary.metadata["strategy_label"] = "Best path"
                primary.metadata["repair_depth"] = depth
                return [primary]

        advanced_hard_rules = bool(
            intent.constraint_teams
            or intent.max_consecutive_away is not None
            or intent.max_consecutive_home is not None
            or str(intent.a4_move_policy or "NORMAL").upper() == "NEVER"
            or any(
                str(r.get("hardness", "")).upper() in {"MUST", "CANNOT"}
                for r in (intent.rules or [])
            )
        )
        if advanced_hard_rules:
            return []
        return super().solve_move_game(intent)

    def solve_move_game_alternatives(
        self,
        intent: Intent,
        primary: Optional[Solution] = None,
    ) -> List[Solution]:
        """Run slower human-tradeoff strategies only when the user asks."""
        if not ORTOOLS_AVAILABLE:
            return []

        target_game = None
        if intent.team_a and intent.team_b and intent.season is not None:
            target_game = self.store.find_game(intent.team_a, intent.team_b, int(intent.season))
        if target_game is None:
            return []

        depth = None
        if primary is not None:
            depth = (primary.metadata or {}).get("repair_depth")
        candidates: List[Solution] = []

        for strategy, label in [
            ("PROTECT_MARQUEE", "Protect marquee games"),
            ("COACH_FIT", "Best coach-preference fit"),
        ]:
            result = self._cp_optimize(
                intent,
                "move",
                repair_depth=depth,
                strategy=strategy,
            )
            if result:
                sol = result[0]
                sol.title = label
                sol.metadata["strategy_label"] = label
                sol.metadata["repair_depth"] = depth
                candidates.append(sol)

        primary_sig = tuple(sorted(
            (m.game_id, int(m.from_week), int(m.to_week))
            for m in (primary.moves if primary else [])
        ))
        unique: Dict[Tuple[Tuple[str, int, int], ...], Solution] = {}
        for sol in candidates:
            sig = tuple(sorted(
                (m.game_id, int(m.from_week), int(m.to_week))
                for m in sol.moves
            ))
            if sig and sig != primary_sig and sig not in unique:
                unique[sig] = sol
        return list(unique.values())[:2]

    def solve_make_conference_even(self, intent: Intent) -> List[Solution]:
        if intent.season is None or intent.target_week is None or not intent.conference:
            return []
        current = self.conference_parity(self.store.copy_games(), int(intent.season), int(intent.target_week)).get(intent.conference, "")
        if current.startswith("EVEN"):
            return [Solution(
                title="No change required",
                moves=[],
                score=100.0,
                explanation=f"{intent.conference} is already even in Week {intent.target_week}: {current}.",
            )]

        # Odd/even requests are intentionally simple: first ask whether exactly
        # ONE game can solve the selected conference/week. Do not launch the
        # national CP-SAT repair merely because other parity issues exist.
        simple = NonConferenceOptimizer.solve_make_conference_even(self, intent)
        one_move = [sol for sol in simple if len(sol.moves) == 1]
        if one_move:
            self.last_solver_status = "Single-move search"
            self.last_solver_seconds = 0.0
            return sorted(one_move, key=lambda s: (-s.score, abs(s.moves[0].to_week - s.moves[0].from_week)))[:5]

        # Only escalate when the direct one-game search has no solution.
        if ORTOOLS_AVAILABLE:
            result = self._cp_optimize(intent, "parity")
            if result:
                return result
        return simple

    def optimize_national(self, intent: Intent) -> List[Solution]:
        if ORTOOLS_AVAILABLE:
            result = self._cp_optimize(intent, "national")
            if result:
                return result
        # Fallback: surface current national health rather than silently failing.
        if intent.season is None:
            return []
        bad = self.parity_violation_count(self.store.copy_games(), int(intent.season), range(0, 14))
        return [Solution(
            title="National optimization requires OR-Tools",
            moves=[], score=max(0, 100 - bad * 5),
            warnings=["OR-Tools is not installed in this runtime. Add ortools to requirements.txt for CP-SAT optimization."],
            explanation=f"The current schedule has {bad} FBS conference/week parity issue(s).",
        )]

    def balance_fcs_games(self, intent: Intent) -> List[Solution]:
        if ORTOOLS_AVAILABLE:
            result = self._cp_optimize(intent, "fcs_balance")
            if result:
                return result
        return self.optimize_national(intent)

    def balance_controlled_games(self, intent: Intent) -> List[Solution]:
        if not intent.conference:
            return []
        if ORTOOLS_AVAILABLE:
            result = self._cp_optimize(intent, "controlled_balance")
            if result:
                return result
        return self.optimize_national(intent)

    def optimize_market(self, intent: Intent) -> List[Solution]:
        """Maximum matching for explicit scheduling-market needs.

        Public FBSchedules data does not contain actual buy/sell intent, so real
        production market optimization requires the authoritative needs table. The
        demo store exercises this path with explicit needs.
        """
        if intent.season is None or not self.store.needs:
            return []
        season = int(intent.season)
        needs = [n for n in self.store.needs if n.season == season]
        if not needs:
            return []

        # This market model is deliberately separate from schedule relocation:
        # it maximizes fulfilled explicit needs while respecting known games.
        if not ORTOOLS_AVAILABLE:
            return []
        model = cp_model.CpModel()
        vars_by_need: Dict[int, List[Tuple[object, Team, Need]]] = {}
        base_games = self.store.copy_games()
        for i, need in enumerate(needs):
            requester = self.store.teams.get(need.team)
            if requester is None:
                continue
            wanted = None
            if need.need_type == "FCS_BUY":
                wanted = "FBS" if requester.subdivision == "FCS" else "FCS"
            for candidate in self.store.teams.values():
                if candidate.name == requester.name:
                    continue
                if wanted and candidate.subdivision != wanted:
                    continue
                if need.need_type == "A4" and (not candidate.is_a4 or not requester.is_a4 or candidate.conference == requester.conference):
                    continue
                if self.store.game_for_team_week(base_games, requester.name, season, need.week):
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, season, need.week):
                    continue
                var = model.NewBoolVar(f"need{i}_{candidate.name}_{need.week}")
                vars_by_need.setdefault(i, []).append((var, candidate, need))
        for choices in vars_by_need.values():
            model.Add(sum(v for v, _, _ in choices) <= 1)
        # A candidate can only accept one newly proposed game in a week.
        cand_week: Dict[Tuple[str, int], List[object]] = {}
        for choices in vars_by_need.values():
            for var, candidate, need in choices:
                cand_week.setdefault((candidate.name, need.week), []).append(var)
        for vars_list in cand_week.values():
            model.Add(sum(vars_list) <= 1)
        model.Maximize(sum(v for choices in vars_by_need.values() for v, _, _ in choices))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = min(2.0, self.time_limit_seconds)
        solver.parameters.num_search_workers = 8
        started = time.perf_counter()
        status = solver.Solve(model)
        self.last_solver_seconds = time.perf_counter() - started
        self.last_solver_status = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []
        selected = []
        for i, choices in vars_by_need.items():
            requester = self.store.teams.get(needs[i].team) if i < len(needs) else None
            for var, candidate, need in choices:
                if solver.Value(var) == 1 and requester:
                    selected.append((requester, candidate, need))
        if not selected:
            return []
        lines = [f"{r.name} ↔ {c.name} · Week {n.week} · {n.need_type}" for r, c, n in selected]
        return [Solution(
            title=f"{len(selected)} market need{'s' if len(selected) != 1 else ''} matched",
            moves=[], score=min(100.0, 70.0 + len(selected) * 4),
            explanation="Explicit scheduling needs matched with CP-SAT: " + "; ".join(lines),
        )]



INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["MOVE_GAME", "MAKE_CONFERENCE_EVEN", "FIND_BUY_GAME", "FIND_A4_GAME", "OPTIMIZE_NATIONAL", "BALANCE_FCS_GAMES", "BALANCE_CONTROLLED_GAMES", "OPTIMIZE_MARKET"]},
        "season": {"type": ["integer", "null"]},
        "target_week": {"type": ["integer", "null"]},
        "conference": {"type": ["string", "null"]},
        "target_weeks": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 13}, "maxItems": 14},
        "conferences": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "all_conferences": {"type": "boolean"},
        "team_a": {"type": ["string", "null"]},
        "team_b": {"type": ["string", "null"]},
        "preserve_fbs_conference_parity": {"type": "boolean"},
        "max_additional_moves": {"type": "integer", "minimum": 0, "maximum": 6},
        "opponent_class": {"type": "string", "enum": ["ANY", "FBS", "FCS", "A4"]},
        "location": {"type": "string", "enum": ["ANY", "HOME", "AWAY"]},
        "max_guarantee": {"type": ["integer", "null"]},
        "summary": {"type": "string"}
    },
    "required": ["action", "season", "target_week", "conference", "target_weeks", "conferences", "all_conferences", "team_a", "team_b", "preserve_fbs_conference_parity", "max_additional_moves", "opponent_class", "location", "max_guarantee", "summary"],
    "additionalProperties": False
}


SYSTEM_INSTRUCTIONS = """You interpret requests for a college-football NON-CONFERENCE scheduling optimizer.
Do not solve the schedule yourself. Convert the user's request into the provided structured intent.
Definitions:
- MOVE_GAME: user names a specific existing matchup and wants it moved to a week.
- MAKE_CONFERENCE_EVEN: user primarily wants an FBS conference to have an even number of teams available for conference play in a week, without requiring a named specific game.
- FIND_BUY_GAME: a school is looking for a buy/guarantee game. For an FBS requester, look for FCS candidates. For an FCS requester, look for potential FBS guarantee-game hosts. target_week may be null when the user asks for options across an entire season.
- FIND_A4_GAME: an A4 school needs an A4 nonconference opponent.
- OPTIMIZE_NATIONAL: user asks to optimize a whole season or solve the biggest national scheduling problems.
  Also use OPTIMIZE_NATIONAL when the user asks to solve parity across multiple conferences or multiple weeks, e.g. "solve all conferences' odd problems in Weeks 1, 2 and 3."
- BALANCE_FCS_GAMES: user wants to improve or balance the number of FBS-vs-FCS games by week.
- BALANCE_CONTROLLED_GAMES: user wants to balance a conference's weekly non-conference/controlled-game inventory.
- OPTIMIZE_MARKET: user asks to optimize or match the overall market / teams-needing-games report.
For a request like 'The SEC is odd in week 2 and I need to move Georgia vs McNeese to week 2 ...', use MOVE_GAME, conference SEC, team_a Georgia, team_b McNeese, target_week 2. Set preserve_fbs_conference_parity true only when the user explicitly asks to avoid creating a new parity problem, keep a conference even, or otherwise preserve parity. For a simple direct move with no parity instruction, set it false so unrelated conference issues are not optimized or repaired.
For every explicitly named week, populate target_weeks. If exactly one week is named, also populate target_week; if multiple weeks are named, target_week should be null.
For every explicitly named conference, populate conferences. If exactly one conference is named, also populate conference. If the user says all/every conferences, set all_conferences true, conferences empty, conference null.
If the year is omitted, season should be null rather than guessed. Never invent teams, dates, guarantee amounts, or constraints.
"""


def _normalize_team(name: str | None, team_names: Iterable[str]) -> str | None:
    if not name:
        return None
    names = list(team_names)
    exact = {n.lower(): n for n in names}
    if name.lower() in exact:
        return exact[name.lower()]
    match = get_close_matches(name, names, n=1, cutoff=0.6)
    return match[0] if match else name


def parse_with_openai(text: str, team_names: Iterable[str]) -> Intent:
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6")
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=text,
        text={
            "format": {
                "type": "json_schema",
                "name": "cfb_nonc_schedule_intent",
                "strict": True,
                "schema": INTENT_SCHEMA,
            }
        },
    )
    data = json.loads(response.output_text)
    data["team_a"] = _normalize_team(data.get("team_a"), team_names)
    data["team_b"] = _normalize_team(data.get("team_b"), team_names)
    data["target_weeks"] = sorted(set(int(w) for w in (data.get("target_weeks") or []) if 0 <= int(w) <= 13))
    if len(data["target_weeks"]) == 1 and data.get("target_week") is None:
        data["target_week"] = data["target_weeks"][0]
    data["conferences"] = list(dict.fromkeys(data.get("conferences") or []))
    if len(data["conferences"]) == 1 and data.get("conference") is None:
        data["conference"] = data["conferences"][0]
    return Intent(**data)


def parse_locally(text: str, team_names: Iterable[str]) -> Intent:
    """Small offline parser so the demo still works without an API key."""
    lower = text.lower()
    year_match = re.search(r"\b(20\d{2})\b", text)
    season = int(year_match.group(1)) if year_match else None

    # Capture both singular and list forms: "Week 2" and "Weeks 1, 2 and 3".
    target_weeks: List[int] = []
    plural = re.search(r"\bweeks?\s+((?:\d{1,2}\s*(?:(?:,|and|&)\s*)?)+)", lower)
    if plural:
        target_weeks.extend(int(v) for v in re.findall(r"\d{1,2}", plural.group(1)))
    target_weeks.extend(int(m.group(1)) for m in re.finditer(r"\bweek\s*(\d{1,2})\b", lower))
    target_weeks = sorted(set(w for w in target_weeks if 0 <= w <= 13))
    week = target_weeks[0] if len(target_weeks) == 1 else None

    known_confs = ["SEC", "ACC", "Big Ten", "Big 12", "AAC", "Mountain West", "Sun Belt", "Conference USA", "MAC", "Pac-12"]
    conferences = [conf for conf in known_confs if conf.lower() in lower]
    all_conferences = bool(re.search(r"\b(?:all|every)\s+(?:fbs\s+)?conferences?\b|\bnational\s+parity\b", lower))
    conference = conferences[0] if len(conferences) == 1 else None

    found = []
    for name in sorted(team_names, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.lower())}\b", lower):
            found.append(name)
    # Preserve mention order when possible.
    found.sort(key=lambda n: lower.find(n.lower()))

    guarantee = None
    money = re.search(r"(?:\$\s*)?([0-9]+(?:\.[0-9]+)?)\s*(million|m|k|thousand)?", lower)
    if "guarantee" in lower or "less than" in lower or "under" in lower:
        if money:
            value = float(money.group(1))
            suffix = (money.group(2) or "").lower()
            if suffix in {"million", "m"}:
                value *= 1_000_000
            elif suffix in {"k", "thousand"}:
                value *= 1_000
            guarantee = int(value)

    buy_request = bool(re.search(r"\bbuy(?:\s+(?:a|an))?\s+game\b|\bbuy-game\b|\bguarantee(?:\s+game)?\b", lower))
    multi_parity_request = (
        (all_conferences or len(target_weeks) > 1 or len(conferences) > 1)
        and bool(re.search(r"\b(?:odd|even|parity|solve|fix|optimi[sz]e)\b", lower))
    )
    if multi_parity_request:
        action = "OPTIMIZE_NATIONAL"
        team_a, team_b = None, None
        opponent_class = "ANY"
    elif ("optimize" in lower or "solve the season" in lower) and not found and not conference:
        action = "OPTIMIZE_NATIONAL"
        team_a, team_b = None, None
        opponent_class = "ANY"
    elif "market report" in lower or "optimize market" in lower or "teams needing games" in lower:
        action = "OPTIMIZE_MARKET"
        team_a, team_b = (found + [None, None])[:2]
        opponent_class = "ANY"
    elif ("fcs" in lower and ("per week" in lower or "by week" in lower or "balance" in lower)) and not buy_request:
        action = "BALANCE_FCS_GAMES"
        team_a, team_b = (found + [None, None])[:2]
        opponent_class = "FCS"
    elif "controlled game" in lower or "controlled games" in lower:
        action = "BALANCE_CONTROLLED_GAMES"
        team_a, team_b = (found + [None, None])[:2]
        opponent_class = "ANY"
    elif ("fcs" in lower or buy_request) and found:
        action = "FIND_BUY_GAME"
        team_a, team_b = found[0], None
        opponent_class = "FCS"
    elif "a4" in lower or "autonomy" in lower:
        action = "FIND_A4_GAME"
        team_a, team_b = (found + [None, None])[:2]
        opponent_class = "A4"
    elif len(found) >= 2 and ("move" in lower or "put" in lower):
        action = "MOVE_GAME"
        team_a, team_b = found[0], found[1]
        opponent_class = "ANY"
    else:
        action = "MAKE_CONFERENCE_EVEN"
        team_a, team_b = (found + [None, None])[:2]
        opponent_class = "ANY"

    explicit_parity_protection = bool(re.search(
        r"(?:without|avoid|don't|do not|preserve|keep).{0,40}(?:parity|odd|even)|(?:parity|odd|even).{0,40}(?:problem|issue|preserve|keep)",
        lower,
    ))

    return Intent(
        action=action,
        season=season,
        target_week=week,
        conference=conference,
        target_weeks=target_weeks,
        conferences=conferences,
        all_conferences=all_conferences,
        team_a=team_a,
        team_b=team_b,
        preserve_fbs_conference_parity=(explicit_parity_protection if action == "MOVE_GAME" else True),
        max_additional_moves=4,
        opponent_class=opponent_class,
        location="HOME" if "home" in lower else "ANY",
        max_guarantee=guarantee,
        summary=text,
    )


def parse_intent(text: str, team_names: Iterable[str]) -> tuple[Intent, str]:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return parse_with_openai(text, team_names), "OpenAI structured intent parser"
        except Exception as exc:
            intent = parse_locally(text, team_names)
            return intent, f"Local fallback (LLM error: {type(exc).__name__})"
    return parse_locally(text, team_names), "Local fallback parser (set OPENAI_API_KEY to enable LLM parsing)"

TEAMS_CSV = r"""name,subdivision,conference,is_a4,parity_managed
Alabama,FBS,SEC,true,true
Arkansas,FBS,SEC,true,true
Auburn,FBS,SEC,true,true
Florida,FBS,SEC,true,true
Georgia,FBS,SEC,true,true
Kentucky,FBS,SEC,true,true
LSU,FBS,SEC,true,true
Ole Miss,FBS,SEC,true,true
Mississippi State,FBS,SEC,true,true
Missouri,FBS,SEC,true,true
Oklahoma,FBS,SEC,true,true
South Carolina,FBS,SEC,true,true
Tennessee,FBS,SEC,true,true
Texas,FBS,SEC,true,true
Texas A&M,FBS,SEC,true,true
Vanderbilt,FBS,SEC,true,true
Virginia Tech,FBS,ACC,true,false
NC State,FBS,ACC,true,false
Kansas State,FBS,Big 12,true,false
Iowa State,FBS,Big 12,true,false
McNeese,FCS,Southland,false,false
Tarleton,FCS,UAC,false,false
Chattanooga,FCS,SoCon,false,false
Samford,FCS,SoCon,false,false
Florida A&M,FCS,SWAC,false,false
ETSU,FCS,SoCon,false,false
UT Martin,FCS,OVC-Big South,false,false
Abilene Christian,FCS,UAC,false,false
Northern Alabama,FCS,UAC,false,false
UTEP,FBS,Conference USA,false,false"""
GAMES_CSV = r"""game_id,season,week,home_team,away_team,moveable,locked,notes
g1,2027,3,Georgia,McNeese,true,false,Target demonstration game
g2,2027,2,McNeese,Tarleton,true,false,Displaced if Georgia-McNeese moves to Week 2
g3,2027,2,Alabama,Chattanooga,true,false,
g4,2027,2,Auburn,Samford,true,false,
g5,2027,2,Florida,Florida A&M,true,false,
g6,2027,2,Tennessee,ETSU,true,false,
g7,2027,2,Texas,UTEP,true,false,
g8,2027,3,Kentucky,UT Martin,true,false,
g9,2027,3,LSU,Samford,true,false,
g10,2027,3,Ole Miss,Chattanooga,true,false,
g11,2027,3,Vanderbilt,ETSU,true,false,
g12,2027,4,Tarleton,Abilene Christian,true,false,Alternate cascade test
g13,2027,5,Virginia Tech,UT Martin,true,false,
g14,2027,6,Kansas State,Northern Alabama,true,false,"""
SLOTS_CSV = r"""team,season,week,status,location
Georgia,2027,2,OPEN,HOME
Georgia,2027,3,FLEX,HOME
Georgia,2027,4,BLOCKED,ANY
McNeese,2027,2,FLEX,AWAY
McNeese,2027,3,FLEX,AWAY
McNeese,2027,4,OPEN,HOME
McNeese,2027,5,OPEN,HOME
Tarleton,2027,2,FLEX,AWAY
Tarleton,2027,3,OPEN,AWAY
Tarleton,2027,4,FLEX,HOME
Tarleton,2027,5,OPEN,AWAY
Abilene Christian,2027,3,OPEN,AWAY
Abilene Christian,2027,4,FLEX,AWAY
Abilene Christian,2027,5,OPEN,AWAY
Alabama,2027,2,FLEX,HOME
Alabama,2027,3,OPEN,HOME
Auburn,2027,2,FLEX,HOME
Auburn,2027,3,OPEN,HOME
Florida,2027,2,FLEX,HOME
Florida,2027,3,OPEN,HOME
Tennessee,2027,2,FLEX,HOME
Tennessee,2027,3,OPEN,HOME
Texas,2027,2,FLEX,HOME
Texas,2027,3,OPEN,HOME
Kentucky,2027,2,OPEN,HOME
Kentucky,2027,3,FLEX,HOME
LSU,2027,2,OPEN,HOME
LSU,2027,3,FLEX,HOME
Ole Miss,2027,2,OPEN,HOME
Ole Miss,2027,3,FLEX,HOME
Vanderbilt,2027,2,OPEN,HOME
Vanderbilt,2027,3,FLEX,HOME
Virginia Tech,2027,2,NEED_A4,ANY
Virginia Tech,2027,3,OPEN,ANY
Virginia Tech,2027,5,FLEX,HOME
NC State,2027,2,NEED_A4,ANY
Kansas State,2027,2,NEED_A4,ANY
Iowa State,2027,2,NEED_A4,ANY
UT Martin,2027,2,OPEN,AWAY
UT Martin,2027,3,FLEX,AWAY
UT Martin,2027,5,FLEX,AWAY
Northern Alabama,2027,2,OPEN,AWAY
Northern Alabama,2027,6,FLEX,AWAY
Chattanooga,2027,2,FLEX,AWAY
Chattanooga,2027,3,FLEX,AWAY
Samford,2027,2,FLEX,AWAY
Samford,2027,3,FLEX,AWAY
Florida A&M,2027,2,FLEX,AWAY
ETSU,2027,2,FLEX,AWAY
ETSU,2027,3,FLEX,AWAY
UTEP,2027,2,FLEX,AWAY"""
NEEDS_CSV = r"""team,season,week,need_type,location,min_guarantee,max_guarantee,notes
UT Martin,2027,2,FCS_BUY,AWAY,550000,,Seeking FBS guarantee game
Northern Alabama,2027,2,FCS_BUY,AWAY,500000,,Seeking FBS guarantee game
Virginia Tech,2027,2,A4,ANY,,,Needs A4 opponent
NC State,2027,2,A4,ANY,,,Needs A4 opponent
Kansas State,2027,2,A4,ANY,,,Needs A4 opponent
Iowa State,2027,2,A4,ANY,,,Needs A4 opponent"""

def _rows(text: str):
    return list(csv.DictReader(io.StringIO(text)))

def _bool(v: str, default=False):
    if v is None or v == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}

def build_demo_store() -> ScheduleStore:
    teams = [Team(
        name=r["name"].strip(),
        subdivision=r["subdivision"].strip().upper(),
        conference=r["conference"].strip(),
        is_a4=_bool(r.get("is_a4")),
        parity_managed=_bool(r.get("parity_managed"), True),
    ) for r in _rows(TEAMS_CSV)]
    games = [Game(
        game_id=r["game_id"].strip(), season=int(r["season"]), week=int(r["week"]),
        home_team=r["home_team"].strip(), away_team=r["away_team"].strip(),
        moveable=_bool(r.get("moveable"), True), locked=_bool(r.get("locked"), False),
        notes=r.get("notes", "").strip(),
    ) for r in _rows(GAMES_CSV)]
    slots = [Slot(
        team=r["team"].strip(), season=int(r["season"]), week=int(r["week"]),
        status=r["status"].strip().upper(), location=(r.get("location") or "ANY").strip().upper(),
    ) for r in _rows(SLOTS_CSV)]
    needs = [Need(
        team=r["team"].strip(), season=int(r["season"]), week=int(r["week"]),
        need_type=r["need_type"].strip().upper(), location=(r.get("location") or "ANY").strip().upper(),
        min_guarantee=int(r["min_guarantee"]) if (r.get("min_guarantee") or "").strip() else None,
        max_guarantee=int(r["max_guarantee"]) if (r.get("max_guarantee") or "").strip() else None,
        notes=r.get("notes", "").strip(),
    ) for r in _rows(NEEDS_CSV)]
    return ScheduleStore(teams, games, slots, needs)



FBSCHEDULES_DIRECTORY = "https://fbschedules.com/future-college-football-schedules/"
FBS_CONFERENCES = {
    "ACC", "American", "AAC", "Big 12", "Big Ten", "Conference USA", "FBS Independent",
    "MAC", "Mountain West", "Pac-12", "SEC", "Sun Belt"
}
A4_CONFERENCES = {"ACC", "Big 12", "Big Ten", "SEC"}
CONFERENCE_NORMALIZATION = {
    "American": "AAC",
    "FBS Independent": "FBS Independent",
    "FCS Independent": "FCS Independent",
    "OVC-Big South": "OVC",
    "OVC-Big South Football Association": "OVC",
}


def _normalize_conference_name(value: str) -> str:
    value = (value or "").strip()
    return CONFERENCE_NORMALIZATION.get(value, value)


def _safe_get(url: str, timeout: int = 20, attempts: int = 3) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPad; CPU OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
            "Version/18.0 Mobile/15E148 Safari/604.1 CFBNonConferenceOptimizer/0.2"
        )
    }
    last_exc = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_exc}")


def _school_name_from_soup(soup: BeautifulSoup, url: str) -> str:
    h1 = soup.find("h1")
    if h1:
        value = h1.get_text(" ", strip=True)
        value = re.sub(r"\s+Football Schedule.*$", "", value, flags=re.I).strip()
        if value:
            return value
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.sub(r"\s+Football Schedule.*$", "", title, flags=re.I).strip()
    if title:
        return title
    return urlparse(url).path.strip("/").split("/")[-1].replace("-", " ").title()


def _conference_from_soup(soup: BeautifulSoup) -> str:
    for label in soup.find_all(["strong", "b"]):
        if "conference" in label.get_text(" ", strip=True).lower():
            parent = label.parent
            if parent:
                link = parent.find("a")
                if link:
                    return _normalize_conference_name(link.get_text(" ", strip=True))
                txt = parent.get_text(" ", strip=True)
                m = re.search(r"Conference:\s*([^|]+?)(?:\s+202\d|$)", txt, flags=re.I)
                if m:
                    return _normalize_conference_name(m.group(1).strip())
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Conference:\s*(ACC|American|Big 12|Big Ten|Conference USA|FBS Independent|MAC|Mountain West|Pac-12|SEC|Sun Belt|Big Sky|CAA|FCS Independent|Ivy League|MEAC|Missouri Valley|NEC|OVC(?:-Big South)?|Patriot League|Pioneer League|SoCon|Southland|SWAC|UAC)", text, flags=re.I)
    if m:
        return _normalize_conference_name(m.group(1))
    return "Unknown"



def _labor_day(year: int) -> date:
    d = date(year, 9, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _week_for_season(date_iso: str | None, season: int) -> int | None:
    """Map a date to college-football Week 0..13 using Labor Day weekend as Week 1."""
    if not date_iso:
        return None
    d = datetime.strptime(date_iso, "%Y-%m-%d").date()
    week1_sat = _labor_day(season) - timedelta(days=2)
    week0_sat = week1_sat - timedelta(days=7)
    days_until_sat = (5 - d.weekday()) % 7
    week_sat = d + timedelta(days=days_until_sat)
    return (week_sat - week0_sat).days // 7


def _week_saturday(season: int, week: int) -> date:
    week1_sat = _labor_day(season) - timedelta(days=2)
    week0_sat = week1_sat - timedelta(days=7)
    return week0_sat + timedelta(days=7 * week)


# Product convention: users see Weeks 1–14. The source parser/solver keeps
# a zero-based 0–13 index internally so we do not have to rewrite the data layer.
def _display_week(week: int) -> int:
    return int(week) + 1


def _internal_week(display_week: int) -> int:
    return int(display_week) - 1


def _week_label(week: int) -> str:
    return f"Week {_display_week(week)}"


def _display_text_weeks(value: object) -> str:
    text = str(value if value is not None else "")
    text = re.sub(r"\bWeek\s+(\d{1,2})\b", lambda m: f"Week {int(m.group(1)) + 1}", text)
    text = re.sub(r"\bW(\d{1,2})\b", lambda m: f"W{int(m.group(1)) + 1}", text)
    return text


def _display_parity_key(value: object) -> str:
    text = str(value if value is not None else "")
    m = re.match(r"^(.*?)\s+W(\d+)$", text)
    if m:
        return f"{m.group(1)} · Week {int(m.group(2)) + 1}"
    return _display_text_weeks(text)


def _school_logo_from_soup(soup: BeautifulSoup, school_name: str, source_url: str) -> str:
    """Best-effort public school image/logo URL from the team page."""
    candidates = []
    school_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", school_name) if len(t) > 2}
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
        if not src or src.startswith("data:"):
            continue
        alt = (img.get("alt") or "").lower()
        classes = " ".join(img.get("class") or []).lower()
        score = 0
        if "football schedule" in alt:
            score += 5
        if school_name.lower() in alt:
            score += 5
        score += sum(1 for token in school_tokens if token in alt)
        if any(k in classes for k in ("team", "logo", "school")):
            score += 2
        if score:
            candidates.append((score, urljoin(source_url, src)))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        return urljoin(source_url, og["content"])
    return ""


def _parse_entries_for_years(soup: BeautifulSoup, current_slug: str, current_name: str, source_url: str, years: Iterable[int]) -> list[dict]:
    rows: list[dict] = []
    for season in sorted({int(y) for y in years}):
        heading = None
        for tag in soup.find_all(["h4", "h3"]):
            if tag.get_text(" ", strip=True) == str(season):
                heading = tag
                break
        if not heading:
            continue
        ul = None
        sibling = heading.next_sibling
        while sibling is not None:
            name = getattr(sibling, "name", None)
            if name in {"h3", "h4"}:
                break
            if name == "ul":
                ul = sibling
                break
            sibling = sibling.next_sibling
        if ul is None:
            ul = heading.find_next("ul")
        if ul is None:
            continue
        for li in ul.find_all("li", recursive=False):
            txt = " ".join(li.get_text(" ", strip=True).split())
            m = re.match(r"^(TBA|\d{2}/\d{2})\s*-\s*(.+)$", txt, flags=re.I)
            if not m:
                continue
            date_token, rest = m.group(1).upper(), m.group(2).strip()
            date_iso = None
            if date_token != "TBA":
                try:
                    date_iso = datetime.strptime(f"{season}/{date_token}", "%Y/%m/%d").date().isoformat()
                except ValueError:
                    date_iso = None
            site = "Home"
            rest_lower = rest.lower()
            if rest_lower.startswith("at "):
                site = "Away"
            elif rest_lower.startswith("vs "):
                site = "Neutral"
            opp_link = None
            for a in li.find_all("a", href=True):
                if "/ncaa/" in a["href"]:
                    opp_link = a
                    break
            opponent_name = opp_link.get_text(" ", strip=True) if opp_link else ""
            opponent_url = urljoin(source_url, opp_link["href"]) if opp_link else ""
            opponent_slug = urlparse(opponent_url).path.strip("/").split("/")[-1] if opponent_url else ""
            if not opponent_name:
                cleaned = re.sub(r"^(?:at|vs)\s+", "", rest, flags=re.I)
                cleaned = re.sub(r"\s*\(in .+\)\s*$", "", cleaned).strip()
                opponent_name = cleaned
            neutral_location = ""
            n = re.search(r"\(in\s+(.+?)\)\s*$", rest, flags=re.I)
            if n:
                neutral_location = n.group(1).strip()
            rows.append({
                "season": season,
                "date": date_iso or "TBA",
                "week": _week_for_season(date_iso, season),
                "current_slug": current_slug,
                "team": current_name,
                "opponent_slug": opponent_slug,
                "opponent": opponent_name,
                "site_for_team": site,
                "neutral_location": neutral_location,
                "source_url": source_url,
            })
    return rows


def _scrape_one_team(url: str, years: tuple[int, ...]) -> tuple[dict, list[dict]]:
    html = _safe_get(url)
    soup = BeautifulSoup(html, "html.parser")
    slug = urlparse(url).path.strip("/").split("/")[-1]
    name = _school_name_from_soup(soup, url)
    conference = _conference_from_soup(soup)
    subdivision = "FBS" if conference in {_normalize_conference_name(c) for c in FBS_CONFERENCES} else "FCS"
    team = {
        "slug": slug,
        "name": name,
        "subdivision": subdivision,
        "conference": conference,
        "is_a4": conference in A4_CONFERENCES,
        "parity_managed": subdivision == "FBS" and conference != "FBS Independent",
        "source_url": url,
        "logo_url": _school_logo_from_soup(soup, name, url),
    }
    return team, _parse_entries_for_years(soup, slug, name, url, years)


def _dedupe_scraped_games(raw_rows: list[dict], team_by_slug: dict[str, dict]) -> pd.DataFrame:
    known_dates: dict[tuple[int, str, str], set[str]] = {}
    for r in raw_rows:
        a = r["current_slug"] or r["team"].lower()
        b = r["opponent_slug"] or r["opponent"].lower()
        pair = tuple(sorted((a, b)))
        if r["date"] != "TBA":
            known_dates.setdefault((int(r["season"]), *pair), set()).add(r["date"])
    games: dict[tuple, dict] = {}
    for r in raw_rows:
        season = int(r["season"])
        a_key = r["current_slug"] or r["team"].lower()
        b_key = r["opponent_slug"] or r["opponent"].lower()
        pair = tuple(sorted((a_key, b_key)))
        event_date = r["date"]
        dates = known_dates.get((season, *pair), set())
        if event_date == "TBA" and len(dates) == 1:
            event_date = next(iter(dates))
        key = (season, pair, event_date)
        current_meta = team_by_slug.get(r["current_slug"], {})
        opp_meta = team_by_slug.get(r["opponent_slug"], {})
        current_name = current_meta.get("name", r["team"])
        opp_name = opp_meta.get("name", r["opponent"])
        if r["site_for_team"] == "Away":
            home, away = opp_name, current_name
            home_meta, away_meta = opp_meta, current_meta
            neutral = False
        elif r["site_for_team"] == "Neutral":
            home, away = current_name, opp_name
            home_meta, away_meta = current_meta, opp_meta
            neutral = True
        else:
            home, away = current_name, opp_name
            home_meta, away_meta = current_meta, opp_meta
            neutral = False
        if key not in games:
            games[key] = {
                "season": season,
                "date": event_date,
                "week": _week_for_season(event_date if event_date != "TBA" else None, season),
                "home_team": home,
                "away_team": away,
                "neutral": neutral,
                "neutral_location": r["neutral_location"],
                "home_subdivision": home_meta.get("subdivision", "Unknown"),
                "away_subdivision": away_meta.get("subdivision", "Unknown"),
                "home_conference": home_meta.get("conference", "Unknown"),
                "away_conference": away_meta.get("conference", "Unknown"),
                "home_logo": home_meta.get("logo_url", ""),
                "away_logo": away_meta.get("logo_url", ""),
                "source_urls": {r["source_url"]},
            }
        else:
            games[key]["source_urls"].add(r["source_url"])
            if r["neutral_location"] and not games[key]["neutral_location"]:
                games[key]["neutral_location"] = r["neutral_location"]
            if r["site_for_team"] == "Neutral":
                games[key]["neutral"] = True
    rows = []
    for g in games.values():
        hs, as_ = g["home_subdivision"], g["away_subdivision"]
        hc, ac = g["home_conference"], g["away_conference"]
        if hs == "FBS" and as_ == "FBS" and hc in A4_CONFERENCES and ac in A4_CONFERENCES:
            game_type = "A4 vs A4"
        elif {hs, as_} == {"FBS", "FCS"}:
            game_type = "FBS vs FCS"
        elif hs == as_ == "FBS":
            game_type = "FBS vs FBS"
        elif hs == as_ == "FCS":
            game_type = "FCS vs FCS"
        else:
            game_type = "Other/Unknown"
        g["matchup_type"] = game_type
        g["a4_vs_a4"] = game_type == "A4 vs A4"
        g["source_urls"] = "; ".join(sorted(g["source_urls"]))
        rows.append(g)
    df = pd.DataFrame(rows)
    if len(df):
        week_sort = pd.to_numeric(df["week"], errors="coerce").fillna(99)
        df = df.assign(_week_sort=week_sort).sort_values(["season", "_week_sort", "date", "home_team", "away_team"]).drop(columns="_week_sort").reset_index(drop=True)
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def scrape_fbschedules_public(years: tuple[int, ...] = tuple(range(2027, 2038))) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    directory_html = _safe_get(FBSCHEDULES_DIRECTORY)
    soup = BeautifulSoup(directory_html, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(FBSCHEDULES_DIRECTORY, a["href"])
        parsed = urlparse(full)
        if parsed.netloc.endswith("fbschedules.com") and re.fullmatch(r"/ncaa/[a-z0-9-]+/?", parsed.path):
            urls.add(full.split("?")[0])
    urls = sorted(urls)
    teams, raw_games, errors = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(_scrape_one_team, url, years): url for url in urls}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                team, games = future.result()
                teams.append(team)
                raw_games.extend(games)
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
    teams_df = pd.DataFrame(teams).drop_duplicates(subset=["slug"]).sort_values(["subdivision", "conference", "name"]).reset_index(drop=True)
    team_by_slug = {r["slug"]: r for r in teams}
    games_df = _dedupe_scraped_games(raw_games, team_by_slug)
    return teams_df, games_df, errors


def build_real_store(teams_df: pd.DataFrame, games_df: pd.DataFrame, season: int) -> ScheduleStore:
    """Public prototype store. Availability and moveability are inferred, not authoritative."""
    teams = [
        Team(
            name=str(r["name"]),
            subdivision=str(r["subdivision"]),
            conference=str(r["conference"]),
            is_a4=bool(r["is_a4"]),
            parity_managed=bool(r["parity_managed"]),
        )
        for _, r in teams_df.iterrows()
    ]
    valid_names = {t.name for t in teams}
    games: List[Game] = []
    season_games = games_df[games_df["season"] == season] if len(games_df) else games_df
    for i, r in season_games.iterrows():
        week = r.get("week")
        if pd.isna(week):
            continue
        week = int(week)
        if week < 0 or week > 13 or r["home_team"] not in valid_names or r["away_team"] not in valid_names:
            continue
        neutral = bool(r.get("neutral", False))
        games.append(Game(
            game_id=f"real{season}_{i+1}",
            season=season,
            week=week,
            home_team=str(r["home_team"]),
            away_team=str(r["away_team"]),
            moveable=True,
            locked=False,
            neutral=neutral,
            campus_home_team="" if neutral else str(r["home_team"]),
            game_status="CONTRACTED",
            moveability="UNKNOWN",
            game_type="NONCONFERENCE",
            source="FBSchedules public future-opponent data",
            confidence="INFERRED",
            date_text=str(r.get("date", "") or ""),
            notes="Prototype assumption: moveability and open-week availability are inferred, not authoritative.",
        ))
    slots = [
        Slot(team=t.name, season=season, week=w, status="OPEN", location="ANY")
        for t in teams for w in range(0, 14)
    ]
    return ScheduleStore(teams, games, slots, needs=[])




def build_authoritative_store(
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
    slots_df: Optional[pd.DataFrame],
    season: int,
    needs_df: Optional[pd.DataFrame] = None,
) -> ScheduleStore:
    """Build a store from validated administrator-supplied data."""
    teams: List[Team] = []
    for _, r in teams_df.iterrows():
        teams.append(Team(
            name=str(r["name"]),
            subdivision=str(r.get("subdivision", "FBS")).upper(),
            conference=str(r.get("conference", "Independent")),
            is_a4=bool(r.get("is_a4", False)),
            parity_managed=bool(r.get("parity_managed", True)),
        ))
    valid_names = {t.name for t in teams}

    games: List[Game] = []
    season_rows = games_df[games_df["season"] == season] if len(games_df) else games_df
    for i, r in season_rows.iterrows():
        home = str(r["home_team"])
        away = str(r["away_team"])
        if home not in valid_names or away not in valid_names:
            continue
        raw_week = r.get("week")
        if pd.isna(raw_week):
            # Week-TBA commitments belong in schedule intelligence, not the
            # optimization graph, until a real week is supplied.
            continue
        week = int(raw_week)
        moveability = str(r.get("moveability", "MOVABLE") or "MOVABLE").upper()
        game_type = str(r.get("game_type", "NONCONFERENCE") or "NONCONFERENCE").upper()
        explicit_movable = moveability in {"MOVABLE", "FLEXIBLE"}
        locked = (not explicit_movable) or game_type == "CONFERENCE"
        earliest = r.get("earliest_week")
        latest = r.get("latest_week")
        guarantee = r.get("guarantee")
        try:
            guarantee = None if pd.isna(guarantee) or guarantee == "" else float(guarantee)
        except Exception:
            guarantee = None
        try:
            earliest = None if pd.isna(earliest) or earliest == "" else int(earliest)
        except Exception:
            earliest = None
        try:
            latest = None if pd.isna(latest) or latest == "" else int(latest)
        except Exception:
            latest = None

        games.append(Game(
            game_id=str(r.get("game_id") or f"auth{season}_{i+1}"),
            season=season,
            week=week,
            home_team=home,
            away_team=away,
            moveable=explicit_movable and not locked,
            locked=locked,
            neutral=bool(r.get("neutral", False)),
            campus_home_team=str(r.get("campus_home_team", "") or ""),
            game_status=str(r.get("game_status", "CONTRACTED") or "CONTRACTED").upper(),
            moveability=moveability,
            game_type=game_type,
            guarantee=guarantee,
            contract_link=str(r.get("contract_link", "") or ""),
            earliest_week=earliest,
            latest_week=latest,
            source=str(r.get("source", "Administrator import") or "Administrator import"),
            last_verified=str(r.get("last_verified", "") or ""),
            confidence=str(r.get("confidence", "AUTHORITATIVE") or "AUTHORITATIVE").upper(),
            date_text=str(r.get("date", "") or ""),
            notes=str(r.get("notes", "") or ""),
        ))

    explicit_slots: Dict[Tuple[str, int], Slot] = {}
    if slots_df is not None and len(slots_df):
        subset = slots_df[slots_df["season"] == season]
        for _, r in subset.iterrows():
            team = str(r["team"])
            if team not in valid_names:
                continue
            explicit_slots[(team, int(r["week"]))] = Slot(
                team=team,
                season=season,
                week=int(r["week"]),
                status=str(r.get("status", "OPEN") or "OPEN").upper(),
                location=str(r.get("location", "ANY") or "ANY").upper(),
            )

    slots: List[Slot] = []
    for t in teams:
        for w in range(14):
            slots.append(explicit_slots.get(
                (t.name, w),
                Slot(team=t.name, season=season, week=w, status="OPEN", location="ANY"),
            ))
    needs: List[Need] = []
    if needs_df is not None and len(needs_df):
        subset = needs_df[
            (pd.to_numeric(needs_df["season"], errors="coerce") == int(season))
            & (needs_df["status"].astype(str).str.upper().isin(["OPEN", "ACTIVE", "HOLD"]))
        ]
        for _, r in subset.iterrows():
            raw_week = r.get("week")
            if pd.isna(raw_week):
                continue
            needs.append(Need(
                team=str(r["team"]),
                season=int(season),
                week=int(raw_week),
                need_type=str(r.get("need_type", "") or "").upper(),
                location=str(r.get("location", "ANY") or "ANY").upper(),
                min_guarantee=None if pd.isna(r.get("min_guarantee")) else int(r.get("min_guarantee")),
                max_guarantee=None if pd.isna(r.get("max_guarantee")) else int(r.get("max_guarantee")),
                notes=str(r.get("notes", "") or ""),
            ))
    return ScheduleStore(teams, games, slots, needs=needs)



def _html_escape(value: object) -> str:
    import html
    return html.escape(str(value) if value is not None else "")


def _team_game_for_row(games_df: pd.DataFrame, team: str, season: int, week: int) -> Optional[pd.Series]:
    if games_df is None or games_df.empty:
        return None
    subset = games_df[(games_df["season"] == season) & (pd.to_numeric(games_df["week"], errors="coerce") == week)]
    subset = subset[(subset["home_team"] == team) | (subset["away_team"] == team)]
    return None if subset.empty else subset.iloc[0]


def _opponent_view(game: pd.Series, team: str) -> tuple[str, str, str]:
    neutral = bool(game.get("neutral", False))
    if str(game["home_team"]) == team:
        return str(game["away_team"]), str(game.get("away_logo", "") or ""), "N" if neutral else "H"
    return str(game["home_team"]), str(game.get("home_logo", "") or ""), "N" if neutral else "A"



def _logo_html(logo: str, opponent: str, size: int = 42) -> str:
    initials = "".join(x[0] for x in re.findall(r"[A-Za-z0-9]+", opponent)[:2]).upper() or "?"
    fallback = (
        f'<span class="logo-fallback" style="width:{size}px;height:{size}px">{initials}</span>'
    )
    if logo and str(logo).lower() != "nan":
        return (
            f'<img draggable="false" src="{_html_escape(logo)}" alt="{_html_escape(opponent)} logo" '
            f'class="team-logo" style="width:{size}px;height:{size}px" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
            f'<span class="logo-fallback" style="display:none;width:{size}px;height:{size}px">{initials}</span>'
        )
    return fallback


def _site_badge(site: str) -> str:
    labels = {"H": "HOME", "A": "AWAY", "N": "NEUTRAL"}
    return f'<span class="site-badge site-{site.lower()}">{labels.get(site, site)}</span>'


def render_conference_calendar(games_df: pd.DataFrame, teams_df: pd.DataFrame, season: int, conference: str) -> None:
    members = sorted(
        teams_df[(teams_df["subdivision"] == "FBS") & (teams_df["conference"] == conference)]["name"].tolist()
    )
    if not members:
        st.info("No FBS schools found for that conference in the current public snapshot.")
        return

    headers = []
    for week in range(14):
        sat = _week_saturday(season, week)
        headers.append(
            f'<th><span class="week-label">W{_display_week(week)}</span><span class="week-date">{sat.strftime("%b %d").replace(" 0", " ")}</span></th>'
        )

    team_logo_map = {str(r["name"]): str(r.get("logo_url", "") or "") for _, r in teams_df.iterrows()}
    rows_html = []
    for team in members:
        cells = []
        for week in range(14):
            game = _team_game_for_row(games_df, team, season, week)
            if game is None:
                cells.append('<td class="empty"><span class="open-dot">•</span></td>')
                continue
            opp, logo, site = _opponent_view(game, team)
            short = opp if len(opp) <= 13 else opp[:11] + "…"
            game_date = str(game.get("date", "") or "")
            if game_date and game_date != "TBA":
                try:
                    game_date = datetime.strptime(game_date, "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")
                except Exception:
                    pass
            cells.append(
                '<td class="game-cell">'
                '<div class="game-tile">'
                + _logo_html(logo, opp, 38)
                + f'<div class="opp" title="{_html_escape(opp)}">{_html_escape(short)}</div>'
                + f'<div class="mini-meta"><span class="mini-site site-{site.lower()}">{site}</span><span>{_html_escape(game_date)}</span></div>'
                + '</div></td>'
            )
        row_logo = _logo_html(team_logo_map.get(team, ""), team, 30)
        rows_html.append(
            f'<tr><th class="school"><div class="school-line">{row_logo}<span>{_html_escape(team)}</span></div></th>{"".join(cells)}</tr>'
        )

    st.markdown(
        '<div class="calendar-shell"><div class="calendar-scroll"><table class="gc">'
        '<thead><tr><th class="school school-head">SCHOOL</th>'
        + ''.join(headers)
        + '</tr></thead><tbody>'
        + ''.join(rows_html)
        + '</tbody></table></div></div>',
        unsafe_allow_html=True,
    )

    tba = games_df[(games_df["season"] == season) & (games_df["date"] == "TBA")]
    tba = tba[(tba["home_team"].isin(members)) | (tba["away_team"].isin(members))]
    if len(tba):
        with st.expander(f"{len(tba)} TBA non-conference matchup{'s' if len(tba) != 1 else ''}"):
            st.dataframe(
                tba[["away_team", "home_team", "neutral", "matchup_type"]],
                use_container_width=True,
                hide_index=True,
            )


def render_team_calendar(games_df: pd.DataFrame, teams_df: pd.DataFrame, season: int, team: str) -> None:
    team_meta_rows = teams_df[teams_df["name"] == team]
    team_logo = ""
    conference = ""
    subdivision = ""
    if len(team_meta_rows):
        meta = team_meta_rows.iloc[0]
        team_logo = str(meta.get("logo_url", "") or "")
        conference = str(meta.get("conference", "") or "")
        subdivision = str(meta.get("subdivision", "") or "")

    st.markdown(
        '<div class="team-hero">'
        f'<div>{_logo_html(team_logo, team, 72)}</div>'
        f'<div><div class="team-hero-name">{_html_escape(team)}</div>'
        f'<div class="team-hero-meta">{_html_escape(conference)} · {_html_escape(subdivision)} · {season}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    cards = []
    for week in range(14):
        sat = _week_saturday(season, week)
        game = _team_game_for_row(games_df, team, season, week)
        if game is None:
            body = (
                '<div class="tc-empty-icon">＋</div>'
                '<div class="tc-open">No known non-conference game</div>'
                '<div class="tc-open-sub">Potential scheduling slot</div>'
            )
            state_class = " is-open"
        else:
            opp, logo, site = _opponent_view(game, team)
            date_text = str(game.get("date", ""))
            if date_text and date_text != "TBA":
                try:
                    date_text = datetime.strptime(date_text, "%Y-%m-%d").strftime("%A, %b %d").replace(" 0", " ")
                except Exception:
                    pass
            body = (
                '<div class="tc-logo">' + _logo_html(logo, opp, 58) + '</div>'
                f'<div class="tc-opp">{_html_escape(opp)}</div>'
                f'<div class="tc-date-detail">{_html_escape(date_text)}</div>'
                + _site_badge(site)
            )
            state_class = ""
        cards.append(
            f'<div class="tc-card{state_class}"><div class="tc-card-top"><span>WEEK {_display_week(week)}</span><span>{sat.strftime("%b %d").replace(" 0", " ")}</span></div>{body}</div>'
        )

    st.markdown('<div class="tc-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

    tba = games_df[(games_df["season"] == season) & (games_df["date"] == "TBA")]
    tba = tba[(tba["home_team"] == team) | (tba["away_team"] == team)]
    if len(tba):
        st.markdown('<div class="section-kicker">TBA COMMITMENTS</div>', unsafe_allow_html=True)
        for _, game in tba.iterrows():
            opp, logo, site = _opponent_view(game, team)
            st.markdown(
                '<div class="tba-row">'
                + _logo_html(logo, opp, 44)
                + f'<div><strong>{_html_escape(opp)}</strong><br><span>{"Neutral" if site == "N" else ("Home" if site == "H" else "Away")} · Date TBA</span></div>'
                + '</div>',
                unsafe_allow_html=True,
            )


def _workspace_move_key(season: int) -> str:
    return f"cfb_nonc_workspace_moves_{int(season)}"


def _workspace_moves(season: int) -> Dict[str, int]:
    raw = st.session_state.get(_workspace_move_key(season), {})
    return {str(k): int(v) for k, v in dict(raw).items()}


def _set_workspace_move(season: int, game_id: str, week: int) -> None:
    moves = _workspace_moves(season)
    moves[str(game_id)] = int(week)
    st.session_state[_workspace_move_key(season)] = moves


def _clear_workspace_moves(season: int) -> None:
    st.session_state[_workspace_move_key(season)] = {}


def _conference_nonconf_state(store: ScheduleStore, games: Dict[str, Game], season: int, conference: str, week: int) -> Dict[str, object]:
    """Return the simple weekly state administrators actually care about."""
    members = store.conference_members(conference)
    member_names = {t.name for t in members}
    nonconf_teams: Set[str] = set()
    game_ids: List[str] = []
    for game in games.values():
        if game.season != season or game.week != week:
            continue
        home = store.teams.get(game.home_team)
        away = store.teams.get(game.away_team)
        if home and away and home.subdivision == away.subdivision == "FBS" and home.conference == away.conference == conference:
            continue
        involved = False
        if game.home_team in member_names:
            nonconf_teams.add(game.home_team)
            involved = True
        if game.away_team in member_names:
            nonconf_teams.add(game.away_team)
            involved = True
        if involved:
            game_ids.append(game.game_id)
    nonconf_count = len(nonconf_teams)
    available = max(0, len(members) - nonconf_count)
    return {
        "conference_size": len(members),
        "nonconf_count": nonconf_count,
        "available": available,
        "is_even": available % 2 == 0,
        "nonconf_teams": sorted(nonconf_teams),
        "game_ids": game_ids,
    }


def _odd_parity_keys(optimizer: AdvancedNonConferenceOptimizer, games: Dict[str, Game], season: int) -> Set[Tuple[str, int]]:
    bad: Set[Tuple[str, int]] = set()
    for w in range(0, 14):
        for conf, status in optimizer.conference_parity(games, season, w).items():
            if status.startswith("ODD"):
                bad.add((conf, w))
    return bad


def _simple_parity_candidates(
    store: ScheduleStore,
    optimizer: AdvancedNonConferenceOptimizer,
    season: int,
    conference: str,
    target_week: int,
) -> Dict[str, List[Dict[str, object]]]:
    """Rank one-game fixes before invoking a network optimizer.

    add: move one game involving exactly one conference member INTO target week.
    remove: move one such game OUT of target week.

    A candidate is "clean" when it fixes the requested conference/week without
    turning any currently-even FBS conference/week into a new odd state.
    """
    base = store.copy_games()
    base_bad = _odd_parity_keys(optimizer, base, season)
    members = {t.name for t in store.conference_members(conference)}
    out: Dict[str, List[Dict[str, object]]] = {"add": [], "remove": []}

    def conf_coeff(game: Game) -> int:
        home = store.teams.get(game.home_team)
        away = store.teams.get(game.away_team)
        if home and away and home.subdivision == away.subdivision == "FBS" and home.conference == away.conference == conference:
            return 0
        return int(game.home_team in members) + int(game.away_team in members)

    def evaluate(game: Game, to_week: int, direction: str) -> Optional[Dict[str, object]]:
        if game.locked or not game.moveable or to_week == game.week:
            return None
        # A one-game fix must represent exactly one conference team appearance.
        if conf_coeff(game) != 1:
            return None
        # Both teams must be free of another known game in the destination week.
        for team in (game.home_team, game.away_team):
            if store.game_for_team_week(base, team, season, to_week, exclude_game_id=game.game_id):
                return None
            if not store.slot_allows_game(team, season, to_week):
                return None
        after = dict(base)
        after[game.game_id] = replace(game, week=int(to_week))
        target_state = _conference_nonconf_state(store, after, season, conference, target_week)
        if not target_state["is_even"]:
            return None
        after_bad = _odd_parity_keys(optimizer, after, season)
        created = sorted(after_bad - base_bad)
        resolved = sorted(base_bad - after_bad)
        clean = len(created) == 0
        # Prefer clean moves, then moves that create the fewest new issues,
        # then moves that resolve more existing issues, then the shortest date move.
        rank = (
            0 if clean else 1,
            len(created),
            -len(resolved),
            abs(int(to_week) - int(game.week)),
            game.away_team,
            game.home_team,
        )
        return {
            "game_id": game.game_id,
            "game": game,
            "direction": direction,
            "from_week": int(game.week),
            "to_week": int(to_week),
            "clean": clean,
            "created": created,
            "resolved": resolved,
            "rank": rank,
            "target_nonconf": int(target_state["nonconf_count"]),
            "target_available": int(target_state["available"]),
        }

    # ADD ONE: bring one conference non-conference appearance into the target week.
    for game in base.values():
        if game.season != season or game.week == target_week or conf_coeff(game) != 1:
            continue
        candidate = evaluate(game, target_week, "add")
        if candidate:
            out["add"].append(candidate)

    # REMOVE ONE: move one current target-week non-conference game to another mutually open week.
    for game in base.values():
        if game.season != season or game.week != target_week or conf_coeff(game) != 1:
            continue
        for to_week in range(0, 14):
            if to_week == target_week:
                continue
            candidate = evaluate(game, to_week, "remove")
            if candidate:
                out["remove"].append(candidate)

    # Show distinct best alternatives rather than dozens of dates for the same game.
    out["add"].sort(key=lambda x: x["rank"])
    out["remove"].sort(key=lambda x: x["rank"])
    best_remove_by_game: Dict[str, Dict[str, object]] = {}
    for item in out["remove"]:
        best_remove_by_game.setdefault(str(item["game_id"]), item)
    out["remove"] = sorted(best_remove_by_game.values(), key=lambda x: x["rank"])
    return out


def _render_simple_parity_option(candidate: Dict[str, object], season: int, key_prefix: str, rank: int) -> None:
    game: Game = candidate["game"]  # type: ignore[assignment]
    clean = bool(candidate["clean"])
    created = list(candidate["created"])
    resolved = list(candidate["resolved"])
    label = f"{game.away_team} @ {game.home_team}"
    badge = "CLEAN MOVE" if clean else f"{len(created)} TRADEOFF{'S' if len(created) != 1 else ''}"
    badge_class = "good" if clean else "bad"
    detail = f"Week {candidate['from_week']} → Week {candidate['to_week']} · 1 game moved"
    if resolved:
        detail += f" · resolves {len(resolved)} existing parity issue{'s' if len(resolved) != 1 else ''}"
    st.markdown(
        '<div class="result-card" style="margin:0 0 10px">'
        '<div class="result-top">'
        f'<div><div class="result-rank">OPTION {rank}</div><div class="result-title">{_html_escape(label)}</div></div>'
        f'<span class="status-chip {badge_class}">{badge}</span>'
        '</div>'
        f'<div class="result-summary">{_html_escape(detail)}</div>'
        '<div class="result-kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">'
        f'<div class="result-kpi"><div class="result-kpi-label">NON-CONF TEAMS</div><div class="result-kpi-value">{candidate["target_nonconf"]}</div></div>'
        f'<div class="result-kpi"><div class="result-kpi-label">AVAILABLE</div><div class="result-kpi-value">{candidate["target_available"]}</div></div>'
        f'<div class="result-kpi"><div class="result-kpi-label">SECONDARY MOVES</div><div class="result-kpi-value">0</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if created:
        created_text = ", ".join(f"{c} W{w}" for c, w in created[:4])
        st.caption(f"Tradeoff: this one-game move would create {created_text}.")
    if st.button("Use this move", key=f"{key_prefix}_{candidate['game_id']}_{candidate['to_week']}_{rank}", type="primary" if rank == 1 and clean else "secondary", use_container_width=True):
        _set_workspace_move(season, str(candidate["game_id"]), int(candidate["to_week"]))
        st.session_state[f"simple_parity_feedback_{season}"] = f"Applied {label}: Week {candidate['from_week']} → Week {candidate['to_week']}"
        st.rerun()


def apply_workspace_moves(store: ScheduleStore, year_games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Apply interactive what-if moves to the in-memory workspace only.

    This never changes the scraped/public source data. It lets the user drag a
    game on the board, then see that proposed state reflected in calendars,
    parity, and subsequent optimization requests for the current session.
    """
    overrides = _workspace_moves(season)
    if not overrides:
        return year_games.copy()
    for game_id, target_week in overrides.items():
        game = store.games.get(game_id)
        if game is not None and int(game.week) != int(target_week):
            store.games[game_id] = replace(game, week=int(target_week))
    df = year_games.copy()
    if "game_id" not in df.columns:
        return df
    for game_id, target_week in overrides.items():
        mask = df["game_id"].astype(str) == str(game_id)
        if not mask.any():
            continue
        df.loc[mask, "week"] = int(target_week)
        if "date" in df.columns:
            df.loc[mask, "date"] = _week_saturday(season, int(target_week)).isoformat()
        df.loc[mask, "workspace_moved"] = True
    return df


def _find_conflicting_games(store: ScheduleStore, game: Game, target_week: int) -> List[Game]:
    conflicts: List[Game] = []
    for team in (game.home_team, game.away_team):
        other = store.game_for_team_week(store.copy_games(), team, game.season, int(target_week), exclude_game_id=game.game_id)
        if other is not None and all(other.game_id != g.game_id for g in conflicts):
            conflicts.append(other)
    return conflicts


def _direct_move_assessment(store: ScheduleStore, game: Game, target_week: int) -> Dict[str, object]:
    target_week = int(target_week)
    if target_week == int(game.week):
        return {"status": "current", "clean": True, "conflicts": [], "message": "Current week"}
    if target_week < 0 or target_week > 13:
        return {"status": "blocked", "clean": False, "conflicts": [], "message": "Outside the regular-season week range"}
    if not store.slot_allows_game(game.home_team, game.season, target_week) or not store.slot_allows_game(game.away_team, game.season, target_week):
        return {"status": "blocked", "clean": False, "conflicts": [], "message": "One or both teams are blocked on this week"}
    conflicts = _find_conflicting_games(store, game, target_week)
    if conflicts:
        names = ", ".join(f"{g.away_team} @ {g.home_team}" for g in conflicts)
        return {"status": "conflict", "clean": False, "conflicts": conflicts, "message": f"Conflict with {names}"}
    return {"status": "clean", "clean": True, "conflicts": [], "message": "Both teams are available"}


def _sortable_game_token(game: Game) -> str:
    # Only one game is draggable at a time, so the visible matchup can safely
    # serve as the unique sortable item id without exposing internal IDs.
    return f"{game.away_team} @ {game.home_team}"



def _render_move_outcome(kind: str, title: str, body: str, detail: str = "") -> None:
    icon = {"success": "✓", "conflict": "!", "info": "i"}.get(kind, "i")
    detail_html = f'<div class="decision-detail">{_html_escape(detail)}</div>' if detail else ""
    st.markdown(
        f'<div class="decision-card decision-{kind}"><div class="decision-icon">{icon}</div>'
        f'<div><div class="decision-title">{_html_escape(title)}</div><div class="decision-body">{_html_escape(body)}</div>{detail_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_conference_drag_board(
    store: ScheduleStore,
    optimizer: AdvancedNonConferenceOptimizer,
    teams_df: pd.DataFrame,
    season: int,
    conference: str,
) -> None:
    """Direct-manipulation conference move board.

    All known non-conference games involving the selected FBS conference are
    placed directly in Week 0-13 containers. The user drags the actual game
    card across weeks; clean moves are accepted into the what-if workspace,
    while blocked moves snap back and trigger the minimum-change CP-SAT repair.
    """
    members = set(
        teams_df[(teams_df["subdivision"] == "FBS") & (teams_df["conference"] == conference)]["name"]
        .dropna().astype(str).tolist()
    )
    if not members:
        st.info("No FBS schools found for that conference in the current snapshot.")
        return

    feedback_key = f"conference_drag_feedback_{season}_{conference}"
    feedback = st.session_state.pop(feedback_key, None)
    if feedback:
        _render_move_outcome(
            feedback.get("kind", "info"),
            feedback.get("title", "Move evaluated"),
            feedback.get("body", ""),
            feedback.get("detail", ""),
        )
        sols = feedback.get("solutions") or []
        if sols:
            st.markdown('<div class="section-kicker">MINIMUM-CHANGE PATH</div>', unsafe_allow_html=True)
            render_solution(sols[0], 1)

    st.markdown(
        '<div class="board-header"><div>'
        '<div class="section-kicker" style="margin-top:0">INTERACTIVE CONFERENCE BOARD</div>'
        f'<div class="section-title" style="font-size:1.05rem">{_html_escape(conference)} · drag a game directly to another week</div>'
        '<div class="section-copy" style="margin-bottom:0">Every card below is a known non-conference game involving this conference. Drop a card on the week you want. A clean move is accepted immediately; a conflict is rejected and the optimizer returns the fewest secondary changes required.</div>'
        '</div><div class="board-legend"><span class="legend-dot legend-current"></span>Current <span class="legend-dot legend-clean"></span>Accepted <span class="legend-dot legend-conflict"></span>Blocked</div></div>',
        unsafe_allow_html=True,
    )

    if not SORTABLES_AVAILABLE:
        games = []
        for g in store.games.values():
            if int(g.season) != int(season):
                continue
            if g.home_team in members or g.away_team in members:
                games.append(g)
        games.sort(key=lambda g: (int(g.week), g.home_team, g.away_team))
        if not games:
            st.info("No dated non-conference games are loaded for this conference.")
            return

        st.markdown(
            '<div class="section-copy" style="margin:.2rem 0 .8rem 0">'
            'Stable move controls are active for this cloud build. Select the game and destination week; '
            'the same optimizer will accept a clean move or return the minimum-change repair path.'
            '</div>',
            unsafe_allow_html=True,
        )
        option_map = {
            f"W{int(g.week)} · {g.away_team} @ {g.home_team}": g
            for g in games
        }
        chosen_label = st.selectbox(
            "Game to move",
            list(option_map.keys()),
            key=f"conference_fallback_game_{season}_{conference}",
        )
        chosen_game = option_map[chosen_label]
        target_week = st.selectbox(
            "Move to week",
            list(range(14)),
            index=int(chosen_game.week),
            key=f"conference_fallback_week_{season}_{conference}",
        )
        if st.button(
            "Evaluate move",
            use_container_width=True,
            key=f"conference_fallback_go_{season}_{conference}",
        ):
            assessment = _direct_move_assessment(store, chosen_game, int(target_week))
            if assessment.get("clean"):
                old_week = int(chosen_game.week)
                _set_workspace_move(season, chosen_game.game_id, int(target_week))
                st.session_state[feedback_key] = {
                    "kind": "success",
                    "title": "Move accepted",
                    "body": f"{chosen_game.away_team} @ {chosen_game.home_team}: Week {old_week} → Week {int(target_week)}",
                    "detail": "Both teams are clear in the target week. No secondary schedule move is required.",
                }
            else:
                solutions = optimizer.solve(Intent(
                    action="MOVE_GAME",
                    season=season,
                    target_week=int(target_week),
                    team_a=chosen_game.home_team,
                    team_b=chosen_game.away_team,
                    preserve_fbs_conference_parity=False,
                    max_additional_moves=8,
                    summary="Conference-board conflict repair",
                ))
                st.session_state[feedback_key] = {
                    "kind": "conflict",
                    "title": "Move blocked",
                    "body": f"{chosen_game.away_team} @ {chosen_game.home_team} cannot move directly to Week {int(target_week)}.",
                    "detail": str(assessment.get("message", "A scheduling conflict exists.")),
                    "solutions": solutions,
                }
            st.rerun()
        return

    games = []
    for g in store.games.values():
        if int(g.season) != int(season):
            continue
        if g.home_team in members or g.away_team in members:
            games.append(g)
    games.sort(key=lambda g: (int(g.week), g.home_team, g.away_team))
    if not games:
        st.info("No dated non-conference games are loaded for this conference.")
        return

    def conference_side(g: Game) -> tuple[str, str, str]:
        h = g.home_team in members
        a = g.away_team in members
        if h and not a:
            return g.home_team, g.away_team, "H"
        if a and not h:
            return g.away_team, g.home_team, "A"
        # Defensive fallback for unusual same-conference rows in source data.
        return g.home_team, g.away_team, "N"

    token_to_game: Dict[str, Game] = {}
    containers = []
    for week in range(14):
        sat = _week_saturday(season, week).strftime("%b %d").replace(" 0", " ")
        items = []
        for g in games:
            if int(g.week) != week:
                continue
            school, opp, site = conference_side(g)
            token = f"{school} — {opp} ({site})"
            if token in token_to_game:
                token = f"{token} · {g.game_id[-4:]}"
            token_to_game[token] = g
            items.append(token)
        containers.append({"header": f"W{week} · {sat}", "items": items})

    css = [
        ".sortable-component.vertical{display:flex!important;flex-wrap:wrap!important;align-items:stretch!important;gap:8px!important;background:transparent!important;padding:2px!important}",
        ".sortable-component.vertical .sortable-container{box-sizing:border-box!important;flex:1 1 150px!important;min-width:145px!important;max-width:190px!important;margin:0!important;padding:0!important;min-height:122px!important;background:#0a1624!important;border:1px solid #263a53!important;border-radius:12px!important;overflow:hidden!important}",
        ".sortable-container-header{font-size:10px!important;font-weight:850!important;color:#9aabc0!important;background:#0f1e30!important;padding:9px!important;border-bottom:1px solid #263a53!important}",
        ".sortable-container-body{box-sizing:border-box!important;min-height:82px!important;padding:7px!important}",
        ".sortable-item,.sortable-item:hover{box-sizing:border-box!important;font-size:10px!important;line-height:1.25!important;background:#19304a!important;color:#f4f7fb!important;border:1px solid #476887!important;border-radius:9px!important;padding:9px!important;cursor:grab!important;box-shadow:none!important;font-weight:760!important;margin:0 0 6px 0!important;touch-action:none!important;user-select:none!important;-webkit-user-select:none!important}",
        ".sortable-item:active{cursor:grabbing!important}",
        ".active{opacity:.45!important}",
    ]

    nonce = st.session_state.get(f"conference_board_nonce_{season}_{conference}", 0)
    sorted_containers = sort_items(
        containers,
        multi_containers=True,
        direction="vertical",
        custom_style="\n".join(css),
        key=f"conference_schedule_{season}_{conference}_{nonce}",
    )

    new_week_by_token: Dict[str, int] = {}
    for week, container in enumerate(sorted_containers or []):
        for token in container.get("items", []):
            new_week_by_token[str(token)] = int(week)

    moved = []
    for token, g in token_to_game.items():
        target_week = new_week_by_token.get(token, int(g.week))
        if int(target_week) != int(g.week):
            moved.append((token, g, int(target_week)))

    if moved:
        # Evaluate one deliberate drop, then immediately rerun so the board
        # rebuilds from the authoritative what-if workspace.
        _, game, target_week = moved[0]
        assessment = _direct_move_assessment(store, game, target_week)
        if assessment.get("clean"):
            old_week = int(game.week)
            _set_workspace_move(season, game.game_id, target_week)
            st.session_state[feedback_key] = {
                "kind": "success",
                "title": "Move accepted",
                "body": f"{game.away_team} @ {game.home_team}: Week {old_week} → Week {target_week}",
                "detail": "Both teams are clear in the target week. No secondary schedule move is required.",
            }
            st.session_state[f"conference_board_nonce_{season}_{conference}"] = nonce + 1
            st.rerun()
        else:
            solutions = optimizer.solve(Intent(
                action="MOVE_GAME",
                season=season,
                target_week=target_week,
                team_a=game.home_team,
                team_b=game.away_team,
                preserve_fbs_conference_parity=False,
                max_additional_moves=8,
                summary="Direct conference-board conflict repair",
            ))
            st.session_state[feedback_key] = {
                "kind": "conflict",
                "title": "Move blocked",
                "body": f"{game.away_team} @ {game.home_team} cannot move directly to Week {target_week}.",
                "detail": str(assessment.get("message", "A scheduling conflict exists.")),
                "solutions": solutions,
            }
            st.session_state[f"conference_board_nonce_{season}_{conference}"] = nonce + 1
            st.rerun()

    st.caption("Drag the game card itself — not the logo. Clean drops become part of your current what-if workspace; blocked drops return the optimized repair path. The logo matrix below updates after accepted moves.")


def render_drag_move_lab(
    store: ScheduleStore,
    optimizer: AdvancedNonConferenceOptimizer,
    season: int,
    selected_team: str,
) -> None:
    """Direct schedule editor for one team.

    Every dated non-conference game for the selected team is visible on a
    fourteen-week rail. The user drags the game itself; there is no separate
    game selector. Clean drops are accepted into the what-if workspace.
    Conflicted drops are rejected, snap back, and trigger the minimum-change
    CP-SAT repair path.
    """
    feedback = st.session_state.pop(f"move_feedback_{season}", None)
    if feedback:
        _render_move_outcome(
            feedback.get("kind", "info"),
            feedback.get("title", "Move evaluated"),
            feedback.get("body", ""),
            feedback.get("detail", ""),
        )
        sols = feedback.get("solutions") or []
        if sols:
            st.markdown('<div class="section-kicker">MINIMUM-CHANGE PATH</div>', unsafe_allow_html=True)
            render_solution(sols[0], 1)

    team_games = sorted(
        [g for g in store.games.values() if g.season == season and g.involves(selected_team)],
        key=lambda g: (g.week, g.home_team, g.away_team),
    )
    if not team_games:
        st.info("No dated non-conference games are loaded for this team.")
        return

    st.markdown(
        '<div class="board-header"><div><div class="section-kicker" style="margin-top:0">LIVE SCHEDULE EDITOR</div>'
        f'<div class="section-title" style="font-size:1.05rem">{_html_escape(selected_team)} · drag any game to another week</div>'
        '<div class="section-copy" style="margin-bottom:0">Drop a game on the week you want. A clean move is accepted immediately. A conflict is rejected and the optimizer returns the fewest secondary moves needed to make that date work.</div>'
        '</div><div class="board-legend"><span class="legend-dot legend-current"></span>Scheduled <span class="legend-dot legend-clean"></span>Accepted <span class="legend-dot legend-conflict"></span>Blocked</div></div>',
        unsafe_allow_html=True,
    )

    if not SORTABLES_AVAILABLE:
        st.markdown(
            '<div class="section-copy" style="margin:.2rem 0 .8rem 0">'
            'Stable tap controls are active for this cloud build. Choose a game and destination week.'
            '</div>',
            unsafe_allow_html=True,
        )
        option_map = {
            f"W{int(g.week)} · {g.away_team if g.home_team == selected_team else g.home_team} "
            f"({'H' if g.home_team == selected_team else 'A'})": g
            for g in team_games
        }
        chosen_label = st.selectbox(
            "Game to move",
            list(option_map.keys()),
            key=f"team_fallback_game_{season}_{selected_team}",
        )
        chosen_game = option_map[chosen_label]
        target_week = st.selectbox(
            "Move to week",
            list(range(14)),
            index=int(chosen_game.week),
            key=f"team_fallback_week_{season}_{selected_team}",
        )
        if st.button(
            "Evaluate move",
            use_container_width=True,
            key=f"team_fallback_go_{season}_{selected_team}",
        ):
            assessment = _direct_move_assessment(store, chosen_game, int(target_week))
            if assessment.get("clean"):
                old_week = int(chosen_game.week)
                _set_workspace_move(season, chosen_game.game_id, int(target_week))
                st.session_state[f"move_feedback_{season}"] = {
                    "kind": "success",
                    "title": "Move accepted",
                    "body": f"{chosen_game.away_team} @ {chosen_game.home_team}: Week {old_week} → Week {int(target_week)}",
                    "detail": "Both teams are clear in the target week. No secondary schedule move is required.",
                }
            else:
                solutions = optimizer.solve(Intent(
                    action="MOVE_GAME",
                    season=season,
                    target_week=int(target_week),
                    team_a=chosen_game.home_team,
                    team_b=chosen_game.away_team,
                    preserve_fbs_conference_parity=False,
                    max_additional_moves=8,
                    summary="Team-board conflict repair",
                ))
                st.session_state[f"move_feedback_{season}"] = {
                    "kind": "conflict",
                    "title": "Move blocked",
                    "body": f"{chosen_game.away_team} @ {chosen_game.home_team} cannot move directly to Week {int(target_week)}.",
                    "detail": str(assessment.get("message", "A scheduling conflict exists.")),
                    "solutions": solutions,
                }
            st.rerun()
        return

    def token_for(game: Game) -> str:
        opponent = game.away_team if game.home_team == selected_team else game.home_team
        site = "H" if game.home_team == selected_team else "A"
        return f"{opponent} · {site}"

    token_to_game: Dict[str, Game] = {}
    containers = []
    for week in range(14):
        sat = _week_saturday(season, week).strftime("%b %d").replace(" 0", " ")
        items = []
        for game in team_games:
            if int(game.week) != week:
                continue
            token = token_for(game)
            # Defensive uniqueness for same opponent labels in unusual datasets.
            if token in token_to_game:
                token = f"{token} · {game.game_id[-4:]}"
            token_to_game[token] = game
            items.append(token)
        containers.append({"header": f"W{week} · {sat}", "items": items})

    css = [
        ".sortable-component.vertical{display:flex!important;flex-wrap:wrap!important;align-items:stretch!important;gap:8px!important;background:transparent!important;padding:2px!important}",
        ".sortable-component.vertical .sortable-container{box-sizing:border-box!important;flex:1 1 138px!important;min-width:132px!important;max-width:175px!important;margin:0!important;padding:0!important;min-height:106px!important;background:#0b1726!important;border:1px solid #25344a!important;border-radius:12px!important;overflow:hidden!important}",
        ".sortable-container-header{font-size:10px!important;font-weight:850!important;letter-spacing:.02em!important;color:#91a0b4!important;background:#0f1d2d!important;padding:9px 8px!important;border-bottom:1px solid #24334a!important}",
        ".sortable-container-body{box-sizing:border-box!important;min-height:62px!important;padding:7px!important}",
        ".sortable-item,.sortable-item:hover{box-sizing:border-box!important;font-size:10px!important;line-height:1.25!important;background:#1a2b42!important;color:#eef3f8!important;border:1px solid #44617f!important;border-radius:9px!important;padding:10px 9px!important;cursor:grab!important;box-shadow:none!important;font-weight:760!important;margin:0!important;touch-action:none!important;user-select:none!important;-webkit-user-select:none!important}",
        ".sortable-item:active{cursor:grabbing!important}",
        ".active{opacity:.45!important}",
    ]

    nonce = st.session_state.get(f"move_board_nonce_{season}_{selected_team}", 0)
    sorted_containers = sort_items(
        containers,
        multi_containers=True,
        direction="vertical",
        custom_style="\n".join(css),
        key=f"direct_schedule_{season}_{selected_team}_{nonce}",
    )

    new_week_by_token: Dict[str, int] = {}
    for week, container in enumerate(sorted_containers or []):
        for token in container.get("items", []):
            new_week_by_token[str(token)] = int(week)

    moved = []
    for token, game in token_to_game.items():
        target_week = new_week_by_token.get(token, int(game.week))
        if int(target_week) != int(game.week):
            moved.append((token, game, int(target_week)))

    if moved:
        # Process only the first intentional change; rerun immediately before a
        # second drag can be interpreted from the same component state.
        _, game, target_week = moved[0]
        assessment = _direct_move_assessment(store, game, target_week)
        if assessment.get("clean"):
            old_week = int(game.week)
            _set_workspace_move(season, game.game_id, target_week)
            st.session_state[f"move_feedback_{season}"] = {
                "kind": "success",
                "title": "Move accepted",
                "body": f"{game.away_team} @ {game.home_team}: Week {old_week} → Week {target_week}",
                "detail": "Both teams are clear in the target week. No secondary schedule move is required.",
            }
            st.session_state[f"move_board_nonce_{season}_{selected_team}"] = nonce + 1
            st.rerun()
        else:
            solutions = optimizer.solve(Intent(
                action="MOVE_GAME",
                season=season,
                target_week=target_week,
                team_a=game.home_team,
                team_b=game.away_team,
                preserve_fbs_conference_parity=False,
                max_additional_moves=8,
                summary="Direct calendar drag conflict repair",
            ))
            st.session_state[f"move_feedback_{season}"] = {
                "kind": "conflict",
                "title": "Move blocked",
                "body": f"{game.away_team} @ {game.home_team} cannot move directly to Week {target_week}.",
                "detail": str(assessment.get("message", "A scheduling conflict exists.")),
                "solutions": solutions,
            }
            st.session_state[f"move_board_nonce_{season}_{selected_team}"] = nonce + 1
            st.rerun()

    st.caption("Tip: the logo calendar above is a read-only overview. Use this live editor to make what-if moves; accepted moves immediately flow back into every calendar and parity report.")

    with st.expander("Move with controls instead"):
        game_labels = {f"W{g.week} · {g.away_team} @ {g.home_team}": g for g in team_games}
        game_label = st.selectbox("Game", list(game_labels), key=f"manual_game_{season}_{selected_team}")
        game = game_labels[game_label]
        target = st.selectbox("Target week", list(range(14)), index=int(game.week), key=f"manual_target_{season}_{game.game_id}")
        a = _direct_move_assessment(store, game, int(target))
        st.caption(("Clean move: " if a.get("clean") else "Conflict: ") + str(a.get("message", "")))
        if st.button("Evaluate move", key=f"manual_eval_{season}_{game.game_id}", use_container_width=True):
            if int(target) == int(game.week):
                st.info("That game is already in the selected week.")
            elif a.get("clean"):
                _set_workspace_move(season, game.game_id, int(target))
                st.session_state[f"move_feedback_{season}"] = {
                    "kind": "success", "title": "Move accepted",
                    "body": f"{game.away_team} @ {game.home_team}: Week {game.week} → Week {int(target)}",
                    "detail": "Both teams are clear in the target week. No secondary move is required.",
                }
                st.rerun()
            else:
                solutions = optimizer.solve(Intent(action="MOVE_GAME", season=season, target_week=int(target), team_a=game.home_team, team_b=game.away_team, preserve_fbs_conference_parity=False, max_additional_moves=8))
                st.session_state[f"move_feedback_{season}"] = {
                    "kind": "conflict", "title": "Move blocked",
                    "body": f"{game.away_team} @ {game.home_team} cannot move directly to Week {int(target)}.",
                    "detail": str(a.get("message", "A scheduling conflict exists.")), "solutions": solutions,
                }
                st.rerun()


