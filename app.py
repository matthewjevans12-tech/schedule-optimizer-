from __future__ import annotations

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
import streamlit as st
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

# If Streamlit secrets are configured, expose them to the OpenAI client.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
    if "OPENAI_MODEL" in st.secrets:
        os.environ.setdefault("OPENAI_MODEL", st.secrets["OPENAI_MODEL"])
except Exception:
    pass




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

    def involves(self, team: str) -> bool:
        return team in (self.home_team, self.away_team)

    def opponents(self) -> Tuple[str, str]:
        return self.home_team, self.away_team


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

    def find_buy_games(self, intent: Intent) -> List[Solution]:
        """Find public buy/guarantee-game candidates.

        If the requesting school is FBS, candidates are FCS programs with a mutually
        open week. If the requesting school is FCS, candidates are FBS programs that
        have a mutually open week and therefore could be potential guarantee-game
        hosts. Public data shows schedule openings only; it does not prove intent.
        """
        if not intent.team_a or intent.season is None:
            return []
        requester = self.store.teams.get(intent.team_a)
        if not requester:
            return []
        base_games = self.store.copy_games()
        weeks = [intent.target_week] if intent.target_week is not None else list(range(0, 14))
        results: List[Solution] = []

        # In the synthetic demo we have explicit FCS marketplace needs. Use those when
        # the requester is an FBS school and a specific week was supplied.
        if self.store.needs and requester.subdivision == "FBS" and intent.target_week is not None:
            for need in self.store.needs:
                candidate = self.store.teams.get(need.team)
                if not candidate or candidate.subdivision != "FCS":
                    continue
                if need.season != intent.season or need.week != intent.target_week:
                    continue
                if need.location not in {"AWAY", "ANY"}:
                    continue
                if self.store.game_for_team_week(base_games, requester.name, intent.season, intent.target_week):
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, intent.season, intent.target_week):
                    continue
                if intent.max_guarantee is not None and need.min_guarantee is not None and need.min_guarantee > intent.max_guarantee:
                    continue
                ask = f"${need.min_guarantee:,}+" if need.min_guarantee else "not specified"
                results.append(Solution(
                    title=f"Week {intent.target_week} — {candidate.name} buy-game match",
                    moves=[],
                    score=90 if need.min_guarantee is None else max(50, 100 - (need.min_guarantee / max(intent.max_guarantee or need.min_guarantee, 1)) * 25),
                    explanation=f"{candidate.name} is available in Week {intent.target_week} and is seeking an away/buy game. Minimum guarantee: {ask}.",
                ))
            return sorted(results, key=lambda s: (-s.score, s.title))[:20]

        wanted_subdivision = "FCS" if requester.subdivision == "FBS" else "FBS"
        for week in weeks:
            if week is None:
                continue
            # The requesting team itself must have no known game that week.
            if self.store.game_for_team_week(base_games, requester.name, intent.season, int(week)):
                continue
            for candidate in self.store.teams.values():
                if candidate.name == requester.name or candidate.subdivision != wanted_subdivision:
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, intent.season, int(week)):
                    continue
                if requester.subdivision == "FBS":
                    title = f"Week {week} — {candidate.name} FCS candidate"
                    explanation = (f"{requester.name} and {candidate.name} both have no known dated non-conference game in Week {week} "
                                   f"of {intent.season}. This is a public-data candidate for an FBS-hosted buy game; confirm actual interest and guarantee terms in the authoritative scheduling system.")
                    score = 72
                else:
                    title = f"Week {week} — {candidate.name} potential FBS host"
                    explanation = (f"{requester.name} and {candidate.name} both have no known dated non-conference game in Week {week} "
                                   f"of {intent.season}. This makes {candidate.name} a public-data candidate for a guarantee/buy-game opportunity; confirm the FBS school's actual need in the authoritative scheduling system.")
                    score = 74 if candidate.is_a4 else 70
                results.append(Solution(title=title, moves=[], score=score, explanation=explanation))

        # Diversify the year-only result so one week does not consume the whole list.
        results = sorted(results, key=lambda s: (-s.score, s.title))
        if intent.target_week is None:
            by_week: Dict[int, int] = {}
            diversified: List[Solution] = []
            for sol in results:
                m = re.search(r"Week (\d+)", sol.title)
                week = int(m.group(1)) if m else -1
                if by_week.get(week, 0) >= 3:
                    continue
                diversified.append(sol)
                by_week[week] = by_week.get(week, 0) + 1
                if len(diversified) >= 20:
                    break
            return diversified
        return results[:20]

    def find_a4_games(self, intent: Intent) -> List[Solution]:
        if not intent.team_a or intent.season is None or intent.target_week is None:
            return []
        team = self.store.teams.get(intent.team_a)
        if not team or not team.is_a4:
            return []
        base_games = self.store.copy_games()
        if self.store.game_for_team_week(base_games, team.name, intent.season, intent.target_week):
            return []
        results: List[Solution] = []
        if self.store.needs:
            for need in self.store.needs:
                candidate = self.store.teams.get(need.team)
                if not candidate or not candidate.is_a4 or candidate.name == team.name:
                    continue
                if candidate.conference == team.conference:
                    continue
                if need.season != intent.season or need.week != intent.target_week or need.need_type != "A4":
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, intent.season, intent.target_week):
                    continue
                results.append(Solution(
                    title=f"{team.name} vs {candidate.name}",
                    moves=[],
                    score=95,
                    explanation=f"Both programs are A4, are available in Week {intent.target_week}, and {candidate.name} has an A4 need recorded for that week.",
                ))
        else:
            for candidate in self.store.teams.values():
                if not candidate.is_a4 or candidate.name == team.name or candidate.conference == team.conference:
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, intent.season, intent.target_week):
                    continue
                results.append(Solution(
                    title=f"{team.name} vs {candidate.name}",
                    moves=[],
                    score=72,
                    explanation=(f"{candidate.name} is an A4 program with no known dated non-conference game in "
                                 f"Week {intent.target_week} in the public snapshot. Confirm that the school actually needs an A4 game."),
                ))
        return sorted(results, key=lambda s: (-s.score, s.title))[:12]

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
        return "OR-Tools CP-SAT" if ORTOOLS_AVAILABLE else "Deterministic fallback"

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

    def _cp_optimize(self, intent: Intent, mode: str) -> List[Solution]:
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

        # For broad national optimization the full +/-5 week neighborhood is
        # still only a few thousand boolean variables for a normal season.
        wide = mode in {"national", "fcs_balance", "controlled_balance"}
        candidate_weeks: Dict[str, List[int]] = {}
        for game in season_games:
            target = int(intent.target_week) if target_game and game.game_id == target_game.game_id and intent.target_week is not None else None
            candidate_weeks[game.game_id] = self._candidate_weeks_for_cp(game, target_week=target, wide=wide)
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

        changed_vars = []
        distance_terms = []
        for game in season_games:
            current = x.get((game.game_id, game.week))
            changed = model.NewBoolVar(f"changed_{game.game_id}")
            if current is not None:
                model.Add(changed + current == 1)
            else:
                model.Add(changed == 1)
            changed_vars.append(changed)
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
            model.Add(sum(changed_vars) <= 30)

        objective_terms = []
        # Direct administrator-requested moves use a strict minimal-intervention
        # objective. The requested move is already a hard constraint above; once
        # that move is made, CP-SAT should leave every unrelated game alone.
        # Existing national parity problems are NOT an invitation to improve the
        # rest of the schedule during a simple move request. Hard constraints still
        # prevent the requested move from creating a brand-new parity problem in a
        # conference/week that was healthy before the move.
        if mode == "move":
            # One extra moved game must always cost more than any plausible total
            # week-distance savings. This gives us lexicographic behavior:
            #   1) fewest changed games, then 2) shortest cascade.
            objective_terms.append(1_000_000 * sum(changed_vars))
            objective_terms.append(100 * sum(distance_terms))
        else:
            # National/multi-week requests heavily prioritize the exact scope the
            # administrator named, while still discouraging parity problems elsewhere.
            scoped_bad_vars = [v for k, v in parity_bad.items() if k in scope_keys]
            if mode == "national" and scoped_bad_vars:
                objective_terms.append((self.PARITY_PENALTY * 5) * sum(scoped_bad_vars))
            objective_terms.append(self.PARITY_PENALTY * sum(parity_bad.values()))
            objective_terms.append(self.MOVE_PENALTY * sum(changed_vars))
            objective_terms.append(self.DISTANCE_PENALTY * sum(distance_terms))

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
                objective_terms.append(self.BALANCE_PENALTY * dev)

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
                objective_terms.append(self.BALANCE_PENALTY * dev)

        model.Minimize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = 7
        solver.parameters.log_search_progress = False

        started = time.perf_counter()
        status = solver.Solve(model)
        self.last_solver_seconds = time.perf_counter() - started
        self.last_solver_status = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
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
            if status == cp_model.FEASIBLE:
                warnings.append("The solver found a feasible minimal-change solution within the time limit; it did not prove global optimality.")
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
                "season": season,
                "unresolved_issues": self.parity_issue_details(after_games, season, range(0, 14)),
            },
        )]

    def solve_move_game(self, intent: Intent) -> List[Solution]:
        if ORTOOLS_AVAILABLE:
            result = self._cp_optimize(intent, "move")
            if result:
                return result
        return super().solve_move_game(intent)

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
        if ORTOOLS_AVAILABLE:
            result = self._cp_optimize(intent, "parity")
            if result:
                return result
        return super().solve_make_conference_even(intent)

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
    teams = [Team(name=str(r["name"]), subdivision=str(r["subdivision"]), conference=str(r["conference"]), is_a4=bool(r["is_a4"]), parity_managed=bool(r["parity_managed"])) for _, r in teams_df.iterrows()]
    valid_names = {t.name for t in teams}
    games = []
    season_games = games_df[games_df["season"] == season] if len(games_df) else games_df
    for i, r in season_games.iterrows():
        week = r.get("week")
        if pd.isna(week):
            continue
        week = int(week)
        if week < 0 or week > 13 or r["home_team"] not in valid_names or r["away_team"] not in valid_names:
            continue
        games.append(Game(game_id=f"real{season}_{i+1}", season=season, week=week, home_team=str(r["home_team"]), away_team=str(r["away_team"]), moveable=True, locked=False, notes="Public-data MVP assumption: treated as moveable until the authoritative scheduling system supplies true status."))
    slots = [Slot(team=t.name, season=season, week=w, status="OPEN", location="ANY") for t in teams for w in range(0, 14)]
    return ScheduleStore(teams, games, slots, needs=[])


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
            f'<th><span class="week-label">W{week}</span><span class="week-date">{sat.strftime("%b %d").replace(" 0", " ")}</span></th>'
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
            f'<div class="tc-card{state_class}"><div class="tc-card-top"><span>WEEK {week}</span><span>{sat.strftime("%b %d").replace(" 0", " ")}</span></div>{body}</div>'
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
        st.info("Direct drag-and-drop requires streamlit-sortables. Your deployment requirements include it; if this message appears, redeploy requirements.txt.")
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
        ".sortable-component{display:grid!important;grid-template-columns:repeat(7,minmax(138px,1fr));gap:8px;background:transparent!important;padding:2px!important}",
        ".sortable-container{min-height:126px!important;background:#0a1624!important;border:1px solid #263a53!important;border-radius:12px!important;overflow:hidden!important;transition:border-color .15s ease,background .15s ease,box-shadow .15s ease!important}",
        ".sortable-container-header{font-size:10px!important;font-weight:850!important;color:#9aabc0!important;background:#0f1e30!important;padding:9px 9px!important;border-bottom:1px solid #263a53!important}",
        ".sortable-container-body{min-height:84px!important;padding:7px!important}",
        ".sortable-item{font-size:10px!important;line-height:1.25!important;background:linear-gradient(180deg,#19304a,#13263b)!important;color:#f4f7fb!important;border:1px solid #476887!important;border-radius:9px!important;padding:9px!important;cursor:grab!important;box-shadow:0 4px 14px rgba(0,0,0,.16)!important;font-weight:760!important;margin-bottom:6px!important;touch-action:none!important;user-select:none!important;-webkit-user-select:none!important}",
        ".sortable-item:active{cursor:grabbing!important;transform:scale(1.015)!important}",
        ".sortable-ghost{opacity:.42!important;border-color:#70a6ff!important;background:#18345a!important}",
        ".sortable-chosen{border-color:#78adff!important;box-shadow:0 0 0 2px rgba(79,140,255,.18)!important}",
        ".sortable-container:has(.sortable-ghost){border-color:#4f8cff!important;background:#10243c!important;box-shadow:inset 0 0 0 1px rgba(79,140,255,.18)!important}",
    ]

    nonce = st.session_state.get(f"conference_board_nonce_{season}_{conference}", 0)
    sorted_containers = sort_items(
        containers,
        multi_containers=True,
        direction="horizontal",
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
        st.info("Drag-and-drop requires the streamlit-sortables package. The deployment requirements file includes it.")
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
        ".sortable-component{display:grid!important;grid-template-columns:repeat(7,minmax(122px,1fr));gap:8px;background:transparent!important;padding:2px!important}",
        ".sortable-container{min-height:108px!important;background:#0b1726!important;border:1px solid #25344a!important;border-radius:12px!important;overflow:hidden!important;transition:border-color .15s ease,background .15s ease!important}",
        ".sortable-container-header{font-size:10px!important;font-weight:850!important;letter-spacing:.02em!important;color:#91a0b4!important;background:#0f1d2d!important;padding:9px 8px!important;border-bottom:1px solid #24334a!important}",
        ".sortable-container-body{min-height:64px!important;padding:7px!important}",
        ".sortable-item{font-size:10px!important;line-height:1.25!important;background:linear-gradient(180deg,#1a2b42,#142338)!important;color:#eef3f8!important;border:1px solid #44617f!important;border-radius:9px!important;padding:10px 9px!important;cursor:grab!important;box-shadow:0 4px 12px rgba(0,0,0,.13)!important;font-weight:760!important}",
        ".sortable-item:active{cursor:grabbing!important;transform:scale(1.02)!important}",
        ".sortable-container:has(.sortable-item){border-color:#416a9a!important;background:#0e1c2c!important}",
    ]

    nonce = st.session_state.get(f"move_board_nonce_{season}_{selected_team}", 0)
    sorted_containers = sort_items(
        containers,
        multi_containers=True,
        direction="horizontal",
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


st.set_page_config(
    page_title="College Football Non-Conference Scheduling Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(r"""
<style>
:root{
  --g-bg:#07111f;
  --g-bg2:#091522;
  --g-surface:#0d1a2a;
  --g-surface2:#122238;
  --g-surface3:#162940;
  --g-border:#21344d;
  --g-border-soft:rgba(163,187,214,.14);
  --g-text:#f5f8fc;
  --g-muted:#91a3b8;
  --g-muted2:#667a91;
  --g-blue:#4f8cff;
  --g-blue2:#7aabff;
  --g-green:#34c785;
  --g-red:#ef5b67;
  --g-amber:#e6b85c;
  --g-cyan:#4dcbd7;
  --g-shadow:0 18px 55px rgba(0,0,0,.24);
}
html,body,[class*="css"]{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:linear-gradient(180deg,#07111f 0%,#08131f 45%,#07111f 100%);color:var(--g-text)}
.block-container{max-width:1580px;padding-top:1rem;padding-bottom:4rem}
header[data-testid="stHeader"]{background:rgba(7,17,31,.86);backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,.04)}
footer,#MainMenu{visibility:hidden}
[data-testid="stSidebar"]{background:#081421;border-right:1px solid var(--g-border-soft)}

/* Controls */
.stSelectbox label,.stRadio label,.stTextInput label,.stSlider label{font-size:.72rem!important;color:var(--g-muted)!important;font-weight:760!important;letter-spacing:.025em!important}
[data-baseweb="select"]>div,[data-baseweb="input"],textarea{background:#0d1a2a!important;border:1px solid #243750!important;border-radius:10px!important;min-height:44px!important}
.stButton>button{min-height:43px;border-radius:10px!important;font-weight:760!important;letter-spacing:.01em!important;border:1px solid #2b405a!important}
.stButton>button[kind="primary"]{background:linear-gradient(180deg,#4f8cff,#3f78df)!important;border-color:#5b96ff!important;color:white!important;box-shadow:0 9px 22px rgba(79,140,255,.18)}
.stButton>button:hover{border-color:#56769c!important}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;border:1px solid var(--g-border-soft);background:#0a1624;border-radius:12px;padding:4px;margin-bottom:16px;overflow-x:auto}
.stTabs [data-baseweb="tab"]{height:38px;padding:0 14px;color:#8ea0b5;font-weight:720;background:transparent;border-radius:8px;white-space:nowrap}
.stTabs [aria-selected="true"]{color:#fff!important;background:#15263a!important}
.stTabs [data-baseweb="tab-highlight"]{display:none!important}
.stChatInputContainer>div{background:#0d1a2a!important;border:1px solid #2a3d56!important;border-radius:14px!important;box-shadow:0 10px 30px rgba(0,0,0,.16)}
[data-testid="stChatMessage"]{background:transparent;border:0;padding:.35rem 0}
[data-testid="stExpander"]{background:#0d1a2a;border:1px solid var(--g-border-soft);border-radius:12px;overflow:hidden}
[data-testid="stDataFrame"]{border:1px solid var(--g-border-soft);border-radius:12px;overflow:hidden}
.stAlert{border-radius:11px;border:1px solid var(--g-border-soft)}

/* Enterprise masthead */
.brand-row{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:0 0 10px}
.brand-lockup{display:flex;align-items:center;gap:11px;min-width:0}
.brand-mark{width:38px;height:38px;border-radius:9px;background:linear-gradient(145deg,#52a96f,#2d6f48);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:950;font-size:18px;box-shadow:0 8px 22px rgba(52,199,133,.12)}
.brand-name{font-weight:900;letter-spacing:.075em;font-size:1.05rem;color:#fff;line-height:1}
.brand-sub{color:#6f8299;font-size:.68rem;margin-top:4px;font-weight:620}
.brand-status{display:flex;align-items:center;gap:8px;color:#9bb0c6;font-size:.72rem;white-space:nowrap;border:1px solid var(--g-border-soft);background:#0c1928;border-radius:999px;padding:7px 10px}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--g-green);box-shadow:0 0 0 4px rgba(52,199,133,.09)}

.hero{background:linear-gradient(135deg,#101f32 0%,#0c1929 68%,#102137 100%);border:1px solid #233750;border-radius:14px;padding:17px 18px;margin-bottom:13px;display:flex;align-items:center;justify-content:space-between;gap:20px;box-shadow:0 16px 46px rgba(0,0,0,.13)}
.hero-kicker{font-size:.61rem;letter-spacing:.16em;color:#7fa8da;font-weight:850;margin-bottom:6px}
.hero-title{font-size:1.36rem;font-weight:850;letter-spacing:-.025em;line-height:1.12;color:#fff}
.hero-copy{font-size:.77rem;color:#8fa3b9;margin-top:6px;max-width:820px;line-height:1.45}

.metric-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:8px 0 14px}
.metric-card{border:1px solid var(--g-border-soft);background:linear-gradient(180deg,#0e1b2b,#0b1725);border-radius:11px;padding:10px 12px;min-width:0}
.metric-label{font-size:.58rem;color:#71869d;font-weight:850;letter-spacing:.12em;text-transform:uppercase}
.metric-value{font-size:.98rem;color:#fff;font-weight:820;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric-sub{font-size:.64rem;color:#657990;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.section-kicker{font-size:.61rem;letter-spacing:.14em;font-weight:880;color:#7698bf;margin:17px 0 6px;text-transform:uppercase}
.section-title{font-size:1.23rem;font-weight:840;color:#fff;letter-spacing:-.02em}
.section-copy{font-size:.76rem;color:#899db3;line-height:1.48;margin:4px 0 13px;max-width:980px}

/* Conference calendar */
.calendar-shell{border:1px solid #20334c;border-radius:13px;overflow:hidden;background:#091522;box-shadow:0 13px 40px rgba(0,0,0,.12)}
.calendar-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.gc{border-collapse:separate;border-spacing:0;min-width:1410px;width:100%;font-size:10px}
.gc th,.gc td{border-right:1px solid rgba(148,172,199,.11);border-bottom:1px solid rgba(148,172,199,.11);text-align:center;vertical-align:middle}
.gc thead th{position:sticky;top:0;z-index:4;background:#101f31;padding:9px 4px;min-width:82px}
.gc .school{position:sticky;left:0;z-index:5;background:#0d1a2a;min-width:172px;width:172px;text-align:left;padding:8px 10px}
.gc .school-head{background:#101f31!important;color:#71869d;font-size:8px;letter-spacing:.13em}
.gc tr:hover .school,.gc tr:hover td{background-color:#112237}
.school-line{display:flex;align-items:center;gap:9px;font-size:10px;font-weight:760;color:#edf3f9;white-space:nowrap}
.team-logo{object-fit:contain;display:block;flex:0 0 auto}
.team-logo{-webkit-user-drag:none;user-select:none;pointer-events:none}
.logo-fallback{display:flex;border:1px solid rgba(255,255,255,.14);border-radius:50%;align-items:center;justify-content:center;color:#9fb0c3;font-size:9px;font-weight:820;flex:0 0 auto;background:#14243a}
.week-label{display:block;color:#fff;font-size:9px;font-weight:860}.week-date{display:block;color:#71849a;font-size:8px;margin-top:2px;font-weight:650}
.gc td{padding:5px 4px;height:78px;min-width:82px;background:#091522}
.gc td.empty{color:#2d3e54;font-size:15px}.open-dot{opacity:.55}
.game-tile{min-height:66px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4px 2px;border-radius:8px}
.opp{font-weight:760;color:#f1f5f9;font-size:9px;line-height:1.08;margin-top:2px;max-width:75px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mini-meta{display:flex;gap:4px;align-items:center;font-size:7px;color:#75889f;margin-top:3px}.mini-site{font-weight:900;padding:1px 4px;border-radius:4px}.site-h{color:#59d49b}.site-a{color:#74adff}.site-n{color:#e6bc6f}

/* Team calendar */
.team-hero{display:flex;align-items:center;gap:15px;padding:14px 16px;border:1px solid var(--g-border-soft);border-radius:13px;background:linear-gradient(135deg,#0f1e30,#0b1725);margin-bottom:11px}
.team-hero-name{font-size:1.22rem;font-weight:850;color:#fff}.team-hero-meta{font-size:.72rem;color:#8498ae;margin-top:3px}
.tc-grid{display:grid;grid-template-columns:repeat(5,minmax(145px,1fr));gap:8px;margin-top:.5rem}
.tc-card{border:1px solid var(--g-border-soft);border-radius:12px;padding:10px 11px;min-height:158px;text-align:center;background:linear-gradient(180deg,#0f1e2f,#0b1725);display:flex;flex-direction:column;align-items:center;justify-content:flex-start}
.tc-card.is-open{background:#091522;border-style:dashed;opacity:.78}
.tc-card-top{width:100%;display:flex;justify-content:space-between;color:#70849b;font-size:8px;font-weight:850;letter-spacing:.05em;margin-bottom:11px}
.tc-logo{height:56px;display:flex;align-items:center;justify-content:center}.tc-opp{font-size:12px;font-weight:820;color:#fff;line-height:1.12;margin-top:4px}.tc-date-detail{font-size:9px;color:#8195ab;margin:4px 0 7px}
.site-badge{font-size:7px;font-weight:900;letter-spacing:.07em;padding:3px 7px;border-radius:999px;border:1px solid currentColor}.tc-empty-icon{color:#34475e;font-size:21px;margin-top:16px}.tc-open{font-size:9px;color:#70849b;font-weight:720;margin-top:6px}.tc-open-sub{font-size:8px;color:#4e6279;margin-top:3px}
.tba-row{display:flex;align-items:center;gap:10px;border:1px solid var(--g-border-soft);background:#0c1928;border-radius:10px;padding:9px 11px;margin:6px 0;color:#f1f5fa}.tba-row span{font-size:10px;color:#7d90a6}

/* Decision / result system */
.decision-card{display:grid;grid-template-columns:34px 1fr;gap:11px;align-items:flex-start;border-radius:12px;padding:12px 13px;margin:9px 0;border:1px solid var(--g-border-soft);background:#0d1a2a}
.decision-icon{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:950;font-size:15px}
.decision-title{font-size:.84rem;font-weight:840;color:#fff}.decision-body{font-size:.74rem;color:#9aadc1;line-height:1.45;margin-top:2px}.decision-detail{font-size:.68rem;color:#6f849b;margin-top:4px}
.decision-success{border-color:rgba(52,199,133,.32);background:linear-gradient(90deg,rgba(52,199,133,.08),#0d1a2a 32%)}.decision-success .decision-icon{background:rgba(52,199,133,.13);color:#59dda2}
.decision-conflict{border-color:rgba(239,91,103,.35);background:linear-gradient(90deg,rgba(239,91,103,.09),#0d1a2a 32%)}.decision-conflict .decision-icon{background:rgba(239,91,103,.13);color:#ff8e98}
.decision-info{border-color:rgba(79,140,255,.3);background:linear-gradient(90deg,rgba(79,140,255,.08),#0d1a2a 32%)}.decision-info .decision-icon{background:rgba(79,140,255,.12);color:#86b2ff}
.board-header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;padding:13px 14px;border:1px solid var(--g-border-soft);background:#0c1928;border-radius:12px;margin:14px 0 9px}
.board-legend{display:flex;align-items:center;gap:6px;color:#778ca2;font-size:.66rem;white-space:nowrap}.legend-dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-left:6px}.legend-current{background:var(--g-blue)}.legend-clean{background:var(--g-green)}.legend-conflict{background:var(--g-red)}

.result-card{border:1px solid #22364f;border-radius:13px;background:linear-gradient(180deg,#0f1c2c,#0a1624);overflow:hidden;margin:10px 0 14px;box-shadow:0 12px 34px rgba(0,0,0,.12)}
.result-top{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:13px 14px;border-bottom:1px solid var(--g-border-soft)}
.result-rank{font-size:.66rem;color:#71869d;font-weight:850;letter-spacing:.09em}.result-title{font-size:.94rem;font-weight:850;color:#fff;margin-top:2px}.result-score{display:flex;align-items:center;justify-content:center;min-width:58px;height:32px;border-radius:999px;background:rgba(52,199,133,.11);border:1px solid rgba(52,199,133,.25);color:#69dfa9;font-weight:900;font-size:.76rem}
.result-summary{padding:12px 14px;color:#91a5bb;font-size:.75rem;line-height:1.47;border-bottom:1px solid var(--g-border-soft)}
.result-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-bottom:1px solid var(--g-border-soft)}
.result-kpi{padding:10px 13px;border-right:1px solid var(--g-border-soft)}.result-kpi:last-child{border-right:0}.result-kpi-label{font-size:.56rem;color:#677d94;letter-spacing:.11em;font-weight:880}.result-kpi-value{font-size:.82rem;color:#eef4fa;font-weight:810;margin-top:3px}
.move-table{width:100%;border-collapse:collapse;font-size:.72rem}.move-table th{color:#71869d;font-size:.57rem;letter-spacing:.1em;text-align:left;padding:9px 13px;background:#0b1725}.move-table td{padding:10px 13px;border-top:1px solid var(--g-border-soft);color:#dfe7ef}.move-table td:nth-child(2),.move-table td:nth-child(3){white-space:nowrap}.move-arrow{color:#70869e;padding:0 6px}
.result-note{display:flex;gap:9px;align-items:flex-start;padding:10px 13px;border-top:1px solid var(--g-border-soft);color:#7f93aa;font-size:.67rem;line-height:1.4}.result-note strong{color:#a8b9ca}
.issue-summary{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid rgba(239,91,103,.28);background:linear-gradient(90deg,rgba(239,91,103,.07),#0d1a2a 45%);border-radius:12px;padding:11px 13px;margin:9px 0}.issue-summary strong{color:#fff;font-size:.82rem}.issue-summary span{color:#8ea2b8;font-size:.69rem}.issue-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:8px 0}.issue-card{border:1px solid var(--g-border-soft);background:#0a1624;border-radius:10px;padding:9px 10px}.issue-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.issue-key{font-size:.71rem;font-weight:850;color:#eef4fa}.issue-badge{font-size:.58rem;font-weight:900;color:#ff9ba4;border:1px solid rgba(239,91,103,.25);background:rgba(239,91,103,.08);padding:3px 6px;border-radius:999px}.issue-stats{font-size:.64rem;color:#7f93aa;margin-top:5px}.issue-games{font-size:.62rem;color:#aab9c8;margin-top:5px;line-height:1.35}.issue-action{font-size:.61rem;color:#79aaf0;margin-top:6px}@media(max-width:900px){.issue-grid{grid-template-columns:1fr}}
.parity-impact{padding:11px 13px;border-top:1px solid var(--g-border-soft)}.parity-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:7px}.parity-row{display:grid;grid-template-columns:105px 1fr 18px 1fr;align-items:center;gap:7px;border:1px solid var(--g-border-soft);border-radius:9px;background:#0a1624;padding:8px 9px;font-size:.65rem}.parity-key{font-weight:800;color:#bdcad7}.parity-old{color:#8194a9}.parity-new{color:#c8d5e2}.parity-arrow{color:#536a83;text-align:center}

/* Compact pills */
.status-chip{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:5px 8px;font-size:.62rem;font-weight:820;border:1px solid var(--g-border-soft);background:#0b1725;color:#a0b1c3}.status-chip.good{color:#62dca4;border-color:rgba(52,199,133,.24);background:rgba(52,199,133,.07)}.status-chip.bad{color:#ff8e98;border-color:rgba(239,91,103,.24);background:rgba(239,91,103,.07)}.status-chip.neutral{color:#8fb8ff;border-color:rgba(79,140,255,.24);background:rgba(79,140,255,.07)}

@media(max-width:1050px){.block-container{padding-left:.85rem;padding-right:.85rem}.metric-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.tc-grid{grid-template-columns:repeat(3,minmax(140px,1fr))}.result-kpis{grid-template-columns:1fr 1fr}.parity-list{grid-template-columns:1fr}.brand-status{display:none}.board-header{align-items:flex-start;flex-direction:column}}
@media(max-width:640px){.hero{padding:14px;align-items:flex-start;flex-direction:column}.hero-title{font-size:1.2rem}.metric-strip{grid-template-columns:1fr 1fr}.tc-grid{grid-template-columns:repeat(2,minmax(128px,1fr))}.gc .school{min-width:140px;width:140px}.result-kpis{grid-template-columns:1fr 1fr}.board-legend{white-space:normal;flex-wrap:wrap}}
</style>
""", unsafe_allow_html=True)

# ---- Brand ----
st.markdown(
    '<div class="brand-row">'
    '<div class="brand-lockup"><div class="brand-mark">G</div><div>'
    '<div class="brand-name">COLLEGE FOOTBALL</div><div class="brand-sub">NON-CONFERENCE SCHEDULING OPTIMIZER</div>'
    '</div></div>'
    '<div class="brand-status"><span class="status-dot"></span>Optimizer online</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div>'
    '<div class="hero-kicker">NATIONAL SCHEDULING COMMAND CENTER</div>'
    '<div class="hero-title">Plan, test and optimize every non-conference move.</div>'
    '<div class="hero-copy">A single operational workspace for future-game inventory, conference parity, buy-game matching, A4 requirements and minimum-disruption schedule repair.</div>'
    '</div><div><span class="status-chip good">● CP-SAT READY</span></div></div>',
    unsafe_allow_html=True,
)

# ---- Source + year controls ----
st.markdown('<div class="section-kicker">WORKSPACE</div>', unsafe_allow_html=True)
ctrl1, ctrl2 = st.columns([1.4, 1])
with ctrl1:
    source_mode = st.selectbox("Data source", ["Real public schedule data", "Demo"], index=0, label_visibility="visible")

real_teams_df = None
real_games_df = None
scrape_errors = []
if source_mode == "Real public schedule data":
    with st.spinner("Syncing public FBS/FCS future schedules…"):
        try:
            real_teams_df, real_games_df, scrape_errors = scrape_fbschedules_public()
        except Exception as exc:
            st.error(f"The public schedule sync failed: {type(exc).__name__}: {exc}")
            st.stop()
    if real_teams_df is None or real_teams_df.empty:
        st.error("No team data was returned from the public schedule sync.")
        st.stop()
    available_years = sorted(int(y) for y in real_games_df["season"].dropna().unique()) if len(real_games_df) else list(range(2027, 2038))
    with ctrl2:
        default_idx = available_years.index(2028) if 2028 in available_years else 0
        season = st.selectbox("Season", available_years, index=default_idx)
    store = build_real_store(real_teams_df, real_games_df, season)
    year_games = real_games_df[real_games_df["season"] == season].copy()
    year_games["game_id"] = [f"real{season}_{int(idx)+1}" for idx in year_games.index]
else:
    store = build_demo_store()
    with ctrl2:
        season = st.selectbox("Season", sorted({g.season for g in store.games.values()}), index=0)
    year_games = pd.DataFrame([g.__dict__ for g in store.games.values()])

year_games = apply_workspace_moves(store, year_games, season)
optimizer = AdvancedNonConferenceOptimizer(store)

if source_mode == "Real public schedule data":
    fbs_count = int((real_teams_df["subdivision"] == "FBS").sum())
    fcs_count = int((real_teams_df["subdivision"] == "FCS").sum())
    commitments = len(year_games)
    metrics = [
        ("Season", str(season), "Active workspace"),
        ("Teams", f"{len(real_teams_df):,}", f"{fbs_count} FBS · {fcs_count} FCS"),
        ("Known games", f"{commitments:,}", "Dated + TBA commitments"),
        ("Data status", "PUBLIC TEST", "Authoritative intent data not connected"),
    ]
else:
    metrics = [
        ("Season", str(season), "Synthetic test year"),
        ("Teams", f"{len(store.teams):,}", "Demo universe"),
        ("Known games", f"{len(store.games):,}", "Synthetic commitments"),
        ("Data status", "DEMO", "Optimizer test mode"),
    ]
st.markdown(
    '<div class="metric-strip">' + ''.join(
        f'<div class="metric-card"><div class="metric-label">{_html_escape(a)}</div><div class="metric-value">{_html_escape(b)}</div><div class="metric-sub">{_html_escape(c)}</div></div>'
        for a,b,c in metrics
    ) + '</div>',
    unsafe_allow_html=True,
)

if source_mode == "Real public schedule data":
    st.caption("Public-data prototype · Blank dates are potential slots, not confirmed availability. Production scheduling data would supply true needs, flexibility, guarantees, and moveability.")

workspace_count = len(_workspace_moves(season))
if workspace_count:
    wc1, wc2 = st.columns([5, 1])
    with wc1:
        _render_move_outcome("info", "What-if workspace active", f"{workspace_count} manually tested move{'s' if workspace_count != 1 else ''} are applied to this session.", "Calendars, parity and optimizer requests use the proposed workspace state.")
    with wc2:
        if st.button("Reset moves", key=f"reset_workspace_{season}", use_container_width=True):
            _clear_workspace_moves(season)
            st.rerun()


def parity_table(season: int) -> pd.DataFrame:
    rows = []
    for week in range(0, 14):
        parity = optimizer.conference_parity(store.copy_games(), season, week)
        for conference, value in parity.items():
            status = "EVEN" if value.startswith("EVEN") else "ODD"
            rows.append({"Week": week, "Conference": conference, "Status": status, "Detail": value})
    return pd.DataFrame(rows)


def _parity_status_word(value: str) -> str:
    value = str(value or "")
    if value.startswith("EVEN"):
        return "EVEN"
    if value.startswith("ODD"):
        return "ODD"
    return value or "—"


def render_solution(sol, idx: int):
    """Render a decision-oriented enterprise result instead of raw warning boxes."""
    md = getattr(sol, "metadata", {}) or {}
    mode = str(md.get("mode", ""))
    move_count = len(sol.moves)
    additional = md.get("additional_moves")
    solver_status = str(md.get("solver_status", "—"))
    solver_seconds = md.get("solver_seconds", "—")
    scope_before = md.get("scope_before_bad")
    scope_after = md.get("scope_after_bad")
    before_bad = md.get("before_bad_count")
    after_bad = md.get("after_bad_count")

    if mode == "move":
        outcome = "Clean move" if move_count == 1 and int(additional or 0) == 0 else "Repair path"
        impact_value = "0 extra" if int(additional or 0) == 0 else f"{int(additional)} extra"
        impact_label = "SECONDARY MOVES"
    else:
        outcome = "Optimized"
        impact_value = f"{scope_after if scope_after is not None else (after_bad if after_bad is not None else '—')} remain"
        impact_label = "REQUESTED ISSUES"

    kpi_scope = "—"
    if scope_before is not None and scope_after is not None:
        kpi_scope = f"{scope_before} → {scope_after}"
    elif before_bad is not None and after_bad is not None:
        kpi_scope = f"{before_bad} → {after_bad}"

    st.markdown(
        '<div class="result-card">'
        '<div class="result-top">'
        f'<div><div class="result-rank">OPTION {idx}</div><div class="result-title">{_html_escape(sol.title)}</div></div>'
        f'<div class="result-score">{sol.score:.0f}/100</div>'
        '</div>'
        f'<div class="result-summary">{_html_escape(sol.explanation)}</div>'
        '<div class="result-kpis">'
        f'<div class="result-kpi"><div class="result-kpi-label">OUTCOME</div><div class="result-kpi-value">{_html_escape(outcome)}</div></div>'
        f'<div class="result-kpi"><div class="result-kpi-label">GAMES MOVED</div><div class="result-kpi-value">{move_count}</div></div>'
        f'<div class="result-kpi"><div class="result-kpi-label">{_html_escape(impact_label)}</div><div class="result-kpi-value">{_html_escape(impact_value)}</div></div>'
        f'<div class="result-kpi"><div class="result-kpi-label">PARITY / SCOPE</div><div class="result-kpi-value">{_html_escape(kpi_scope)}</div></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if sol.moves:
        rows = []
        for m in sol.moves:
            rows.append(
                '<tr>'
                f'<td><strong>{_html_escape(m.away_team)} @ {_html_escape(m.home_team)}</strong></td>'
                f'<td>Week {m.from_week}</td>'
                '<td class="move-arrow">→</td>'
                f'<td><strong>Week {m.to_week}</strong></td>'
                '</tr>'
            )
        st.markdown(
            '<div class="result-card" style="margin-top:-6px">'
            '<table class="move-table"><thead><tr><th>GAME</th><th>CURRENT</th><th></th><th>PROPOSED</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        _render_move_outcome("success", "No schedule movement required", "The requested condition is already satisfied in the current workspace.")

    # Keep technical solver notes quiet. Remaining parity counts are rendered
    # below as an inspectable issue register instead of an unexplained warning.
    for warning in sol.warnings:
        warning_text = str(warning)
        if "remain" in warning_text.lower() and "parity" in warning_text.lower():
            continue
        if "global optimality" in warning_text.lower():
            note = "Best solution found within the interactive time limit. Global optimality was not required before returning the recommendation."
            with st.expander("Solver details", expanded=False):
                st.caption(note)
        else:
            _render_move_outcome("info", "Solver note", warning_text)

    unresolved = list(md.get("unresolved_issues") or [])
    if unresolved:
        by_conf: Dict[str, int] = {}
        for issue in unresolved:
            conf = str(issue.get("conference", "Unknown"))
            by_conf[conf] = by_conf.get(conf, 0) + 1
        conf_summary = " · ".join(f"{c} {n}" for c, n in sorted(by_conf.items(), key=lambda x: (-x[1], x[0])))
        st.markdown(
            '<div class="issue-summary"><div><strong>' + str(len(unresolved)) + ' unresolved conference/week parity issues</strong><br>'
            '<span>Every remaining issue is listed below — conference, week, current state and the games creating the non-conference inventory.</span></div>'
            '<span>' + _html_escape(conf_summary) + '</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"View all {len(unresolved)} unresolved issues", expanded=False):
            cards = []
            for issue in unresolved:
                games = list(issue.get("games") or [])
                game_text = " · ".join(games[:4]) if games else "No dated non-conference games identified"
                if len(games) > 4:
                    game_text += f" · +{len(games)-4} more"
                cards.append(
                    '<div class="issue-card">'
                    '<div class="issue-head"><div class="issue-key">' + _html_escape(issue.get("conference")) + ' · Week ' + _html_escape(issue.get("week")) + '</div><div class="issue-badge">ODD</div></div>'
                    '<div class="issue-stats">' + _html_escape(issue.get("available")) + ' conference teams available · ' + _html_escape(issue.get("nonconf_count")) + ' teams in non-conference games</div>'
                    '<div class="issue-games">' + _html_escape(game_text) + '</div>'
                    '<div class="issue-action">' + _html_escape(issue.get("next_action")) + '</div>'
                    '</div>'
                )
            st.markdown('<div class="issue-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

            option_map = {
                f"{i.get('conference')} · Week {i.get('week')} — {i.get('available')} available": i
                for i in unresolved
            }
            selected_issue_label = st.selectbox("Analyze one unresolved issue", list(option_map), key=f"issue_analyze_{idx}_{md.get('season', 'x')}")
            selected_issue = option_map[selected_issue_label]
            if st.button("Find the least-disruptive fix", key=f"issue_fix_{idx}_{selected_issue.get('conference')}_{selected_issue.get('week')}", use_container_width=True):
                fix_intent = Intent(
                    action="MAKE_CONFERENCE_EVEN",
                    season=int(md.get("season") or season),
                    target_week=int(selected_issue.get("week")),
                    conference=str(selected_issue.get("conference")),
                    preserve_fbs_conference_parity=True,
                    max_additional_moves=4,
                    summary="Resolve selected parity issue",
                )
                fixes = optimizer.solve(fix_intent)
                if fixes:
                    st.markdown('<div class="section-kicker">BEST NEXT FIX</div>', unsafe_allow_html=True)
                    # Use a compact move table here to avoid recursively opening another issue register.
                    fix = fixes[0]
                    _render_move_outcome("success", "Least-disruptive path found", fix.explanation)
                    if fix.moves:
                        fix_df = pd.DataFrame([{
                            "Game": f"{m.away_team} @ {m.home_team}",
                            "Current": f"Week {m.from_week}",
                            "Proposed": f"Week {m.to_week}",
                        } for m in fix.moves])
                        st.dataframe(fix_df, use_container_width=True, hide_index=True)
                else:
                    _render_move_outcome("conflict", "No direct fix found", "The current public-data constraint graph could not resolve this issue within the move limit.")

    if sol.parity_after:
        changed = []
        keys = sorted(set(sol.parity_before) | set(sol.parity_after))
        for key in keys:
            before = sol.parity_before.get(key, "—")
            after = sol.parity_after.get(key, "—")
            if before != after:
                changed.append((key, before, after))
        if changed:
            rows = []
            for key, before, after in changed[:24]:
                rows.append(
                    '<div class="parity-row">'
                    f'<div class="parity-key">{_html_escape(key)}</div>'
                    f'<div class="parity-old">{_html_escape(before)}</div>'
                    '<div class="parity-arrow">→</div>'
                    f'<div class="parity-new">{_html_escape(after)}</div>'
                    '</div>'
                )
            st.markdown(
                '<div class="result-card"><div class="parity-impact"><div class="section-kicker" style="margin-top:0">PARITY IMPACT</div>'
                '<div class="section-copy" style="margin-bottom:6px">Only conference/week states changed by this recommendation are shown.</div>'
                '<div class="parity-list">' + ''.join(rows) + '</div></div></div>',
                unsafe_allow_html=True,
            )

    if solver_status != "—":
        st.caption(f"{solver_status} · {solver_seconds}s · {getattr(optimizer, 'engine_name', 'Optimizer')}")


# ---- Product navigation ----
tab_chat, tab_calendar, tab_opt, tab_health, tab_schedule, tab_needs = st.tabs([
    "Ask Optimizer", "Schedule Board", "Optimization Center", "Conference Health", "Schedule Data", "Open Market"
])

with tab_chat:
    st.markdown('<div class="section-kicker">ASK THE OPTIMIZER</div><div class="section-title">What are you trying to accomplish?</div>', unsafe_allow_html=True)
    if source_mode == "Demo":
        helper = "Try: Move Georgia vs McNeese to Week 2 and solve the displaced Tarleton game without creating a new FBS parity problem."
    else:
        helper = f"Ask naturally — for example: ‘Get the SEC even in Week 2,’ ‘Grambling needs to buy a game in 2029,’ or ‘Find Georgia an A4 opponent in {season}.’"
    st.markdown(f'<div class="section-copy">{_html_escape(helper)}</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Describe the scheduling problem…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        intent, parser_name = parse_intent(prompt, store.teams.keys())
        if intent.season is None:
            intent.season = season
        with st.chat_message("assistant"):
            run_optimizer = optimizer
            if source_mode == "Real public schedule data" and intent.season != season:
                run_store = build_real_store(real_teams_df, real_games_df, int(intent.season))
                run_optimizer = AdvancedNonConferenceOptimizer(run_store)
            started = time.perf_counter()
            with st.spinner(f"Searching {intent.season} scheduling options…"):
                solutions = run_optimizer.solve(intent)
            elapsed = time.perf_counter() - started
            parser_label = "AI intent" if "openai" in parser_name.lower() else "Local intent"
            st.caption(f"{parser_label} · Optimizer {elapsed:.2f}s")
            if intent.action == "OPTIMIZE_NATIONAL" and (intent.target_weeks or intent.all_conferences or intent.conferences):
                scope_bits = []
                if intent.all_conferences:
                    scope_bits.append("All FBS conferences")
                elif intent.conferences:
                    scope_bits.append(", ".join(intent.conferences))
                if intent.target_weeks:
                    scope_bits.append("Weeks " + ", ".join(str(w) for w in intent.target_weeks))
                _render_move_outcome("info", "Optimization scope", " · ".join(scope_bits))
            with st.expander("How the optimizer interpreted your request", expanded=False):
                st.json(intent.__dict__)
            if not solutions:
                _render_move_outcome("conflict", "No feasible result", "The optimizer could not satisfy the request with the currently loaded constraints.", "Public data does not yet include true school intent, contract flexibility or guarantee requirements.")
            else:
                _render_move_outcome("success", "Solution found", f"{len(solutions)} feasible option{'s' if len(solutions) != 1 else ''} evaluated and ranked.")
                for i, sol in enumerate(solutions, start=1):
                    render_solution(sol, i)

with tab_calendar:
    st.markdown('<div class="section-kicker">SCHEDULE BOARD</div><div class="section-title">See the season. Test the move.</div><div class="section-copy">Conference mode now includes a direct-manipulation Week 0–13 board: drag any game card straight to the week you want. The logo matrix below remains the team-by-week visual reference. H = home, A = away, N = neutral.</div>', unsafe_allow_html=True)
    if source_mode != "Real public schedule data":
        st.info("Calendar view uses the real public scheduling dataset. Switch the data source above to Real public schedule data.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            view_mode = st.radio("View", ["Conference", "Team"], horizontal=True)
        if view_mode == "Conference":
            conferences = sorted(real_teams_df[(real_teams_df["subdivision"] == "FBS") & (real_teams_df["conference"] != "Unknown")]["conference"].dropna().unique())
            default_conf = conferences.index("SEC") if "SEC" in conferences else 0
            with c2:
                conference = st.selectbox("Conference", conferences, index=default_conf)
            st.markdown(f'<div class="section-kicker">{_html_escape(conference)} · {season}</div>', unsafe_allow_html=True)
            render_conference_drag_board(store, optimizer, real_teams_df, season, conference)
            st.markdown('<div class="section-kicker" style="margin-top:1rem">TEAM-BY-WEEK MATRIX</div>', unsafe_allow_html=True)
            render_conference_calendar(year_games, real_teams_df, season, conference)
        else:
            team_names = sorted(real_teams_df["name"].dropna().unique())
            default_team = team_names.index("Georgia") if "Georgia" in team_names else 0
            with c2:
                team = st.selectbox("Team", team_names, index=default_team)
            render_team_calendar(year_games, real_teams_df, season, team)
            render_drag_move_lab(store, optimizer, season, team)


with tab_opt:
    st.markdown(
        '<div class="section-kicker">OPTIMIZATION CENTER</div>'
        '<div class="section-title">Turn every scheduling report into a solution</div>'
        '<div class="section-copy">One CP-SAT engine powers the report scenarios below. It minimizes game movement, protects healthy conference/week parity, and applies the selected report objective as a mathematical optimization problem.</div>',
        unsafe_allow_html=True,
    )
    engine_label = optimizer.engine_name
    engine_state = "READY" if ORTOOLS_AVAILABLE else "FALLBACK"
    st.markdown(
        f'<div class="metric-strip">'
        f'<div class="metric-card"><div class="metric-label">ENGINE</div><div class="metric-value">{_html_escape(engine_label)}</div><div class="metric-sub">Advanced constraint programming</div></div>'
        f'<div class="metric-card"><div class="metric-label">STATUS</div><div class="metric-value">{engine_state}</div><div class="metric-sub">Interactive solve target: ≤ {optimizer.time_limit_seconds:.0f}s</div></div>'
        f'<div class="metric-card"><div class="metric-label">OBJECTIVE</div><div class="metric-value">MINIMUM DISRUPTION</div><div class="metric-sub">Parity → moves → date distance</div></div>'
        f'<div class="metric-card"><div class="metric-label">SEASON</div><div class="metric-value">{season}</div><div class="metric-sub">Active optimization workspace</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if not ORTOOLS_AVAILABLE:
        _render_move_outcome("conflict", "Advanced solver unavailable", "OR-Tools is not installed in this runtime.", "Commit the updated requirements.txt and Streamlit will install CP-SAT automatically.")

    report = st.selectbox(
        "Scheduling report / scenario",
        [
            "Odd / Even",
            "Scheduled Games / Move Repair",
            "# of Controlled Games",
            "# of FCS Games / Week",
            "Market Report",
            "Teams Needing Games",
            "Pending Games",
            "Bye Report",
            "Optimize National Schedule",
        ],
        key="optimization_report",
    )

    def _run_and_render(intent: Intent, button_key: str, label: str = "Run optimization"):
        if st.button(label, key=button_key, type="primary", use_container_width=True):
            started = time.perf_counter()
            with st.spinner("Solving the feasible scheduling graph…"):
                sols = optimizer.solve(intent)
            elapsed = time.perf_counter() - started
            st.caption(f"{optimizer.engine_name} · {optimizer.last_solver_status} · {elapsed:.2f}s")
            if not sols:
                _render_move_outcome("conflict", "No feasible solution", "The current constraints do not permit a valid schedule change.", "Try a different target week or connect authoritative intent/flexibility data for a richer feasible graph.")
            else:
                for i, sol in enumerate(sols, 1):
                    render_solution(sol, i)

    conferences = optimizer.store.fbs_conferences()
    default_sec = conferences.index("SEC") if "SEC" in conferences else 0

    if report == "Odd / Even":
        st.markdown('<div class="section-kicker">ODD / EVEN</div><div class="section-copy">Force the selected conference/week back to an even number of teams available for conference play while preventing new parity problems in weeks that are currently healthy.</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            conf = st.selectbox("Conference", conferences, index=default_sec, key="opt_parity_conf")
        with b:
            week = st.selectbox("Week", list(range(0, 14)), index=2, key="opt_parity_week")
        current = optimizer.conference_parity(store.copy_games(), season, week).get(conf, "Unknown")
        _render_move_outcome("info", f"{conf} · Week {week}", current, "This is the current conference parity state before optimization.")
        _run_and_render(Intent(action="MAKE_CONFERENCE_EVEN", season=season, target_week=week, conference=conf, max_additional_moves=6, summary="Optimization Center odd/even"), "run_parity")

    elif report == "Scheduled Games / Move Repair":
        st.markdown('<div class="section-kicker">SCHEDULED GAMES</div><div class="section-copy">Choose a known non-conference game and force it into a new week. The optimizer first attempts the requested move only. It relocates other games only when they are directly displaced or when you explicitly ask to preserve conference parity.</div>', unsafe_allow_html=True)
        season_games = sorted([g for g in store.games.values() if g.season == season], key=lambda g: (g.week, g.home_team, g.away_team))
        if not season_games:
            st.info("No dated games are loaded for this season.")
        else:
            labels = {f"W{g.week} · {g.away_team} @ {g.home_team}": g for g in season_games}
            game_label = st.selectbox("Game", list(labels), key="opt_move_game")
            target_week = st.selectbox("Move to week", list(range(0, 14)), index=min(13, labels[game_label].week + 1), key="opt_move_week")
            g = labels[game_label]
            _run_and_render(Intent(action="MOVE_GAME", season=season, target_week=target_week, team_a=g.home_team, team_b=g.away_team, preserve_fbs_conference_parity=False, max_additional_moves=6, summary="Optimization Center scheduled-game repair"), "run_move")

    elif report == "# of Controlled Games":
        st.markdown('<div class="section-kicker">CONTROLLED GAME DISTRIBUTION</div><div class="section-copy">Balance a conference’s weekly non-conference inventory while preserving currently healthy FBS parity. In the public-data prototype this uses known non-conference team appearances as the controlled-inventory proxy.</div>', unsafe_allow_html=True)
        conf = st.selectbox("Conference", conferences, index=default_sec, key="opt_control_conf")
        _run_and_render(Intent(action="BALANCE_CONTROLLED_GAMES", season=season, conference=conf, preserve_fbs_conference_parity=True, max_additional_moves=12, summary="Balance controlled games"), "run_controlled", "Optimize weekly inventory")

    elif report == "# of FCS Games / Week":
        st.markdown('<div class="section-kicker">FCS GAMES / WEEK</div><div class="section-copy">Redistribute known FBS–FCS games toward a more even weekly cadence while minimizing moves and keeping FBS conferences schedulable.</div>', unsafe_allow_html=True)
        _run_and_render(Intent(action="BALANCE_FCS_GAMES", season=season, preserve_fbs_conference_parity=True, max_additional_moves=18, summary="Balance FCS games by week"), "run_fcs_balance", "Optimize FCS distribution")

    elif report == "Optimize National Schedule":
        st.markdown('<div class="section-kicker">NATIONAL OPTIMIZATION</div><div class="section-copy">Solve the season as one network. The objective first minimizes FBS conference/week parity failures, then minimizes games moved and distance from current dates.</div>', unsafe_allow_html=True)
        _run_and_render(Intent(action="OPTIMIZE_NATIONAL", season=season, preserve_fbs_conference_parity=False, max_additional_moves=30, summary="Optimize national non-conference schedule"), "run_national", f"Optimize {season}")

    elif report == "Market Report":
        st.markdown('<div class="section-kicker">MARKET REPORT</div><div class="section-copy">Production mode maximizes fulfilled explicit buy/sell/A4 needs subject to mutual date availability. This requires the authoritative needs table; public schedule pages only show commitments, not school intent.</div>', unsafe_allow_html=True)
        if store.needs:
            st.dataframe(pd.DataFrame([n.__dict__ for n in store.needs if n.season == season]), use_container_width=True, hide_index=True)
            _run_and_render(Intent(action="OPTIMIZE_MARKET", season=season, preserve_fbs_conference_parity=True, summary="Optimize market report"), "run_market", "Optimize market matches")
        else:
            st.info("The solver path is built, but the public-data mode has no true ‘looking to buy/sell’ flags. Once connected to the authoritative scheduling data, these report rows become explicit optimization demand.")

    elif report == "Teams Needing Games":
        st.markdown('<div class="section-kicker">TEAMS NEEDING GAMES</div><div class="section-copy">This scenario needs authoritative NEED_FBS / NEED_FCS / NEED_A4 inventory. Public blank dates cannot safely be treated as a school asking for a game.</div>', unsafe_allow_html=True)
        if store.needs:
            st.dataframe(pd.DataFrame([n.__dict__ for n in store.needs if n.season == season]), use_container_width=True, hide_index=True)
        else:
            st.info("Data adapter ready: connect the Teams Needing Games report or underlying needs table to activate national maximum matching.")

    elif report == "Pending Games":
        st.markdown('<div class="section-kicker">PENDING GAMES</div><div class="section-copy">Pending games should become soft reservations in the production model: protected more strongly than an open slot but still movable if a higher-value national solution requires it.</div>', unsafe_allow_html=True)
        st.info("Public FBSchedules data does not expose authoritative pending-game status. The production data adapter should map pending rows into weighted soft constraints.")

    elif report == "Bye Report":
        st.markdown('<div class="section-kicker">BYE REPORT</div><div class="section-copy">A true bye optimizer requires the complete conference + non-conference schedule. A blank non-conference week is not necessarily a bye because a conference game may occupy it.</div>', unsafe_allow_html=True)
        st.info("The CP-SAT model is ready to accept blocked/bye/conference-game weeks from the authoritative scheduling system. Public non-conference data alone is intentionally not treated as authoritative bye data.")

with tab_health:
    st.markdown('<div class="section-kicker">CONFERENCE HEALTH</div><div class="section-title">Weekly FBS scheduling parity</div><div class="section-copy">After removing teams with known dated non-conference games, is each conference left with an even number of teams available for conference play?</div>', unsafe_allow_html=True)
    df = parity_table(season)
    if len(df):
        pivot = df.pivot(index="Conference", columns="Week", values="Status")
        st.dataframe(pivot, use_container_width=True)
        odd_rows = df[df["Status"] == "ODD"]
        if len(odd_rows):
            st.markdown('<div class="section-kicker">ODD-WEEK FLAGS</div>', unsafe_allow_html=True)
            st.dataframe(odd_rows[["Conference", "Week", "Detail"]], use_container_width=True, hide_index=True)

with tab_schedule:
    st.markdown('<div class="section-kicker">SCHEDULE DATA</div><div class="section-title">Known non-conference commitments</div>', unsafe_allow_html=True)
    if source_mode == "Real public schedule data":
        display_cols = ["date", "week", "away_team", "home_team", "neutral", "matchup_type", "away_conference", "home_conference"]
        year_df = year_games.copy()
        st.dataframe(year_df[[c for c in display_cols if c in year_df.columns]], use_container_width=True, hide_index=True, height=520)
        csv_bytes = year_df.to_csv(index=False).encode("utf-8")
        st.download_button(f"Download {season} CSV", csv_bytes, f"cfb_nonc_optimizer_{season}_public_snapshot.csv", "text/csv", use_container_width=False)
        if scrape_errors:
            with st.expander(f"Data warnings ({len(scrape_errors)})"):
                st.code("\n".join(scrape_errors[:100]))
    else:
        games_df = pd.DataFrame([g.__dict__ for g in store.games.values()])
        st.dataframe(games_df.sort_values(["season", "week", "home_team"]), use_container_width=True, hide_index=True)

with tab_needs:
    st.markdown('<div class="section-kicker">OPEN MARKET</div><div class="section-title">Potential scheduling inventory</div><div class="section-copy">Public data can identify teams with no known dated non-conference commitment. Production scheduling data would distinguish truly open, flexible, buy-game, A4, and blocked inventory.</div>', unsafe_allow_html=True)
    if source_mode == "Real public schedule data":
        candidate_week = st.select_slider("Week", options=list(range(0, 14)), value=2)
        base_games = store.copy_games()
        rows = []
        for team in store.teams.values():
            occupied = store.game_for_team_week(base_games, team.name, season, candidate_week) is not None
            if not occupied:
                rows.append({"Team": team.name, "Subdivision": team.subdivision, "Conference": team.conference, "A4": team.is_a4})
        cand = pd.DataFrame(rows).sort_values(["Subdivision", "Conference", "Team"])
        k1, k2, k3 = st.columns(3)
        with k1: subdivision_filter = st.selectbox("Level", ["All", "FBS", "FCS"])
        with k2: conf_opts = ["All"] + sorted(cand["Conference"].dropna().unique().tolist())
        with k2: conference_filter = st.selectbox("Conference filter", conf_opts)
        with k3: a4_only = st.checkbox("A4 only")
        if subdivision_filter != "All": cand = cand[cand["Subdivision"] == subdivision_filter]
        if conference_filter != "All": cand = cand[cand["Conference"] == conference_filter]
        if a4_only: cand = cand[cand["A4"] == True]
        st.dataframe(cand, use_container_width=True, hide_index=True, height=520)
    else:
        needs_df = pd.DataFrame([n.__dict__ for n in store.needs])
        if len(needs_df):
            st.dataframe(needs_df[needs_df["season"] == season], use_container_width=True, hide_index=True)
        else:
            st.info("No needs loaded.")
