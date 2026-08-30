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


@dataclass
class Intent:
    action: str
    season: Optional[int] = None
    target_week: Optional[int] = None
    conference: Optional[str] = None
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

    def solve_make_conference_even(self, intent: Intent) -> List[Solution]:
        if not intent.conference or intent.season is None or intent.target_week is None:
            return []
        base = self.store.copy_games()
        current = self.conference_parity(base, intent.season, intent.target_week).get(intent.conference)
        if current and current.startswith("EVEN"):
            return [Solution(title="Already even", moves=[], score=100, explanation=f"{intent.conference} is already even in Week {intent.target_week}: {current}.")]

        # Search moveable nonconference games involving a member of the conference.
        members = {t.name for t in self.store.conference_members(intent.conference)}
        candidates = [g for g in base.values() if g.season == intent.season and g.moveable and not g.locked and (g.home_team in members or g.away_team in members)]
        solutions: List[Solution] = []

        # Option A: move a conference member's nonconf game INTO target week.
        for game in candidates:
            if game.week == intent.target_week:
                continue
            temp_intent = Intent(
                action="MOVE_GAME", season=intent.season, target_week=intent.target_week,
                conference=intent.conference, team_a=game.home_team, team_b=game.away_team,
                preserve_fbs_conference_parity=True, max_additional_moves=intent.max_additional_moves,
                summary=intent.summary,
            )
            for sol in self.solve_move_game(temp_intent):
                status = sol.parity_after.get(f"{intent.conference} W{intent.target_week}", "")
                if status.startswith("EVEN"):
                    solutions.append(sol)

        # Option B: move an existing target-week nonconference game OUT.
        for game in candidates:
            if game.week != intent.target_week:
                continue
            for alt_week in self._candidate_weeks(game):
                temp_intent = Intent(
                    action="MOVE_GAME", season=intent.season, target_week=alt_week,
                    conference=intent.conference, team_a=game.home_team, team_b=game.away_team,
                    preserve_fbs_conference_parity=True, max_additional_moves=intent.max_additional_moves,
                    summary=intent.summary,
                )
                for sol in self.solve_move_game(temp_intent):
                    status = sol.parity_after.get(f"{intent.conference} W{intent.target_week}", "")
                    if status.startswith("EVEN"):
                        solutions.append(sol)
                if len(solutions) >= 20:
                    break

        uniq = {}
        for sol in solutions:
            sig = tuple(sorted((m.game_id, m.from_week, m.to_week) for m in sol.moves))
            if sig not in uniq or sol.score > uniq[sig].score:
                uniq[sig] = sol
        return sorted(uniq.values(), key=lambda s: (-s.score, len(s.moves)))[:5]

    def find_buy_games(self, intent: Intent) -> List[Solution]:
        if not intent.team_a or intent.season is None or intent.target_week is None:
            return []
        host = self.store.teams.get(intent.team_a)
        if not host or host.subdivision != "FBS":
            return []
        base_games = self.store.copy_games()
        if self.store.game_for_team_week(base_games, host.name, intent.season, intent.target_week):
            return []

        results: List[Solution] = []
        if self.store.needs:
            for need in self.store.needs:
                candidate = self.store.teams.get(need.team)
                if not candidate or candidate.subdivision != "FCS":
                    continue
                if need.season != intent.season or need.week != intent.target_week:
                    continue
                if need.location not in {"AWAY", "ANY"}:
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, intent.season, intent.target_week):
                    continue
                if intent.max_guarantee is not None and need.min_guarantee is not None and need.min_guarantee > intent.max_guarantee:
                    continue
                ask = f"${need.min_guarantee:,}+" if need.min_guarantee else "not specified"
                results.append(Solution(
                    title=f"{candidate.name} buy-game match",
                    moves=[],
                    score=90 if need.min_guarantee is None else max(50, 100 - (need.min_guarantee / max(intent.max_guarantee or need.min_guarantee, 1)) * 25),
                    explanation=f"{candidate.name} is available in Week {intent.target_week} and is seeking an away/buy game. Minimum guarantee: {ask}.",
                ))
        else:
            # Public-data fallback: identify FCS schools with no known dated game in that week.
            # This is a candidate list, not proof that the FCS program is actively seeking a buy game.
            for candidate in self.store.teams.values():
                if candidate.subdivision != "FCS":
                    continue
                if self.store.game_for_team_week(base_games, candidate.name, intent.season, intent.target_week):
                    continue
                results.append(Solution(
                    title=f"{candidate.name} — public-data candidate",
                    moves=[],
                    score=70,
                    explanation=(f"{candidate.name} has no known dated non-conference game in Week {intent.target_week} "
                                 "in the public snapshot. Confirm true availability and buy-game interest in Gridiron."),
                ))
        return sorted(results, key=lambda s: (-s.score, s.title))[:12]

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

    def _explain_moves(moves: List[Move], before: int, after: int) -> str:
        chain = " → ".join(f"{m.home_team}-{m.away_team} W{m.from_week}→W{m.to_week}" for m in moves)
        if after < before:
            parity = f"The move reduces affected FBS parity issues from {before} to {after}."
        elif after == before:
            parity = f"The move does not increase FBS parity issues ({after} remain in the affected weeks)."
        else:
            parity = f"The move increases parity issues from {before} to {after}."
        return f"Move chain: {chain}. {parity}"





INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["MOVE_GAME", "MAKE_CONFERENCE_EVEN", "FIND_BUY_GAME", "FIND_A4_GAME"]},
        "season": {"type": ["integer", "null"]},
        "target_week": {"type": ["integer", "null"]},
        "conference": {"type": ["string", "null"]},
        "team_a": {"type": ["string", "null"]},
        "team_b": {"type": ["string", "null"]},
        "preserve_fbs_conference_parity": {"type": "boolean"},
        "max_additional_moves": {"type": "integer", "minimum": 0, "maximum": 6},
        "opponent_class": {"type": "string", "enum": ["ANY", "FBS", "FCS", "A4"]},
        "location": {"type": "string", "enum": ["ANY", "HOME", "AWAY"]},
        "max_guarantee": {"type": ["integer", "null"]},
        "summary": {"type": "string"}
    },
    "required": ["action", "season", "target_week", "conference", "team_a", "team_b", "preserve_fbs_conference_parity", "max_additional_moves", "opponent_class", "location", "max_guarantee", "summary"],
    "additionalProperties": False
}


SYSTEM_INSTRUCTIONS = """You interpret requests for a college-football NON-CONFERENCE scheduling optimizer.
Do not solve the schedule yourself. Convert the user's request into the provided structured intent.
Definitions:
- MOVE_GAME: user names a specific existing matchup and wants it moved to a week.
- MAKE_CONFERENCE_EVEN: user primarily wants an FBS conference to have an even number of teams available for conference play in a week, without requiring a named specific game.
- FIND_BUY_GAME: an FBS school needs an FCS/buy-game opponent.
- FIND_A4_GAME: an A4 school needs an A4 nonconference opponent.
For a request like 'The SEC is odd in week 2 and I need to move Georgia vs McNeese to week 2 ...', use MOVE_GAME, conference SEC, team_a Georgia, team_b McNeese, target_week 2, and preserve parity true.
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
                "name": "gridiron_schedule_intent",
                "strict": True,
                "schema": INTENT_SCHEMA,
            }
        },
    )
    data = json.loads(response.output_text)
    data["team_a"] = _normalize_team(data.get("team_a"), team_names)
    data["team_b"] = _normalize_team(data.get("team_b"), team_names)
    return Intent(**data)


def parse_locally(text: str, team_names: Iterable[str]) -> Intent:
    """Small offline parser so the demo still works without an API key."""
    lower = text.lower()
    year_match = re.search(r"\b(20\d{2})\b", text)
    week_match = re.search(r"\bweek\s*(\d{1,2})\b", lower)
    season = int(year_match.group(1)) if year_match else None
    week = int(week_match.group(1)) if week_match else None

    conference = None
    for conf in ["SEC", "ACC", "Big Ten", "Big 12", "AAC", "Mountain West", "Sun Belt", "Conference USA", "MAC", "Pac-12"]:
        if conf.lower() in lower:
            conference = conf
            break

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

    if ("fcs" in lower or "buy game" in lower or "buy-game" in lower) and found:
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

    return Intent(
        action=action,
        season=season,
        target_week=week,
        conference=conference,
        team_a=team_a,
        team_b=team_b,
        preserve_fbs_conference_parity=True,
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
            "Version/18.0 Mobile/15E148 Safari/604.1 GridironOptimizerMVP/0.2"
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


def _week_for_2028(date_iso: str | None) -> int | None:
    if not date_iso:
        return None
    d = datetime.strptime(date_iso, "%Y-%m-%d").date()
    first_sat = date(2028, 8, 26)  # Week Zero
    days_until_sat = (5 - d.weekday()) % 7  # Monday=0, Saturday=5
    week_sat = d + timedelta(days=days_until_sat)
    return (week_sat - first_sat).days // 7


def _parse_2028_entries(soup: BeautifulSoup, current_slug: str, current_name: str, source_url: str) -> list[dict]:
    heading = None
    for tag in soup.find_all(["h4", "h3"]):
        if tag.get_text(" ", strip=True) == "2028":
            heading = tag
            break
    if not heading:
        return []

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
        return []

    rows = []
    for li in ul.find_all("li", recursive=False):
        txt = " ".join(li.get_text(" ", strip=True).split())
        m = re.match(r"^(TBA|\d{2}/\d{2})\s*-\s*(.+)$", txt, flags=re.I)
        if not m:
            continue
        date_token, rest = m.group(1).upper(), m.group(2).strip()
        date_iso = None
        if date_token != "TBA":
            try:
                date_iso = datetime.strptime(f"2028/{date_token}", "%Y/%m/%d").date().isoformat()
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
            "season": 2028,
            "date": date_iso or "TBA",
            "week": _week_for_2028(date_iso),
            "current_slug": current_slug,
            "team": current_name,
            "opponent_slug": opponent_slug,
            "opponent": opponent_name,
            "site_for_team": site,
            "neutral_location": neutral_location,
            "source_url": source_url,
        })
    return rows


def _scrape_one_team(url: str) -> tuple[dict, list[dict]]:
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
    }
    return team, _parse_2028_entries(soup, slug, name, url)


def _dedupe_scraped_games(raw_rows: list[dict], team_by_slug: dict[str, dict]) -> pd.DataFrame:
    known_dates: dict[tuple[str, str], set[str]] = {}
    for r in raw_rows:
        a = r["current_slug"] or r["team"].lower()
        b = r["opponent_slug"] or r["opponent"].lower()
        pair = tuple(sorted((a, b)))
        if r["date"] != "TBA":
            known_dates.setdefault(pair, set()).add(r["date"])

    games: dict[tuple, dict] = {}
    for r in raw_rows:
        a_key = r["current_slug"] or r["team"].lower()
        b_key = r["opponent_slug"] or r["opponent"].lower()
        pair = tuple(sorted((a_key, b_key)))
        event_date = r["date"]
        if event_date == "TBA" and len(known_dates.get(pair, set())) == 1:
            event_date = next(iter(known_dates[pair]))
        key = (pair, event_date)

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
                "season": 2028,
                "date": event_date,
                "week": _week_for_2028(event_date if event_date != "TBA" else None),
                "home_team": home,
                "away_team": away,
                "neutral": neutral,
                "neutral_location": r["neutral_location"],
                "home_subdivision": home_meta.get("subdivision", "Unknown"),
                "away_subdivision": away_meta.get("subdivision", "Unknown"),
                "home_conference": home_meta.get("conference", "Unknown"),
                "away_conference": away_meta.get("conference", "Unknown"),
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
        df = df.assign(_week_sort=week_sort).sort_values(["_week_sort", "date", "home_team", "away_team"]).drop(columns="_week_sort").reset_index(drop=True)
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def scrape_fbschedules_2028() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    directory_html = _safe_get(FBSCHEDULES_DIRECTORY)
    soup = BeautifulSoup(directory_html, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(FBSCHEDULES_DIRECTORY, a["href"])
        parsed = urlparse(full)
        if parsed.netloc.endswith("fbschedules.com") and re.fullmatch(r"/ncaa/[a-z0-9-]+/?", parsed.path):
            urls.add(full.split("?")[0])
    urls = sorted(urls)

    teams = []
    raw_games = []
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(_scrape_one_team, url): url for url in urls}
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


def build_real_2028_store(teams_df: pd.DataFrame, games_df: pd.DataFrame) -> ScheduleStore:
    teams = [Team(
        name=str(r["name"]),
        subdivision=str(r["subdivision"]),
        conference=str(r["conference"]),
        is_a4=bool(r["is_a4"]),
        parity_managed=bool(r["parity_managed"]),
    ) for _, r in teams_df.iterrows()]
    valid_names = {t.name for t in teams}
    games = []
    for i, r in games_df.iterrows():
        week = r.get("week")
        if pd.isna(week):
            continue
        week = int(week)
        if week < 0 or week > 13:
            continue
        if r["home_team"] not in valid_names or r["away_team"] not in valid_names:
            continue
        games.append(Game(
            game_id=f"real2028_{i+1}", season=2028, week=week,
            home_team=str(r["home_team"]), away_team=str(r["away_team"]),
            moveable=True, locked=False,
            notes="Public-data MVP assumption: treated as moveable until Gridiron supplies true status.",
        ))
    # Public-data mode does not know conference dates or school intent. All weeks are candidate
    # slots; actual conflicts are handled by the game graph. Gridiron will replace these assumptions.
    slots = [Slot(team=t.name, season=2028, week=w, status="OPEN", location="ANY")
             for t in teams for w in range(0, 14)]
    return ScheduleStore(teams, games, slots, needs=[])

st.set_page_config(page_title="Gridiron Optimizer MVP", page_icon="🏈", layout="wide")

st.title("🏈 Gridiron Optimizer — MVP")
st.caption("Chat-first non-conference scheduling. The LLM interprets intent; the deterministic engine validates and solves.")

with st.sidebar:
    st.subheader("Data source")
    source_mode = st.radio("Choose data", ["Demo", "Real 2028 public snapshot"], index=0)
    st.caption("Real mode scrapes the public FBSchedules future-opponents pages and caches the result for six hours.")

real_teams_df = None
real_games_df = None
scrape_errors = []

if source_mode == "Real 2028 public snapshot":
    st.warning(
        "Public-data test mode: FBSchedules future schedules are tentative. The optimizer treats known non-conference "
        "games as moveable and dates without a known non-conference game as candidate slots. It does NOT know a school's "
        "actual Gridiron availability/need status yet."
    )
    st.info(
        "Use this for MVP testing, not as a production data feed. FBSchedules states that its schedule data is compiled "
        "from public sources and asks users not to republish its schedules. A production Gridiron integration should use "
        "Gridiron's own data or licensed/official school and conference sources."
    )
    with st.spinner("Loading 2028 FBS/FCS public scheduling snapshot — first load can take 1–3 minutes..."):
        try:
            real_teams_df, real_games_df, scrape_errors = scrape_fbschedules_2028()
        except Exception as exc:
            st.error(f"The live scrape failed: {type(exc).__name__}: {exc}")
            st.caption("Switch back to Demo mode while we troubleshoot the public site connection.")
            st.stop()
    if real_teams_df is None or real_teams_df.empty:
        st.error("No team data was returned from the public scrape.")
        st.stop()
    store = build_real_2028_store(real_teams_df, real_games_df)
    season = 2028
    with st.sidebar:
        st.success(f"Loaded {len(real_teams_df):,} teams / {len(real_games_df):,} unique 2028 commitments")
        if scrape_errors:
            st.warning(f"{len(scrape_errors)} team page(s) could not be read")
else:
    store = build_demo_store()
    with st.sidebar:
        season = st.selectbox("Active season", sorted({g.season for g in store.games.values()}), index=0)
        st.caption("Demo data is synthetic and designed around the Georgia–McNeese–Tarleton use case.")

optimizer = NonConferenceOptimizer(store)

with st.sidebar:
    st.divider()
    st.subheader("MVP scope")
    st.markdown("""
- Move a non-conference game and solve displaced games
- Protect FBS conference weekly parity
- Find public FCS availability candidates
- Find public A4-vs-A4 availability candidates
- No contract amendment generation
    """)


def parity_table(season: int) -> pd.DataFrame:
    rows = []
    for week in range(0, 14):
        parity = optimizer.conference_parity(store.copy_games(), season, week)
        for conference, value in parity.items():
            rows.append({"Week": week, "Conference": conference, "Status": value})
    return pd.DataFrame(rows)


def render_solution(sol, idx: int):
    label = f"#{idx} — {sol.title} · Score {sol.score:.1f}"
    with st.expander(label, expanded=(idx == 1)):
        st.write(sol.explanation)
        if sol.moves:
            df = pd.DataFrame([{
                "Game": f"{m.away_team} @ {m.home_team}",
                "From": f"Week {m.from_week}",
                "To": f"Week {m.to_week}",
            } for m in sol.moves])
            st.dataframe(df, use_container_width=True, hide_index=True)
        if sol.warnings:
            for warning in sol.warnings:
                st.warning(warning)
        if sol.parity_after:
            changed = []
            keys = sorted(set(sol.parity_before) | set(sol.parity_after))
            for key in keys:
                before = sol.parity_before.get(key, "—")
                after = sol.parity_after.get(key, "—")
                if before != after:
                    changed.append({"Conference / Week": key, "Before": before, "After": after})
            if changed:
                st.markdown("**Parity impact**")
                st.dataframe(pd.DataFrame(changed), use_container_width=True, hide_index=True)


tab_chat, tab_health, tab_schedule, tab_needs = st.tabs(["Ask Gridiron", "Conference Parity", "Schedule Data", "Open Candidates"])

with tab_chat:
    st.markdown("### What are you trying to accomplish?")
    if source_mode == "Demo":
        st.caption("Try: “In 2027 the SEC is odd in Week 2. Move Georgia vs McNeese to Week 2 and figure out where to put Tarleton without creating a new FBS parity problem.”")
    else:
        st.caption("Try a real 2028 matchup from Schedule Data, or ask: “Which FBS conferences are odd in Week 2?” / “Find Alabama an FCS candidate in Week 4.”")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Describe the scheduling problem...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        intent, parser_name = parse_intent(prompt, store.teams.keys())
        if intent.season is None:
            intent.season = season

        with st.chat_message("assistant"):
            st.caption(f"Intent parser: {parser_name}")
            with st.expander("Interpreted scheduling request"):
                st.json(intent.__dict__)

            solutions = optimizer.solve(intent)
            if not solutions:
                st.error("I couldn't find a feasible solution in the current scheduling data. In real-data mode, remember that TBA games have no week assignment and Gridiron-specific availability/needs are not yet loaded.")
            else:
                if intent.action == "MOVE_GAME":
                    st.success(f"Found {len(solutions)} feasible solution{'s' if len(solutions) != 1 else ''}. Ranked by fewest moves, week displacement, and FBS parity impact.")
                else:
                    st.success(f"Found {len(solutions)} match/solution{'s' if len(solutions) != 1 else ''}.")
                for i, sol in enumerate(solutions, start=1):
                    render_solution(sol, i)

with tab_health:
    st.markdown("### FBS conference parity by week")
    df = parity_table(season)
    if len(df):
        pivot = df.pivot(index="Conference", columns="Week", values="Status")
        st.dataframe(pivot, use_container_width=True)
    st.caption("Parity = after teams with known dated non-conference games are removed, is the remaining FBS conference inventory even for that week?")

with tab_schedule:
    st.markdown("### Non-conference games")
    if source_mode == "Real 2028 public snapshot":
        display_cols = ["date", "week", "away_team", "home_team", "neutral", "matchup_type", "away_conference", "home_conference", "source_urls"]
        st.dataframe(real_games_df[display_cols], use_container_width=True, hide_index=True)
        csv_bytes = real_games_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download 2028 public snapshot CSV", csv_bytes, "gridiron_2028_public_snapshot.csv", "text/csv")
        if scrape_errors:
            with st.expander(f"Scrape warnings ({len(scrape_errors)})"):
                st.code("\n".join(scrape_errors[:100]))
    else:
        games_df = pd.DataFrame([g.__dict__ for g in store.games.values()])
        st.dataframe(games_df.sort_values(["season", "week", "home_team"]), use_container_width=True, hide_index=True)
        st.markdown("### Explicit date availability")
        slots_df = pd.DataFrame([s.__dict__ for s in store.slots.values()])
        st.dataframe(slots_df[slots_df["season"] == season].sort_values(["team", "week"]), use_container_width=True, hide_index=True)

with tab_needs:
    if source_mode == "Real 2028 public snapshot":
        st.markdown("### Public availability candidates")
        st.write("This scrape cannot tell us who *wants* a game. It can only show who has no known dated non-conference commitment in a week. Gridiron's intent/need data is the missing production layer.")
        candidate_week = st.selectbox("Week", list(range(0, 14)), index=2)
        base_games = store.copy_games()
        rows = []
        for team in store.teams.values():
            occupied = store.game_for_team_week(base_games, team.name, 2028, candidate_week) is not None
            if not occupied:
                rows.append({"Team": team.name, "Subdivision": team.subdivision, "Conference": team.conference, "A4": team.is_a4})
        st.dataframe(pd.DataFrame(rows).sort_values(["Subdivision", "Conference", "Team"]), use_container_width=True, hide_index=True)
    else:
        st.markdown("### Marketplace / scheduling needs")
        needs_df = pd.DataFrame([n.__dict__ for n in store.needs])
        if len(needs_df):
            st.dataframe(needs_df[needs_df["season"] == season], use_container_width=True, hide_index=True)
        else:
            st.info("No needs loaded.")
