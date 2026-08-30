import pytest

from optimizer_engine import (
    AdvancedNonConferenceOptimizer,
    Game,
    Intent,
    ORTOOLS_AVAILABLE,
    ScheduleStore,
    Slot,
    Team,
)


def make_store(games):
    teams = [
        Team("A", "FBS", "SEC", True, True),
        Team("B", "FCS", "FCS", False, False),
        Team("C", "FBS", "SEC", True, True),
        Team("D", "FCS", "FCS", False, False),
        Team("E", "FBS", "ACC", True, True),
        Team("F", "FBS", "ACC", True, True),
    ]
    slots = [Slot(t.name, 2028, w, "OPEN", "ANY") for t in teams for w in range(14)]
    return ScheduleStore(teams, games, slots, [])


def test_neutral_designated_home_is_not_campus_home_semantically():
    game = Game(
        "neutral", 2028, 2, "A", "C",
        neutral=True, campus_home_team="", moveable=False, locked=True,
        game_type="CONFERENCE",
    )
    assert game.site_for("A") == "NEUTRAL"
    assert game.site_for("C") == "NEUTRAL"


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed in local artifact runtime")
def test_clean_move_is_one_move():
    g = Game("g1", 2028, 1, "A", "B", neutral=False, campus_home_team="A")
    store = make_store([g])
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3)
    sol = opt.solve_move_game(Intent(
        action="MOVE_GAME", season=2028, target_week=2,
        team_a="A", team_b="B", max_additional_moves=5
    ))
    assert sol
    assert len(sol[0].moves) == 1
    assert sol[0].moves[0].game_id == "g1"
    assert sol[0].moves[0].to_week == 2


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed in local artifact runtime")
def test_locked_conflict_never_moves():
    target = Game("target", 2028, 1, "A", "B")
    locked = Game("locked", 2028, 2, "C", "B", moveable=False, locked=True, moveability="LOCKED")
    store = make_store([target, locked])
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3)
    sol = opt.solve_move_game(Intent(
        action="MOVE_GAME", season=2028, target_week=2,
        team_a="A", team_b="B", max_additional_moves=5
    ))
    assert not sol


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed in local artifact runtime")
def test_neutral_cannot_satisfy_campus_home_rule():
    neutral = Game(
        "neutral", 2028, 2, "A", "C",
        neutral=True, campus_home_team="", moveable=False, locked=True,
        game_type="CONFERENCE"
    )
    target = Game("target", 2028, 1, "A", "B")
    store = make_store([neutral, target])
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3)
    rules = [{
        "rule_id": "r1", "hardness": "MUST",
        "rule_type": "MIN_CAMPUS_HOME_IN_RANGE",
        "team": "A", "start_week": 2, "end_week": 2, "value": 1, "active": True
    }]
    sol = opt.solve_move_game(Intent(
        action="MOVE_GAME", season=2028, target_week=3,
        team_a="A", team_b="B", rules=rules
    ))
    assert not sol


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed in local artifact runtime")
def test_preference_cannot_buy_extra_move():
    target = Game("target", 2028, 1, "A", "B")
    unrelated = Game("other", 2028, 5, "C", "D")
    store = make_store([target, unrelated])
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3)
    sol = opt.solve_move_game(Intent(
        action="MOVE_GAME", season=2028, target_week=2,
        team_a="A", team_b="B",
        rules=[{
            "rule_id": "p1", "hardness": "PREFER", "rule_type": "PREFER_KEEP_BYE",
            "team": "C", "start_week": 5, "end_week": 5, "value": 1, "active": True
        }]
    ))
    assert sol
    assert len(sol[0].moves) == 1
    assert sol[0].moves[0].game_id == "target"


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed in local artifact runtime")
def test_generic_max_consecutive_away_is_hard():
    a1 = Game("a1", 2028, 0, "E", "A", moveable=False, locked=True, game_type="CONFERENCE")
    a2 = Game("a2", 2028, 1, "F", "A", moveable=False, locked=True, game_type="CONFERENCE")
    target = Game("target", 2028, 5, "B", "A")
    store = make_store([a1, a2, target])
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3)
    rules = [{
        "rule_id": "r", "hardness": "CANNOT", "rule_type": "MAX_CONSECUTIVE_AWAY",
        "team": "A", "start_week": 0, "end_week": 4, "value": 2, "active": True
    }]
    sol = opt.solve_move_game(Intent(
        action="MOVE_GAME", season=2028, target_week=2,
        team_a="A", team_b="B", rules=rules
    ))
    assert not sol
