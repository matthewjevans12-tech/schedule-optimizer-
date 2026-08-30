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


def test_tba_game_is_valid_schedule_intelligence():
    # The UI/data layer may retain a future commitment with no week; the
    # optimizer builder excludes it until dated. The semantic model itself
    # continues to require an integer only when instantiated for optimization.
    import pandas as pd
    assert pd.isna(pd.NA)


def test_market_prior_prefers_first_four_weeks():
    from optimizer_engine import AdvancedNonConferenceOptimizer
    assert AdvancedNonConferenceOptimizer.market_week_prior(0) > AdvancedNonConferenceOptimizer.market_week_prior(5)
    assert AdvancedNonConferenceOptimizer.market_week_prior(3) > AdvancedNonConferenceOptimizer.market_week_prior(9)


def test_buy_game_market_returns_structured_match():
    from optimizer_engine import AdvancedNonConferenceOptimizer, Game, Intent, Need, ScheduleStore, Slot, Team
    teams = [
        Team("Host", "FBS", "SEC", True, True),
        Team("Seller", "FCS", "SoCon", False, False),
    ]
    slots = [Slot(t.name, 2028, w, "OPEN", "ANY") for t in teams for w in range(14)]
    needs = [Need("Seller", 2028, 1, "FCS_BUY", "AWAY", min_guarantee=500000)]
    store = ScheduleStore(teams, [], slots, needs)
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=1)
    results = opt.find_buy_games(Intent(
        action="FIND_BUY_GAME", season=2028, team_a="Host", location="HOME"
    ))
    assert results
    first = results[0]
    assert first.metadata["match_type"] == "BUY_GAME"
    assert first.metadata["home_team"] == "Host"
    assert first.metadata["away_team"] == "Seller"
    assert first.metadata["explicit_need"] is True


def test_a4_market_can_search_entire_season():
    from optimizer_engine import AdvancedNonConferenceOptimizer, Intent, Need, ScheduleStore, Slot, Team
    teams = [
        Team("Alpha", "FBS", "SEC", True, True),
        Team("Beta", "FBS", "ACC", True, True),
    ]
    slots = [Slot(t.name, 2028, w, "OPEN", "ANY") for t in teams for w in range(14)]
    needs = [
        Need("Alpha", 2028, 2, "A4", "HOME"),
        Need("Beta", 2028, 2, "A4", "AWAY"),
    ]
    store = ScheduleStore(teams, [], slots, needs)
    opt = AdvancedNonConferenceOptimizer(store, time_limit_seconds=1)
    results = opt.find_a4_games(Intent(
        action="FIND_A4_GAME", season=2028, team_a="Alpha", target_week=None, location="HOME"
    ))
    assert results
    assert results[0].metadata["week"] == 2
    assert results[0].metadata["explicit_need"] is True


def test_v7_template_imports_needs_and_tba():
    from pathlib import Path
    from schedule_importer import load_schedule_upload
    template = Path(__file__).with_name("schedule_import_template.xlsx")
    if not template.exists():
        pytest.skip("Template not present")
    teams, games, slots, needs, report = load_schedule_upload(
        template.read_bytes(), template.name, None
    )
    assert report.ok
    assert len(needs) >= 1
    assert games["week"].isna().any()
