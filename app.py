from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

from optimizer_engine import (
    AdvancedNonConferenceOptimizer,
    Game,
    Intent,
    Move,
    ScheduleStore,
    Solution,
    Slot,
    Team,
    build_authoritative_store,
    build_demo_store,
    build_real_store,
    scrape_fbschedules_public,
)
from schedule_importer import load_schedule_upload, make_template_bytes
from workspace_db import WorkspaceDB


st.set_page_config(
    page_title="College Football Scheduling Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html,body,.stApp,[class*="css"]{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif!important;
}
.stApp{background:#fff;color:#202124}
.block-container{max-width:1080px;padding-top:2.2rem;padding-bottom:5rem}
h1,h2,h3{color:#202124!important;letter-spacing:-.025em}
[data-testid="stMarkdownContainer"] p{font-size:17px;line-height:1.55;color:#5f6368}
.simple-title{text-align:center;font-size:32px;font-weight:650;letter-spacing:-.035em;color:#202124;margin-top:8px}
.simple-sub{text-align:center;font-size:17px;color:#5f6368;max-width:720px;margin:10px auto 30px;line-height:1.5}
.data-pill{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border-radius:999px;border:1px solid #dadce0;font-size:14px;color:#5f6368;background:#fff}
.data-pill.good{border-color:#c8e6c9;background:#f3fbf5;color:#137333}
.data-pill.warn{border-color:#fdd663;background:#fff8e1;color:#8a5a00}
.page-title{font-size:28px;font-weight:650;color:#202124;letter-spacing:-.03em;margin:20px 0 5px}
.page-copy{font-size:17px;line-height:1.5;color:#5f6368;max-width:760px;margin-bottom:26px}
.stTabs [data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid #e8eaed}
.stTabs [data-baseweb="tab"]{font-size:17px;font-weight:600;padding:14px 2px;height:auto}
.stTabs [aria-selected="true"]{color:#1a73e8!important}
.stButton>button{min-height:50px;border-radius:25px;font-size:16px;font-weight:650;padding:0 24px;box-shadow:none}
.stButton>button[kind="primary"],.stButton>button[data-testid="baseButton-primary"]{background:#1a73e8;border-color:#1a73e8;color:#fff}
.stSelectbox label,.stRadio label,.stTextInput label,.stSlider label,.stMultiSelect label,.stTextArea label,.stCheckbox label,.stNumberInput label,.stFileUploader label{
  font-size:16px!important;color:#3c4043!important;font-weight:600!important;text-transform:none!important;letter-spacing:0!important
}
div[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea,.stMultiSelect [data-baseweb="select"]>div,.stNumberInput input{
  background:#fff!important;border:1px solid #dadce0!important;border-radius:16px!important;min-height:52px!important;color:#202124!important;box-shadow:none!important;font-size:16px!important
}
[data-testid="stExpander"]{border:1px solid #e8eaed;border-radius:16px;background:#fff;box-shadow:none;margin:12px 0}
[data-testid="stExpander"] summary{font-size:16px;font-weight:600;color:#3c4043}
.answer{border:1px solid #dadce0;border-radius:22px;padding:24px;margin:20px 0;background:#fff}
.answer-label{font-size:14px;font-weight:700;color:#1a73e8;margin-bottom:7px}
.answer-title{font-size:25px;font-weight:650;color:#202124;letter-spacing:-.02em}
.answer-meta{font-size:16px;color:#5f6368;margin-top:5px}
.move-row{display:grid;grid-template-columns:34px 1fr auto;gap:14px;align-items:center;padding:17px 0;border-top:1px solid #eceff1}
.move-num{width:30px;height:30px;border-radius:50%;background:#f1f3f4;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#5f6368}
.move-game{font-size:17px;font-weight:650;color:#202124}.move-why{font-size:14px;color:#5f6368;margin-top:3px}
.move-week{font-size:16px;font-weight:650;color:#1a73e8;white-space:nowrap}
.checks{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0}
.check{font-size:14px;color:#3c4043;padding:7px 10px;border-radius:999px;background:#f8f9fa;border:1px solid #e8eaed}
.check.ok{color:#137333;background:#f3fbf5;border-color:#c8e6c9}.check.warn{color:#8a5a00;background:#fff8e1;border-color:#fdd663}
.rule-row{border:1px solid #e8eaed;border-radius:14px;padding:13px 15px;margin:8px 0;background:#fff}
.rule-title{font-size:15px;font-weight:650;color:#202124}.rule-meta{font-size:14px;color:#5f6368;margin-top:3px}
.year-row{display:grid;grid-template-columns:90px 1fr;gap:18px;padding:18px 0;border-top:1px solid #eceff1}
.year-label{font-size:22px;font-weight:650;color:#202124}
.game-chips{display:flex;gap:10px;flex-wrap:wrap}
.game-chip{border:1px solid #dadce0;border-radius:14px;padding:10px 12px;min-width:150px}
.game-chip-week{font-size:14px;color:#5f6368}.game-chip-opp{font-size:16px;font-weight:650;color:#202124;margin-top:2px}.game-chip-site{font-size:14px;color:#5f6368;margin-top:2px}
.status-line{font-size:15px;color:#5f6368;margin:8px 0 18px}
[data-testid="stDataFrame"]{font-size:15px}
small,.stCaption,[data-testid="stCaptionContainer"]{font-size:14px!important;color:#80868b!important}
@media(max-width:800px){
 .block-container{padding-top:1.2rem}.simple-title{font-size:27px}.page-title{font-size:25px}
 .year-row{grid-template-columns:1fr;gap:8px}.move-row{grid-template-columns:32px 1fr}.move-week{grid-column:2}
}
</style>
""", unsafe_allow_html=True)


# --------------------------- helpers ---------------------------------

@st.cache_resource
def get_db() -> WorkspaceDB:
    db_url = None
    try:
        db_url = st.secrets.get("DATABASE_URL")
    except Exception:
        db_url = None
    return WorkspaceDB(db_url)


def display_week(internal_week: int) -> int:
    return int(internal_week) + 1


def internal_week(display: int) -> int:
    return int(display) - 1


def game_label(game: Game) -> str:
    site = "Neutral" if game.neutral else "@" if False else ""
    return f"Week {display_week(game.week)} · {game.away_team} @ {game.home_team}"


def store_with_locked(base: ScheduleStore, locked_ids: Set[str]) -> ScheduleStore:
    games = []
    for g in base.games.values():
        if g.game_id in locked_ids:
            games.append(replace(g, locked=True, moveable=False, moveability="LOCKED"))
        else:
            games.append(g)
    return ScheduleStore(
        list(base.teams.values()),
        games,
        list(base.slots.values()),
        list(base.needs),
    )


def solution_signature(sol: Solution) -> Tuple[Tuple[str, int, int], ...]:
    return tuple(sorted((m.game_id, int(m.from_week), int(m.to_week)) for m in sol.moves))


def apply_moves(games: Dict[str, Game], sol: Solution) -> Dict[str, Game]:
    out = dict(games)
    for move in sol.moves:
        if move.game_id in out:
            out[move.game_id] = replace(out[move.game_id], week=int(move.to_week))
    return out


def odd_keys(engine: AdvancedNonConferenceOptimizer, games: Dict[str, Game], season: int) -> Set[Tuple[str, int]]:
    keys = set()
    for w in range(14):
        for conf, value in engine.conference_parity(games, season, w).items():
            if str(value).startswith("ODD"):
                keys.add((conf, w))
    return keys


def data_label(mode: str, report_ok: bool = True) -> Tuple[str, str]:
    if mode == "Authoritative upload" and report_ok:
        return "Authoritative administrator data", "good"
    if mode == "Public prototype":
        return "Partial / inferred public data", "warn"
    return "Demo data", "warn"


def rule_summary(rule: Dict[str, object]) -> str:
    typ = str(rule.get("rule_type", "")).replace("_", " ").title()
    team = str(rule.get("team", "") or "")
    hard = str(rule.get("hardness", "MUST")).title()
    start = display_week(int(rule.get("start_week", 0)))
    end = display_week(int(rule.get("end_week", 13)))
    value = rule.get("value")
    if typ in {"Lock Game", "Avoid Move Game"}:
        return f"{hard}: {typ}"
    if start == end:
        rng = f"Week {start}"
    else:
        rng = f"Weeks {start}–{end}"
    suffix = f" · {value}" if value not in (None, "") else ""
    return f"{hard}: {team} · {typ} · {rng}{suffix}"


def persistent_profile_rules(db: WorkspaceDB, team: str) -> List[Dict[str, object]]:
    item = db.get("school_profile", team)
    return list((item or {}).get("rules", []))


def save_profile_rules(db: WorkspaceDB, team: str, rules: List[Dict[str, object]]):
    db.put("school_profile", team, {"team": team, "rules": rules})


def constraint_builder(
    db: WorkspaceDB,
    *,
    prefix: str,
    primary_team: str,
    teams: List[str],
    games: List[Game],
) -> Tuple[List[Dict[str, object]], Set[str], Set[str], str]:
    """Generic Must/Cannot/Prefer builder. Returns rules, locks, avoids, context."""
    saved = persistent_profile_rules(db, primary_team)
    state_key = f"{prefix}_rules_{primary_team}"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(saved)
    rules: List[Dict[str, object]] = st.session_state[state_key]

    if rules:
        st.markdown("**Active rules**")
        for idx, rule in enumerate(list(rules)):
            c1, c2 = st.columns([8, 1])
            with c1:
                st.markdown(
                    f'<div class="rule-row"><div class="rule-title">{rule_summary(rule)}</div>'
                    f'<div class="rule-meta">{rule.get("note","")}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("×", key=f"{prefix}_remove_rule_{idx}", help="Remove rule"):
                    rules.pop(idx)
                    st.session_state[state_key] = rules
                    st.rerun()

    with st.expander("Add rule", expanded=False):
        hard = st.selectbox("Type", ["MUST", "CANNOT", "PREFER"], key=f"{prefix}_hardness")
        rule_types = [
            "MAX_CONSECUTIVE_AWAY",
            "MAX_CONSECUTIVE_HOME",
            "MIN_CAMPUS_HOME_IN_RANGE",
            "MAX_WEEKS_WITHOUT_CAMPUS_HOME",
            "MUST_CAMPUS_HOME_WEEK",
            "CANNOT_AWAY_WEEK",
            "PROTECT_BYE_WEEK",
            "MAX_CONSECUTIVE_A4",
        ]
        if hard == "PREFER":
            rule_types = ["PREFER_KEEP_BYE", "PREFER_CAMPUS_HOME_WEEK"]
        rtype = st.selectbox(
            "Rule",
            rule_types,
            format_func=lambda x: x.replace("_", " ").title(),
            key=f"{prefix}_rtype",
        )
        team = st.selectbox("Team", teams, index=teams.index(primary_team) if primary_team in teams else 0, key=f"{prefix}_rteam")
        if rtype in {"MUST_CAMPUS_HOME_WEEK", "CANNOT_AWAY_WEEK", "PROTECT_BYE_WEEK", "PREFER_KEEP_BYE", "PREFER_CAMPUS_HOME_WEEK"}:
            week = st.selectbox("Week", list(range(1, 15)), key=f"{prefix}_singleweek")
            start_w = end_w = internal_week(week)
        else:
            wrange = st.slider("Week range", 1, 14, value=(1, 5), key=f"{prefix}_rrange")
            start_w, end_w = internal_week(wrange[0]), internal_week(wrange[1])
        value = 1
        if rtype in {"MAX_CONSECUTIVE_AWAY", "MAX_CONSECUTIVE_HOME", "MAX_CONSECUTIVE_A4"}:
            value = st.selectbox("Maximum", [1, 2, 3, 4], index=1, key=f"{prefix}_rvalue")
        elif rtype in {"MIN_CAMPUS_HOME_IN_RANGE"}:
            value = st.selectbox("Minimum campus home games", [1, 2, 3, 4], key=f"{prefix}_rvalue2")
        elif rtype == "MAX_WEEKS_WITHOUT_CAMPUS_HOME":
            value = st.selectbox("Maximum weeks without a campus home game", [1, 2, 3, 4, 5], index=2, key=f"{prefix}_rvalue3")
        note = st.text_input("Why this matters (optional)", key=f"{prefix}_rnote")
        if st.button("Add", use_container_width=True, key=f"{prefix}_add_rule"):
            rules.append({
                "rule_id": f"{prefix}_{len(rules)+1}_{datetime.now().timestamp()}",
                "hardness": hard,
                "rule_type": rtype,
                "team": team,
                "start_week": start_w,
                "end_week": end_w,
                "value": int(value),
                "weight": 15,
                "active": True,
                "note": note,
            })
            st.session_state[state_key] = rules
            st.rerun()

    game_map = {game_label(g): g.game_id for g in games}
    protected_labels = st.multiselect(
        "Protect these games — never move",
        list(game_map.keys()),
        key=f"{prefix}_protected",
    )
    avoid_labels = st.multiselect(
        "Avoid moving these games if possible",
        [x for x in game_map.keys() if x not in protected_labels],
        key=f"{prefix}_avoid",
    )
    context = st.text_area(
        "Coach / AD context",
        placeholder="Example: Coach strongly prefers a campus home game before rivalry week.",
        key=f"{prefix}_context",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save active rules as school defaults", use_container_width=True, key=f"{prefix}_save_profile"):
            save_profile_rules(db, primary_team, rules)
            st.success("School profile saved.")
    with c2:
        if st.button("Clear active rules", use_container_width=True, key=f"{prefix}_clear_rules"):
            st.session_state[state_key] = []
            st.rerun()

    return rules, {game_map[x] for x in protected_labels}, {game_map[x] for x in avoid_labels}, context


def build_move_intent(
    game: Game,
    target_week: int,
    rules: List[Dict[str, object]],
    avoid_ids: Set[str],
    context: str,
    preserve_parity: bool,
) -> Intent:
    return Intent(
        action="MOVE_GAME",
        season=int(game.season),
        target_week=int(target_week),
        team_a=game.home_team,
        team_b=game.away_team,
        preserve_fbs_conference_parity=bool(preserve_parity),
        max_additional_moves=10,
        summary="Repair game",
        avoid_game_ids=sorted(avoid_ids),
        coach_context=context,
        rules=list(rules),
    )


def render_result(
    engine: AdvancedNonConferenceOptimizer,
    sol: Solution,
    *,
    season: int,
    data_status: str,
    label: str = "BEST PATH",
):
    md = dict(sol.metadata or {})
    before = engine.store.copy_games()
    after = apply_moves(before, sol)
    new_odd = odd_keys(engine, after, season) - odd_keys(engine, before, season)
    proven = bool(md.get("lexicographic_proven", False))

    st.markdown(
        f'<div class="answer"><div class="answer-label">{label}</div>'
        f'<div class="answer-title">{len(sol.moves)} game change{"s" if len(sol.moves)!=1 else ""}</div>'
        f'<div class="answer-meta">{"Minimum number of changes proven" if proven else "Best path found within the solve window"} · {data_status}</div>',
        unsafe_allow_html=True,
    )
    for i, move in enumerate(sol.moves, 1):
        st.markdown(
            f'<div class="move-row"><div class="move-num">{i}</div><div>'
            f'<div class="move-game">{move.away_team} @ {move.home_team}</div>'
            f'<div class="move-why">Move from Week {display_week(move.from_week)}</div></div>'
            f'<div class="move-week">→ Week {display_week(move.to_week)}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="checks">'
        '<span class="check ok">✓ Hard constraints satisfied</span>'
        f'<span class="check {"ok" if not new_odd else "warn"}">'
        f'{"✓ No new parity issue" if not new_odd else f"⚠ {len(new_odd)} new parity issue(s)"}</span>'
        f'<span class="check">Data: {data_status}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if sol.warnings:
        for warning in sol.warnings:
            st.caption(warning)


def scenario_payload(sol: Solution, season: int, title: str, data_status: str) -> Dict[str, object]:
    return {
        "title": title,
        "season": season,
        "data_status": data_status,
        "moves": [
            {
                "game_id": m.game_id,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "from_week": int(m.from_week),
                "to_week": int(m.to_week),
            }
            for m in sol.moves
        ],
        "games_moved": len(sol.moves),
        "metadata": dict(sol.metadata or {}),
    }


def feedback_ui(db: WorkspaceDB, season: int, game: Optional[Game], sol: Solution, key: str):
    with st.expander("This path would not work in the real world", expanded=False):
        reason = st.selectbox(
            "Why?",
            ["Coach preference", "Contract issue", "Travel issue", "Game cannot move", "Financial issue", "Other"],
            key=f"{key}_reason",
        )
        notes = st.text_area("What are we missing?", key=f"{key}_notes")
        if st.button("Save feedback", use_container_width=True, key=f"{key}_save_feedback"):
            db.add_feedback(
                season=season,
                team=game.home_team if game else "",
                game_id=game.game_id if game else "",
                reason=reason,
                notes=notes,
                payload=scenario_payload(sol, season, "Rejected path", ""),
            )
            st.success("Feedback saved. This is the structured knowledge the product should accumulate.")


def render_all_years(games_df: pd.DataFrame, team: str):
    subset = games_df[(games_df["home_team"] == team) | (games_df["away_team"] == team)].copy()
    if subset.empty:
        st.info("No known future games for this team.")
        return
    years = sorted(int(y) for y in subset["season"].dropna().unique())
    for year in years:
        ys = subset[subset["season"] == year].copy().sort_values(["week", "date"])
        chips = []
        for _, row in ys.iterrows():
            neutral = bool(row.get("neutral", False))
            if str(row["home_team"]) == team:
                opp = str(row["away_team"])
                site = "Neutral" if neutral else "Home"
            else:
                opp = str(row["home_team"])
                site = "Neutral" if neutral else "Away"
            chips.append(
                f'<div class="game-chip"><div class="game-chip-week">Week {display_week(int(row["week"]))}</div>'
                f'<div class="game-chip-opp">{opp}</div><div class="game-chip-site">{site}</div></div>'
            )
        st.markdown(
            f'<div class="year-row"><div class="year-label">{year}</div><div class="game-chips">{"".join(chips)}</div></div>',
            unsafe_allow_html=True,
        )


# ------------------------ workspace / data ----------------------------

db = get_db()

st.markdown('<div class="simple-title">College Football Scheduling Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="simple-sub">Find the fewest realistic changes required to get from the schedule you have to the schedule you want.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Workspace")
    data_mode = st.radio("Schedule data", ["Authoritative upload", "Public prototype", "Demo"], key="data_mode")

public_teams_df = None
public_games_df = None
store = None
all_games_df = pd.DataFrame()
all_teams_df = pd.DataFrame()
slots_df = pd.DataFrame()
report_ok = True

if data_mode == "Authoritative upload":
    with st.sidebar:
        st.download_button(
            "Download Excel template",
            data=make_template_bytes(),
            file_name="college_football_schedule_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        uploaded = st.file_uploader("Upload schedule", type=["xlsx", "xlsm", "csv"])

    # Public metadata is optional help for CSV team matching.
    if uploaded is not None:
        raw = uploaded.getvalue()
        # Excel is the preferred authoritative format because it carries Teams.
        # Only CSV needs public metadata assistance.
        if uploaded.name.lower().endswith(".csv"):
            try:
                public_teams_df, _, _ = scrape_fbschedules_public(tuple(range(2027, 2038)))
            except Exception:
                public_teams_df = None
        else:
            public_teams_df = None

        teams_df, games_df, slots_df, report = load_schedule_upload(raw, uploaded.name, public_teams_df)
        report_ok = report.ok
        with st.sidebar:
            if report.errors:
                for e in report.errors:
                    st.error(e)
            if report.warnings:
                for w in report.warnings:
                    st.warning(w)
            for msg in report.info:
                st.caption(msg)
        if report.ok:
            all_teams_df = teams_df
            all_games_df = games_df
            # Persist normalized snapshot if DB is durable; useful in pilot too.
            db.put("data_snapshot", "latest", {
                "teams": teams_df.to_dict("records"),
                "games": games_df.to_dict("records"),
                "slots": slots_df.to_dict("records"),
                "source_name": uploaded.name,
                "saved_at": datetime.now().isoformat(),
            })
    else:
        snapshot = db.get("data_snapshot", "latest")
        if snapshot:
            all_teams_df = pd.DataFrame(snapshot.get("teams", []))
            all_games_df = pd.DataFrame(snapshot.get("games", []))
            slots_df = pd.DataFrame(snapshot.get("slots", []))
            with st.sidebar:
                st.caption(f"Using saved snapshot: {snapshot.get('source_name','latest')}")
        else:
            st.info("Upload the authoritative schedule workbook to begin. The Excel template includes Teams, Games, and optional Slots.")
            st.stop()

elif data_mode == "Public prototype":
    with st.spinner("Loading public future-opponent data…"):
        public_teams_df, public_games_df, scrape_errors = scrape_fbschedules_public(tuple(range(2027, 2038)))
    all_teams_df = public_teams_df
    all_games_df = public_games_df

else:
    demo = build_demo_store()
    all_teams_df = pd.DataFrame([asdict(t) for t in demo.teams.values()])
    all_games_df = pd.DataFrame([asdict(g) for g in demo.games.values()])

available_years = sorted(int(y) for y in all_games_df["season"].dropna().unique())
if not available_years:
    st.error("No seasons are available in the current data.")
    st.stop()

with st.sidebar:
    default_year = 2028 if 2028 in available_years else available_years[0]
    season = st.selectbox("Active season", available_years, index=available_years.index(default_year))

if data_mode == "Authoritative upload":
    store = build_authoritative_store(all_teams_df, all_games_df, slots_df, int(season))
elif data_mode == "Public prototype":
    store = build_real_store(all_teams_df, all_games_df, int(season))
else:
    # Demo currently contains a single modeled season; rebuild directly.
    demo = build_demo_store()
    store = demo

engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=6.0)
status_text, status_class = data_label(data_mode, report_ok)
st.markdown(
    f'<div style="text-align:center;margin-top:-15px;margin-bottom:24px">'
    f'<span class="data-pill {status_class}">● {status_text}</span></div>',
    unsafe_allow_html=True,
)

if not db.durable:
    with st.sidebar:
        st.caption("Pilot persistence: local SQLite. Add DATABASE_URL in Streamlit Secrets for durable Postgres storage.")


# ------------------------------ UI -----------------------------------

tab_repair, tab_schedules = st.tabs(["Repair", "Schedules"])

with tab_repair:
    st.markdown('<div class="page-title">Repair</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">One outcome at a time. Add only the real-world constraints that matter.</div>',
        unsafe_allow_html=True,
    )
    repair_scope = st.radio("Repair", ["Game", "Conference"], horizontal=True)

    season_games = sorted(
        list(store.games.values()),
        key=lambda g: (g.week, g.home_team, g.away_team),
    )
    teams = sorted(store.teams.keys())

    if repair_scope == "Game":
        movable_games = [g for g in season_games if str(g.game_type).upper() != "CONFERENCE"]
        game_teams = sorted({t for g in movable_games for t in (g.home_team, g.away_team)})
        if not movable_games:
            st.info("No non-conference games are loaded for this season.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                default_team = game_teams.index("Georgia") if "Georgia" in game_teams else 0
                selected_team = st.selectbox("School", game_teams, index=default_team)
            team_games = [g for g in movable_games if g.involves(selected_team)]
            labels = {game_label(g): g for g in team_games}
            with c2:
                selected_game = labels[st.selectbox("Game", list(labels.keys()))]

            target_display = st.selectbox(
                "Move to",
                list(range(1, 15)),
                index=int(selected_game.week),
            )
            target_week = internal_week(target_display)

            with st.expander("Add constraints or preferences", expanded=False):
                other_games = [g for g in season_games if g.game_id != selected_game.game_id]
                rules, protected_ids, avoid_ids, context = constraint_builder(
                    db,
                    prefix=f"game_{season}_{selected_game.game_id}",
                    primary_team=selected_team,
                    teams=teams,
                    games=other_games,
                )
                preserve_parity = st.checkbox("Cannot create a new FBS conference parity problem", value=False)

            if data_mode == "Public prototype":
                st.markdown(
                    '<div class="status-line">Public mode can test product behavior, but game moveability and open weeks are inferred.</div>',
                    unsafe_allow_html=True,
                )

            if st.button("Find best path", type="primary", use_container_width=True):
                if target_week == selected_game.week:
                    st.info("That game is already in the selected week.")
                else:
                    run_store = store_with_locked(store, protected_ids)
                    run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=6.0)
                    intent = build_move_intent(selected_game, target_week, rules, avoid_ids, context, preserve_parity)
                    with st.spinner("Finding the smallest realistic repair chain…"):
                        results = run_engine.solve_move_game(intent)
                    st.session_state["last_game_result"] = {
                        "intent": intent,
                        "result": results[0] if results else None,
                        "game": selected_game,
                        "protected": protected_ids,
                    }

            state = st.session_state.get("last_game_result")
            if state and state.get("game") and state["game"].game_id == selected_game.game_id:
                sol = state.get("result")
                intent = state.get("intent")
                run_store = store_with_locked(store, set(state.get("protected") or set()))
                run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=6.0)

                if sol:
                    render_result(run_engine, sol, season=int(season), data_status=status_text)
                    c1, c2 = st.columns(2)
                    with c1:
                        scenario_name = st.text_input("Scenario name", value=f"{selected_game.home_team}-{selected_game.away_team} repair")
                        if st.button("Save scenario", use_container_width=True):
                            db.put("scenario", scenario_name, scenario_payload(sol, int(season), scenario_name, status_text))
                            st.success("Scenario saved.")
                    with c2:
                        if st.button("See alternative strategies", use_container_width=True):
                            with st.spinner("Testing human tradeoff strategies…"):
                                alts = run_engine.solve_move_game_alternatives(intent, sol)
                            st.session_state["last_alternatives"] = alts

                    alts = st.session_state.get("last_alternatives", [])
                    if alts:
                        with st.expander("Alternative strategies", expanded=True):
                            for i, alt in enumerate(alts, 1):
                                render_result(run_engine, alt, season=int(season), data_status=status_text, label=str((alt.metadata or {}).get("strategy_label","ALTERNATIVE")).upper())

                    with st.expander("Why this works", expanded=False):
                        st.write(sol.explanation)
                        stages = list((sol.metadata or {}).get("lexicographic_stages") or [])
                        if stages:
                            st.dataframe(pd.DataFrame([{
                                "Priority": s.get("stage"),
                                "Result": s.get("value"),
                                "Proof": "Proven" if s.get("proven") else ("Skipped" if s.get("status") == "SKIPPED" else "Best found"),
                            } for s in stages]), use_container_width=True, hide_index=True)
                    feedback_ui(db, int(season), selected_game, sol, f"feedback_{selected_game.game_id}")
                elif state:
                    st.error("No path satisfies every active hard constraint.")
                    if st.button("Explain why blocked", use_container_width=True):
                        # On-demand one-rule relaxation diagnosis.
                        relaxations = []
                        base_rules = list(intent.rules or [])
                        hard_rules = [r for r in base_rules if str(r.get("hardness","")).upper() in {"MUST","CANNOT"}]
                        for idx, rule in enumerate(hard_rules[:6]):
                            test_rules = [r for r in base_rules if r is not rule]
                            test_intent = replace(intent, rules=test_rules)
                            test_results = run_engine.solve_move_game(test_intent)
                            if test_results:
                                relaxations.append((rule_summary(rule), len(test_results[0].moves)))
                        if protected_ids:
                            unlocked_engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3.0)
                            unlocked = unlocked_engine.solve_move_game(intent)
                            if unlocked:
                                relaxations.append(("Allow one or more protected games to move", len(unlocked[0].moves)))
                        st.session_state["block_relaxations"] = relaxations
                    relaxations = st.session_state.get("block_relaxations", [])
                    if relaxations:
                        st.markdown("**Smallest tested relaxations that restore feasibility**")
                        for label, count in relaxations:
                            st.write(f"• {label} → feasible with {count} game change{'s' if count != 1 else ''}")

    else:
        conferences = engine.store.fbs_conferences()
        default_conf = conferences.index("SEC") if "SEC" in conferences else 0
        selected_conf = st.selectbox("Conference", conferences, index=default_conf)
        selected_display_weeks = st.multiselect(
            "Weeks that must be even",
            list(range(1, 15)),
            default=[1, 2, 3],
        )
        selected_weeks = [internal_week(w) for w in selected_display_weeks]

        members = sorted(t.name for t in store.conference_members(selected_conf))
        member_set = set(members)
        conf_games = [
            g for g in season_games
            if g.home_team in member_set or g.away_team in member_set
        ]

        current_odd = []
        for w in selected_weeks:
            value = engine.conference_parity(store.copy_games(), int(season), w).get(selected_conf, "")
            if str(value).startswith("ODD"):
                current_odd.append(w)
        if selected_weeks:
            if current_odd:
                st.markdown(
                    '<div class="status-line"><strong>'
                    f'{len(current_odd)} selected week{"s" if len(current_odd)!=1 else ""} need repair:</strong> '
                    + ", ".join(f"Week {display_week(w)}" for w in current_odd)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.success("Every selected week is already even.")

        with st.expander("Add constraints or preferences", expanded=False):
            rules, protected_ids, avoid_ids, context = constraint_builder(
                db,
                prefix=f"conf_{season}_{selected_conf}",
                primary_team=members[0] if members else teams[0],
                teams=members if members else teams,
                games=conf_games,
            )

        if st.button("Find best conference plan", type="primary", use_container_width=True):
            run_store = store_with_locked(store, protected_ids)
            run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=12.0)
            intent = Intent(
                action="OPTIMIZE_NATIONAL",
                season=int(season),
                target_weeks=selected_weeks,
                conferences=[selected_conf],
                conference=selected_conf,
                all_conferences=False,
                preserve_fbs_conference_parity=True,
                max_additional_moves=60,
                avoid_game_ids=sorted(avoid_ids),
                coach_context=context,
                rules=rules,
                summary="Repair conference selected weeks",
            )
            with st.spinner("Enforcing selected weeks, then minimizing game changes…"):
                plans = run_engine.optimize_national(intent)
            st.session_state["last_conf_result"] = {
                "intent": intent,
                "result": plans[0] if plans else None,
                "protected": protected_ids,
                "conference": selected_conf,
            }

        state = st.session_state.get("last_conf_result")
        if state and state.get("conference") == selected_conf:
            sol = state.get("result")
            run_store = store_with_locked(store, set(state.get("protected") or set()))
            run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=12.0)
            if sol and not bool((sol.metadata or {}).get("infeasible")):
                render_result(run_engine, sol, season=int(season), data_status=status_text, label="BEST CONFERENCE PLAN")
                scenario_name = st.text_input("Scenario name", value=f"{selected_conf} Weeks {'-'.join(map(str,selected_display_weeks))}")
                if st.button("Save conference scenario", use_container_width=True):
                    db.put("scenario", scenario_name, scenario_payload(sol, int(season), scenario_name, status_text))
                    st.success("Scenario saved.")
                feedback_ui(db, int(season), None, sol, f"feedback_conf_{selected_conf}")
            elif state:
                st.error("No plan satisfies every selected week and every active hard constraint.")
                st.caption("The optimizer will not label a partial parity repair as successful.")
                if st.button("Explain why blocked", use_container_width=True, key=f"explain_conf_{selected_conf}"):
                    base_intent = state.get("intent")
                    relaxations = []
                    hard_rules = [
                        r for r in (base_intent.rules or [])
                        if str(r.get("hardness", "")).upper() in {"MUST", "CANNOT"}
                    ]
                    for rule in hard_rules[:6]:
                        test_rules = [r for r in (base_intent.rules or []) if r is not rule]
                        test_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=3.0)
                        test_intent = replace(base_intent, rules=test_rules)
                        test = test_engine.optimize_national(test_intent)
                        if test and not bool((test[0].metadata or {}).get("infeasible")):
                            relaxations.append((rule_summary(rule), len(test[0].moves)))
                    if state.get("protected"):
                        unlocked_engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=3.0)
                        unlocked = unlocked_engine.optimize_national(base_intent)
                        if unlocked and not bool((unlocked[0].metadata or {}).get("infeasible")):
                            relaxations.append(("Allow one or more protected games to move", len(unlocked[0].moves)))
                    st.session_state[f"conf_relax_{selected_conf}"] = relaxations

                conf_relax = st.session_state.get(f"conf_relax_{selected_conf}", [])
                if conf_relax:
                    st.markdown("**Smallest tested relaxations that restore feasibility**")
                    for label, count in conf_relax:
                        st.write(f"• {label} → feasible with {count} game change{'s' if count != 1 else ''}")


with tab_schedules:
    st.markdown('<div class="page-title">Schedules</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">Use the schedule as the intelligence layer: see one team across every loaded year, inspect a conference, and compare saved scenarios.</div>',
        unsafe_allow_html=True,
    )

    view = st.radio("View", ["Team — all years", "Conference — active season", "Scenarios"], horizontal=True)

    if view == "Team — all years":
        team_names = sorted(all_teams_df["name"].dropna().astype(str).unique())
        default_team = team_names.index("Georgia") if "Georgia" in team_names else 0
        schedule_team = st.selectbox("Team", team_names, index=default_team)
        profile_rules = persistent_profile_rules(db, schedule_team)
        if profile_rules:
            with st.expander(f"{len(profile_rules)} saved scheduling rule{'s' if len(profile_rules)!=1 else ''}", expanded=False):
                for r in profile_rules:
                    st.write("• " + rule_summary(r))
        render_all_years(all_games_df, schedule_team)

    elif view == "Conference — active season":
        conferences = engine.store.fbs_conferences()
        default_conf = conferences.index("SEC") if "SEC" in conferences else 0
        conf = st.selectbox("Conference", conferences, index=default_conf)
        members = sorted(t.name for t in store.conference_members(conf))
        rows = []
        for team in members:
            games = sorted([g for g in store.games.values() if g.involves(team)], key=lambda g: g.week)
            row = {"School": team}
            for w in range(14):
                game = next((g for g in games if g.week == w), None)
                if game:
                    opp = game.away_team if game.home_team == team else game.home_team
                    site = game.site_for(team)
                    row[f"W{display_week(w)}"] = f"{'vs' if site=='HOME' else '@' if site=='AWAY' else 'N'} {opp}"
                else:
                    row[f"W{display_week(w)}"] = ""
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=650)
        odd_rows = []
        for w in range(14):
            value = engine.conference_parity(store.copy_games(), int(season), w).get(conf, "")
            if str(value).startswith("ODD"):
                odd_rows.append({"Week": display_week(w), "State": value})
        if odd_rows:
            with st.expander(f"{len(odd_rows)} modeled odd week{'s' if len(odd_rows)!=1 else ''}", expanded=False):
                st.dataframe(pd.DataFrame(odd_rows), use_container_width=True, hide_index=True)

    else:
        scenarios = db.list("scenario")
        if not scenarios:
            st.info("Save a repair result and it will appear here.")
        else:
            names = [s["key"] for s in scenarios]
            selected = st.selectbox("Scenario", names)
            item = next(s for s in scenarios if s["key"] == selected)
            payload = item["payload"]
            st.markdown(f"### {payload.get('title', selected)}")
            st.caption(f"Season {payload.get('season')} · {payload.get('data_status')}")
            moves = payload.get("moves", [])
            if moves:
                st.dataframe(pd.DataFrame([{
                    "Game": f"{m['away_team']} @ {m['home_team']}",
                    "Current": f"Week {display_week(m['from_week'])}",
                    "Proposed": f"Week {display_week(m['to_week'])}",
                } for m in moves]), use_container_width=True, hide_index=True)

            if len(scenarios) >= 2:
                with st.expander("Compare two scenarios", expanded=False):
                    a = st.selectbox("Scenario A", names, index=0, key="scenario_a")
                    b = st.selectbox("Scenario B", names, index=1 if len(names)>1 else 0, key="scenario_b")
                    pa = next(s["payload"] for s in scenarios if s["key"] == a)
                    pb = next(s["payload"] for s in scenarios if s["key"] == b)
                    comp = pd.DataFrame([
                        {"Measure": "Games moved", a: pa.get("games_moved", 0), b: pb.get("games_moved", 0)},
                        {"Measure": "Disruption cost", a: pa.get("metadata",{}).get("disruption_cost", "—"), b: pb.get("metadata",{}).get("disruption_cost", "—")},
                        {"Measure": "Data status", a: pa.get("data_status", ""), b: pb.get("data_status", "")},
                        {"Measure": "Minimum-change proof", a: "Proven" if pa.get("metadata",{}).get("lexicographic_proven") else "Best found", b: "Proven" if pb.get("metadata",{}).get("lexicographic_proven") else "Best found"},
                    ])
                    st.dataframe(comp, use_container_width=True, hide_index=True)

st.caption(
    "Pilot-ready architecture · Set DATABASE_URL for durable Postgres persistence. "
    "Public prototype data is intentionally labeled as partial/inferred."
)
