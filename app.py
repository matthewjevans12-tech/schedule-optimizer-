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
    Need,
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
.workspace-bar{border:1px solid #e8eaed;border-radius:18px;padding:14px 16px;margin:4px 0 22px;background:#fafafa}
.tx-card{border:1px solid #dadce0;border-radius:18px;padding:18px;margin:12px 0;background:#fff}
.tx-title{font-size:18px;font-weight:650;color:#202124}.tx-meta{font-size:14px;color:#5f6368;margin-top:4px}
.tx-status{display:inline-block;padding:5px 9px;border-radius:999px;font-size:13px;font-weight:650;background:#f1f3f4;color:#5f6368;margin-top:10px}
.tx-status.pending{background:#fff8e1;color:#8a5a00}.tx-status.completed{background:#f3fbf5;color:#137333}.tx-status.rejected{background:#fce8e6;color:#c5221f}
.approval-grid{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}.approval{font-size:13px;padding:6px 9px;border:1px solid #e8eaed;border-radius:999px;color:#5f6368}
.approval.accepted{background:#f3fbf5;color:#137333;border-color:#c8e6c9}.approval.pending{background:#fff8e1;color:#8a5a00;border-color:#fdd663}.approval.rejected{background:#fce8e6;color:#c5221f;border-color:#f4c7c3}
.approval.changes_requested{background:#fce8e6;color:#b06000;border-color:#fdd663}
.outcome-note{border-left:3px solid #1a73e8;background:#f8fbff;padding:12px 14px;border-radius:0 12px 12px 0;font-size:15px;color:#5f6368;margin:10px 0 18px}
.market-card{border:1px solid #e8eaed;border-radius:18px;padding:17px 18px;margin:11px 0;background:#fff}
.market-title{font-size:18px;font-weight:650;color:#202124}.market-meta{font-size:14px;color:#5f6368;margin-top:5px;line-height:1.45}
.market-high{display:inline-block;margin-top:9px;padding:5px 8px;border-radius:999px;background:#f3fbf5;color:#137333;font-size:13px;font-weight:650}
.impact-school{font-size:17px;font-weight:650;color:#202124;margin-top:15px}.impact-row{font-size:15px;color:#5f6368;margin:4px 0}
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


def _safe_internal_week(value) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        week = int(value)
        return week if 0 <= week <= 13 else None
    except Exception:
        return None


def render_all_years(games_df: pd.DataFrame, team: str):
    """Render every loaded year. Week-TBA commitments are first-class records."""
    subset = games_df[(games_df["home_team"] == team) | (games_df["away_team"] == team)].copy()
    if subset.empty:
        st.info("No known future games for this team.")
        return

    years = sorted(int(y) for y in subset["season"].dropna().unique())
    for year in years:
        ys = subset[subset["season"] == year].copy()
        ys["_week_sort"] = ys["week"].apply(lambda v: _safe_internal_week(v) if _safe_internal_week(v) is not None else 99)
        if "date" not in ys.columns:
            ys["date"] = ""
        ys = ys.sort_values(["_week_sort", "date", "home_team", "away_team"])

        chips = []
        for _, row in ys.iterrows():
            neutral = bool(row.get("neutral", False))
            if str(row["home_team"]) == team:
                opp = str(row["away_team"])
                site = "Neutral" if neutral else "Home"
            else:
                opp = str(row["home_team"])
                site = "Neutral" if neutral else "Away"

            internal = _safe_internal_week(row.get("week"))
            week_text = f"Week {display_week(internal)}" if internal is not None else "Week TBA"
            date_text = str(row.get("date", "") or "").strip()
            meta = site + (f" · {date_text}" if date_text and date_text.lower() not in {"nan", "none"} else "")

            chips.append(
                f'<div class="game-chip"><div class="game-chip-week">{week_text}</div>'
                f'<div class="game-chip-opp">{opp}</div><div class="game-chip-site">{meta}</div></div>'
            )
        st.markdown(
            f'<div class="year-row"><div class="year-label">{year}</div><div class="game-chips">{"".join(chips)}</div></div>',
            unsafe_allow_html=True,
        )



def _tx_items(db: WorkspaceDB) -> List[Dict[str, object]]:
    """Transaction listing compatible with every V5+ WorkspaceDB."""
    return db.list("transaction")


def _tx_get(db: WorkspaceDB, tx_id: str) -> Optional[Dict[str, object]]:
    return db.get("transaction", tx_id)


def _tx_save(db: WorkspaceDB, tx_id: str, tx: Dict[str, object]) -> None:
    body = dict(tx)
    body["transaction_id"] = tx_id
    body["updated_at"] = datetime.now().isoformat()
    db.put("transaction", tx_id, body)


def _tx_create(db: WorkspaceDB, payload: Dict[str, object]) -> str:
    body = dict(payload)
    stamp = datetime.now()
    raw = json.dumps(body, sort_keys=True, default=str)
    tx_id = str(body.get("transaction_id") or f"tx_{stamp.strftime('%Y%m%d%H%M%S')}_{abs(hash(raw)) % 100000:05d}")
    body["transaction_id"] = tx_id
    body.setdefault("created_at", stamp.isoformat())
    body.setdefault("history", [])
    _tx_save(db, tx_id, body)
    return tx_id


def _tx_action(
    db: WorkspaceDB,
    tx_id: str,
    *,
    actor: str,
    action: str,
    note: str = "",
    extra: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    tx = _tx_get(db, tx_id)
    if not tx:
        return None
    history = list(tx.get("history", []))
    history.append({
        "at": datetime.now().isoformat(),
        "actor": actor,
        "action": action,
        "note": note,
        "extra": extra or {},
    })
    tx["history"] = history
    _tx_save(db, tx_id, tx)
    return tx


def _tx_recalculate_status(tx: Dict[str, object]) -> Dict[str, object]:
    status = str(tx.get("status", "")).upper()
    if status in {"REJECTED", "SUPERSEDED"}:
        return tx
    school_approvals = dict(tx.get("school_approvals", {}))
    conference_approvals = dict(tx.get("conference_approvals", {}))
    if any(str(v).upper() == "REJECTED" for v in school_approvals.values()):
        tx["status"] = "REJECTED"
        return tx
    schools_done = bool(school_approvals) and all(
        str(v).upper() == "ACCEPTED" for v in school_approvals.values()
    )
    conferences_done = all(
        str(v).upper() == "ACCEPTED" for v in conference_approvals.values()
    )
    if schools_done and conferences_done:
        tx["status"] = "COMPLETED"
        tx["completed_at"] = datetime.now().isoformat()
    else:
        tx["status"] = "PENDING"
    return tx


def _tx_school_approval(
    db: WorkspaceDB,
    tx_id: str,
    school: str,
    status: str,
    note: str = "",
) -> Optional[Dict[str, object]]:
    tx = _tx_get(db, tx_id)
    if not tx:
        return None
    approvals = dict(tx.get("school_approvals", {}))
    approvals[school] = str(status).upper()
    tx["school_approvals"] = approvals
    tx = _tx_recalculate_status(tx)
    _tx_save(db, tx_id, tx)
    _tx_action(
        db, tx_id, actor=school,
        action=f"SCHOOL_{str(status).upper()}",
        note=note,
    )
    return _tx_get(db, tx_id)


def _tx_conference_approval(
    db: WorkspaceDB,
    tx_id: str,
    conference: str,
    status: str,
    note: str = "",
) -> Optional[Dict[str, object]]:
    tx = _tx_get(db, tx_id)
    if not tx:
        return None
    approvals = dict(tx.get("conference_approvals", {}))
    approvals[conference] = str(status).upper()
    tx["conference_approvals"] = approvals
    tx = _tx_recalculate_status(tx)
    _tx_save(db, tx_id, tx)
    _tx_action(
        db, tx_id, actor=conference,
        action=f"CONFERENCE_{str(status).upper()}",
        note=note,
    )
    return _tx_get(db, tx_id)


def _tx_set_status(
    db: WorkspaceDB,
    tx_id: str,
    status: str,
    *,
    actor: str = "System",
    note: str = "",
) -> Optional[Dict[str, object]]:
    tx = _tx_get(db, tx_id)
    if not tx:
        return None
    tx["status"] = str(status).upper()
    _tx_save(db, tx_id, tx)
    _tx_action(db, tx_id, actor=actor, action=f"STATUS_{str(status).upper()}", note=note)
    return _tx_get(db, tx_id)


def completed_transactions(db: WorkspaceDB) -> List[Dict[str, object]]:
    return [
        item["payload"]
        for item in _tx_items(db)
        if str(item["payload"].get("status", "")).upper() == "COMPLETED"
    ]


def apply_completed_transactions_to_dataframe(games_df: pd.DataFrame, db: WorkspaceDB) -> pd.DataFrame:
    """Overlay completed moves and newly scheduled games, idempotently."""
    if games_df is None:
        games_df = pd.DataFrame()
    out = games_df.copy()

    for tx in completed_transactions(db):
        # Move existing games.
        for move in tx.get("moves", []):
            season = int(move["season"])
            home = str(move["home_team"])
            away = str(move["away_team"])
            to_week = int(move["to_week"])
            game_id = str(move.get("game_id", "") or "")

            if out.empty:
                continue
            mask = (
                (pd.to_numeric(out["season"], errors="coerce") == season)
                & (out["home_team"].astype(str) == home)
                & (out["away_team"].astype(str) == away)
            )
            if "game_id" in out.columns and game_id:
                id_mask = out["game_id"].astype(str) == game_id
                if id_mask.any():
                    mask = id_mask
            idx = out.index[mask]
            if len(idx):
                out.loc[idx[0], "week"] = to_week

        # Add newly agreed games.
        for added in tx.get("add_games", []):
            season = int(added["season"])
            home = str(added["home_team"])
            away = str(added["away_team"])
            game_id = str(added.get("game_id", "") or "")

            exists = False
            if not out.empty:
                if "game_id" in out.columns and game_id:
                    exists = bool((out["game_id"].astype(str) == game_id).any())
                if not exists:
                    exists = bool((
                        (pd.to_numeric(out["season"], errors="coerce") == season)
                        & (out["home_team"].astype(str) == home)
                        & (out["away_team"].astype(str) == away)
                    ).any())
            if exists:
                continue

            row = {c: None for c in out.columns} if len(out.columns) else {}
            row.update({
                "game_id": game_id or f"transaction_{tx.get('transaction_id','')}",
                "season": season,
                "week": int(added["week"]),
                "date": added.get("date", ""),
                "home_team": home,
                "away_team": away,
                "neutral": bool(added.get("neutral", False)),
                "campus_home_team": "" if bool(added.get("neutral", False)) else home,
                "game_status": "CONTRACTED",
                "moveability": "UNKNOWN",
                "game_type": str(added.get("game_type", "NONCONFERENCE")),
                "source": "Completed scheduling transaction",
                "confidence": "AUTHORITATIVE",
                "notes": f"Approved transaction {tx.get('transaction_id','')}",
            })
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    return out


def db_need_records(db: WorkspaceDB, season: Optional[int] = None, school: Optional[str] = None) -> List[Dict[str, object]]:
    records = []
    for item in db.list("school_need"):
        payload = dict(item["payload"])
        if season is not None and int(payload.get("season", -1)) != int(season):
            continue
        if school is not None and str(payload.get("team")) != str(school):
            continue
        if str(payload.get("status", "OPEN")).upper() not in {"OPEN", "ACTIVE", "HOLD"}:
            continue
        records.append(payload)
    return records


def enrich_store_with_db_needs(base: ScheduleStore, db: WorkspaceDB, season: int) -> ScheduleStore:
    existing = list(base.needs)
    seen = {(n.team, int(n.season), int(n.week), str(n.need_type).upper(), str(n.location).upper()) for n in existing}
    for item in db_need_records(db, season=season):
        key = (
            str(item["team"]), int(item["season"]), int(item["week"]),
            str(item["need_type"]).upper(), str(item.get("location", "ANY")).upper(),
        )
        if key in seen:
            continue
        existing.append(Need(
            team=key[0],
            season=key[1],
            week=key[2],
            need_type=key[3],
            location=key[4],
            min_guarantee=None if item.get("min_guarantee") in (None, "") else int(item["min_guarantee"]),
            max_guarantee=None if item.get("max_guarantee") in (None, "") else int(item["max_guarantee"]),
            notes=str(item.get("notes", "") or ""),
        ))
        seen.add(key)
    return ScheduleStore(
        list(base.teams.values()),
        list(base.games.values()),
        list(base.slots.values()),
        existing,
    )


def rules_for_schools(db: WorkspaceDB, schools: List[str]) -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    seen = set()
    for school in schools:
        for rule in persistent_profile_rules(db, school):
            rid = str(rule.get("rule_id") or json.dumps(rule, sort_keys=True, default=str))
            if rid not in seen:
                rules.append(rule)
                seen.add(rid)
    return rules


def conference_policy(db: WorkspaceDB, conference: str) -> Dict[str, object]:
    return db.get("conference_policy", conference) or {
        "conference": conference,
        "enforce_no_new_parity": True,
        "require_manual_approval": False,
        "auto_complete_after_school_approvals": True,
    }


def all_conference_policies(db: WorkspaceDB, store: ScheduleStore) -> Dict[str, Dict[str, object]]:
    return {conf: conference_policy(db, conf) for conf in store.fbs_conferences()}


def governing_conferences(store: ScheduleStore, schools: List[str]) -> List[str]:
    return sorted({
        store.teams[s].conference
        for s in schools
        if s in store.teams
        and store.teams[s].subdivision == "FBS"
        and store.teams[s].conference not in {"", "Unknown", "Independent"}
    })


def school_approvals(affected: List[str], proposer: str) -> Dict[str, str]:
    return {
        school: ("ACCEPTED" if school == proposer else "PENDING")
        for school in affected
    }


def transaction_from_solution(
    *,
    sol: Solution,
    proposer: str,
    season: int,
    data_status: str,
    store: ScheduleStore,
    conference_policies: Dict[str, Dict[str, object]],
    context: str = "",
    objective: Optional[Dict[str, object]] = None,
    rules: Optional[List[Dict[str, object]]] = None,
    supersedes: str = "",
) -> Dict[str, object]:
    affected = sorted({s for m in sol.moves for s in (m.home_team, m.away_team)})
    conferences = governing_conferences(store, affected)
    conf_approvals = {
        conf: "PENDING"
        for conf in conferences
        if bool(conference_policies.get(conf, {}).get("require_manual_approval", False))
    }
    return {
        "status": "PENDING",
        "season": int(season),
        "proposer": proposer,
        "affected_schools": affected,
        "school_approvals": school_approvals(affected, proposer),
        "governing_conferences": conferences,
        "conference_approvals": conf_approvals,
        "data_status": data_status,
        "coach_context": context,
        "moves": [
            {
                "game_id": m.game_id,
                "season": int(season),
                "home_team": m.home_team,
                "away_team": m.away_team,
                "from_week": int(m.from_week),
                "to_week": int(m.to_week),
            }
            for m in sol.moves
        ],
        "add_games": [],
        "objective": objective or {"type": "MOVE_GAME"},
        "rules": list(rules or []),
        "supersedes": supersedes,
        "proof": {
            "lexicographic_proven": bool((sol.metadata or {}).get("lexicographic_proven", False)),
            "games_moved": len(sol.moves),
            "disruption_cost": (sol.metadata or {}).get("disruption_cost"),
        },
        "history": [{
            "at": datetime.now().isoformat(),
            "actor": proposer,
            "action": "PROPOSED",
            "note": context,
        }],
    }


def parity_impact_of_new_game(
    store: ScheduleStore,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> Dict[str, object]:
    engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=2.0)
    before = odd_keys(engine, store.copy_games(), season)
    games = store.copy_games()
    gid = f"candidate_{season}_{week}_{home_team}_{away_team}"
    games[gid] = Game(
        game_id=gid,
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
        moveable=False,
        locked=True,
        campus_home_team=home_team,
        game_status="CONCEPT",
        moveability="LOCKED",
        game_type="NONCONFERENCE",
        source="Candidate transaction",
        confidence="INFERRED",
    )
    after = odd_keys(engine, games, season)
    return {
        "new_issues": sorted(after - before),
        "resolved_issues": sorted(before - after),
        "before": before,
        "after": after,
    }


def transaction_from_match(
    *,
    match: Solution,
    proposer: str,
    season: int,
    data_status: str,
    store: ScheduleStore,
    conference_policies: Dict[str, Dict[str, object]],
    need_type: str,
    supersedes: str = "",
) -> Dict[str, object]:
    md = dict(match.metadata or {})
    home = str(md["home_team"])
    away = str(md["away_team"])
    week = int(md["week"])
    affected = sorted({home, away})
    conferences = governing_conferences(store, affected)
    conf_approvals = {
        conf: "PENDING"
        for conf in conferences
        if bool(conference_policies.get(conf, {}).get("require_manual_approval", False))
    }
    impact = parity_impact_of_new_game(store, home, away, season, week)
    return {
        "status": "PENDING",
        "season": int(season),
        "proposer": proposer,
        "affected_schools": affected,
        "school_approvals": school_approvals(affected, proposer),
        "governing_conferences": conferences,
        "conference_approvals": conf_approvals,
        "data_status": data_status,
        "moves": [],
        "add_games": [{
            "game_id": f"new_{season}_{week}_{home}_{away}",
            "season": int(season),
            "week": week,
            "home_team": home,
            "away_team": away,
            "neutral": False,
            "game_type": "FCS_GUARANTEE" if need_type == "BUY_GAME" else "A4",
        }],
        "objective": {
            "type": "NEW_GAME",
            "match_type": need_type,
            "week": week,
            "home_team": home,
            "away_team": away,
        },
        "rules": rules_for_schools(db, affected),
        "supersedes": supersedes,
        "proof": {
            "games_moved": 0,
            "new_game": True,
            "new_parity_issues": len(impact["new_issues"]),
            "resolved_parity_issues": len(impact["resolved_issues"]),
        },
        "history": [{
            "at": datetime.now().isoformat(),
            "actor": proposer,
            "action": "PROPOSED_NEW_GAME",
            "note": match.explanation,
        }],
    }


def confirmation_text(tx: Dict[str, object]) -> str:
    lines = [
        f"Subject: {tx.get('season')} Non-Conference Schedule Change — Approved",
        "",
        "All affected institutions have approved the following non-conference scheduling transaction:",
        "",
    ]
    for m in tx.get("moves", []):
        lines.append(
            f"- {m['away_team']} @ {m['home_team']}: "
            f"Week {display_week(int(m['from_week']))} to Week {display_week(int(m['to_week']))}"
        )
    for g in tx.get("add_games", []):
        lines.append(
            f"- New game: {g['away_team']} @ {g['home_team']} — Week {display_week(int(g['week']))}"
        )
    lines += [
        "",
        "Approved by: " + ", ".join(
            f"{school} ({status})"
            for school, status in tx.get("school_approvals", {}).items()
        ),
        "",
        f"Transaction ID: {tx.get('transaction_id','')}",
    ]
    return "\n".join(lines)


def render_school_impacts(sol: Solution):
    impacts: Dict[str, List[str]] = {}
    for move in sol.moves:
        for school in (move.home_team, move.away_team):
            impacts.setdefault(school, []).append(
                f"{move.away_team} @ {move.home_team}: Week {display_week(move.from_week)} → Week {display_week(move.to_week)}"
            )
    if not impacts:
        return
    st.markdown("**Affected schools**")
    for school in sorted(impacts):
        st.markdown(f'<div class="impact-school">{school}</div>', unsafe_allow_html=True)
        for text in impacts[school]:
            st.markdown(f'<div class="impact-row">{text}</div>', unsafe_allow_html=True)


def transaction_card(tx: Dict[str, object], viewer: str):
    status = str(tx.get("status", "PENDING")).upper()
    css = (
        "completed" if status == "COMPLETED"
        else "rejected" if status in {"REJECTED", "SUPERSEDED"}
        else "pending"
    )
    moves = list(tx.get("moves", []))
    additions = list(tx.get("add_games", []))
    count = len(moves) + len(additions)

    if additions:
        first = additions[0]
        title = f"{first.get('away_team','')} @ {first.get('home_team','')}"
    elif len(moves) == 1:
        title = f"{moves[0].get('away_team','')} @ {moves[0].get('home_team','')}"
    else:
        title = f"{len(moves)}-game coordinated repair"

    st.markdown(
        f'<div class="tx-card"><div class="tx-title">{title}</div>'
        f'<div class="tx-meta">Season {tx.get("season")} · Proposed by {tx.get("proposer")} · '
        f'{count} schedule action{"s" if count != 1 else ""}</div>'
        f'<span class="tx-status {css}">{status}</span>',
        unsafe_allow_html=True,
    )

    approvals = dict(tx.get("school_approvals", {}))
    if approvals:
        approval_html = []
        for school, astatus in approvals.items():
            acss = str(astatus).lower()
            approval_html.append(f'<span class="approval {acss}">{school}: {astatus}</span>')
        st.markdown('<div class="approval-grid">' + "".join(approval_html) + '</div>', unsafe_allow_html=True)

    # Five-second school impact: show viewer's changes first.
    if viewer in approvals:
        own = [
            m for m in moves
            if viewer in {m.get("home_team"), m.get("away_team")}
        ]
        own_add = [
            g for g in additions
            if viewer in {g.get("home_team"), g.get("away_team")}
        ]
        if own or own_add:
            st.markdown(f"**Your impact — {viewer}**")
            for m in own:
                st.write(
                    f"• {m['away_team']} @ {m['home_team']}: "
                    f"Week {display_week(int(m['from_week']))} → Week {display_week(int(m['to_week']))}"
                )
            for g in own_add:
                st.write(
                    f"• Add {g['away_team']} @ {g['home_team']} in Week {display_week(int(g['week']))}"
                )

    with st.expander("Full coordinated plan", expanded=viewer not in approvals):
        for i, m in enumerate(moves, 1):
            st.write(
                f"{i}. {m['away_team']} @ {m['home_team']}: "
                f"Week {display_week(int(m['from_week']))} → Week {display_week(int(m['to_week']))}"
            )
        for g in additions:
            st.write(f"Add {g['away_team']} @ {g['home_team']} — Week {display_week(int(g['week']))}")

    confs = list(tx.get("governing_conferences", []))
    if confs:
        st.caption("Automated conference guardrails: " + ", ".join(confs))
    st.markdown("</div>", unsafe_allow_html=True)


def market_result_card(
    match: Solution,
    *,
    store: ScheduleStore,
    season: int,
    policies: Dict[str, Dict[str, object]],
) -> Tuple[bool, Dict[str, object]]:
    md = dict(match.metadata or {})
    home = str(md["home_team"])
    away = str(md["away_team"])
    week = int(md["week"])
    impact = parity_impact_of_new_game(store, home, away, season, week)

    relevant_confs = governing_conferences(store, [home, away])
    blocked = bool(impact["new_issues"]) and any(
        bool(policies.get(c, {}).get("enforce_no_new_parity", True))
        for c in relevant_confs
    )

    st.markdown(
        f'<div class="market-card"><div class="market-title">{away} @ {home} · Week {display_week(week)}</div>'
        f'<div class="market-meta">{match.explanation}</div>'
        + (
            '<span class="market-high">High-opportunity Weeks 1–4</span>'
            if str(md.get("market_liquidity")) == "HIGH" else ""
        )
        + '</div>',
        unsafe_allow_html=True,
    )
    if md.get("explicit_need"):
        st.caption("✓ Compatible explicit school need is recorded.")
    if impact["resolved_issues"]:
        st.success(f"This matchup resolves {len(impact['resolved_issues'])} modeled conference parity issue(s).")
    if impact["new_issues"]:
        st.warning(f"This matchup creates {len(impact['new_issues'])} modeled parity issue(s).")
    if blocked:
        st.caption("Conference guardrails prevent proposing this matchup as currently structured.")
    return blocked, impact


def easiest_relocation(
    store: ScheduleStore,
    game: Game,
    *,
    rules: List[Dict[str, object]],
    preserve_parity: bool,
) -> Optional[Tuple[int, Solution]]:
    candidates: List[Tuple[Tuple[int, int, int], int, Solution]] = []
    for target in range(14):
        if target == game.week:
            continue
        run_engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=2.2)
        result = run_engine.solve_move_game(Intent(
            action="MOVE_GAME",
            season=int(game.season),
            target_week=target,
            team_a=game.home_team,
            team_b=game.away_team,
            preserve_fbs_conference_parity=preserve_parity,
            max_additional_moves=10,
            rules=rules,
        ))
        if not result:
            continue
        sol = result[0]
        # Minimum changes first; then early-market preference; then displacement.
        key = (
            len(sol.moves),
            0 if target <= 3 else 1,
            abs(target - game.week),
        )
        candidates.append((key, target, sol))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def save_school_need(
    db: WorkspaceDB,
    *,
    team: str,
    season: int,
    display_weeks: List[int],
    need_type: str,
    location: str,
    min_guarantee: Optional[int],
    max_guarantee: Optional[int],
    notes: str,
):
    for display in display_weeks:
        week = internal_week(display)
        key = f"{team}|{season}|{need_type}|{week}"
        db.put("school_need", key, {
            "team": team,
            "season": int(season),
            "week": week,
            "need_type": need_type,
            "location": location,
            "min_guarantee": min_guarantee,
            "max_guarantee": max_guarantee,
            "status": "OPEN",
            "notes": notes,
        })


def try_counterproposal(
    *,
    tx: Dict[str, object],
    school: str,
    game_id: str,
    requested_week: int,
    store: ScheduleStore,
    db: WorkspaceDB,
    policies: Dict[str, Dict[str, object]],
    data_status: str,
) -> Tuple[Optional[str], str]:
    """Automatically re-solve a school's suggested alternative."""
    objective = dict(tx.get("objective", {}))
    affected = list(tx.get("affected_schools", []))
    combined_rules = rules_for_schools(db, affected)
    old_tx_id = str(tx.get("transaction_id", ""))

    # Direct new-game counterproposal: move the proposed new matchup to another week.
    if objective.get("type") == "NEW_GAME":
        additions = list(tx.get("add_games", []))
        if not additions:
            return None, "No proposed new game was found."
        g = additions[0]
        home, away = str(g["home_team"]), str(g["away_team"])
        if store.game_for_team_week(store.copy_games(), home, int(tx["season"]), requested_week):
            return None, f"{home} already has a known game in Week {display_week(requested_week)}."
        if store.game_for_team_week(store.copy_games(), away, int(tx["season"]), requested_week):
            return None, f"{away} already has a known game in Week {display_week(requested_week)}."

        impact = parity_impact_of_new_game(store, home, away, int(tx["season"]), requested_week)
        confs = governing_conferences(store, [home, away])
        if impact["new_issues"] and any(
            bool(policies.get(c, {}).get("enforce_no_new_parity", True)) for c in confs
        ):
            return None, "That week creates a protected conference parity issue."

        fake_match = Solution(
            title="Counterproposal",
            moves=[],
            score=100,
            explanation=f"{school} suggested Week {display_week(requested_week)}.",
            metadata={
                "home_team": home,
                "away_team": away,
                "week": requested_week,
                "match_type": objective.get("match_type", "NEW_GAME"),
            },
        )
        payload = transaction_from_match(
            match=fake_match,
            proposer=school,
            season=int(tx["season"]),
            data_status=data_status,
            store=store,
            conference_policies=policies,
            need_type=str(objective.get("match_type", "NEW_GAME")),
            supersedes=old_tx_id,
        )
        new_id = _tx_create(db, payload)
        _tx_set_status(db, old_tx_id, "SUPERSEDED", actor=school, note=f"Replaced by {new_id}.")
        return new_id, f"Feasible counterproposal created for Week {display_week(requested_week)}."

    # Find the original game in the current schedule.
    target_game = store.games.get(game_id)
    if not target_game:
        for g in store.games.values():
            if g.game_id == game_id:
                target_game = g
                break
    if not target_game:
        return None, "The selected game is not available in the active schedule."

    preserve = any(
        bool(policies.get(c, {}).get("enforce_no_new_parity", True))
        for c in tx.get("governing_conferences", [])
    )

    if objective.get("type") == "CONFERENCE_EVEN":
        exact_rule = {
            "rule_id": f"counter_{game_id}",
            "hardness": "MUST",
            "rule_type": "GAME_WEEK_WINDOW",
            "game_id": game_id,
            "team": "",
            "start_week": requested_week,
            "end_week": requested_week,
            "value": 1,
            "active": True,
            "note": f"{school} requested this week.",
        }
        conf = str(objective.get("conference"))
        weeks = [int(w) for w in objective.get("target_weeks", [])]
        run_engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=12.0)
        plans = run_engine.optimize_national(Intent(
            action="OPTIMIZE_NATIONAL",
            season=int(tx["season"]),
            target_weeks=weeks,
            conferences=[conf],
            conference=conf,
            all_conferences=False,
            preserve_fbs_conference_parity=True,
            max_additional_moves=60,
            rules=combined_rules + [exact_rule],
            summary="Counterproposal conference repair",
        ))
        if not plans or bool((plans[0].metadata or {}).get("infeasible")):
            return None, "That suggested week cannot preserve the conference's required outcome."
        payload = transaction_from_solution(
            sol=plans[0],
            proposer=school,
            season=int(tx["season"]),
            data_status=data_status,
            store=store,
            conference_policies=policies,
            context=f"{school} counterproposal",
            objective=objective,
            rules=combined_rules + [exact_rule],
            supersedes=old_tx_id,
        )
    else:
        run_engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=6.0)
        results = run_engine.solve_move_game(Intent(
            action="MOVE_GAME",
            season=int(tx["season"]),
            target_week=requested_week,
            team_a=target_game.home_team,
            team_b=target_game.away_team,
            preserve_fbs_conference_parity=preserve,
            max_additional_moves=10,
            rules=combined_rules,
            summary="School counterproposal",
        ))
        if not results:
            return None, "The suggested week is not feasible under current school and conference rules."
        payload = transaction_from_solution(
            sol=results[0],
            proposer=school,
            season=int(tx["season"]),
            data_status=data_status,
            store=store,
            conference_policies=policies,
            context=f"{school} counterproposal",
            objective=objective or {"type": "MOVE_GAME"},
            rules=combined_rules,
            supersedes=old_tx_id,
        )

    new_id = _tx_create(db, payload)
    _tx_set_status(db, old_tx_id, "SUPERSEDED", actor=school, note=f"Replaced by counterproposal {new_id}.")
    return new_id, "A feasible revised transaction was created automatically."


# ------------------------ workspace / data ----------------------------

db = get_db()

st.markdown('<div class="simple-title">College Football Scheduling</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="simple-sub">Solve the scheduling problem, coordinate every affected school, and collect unanimous approval in one place.</div>',
    unsafe_allow_html=True,
)

top1, top2 = st.columns([1.2, 1])
with top1:
    data_mode = st.selectbox(
        "Schedule data",
        ["Public prototype", "Authoritative upload", "Demo"],
        index=0,
        help="Use authoritative data before relying on a recommendation operationally.",
    )

public_teams_df = None
public_games_df = None
store = None
all_games_df = pd.DataFrame()
all_teams_df = pd.DataFrame()
slots_df = pd.DataFrame()
needs_df = pd.DataFrame()
report_ok = True

if data_mode == "Authoritative upload":
    with st.expander("Authoritative schedule data", expanded=True):
        st.download_button(
            "Download Excel template",
            data=make_template_bytes(),
            file_name="college_football_schedule_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        uploaded = st.file_uploader("Upload schedule", type=["xlsx", "xlsm", "csv"])

    if uploaded is not None:
        raw = uploaded.getvalue()
        if uploaded.name.lower().endswith(".csv"):
            try:
                public_teams_df, _, _ = scrape_fbschedules_public(tuple(range(2027, 2038)))
            except Exception:
                public_teams_df = None
        all_teams_df, all_games_df, slots_df, needs_df, report = load_schedule_upload(
            raw, uploaded.name, public_teams_df
        )
        report_ok = report.ok
        for e in report.errors:
            st.error(e)
        for w in report.warnings:
            st.warning(w)
        for msg in report.info:
            st.caption(msg)
        if report.ok:
            db.put("data_snapshot", "latest", {
                "teams": all_teams_df.to_dict("records"),
                "games": all_games_df.to_dict("records"),
                "slots": slots_df.to_dict("records"),
                "needs": needs_df.to_dict("records"),
                "source_name": uploaded.name,
                "saved_at": datetime.now().isoformat(),
            })
    else:
        snapshot = db.get("data_snapshot", "latest")
        if snapshot:
            all_teams_df = pd.DataFrame(snapshot.get("teams", []))
            all_games_df = pd.DataFrame(snapshot.get("games", []))
            slots_df = pd.DataFrame(snapshot.get("slots", []))
            needs_df = pd.DataFrame(snapshot.get("needs", []))
            st.caption(f"Using saved authoritative snapshot: {snapshot.get('source_name','latest')}")
        else:
            st.info("Upload the authoritative schedule workbook, or switch to Public prototype to explore immediately.")
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

all_games_df = apply_completed_transactions_to_dataframe(all_games_df, db)

available_years = sorted(int(y) for y in all_games_df["season"].dropna().unique())
if not available_years:
    st.error("No seasons are available in the current data.")
    st.stop()

with top2:
    default_year = 2028 if 2028 in available_years else available_years[0]
    season = st.selectbox("Active season", available_years, index=available_years.index(default_year))

if data_mode == "Authoritative upload":
    store = build_authoritative_store(
        all_teams_df, all_games_df, slots_df, int(season), needs_df
    )
elif data_mode == "Public prototype":
    store = build_real_store(all_teams_df, all_games_df, int(season))
else:
    store = build_demo_store()

store = enrich_store_with_db_needs(store, db, int(season))
engine = AdvancedNonConferenceOptimizer(store, time_limit_seconds=6.0)
status_text, status_class = data_label(data_mode, report_ok)
policies = all_conference_policies(db, store)

st.markdown(
    f'<div style="text-align:center;margin:2px 0 20px">'
    f'<span class="data-pill {status_class}">● {status_text}</span></div>',
    unsafe_allow_html=True,
)

if not db.durable:
    st.caption("Pilot persistence is local SQLite. Add DATABASE_URL for durable school needs, proposals and audit history.")


# ------------------------------ UI -----------------------------------

perspective = st.radio(
    "Workspace",
    ["School", "Conference"],
    horizontal=True,
    label_visibility="collapsed",
)

season_games = sorted(list(store.games.values()), key=lambda g: (g.week, g.home_team, g.away_team))
all_team_names = sorted(store.teams.keys())

if perspective == "School":
    school_names = all_team_names
    default_school = school_names.index("Georgia") if "Georgia" in school_names else 0
    acting_school = st.selectbox("School", school_names, index=default_school)

    st.markdown(f'<div class="page-title">{acting_school}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">Your schedule, your needs, and every proposal requiring your approval.</div>',
        unsafe_allow_html=True,
    )

    tab_schedule, tab_solve, tab_needs, tab_proposals = st.tabs([
        "My Schedule", "Solve", "Needs & Opportunities", "Proposals"
    ])

    with tab_schedule:
        profile_rules = persistent_profile_rules(db, acting_school)
        if profile_rules:
            st.caption(f"{len(profile_rules)} saved scheduling rule{'s' if len(profile_rules) != 1 else ''}")
        render_all_years(all_games_df, acting_school)

    with tab_solve:
        outcome = st.radio(
            "What do you need to accomplish?",
            ["Move a game", "Make my school open", "Find a buy game", "Find an A4 opponent"],
            horizontal=True,
        )
        acting_conf = store.teams.get(acting_school).conference if acting_school in store.teams else ""
        preserve_parity = bool(conference_policy(db, acting_conf).get("enforce_no_new_parity", True)) if acting_conf else False

        if outcome == "Move a game":
            school_games = [
                g for g in season_games
                if g.involves(acting_school) and str(g.game_type).upper() != "CONFERENCE"
            ]
            if not school_games:
                st.info("No dated non-conference games are available to move in the active season.")
            else:
                labels = {game_label(g): g for g in school_games}
                selected_game = labels[st.selectbox("Game", list(labels.keys()), key=f"solve_move_game_{acting_school}")]
                target_display = st.selectbox(
                    "Move to", list(range(1, 15)), index=int(selected_game.week),
                    key=f"solve_move_week_{acting_school}",
                )
                target_week = internal_week(target_display)

                with st.expander("Constraints & preferences", expanded=False):
                    rules, protected_ids, avoid_ids, context = constraint_builder(
                        db,
                        prefix=f"solve_move_{season}_{acting_school}_{selected_game.game_id}",
                        primary_team=acting_school,
                        teams=all_team_names,
                        games=[g for g in season_games if g.game_id != selected_game.game_id],
                    )

                if st.button("Solve", type="primary", use_container_width=True, key=f"solve_move_{acting_school}"):
                    run_store = store_with_locked(store, protected_ids)
                    run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=6.0)
                    intent = build_move_intent(selected_game, target_week, rules, avoid_ids, context, preserve_parity)
                    with st.spinner("Looking across every affected schedule…"):
                        results = run_engine.solve_move_game(intent)
                    st.session_state[f"solve_result_{acting_school}"] = {
                        "result": results[0] if results else None,
                        "intent": intent,
                        "protected": protected_ids,
                        "context": context,
                        "objective": {
                            "type": "MOVE_GAME",
                            "game_id": selected_game.game_id,
                            "requested_week": target_week,
                        },
                    }

                state = st.session_state.get(f"solve_result_{acting_school}")
                if state:
                    sol = state.get("result")
                    run_store = store_with_locked(store, set(state.get("protected") or set()))
                    run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=6.0)
                    if sol:
                        render_result(run_engine, sol, season=int(season), data_status=status_text)
                        render_school_impacts(sol)
                        if st.button("Send coordinated proposal", type="primary", use_container_width=True, key=f"send_move_{acting_school}"):
                            payload = transaction_from_solution(
                                sol=sol,
                                proposer=acting_school,
                                season=int(season),
                                data_status=status_text,
                                store=run_store,
                                conference_policies=policies,
                                context=str(state.get("context", "")),
                                objective=state.get("objective"),
                                rules=list(state.get("intent").rules or []),
                            )
                            tx_id = _tx_create(db, payload)
                            _tx_action(
            db,
                                tx_id, actor=acting_school, action="SENT_TO_AFFECTED_SCHOOLS",
                                note="Unanimous school approval requested."
                            )
                            st.success(f"Proposal {tx_id} sent to every affected school.")
                    else:
                        st.error("No feasible path satisfies the current school and conference rules.")

        elif outcome == "Make my school open":
            open_display = st.selectbox("I need to be open in", list(range(1, 15)), key=f"open_week_{acting_school}")
            open_week = internal_week(open_display)
            occupied = store.game_for_team_week(store.copy_games(), acting_school, int(season), open_week)
            if occupied is None:
                st.success(f"{acting_school} is already open in Week {open_display}.")
            elif str(occupied.game_type).upper() == "CONFERENCE":
                st.error("That week contains a conference game and cannot be repaired through the nonconference workflow.")
            else:
                st.markdown(f"Current conflict: **{occupied.away_team} @ {occupied.home_team}**")
                rules = rules_for_schools(db, [acting_school])
                if st.button("Find easiest way to open this week", type="primary", use_container_width=True):
                    with st.spinner("Searching the smallest relocation chain…"):
                        answer = easiest_relocation(
                            store, occupied, rules=rules, preserve_parity=preserve_parity
                        )
                    st.session_state[f"open_result_{acting_school}"] = answer
                answer = st.session_state.get(f"open_result_{acting_school}")
                if answer:
                    destination, sol = answer
                    st.markdown(
                        f'<div class="outcome-note">Best path moves the current game to <strong>Week {display_week(destination)}</strong>.</div>',
                        unsafe_allow_html=True,
                    )
                    render_result(engine, sol, season=int(season), data_status=status_text)
                    render_school_impacts(sol)
                    if st.button("Send coordinated proposal", type="primary", use_container_width=True, key=f"send_open_{acting_school}"):
                        payload = transaction_from_solution(
                            sol=sol,
                            proposer=acting_school,
                            season=int(season),
                            data_status=status_text,
                            store=store,
                            conference_policies=policies,
                            objective={"type": "MAKE_SCHOOL_OPEN", "school": acting_school, "week": open_week},
                            rules=rules,
                        )
                        tx_id = _tx_create(db, payload)
                        st.success(f"Proposal {tx_id} sent to every affected school.")

        else:
            match_type = "BUY_GAME" if outcome == "Find a buy game" else "A4"
            if match_type == "A4" and not store.teams[acting_school].is_a4:
                st.warning("This school is not classified as A4 in the current data.")
            else:
                week_choice = st.selectbox(
                    "Week",
                    ["Best available"] + list(range(1, 15)),
                    key=f"market_week_{acting_school}_{match_type}",
                )
                target_week = None if week_choice == "Best available" else internal_week(int(week_choice))
                location = st.selectbox(
                    "Site preference",
                    ["HOME", "ANY", "AWAY"] if match_type == "A4" else ["HOME", "ANY"],
                    key=f"market_location_{acting_school}_{match_type}",
                )
                max_guarantee = None
                if match_type == "BUY_GAME":
                    max_guarantee = st.number_input(
                        "Maximum guarantee (optional)",
                        min_value=0, value=0, step=50000,
                        key=f"market_guarantee_{acting_school}",
                    )
                    if max_guarantee == 0:
                        max_guarantee = None

                st.markdown(
                    '<div class="outcome-note">Weeks 1–4 receive a market-liquidity preference when all else is equal. Later weeks remain valid.</div>',
                    unsafe_allow_html=True,
                )

                if st.button("Find best matches", type="primary", use_container_width=True, key=f"find_market_{acting_school}_{match_type}"):
                    intent = Intent(
                        action="FIND_BUY_GAME" if match_type == "BUY_GAME" else "FIND_A4_GAME",
                        season=int(season),
                        target_week=target_week,
                        team_a=acting_school,
                        location=location,
                        max_guarantee=max_guarantee,
                    )
                    results = engine.solve(intent)
                    st.session_state[f"market_results_{acting_school}_{match_type}"] = results

                results = st.session_state.get(f"market_results_{acting_school}_{match_type}", [])
                if results:
                    for idx, match in enumerate(results[:8]):
                        blocked, impact = market_result_card(
                            match, store=store, season=int(season), policies=policies
                        )
                        if not blocked:
                            if st.button(
                                "Propose matchup",
                                use_container_width=True,
                                key=f"propose_match_{acting_school}_{match_type}_{idx}",
                            ):
                                payload = transaction_from_match(
                                    match=match,
                                    proposer=acting_school,
                                    season=int(season),
                                    data_status=status_text,
                                    store=store,
                                    conference_policies=policies,
                                    need_type=match_type,
                                )
                                tx_id = _tx_create(db, payload)
                                st.success(f"Proposal {tx_id} sent to {match.metadata.get('candidate')}.")

    with tab_needs:
        st.markdown("### Tell the market what you need")
        st.caption("Explicit school needs make matching much stronger than inferring an open date.")
        need_type_label = st.selectbox(
            "Need",
            ["FCS buy game", "A4 opponent"],
            key=f"need_type_{acting_school}",
        )
        need_type = "FCS_BUY" if need_type_label == "FCS buy game" else "A4"
        need_year = st.selectbox(
            "Season",
            available_years,
            index=available_years.index(season),
            key=f"need_year_{acting_school}",
        )
        need_weeks = st.multiselect(
            "Acceptable weeks",
            list(range(1, 15)),
            default=[1, 2, 3, 4],
            key=f"need_weeks_{acting_school}",
        )
        need_location = st.selectbox(
            "Location",
            ["HOME", "ANY", "AWAY"],
            key=f"need_location_{acting_school}",
        )
        min_g = max_g = None
        if need_type == "FCS_BUY":
            g1, g2 = st.columns(2)
            with g1:
                min_value = st.number_input(
                    "Minimum guarantee (optional)", min_value=0, value=0, step=50000,
                    key=f"need_min_g_{acting_school}",
                )
            with g2:
                max_value = st.number_input(
                    "Maximum guarantee (optional)", min_value=0, value=0, step=50000,
                    key=f"need_max_g_{acting_school}",
                )
            min_g = None if min_value == 0 else int(min_value)
            max_g = None if max_value == 0 else int(max_value)
        need_notes = st.text_input("Notes (optional)", key=f"need_notes_{acting_school}")
        if st.button("Publish need", type="primary", use_container_width=True, key=f"publish_need_{acting_school}"):
            if not need_weeks:
                st.warning("Choose at least one acceptable week.")
            else:
                save_school_need(
                    db, team=acting_school, season=int(need_year),
                    display_weeks=need_weeks, need_type=need_type,
                    location=need_location, min_guarantee=min_g,
                    max_guarantee=max_g, notes=need_notes,
                )
                st.success("Need published to the scheduling market.")
                st.rerun()

        current_needs = db_need_records(db, school=acting_school)
        if current_needs:
            st.markdown("### Open needs")
            need_rows = [{
                "Season": n["season"],
                "Week": display_week(int(n["week"])),
                "Need": n["need_type"],
                "Location": n["location"],
                "Min guarantee": n.get("min_guarantee"),
                "Max guarantee": n.get("max_guarantee"),
            } for n in current_needs]
            st.dataframe(pd.DataFrame(need_rows), use_container_width=True, hide_index=True)

    with tab_proposals:
        relevant = [
            item["payload"] for item in _tx_items(db)
            if acting_school in item["payload"].get("affected_schools", [])
            or acting_school == item["payload"].get("proposer")
        ]
        if not relevant:
            st.info("No proposals involve this school yet.")
        else:
            for tx in relevant:
                tx_id = str(tx.get("transaction_id"))
                transaction_card(tx, acting_school)
                status = str(tx.get("status", "PENDING")).upper()
                approvals = dict(tx.get("school_approvals", {}))
                my_status = approvals.get(acting_school)

                if status == "PENDING" and my_status == "PENDING":
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Accept", type="primary", use_container_width=True, key=f"accept_{tx_id}_{acting_school}"):
                            updated = _tx_school_approval(db, tx_id, acting_school, "ACCEPTED")
                            if updated and updated.get("status") == "COMPLETED":
                                st.success("Unanimous approval complete. The schedule has been updated.")
                            else:
                                st.success("Accepted. Waiting on the remaining schools.")
                            st.rerun()
                    with c2:
                        if st.button("Reject", use_container_width=True, key=f"reject_toggle_{tx_id}_{acting_school}"):
                            st.session_state[f"show_reject_{tx_id}_{acting_school}"] = True

                    if st.session_state.get(f"show_reject_{tx_id}_{acting_school}"):
                        reason = st.selectbox(
                            "Why doesn't it work?",
                            ["Coach preference", "Contract issue", "Travel issue", "Game cannot move", "Financial issue", "Other"],
                            key=f"reject_reason_{tx_id}_{acting_school}",
                        )
                        note = st.text_input("Detail", key=f"reject_note_{tx_id}_{acting_school}")
                        if st.button("Confirm rejection", use_container_width=True, key=f"confirm_reject_{tx_id}_{acting_school}"):
                            _tx_school_approval(db, tx_id, acting_school, "REJECTED", f"{reason}: {note}")
                            db.add_feedback(
                                season=int(tx.get("season")),
                                team=acting_school,
                                game_id="",
                                reason=reason,
                                notes=note,
                                payload=tx,
                            )
                            st.rerun()

                    with st.expander("Suggest another week", expanded=False):
                        relevant_moves = [
                            m for m in tx.get("moves", [])
                            if acting_school in {m.get("home_team"), m.get("away_team")}
                        ]
                        additions = [
                            g for g in tx.get("add_games", [])
                            if acting_school in {g.get("home_team"), g.get("away_team")}
                        ]
                        options = {}
                        for m in relevant_moves:
                            options[f"{m['away_team']} @ {m['home_team']}"] = str(m["game_id"])
                        for g in additions:
                            options[f"{g['away_team']} @ {g['home_team']} (new game)"] = str(g["game_id"])
                        if options:
                            chosen_label = st.selectbox("Game", list(options.keys()), key=f"counter_game_{tx_id}_{acting_school}")
                            alt_display = st.selectbox("Suggested week", list(range(1, 15)), key=f"counter_week_{tx_id}_{acting_school}")
                            alt_note = st.text_input("Why?", key=f"counter_note_{tx_id}_{acting_school}")
                            if st.button("Test and send counterproposal", type="primary", use_container_width=True, key=f"counter_send_{tx_id}_{acting_school}"):
                                new_id, message = try_counterproposal(
                                    tx=tx,
                                    school=acting_school,
                                    game_id=options[chosen_label],
                                    requested_week=internal_week(alt_display),
                                    store=store,
                                    db=db,
                                    policies=policies,
                                    data_status=status_text,
                                )
                                if new_id:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)

                elif status == "PENDING" and my_status == "ACCEPTED":
                    st.caption("You have accepted. Waiting on the remaining affected schools.")
                elif status == "COMPLETED":
                    st.success("Completed — unanimous approval is on record and the schedule is updated.")
                    text = confirmation_text(tx)
                    with st.expander("Approval confirmation", expanded=False):
                        st.text_area("Reply-all equivalent", value=text, height=200, disabled=True, key=f"confirm_text_{tx_id}")
                        st.download_button(
                            "Download confirmation",
                            data=text.encode("utf-8"),
                            file_name=f"{tx_id}_approval_confirmation.txt",
                            mime="text/plain",
                            key=f"download_confirmation_{tx_id}",
                        )
                elif status == "SUPERSEDED":
                    st.caption("Superseded by a revised proposal.")
                elif status == "REJECTED":
                    st.error("This proposal was rejected.")

                with st.expander("Audit history", expanded=False):
                    history = list(tx.get("history", []))
                    if history:
                        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

else:
    confs = engine.store.fbs_conferences()
    default_conf = confs.index("SEC") if "SEC" in confs else 0
    acting_conf = st.selectbox("Conference", confs, index=default_conf)

    st.markdown(f'<div class="page-title">{acting_conf} oversight</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">Solve conference-wide problems, send one coordinated proposal, and let the affected schools negotiate and approve it inside the platform.</div>',
        unsafe_allow_html=True,
    )

    conf_solve, conf_transactions, conf_governance, conf_schedules = st.tabs([
        "Solve", "Transactions", "Governance", "Schedules"
    ])

    with conf_solve:
        st.markdown("### Make conference weeks even")
        selected_display_weeks = st.multiselect(
            "Weeks that must be even",
            list(range(1, 15)),
            default=[1, 2, 3],
        )
        selected_weeks = [internal_week(w) for w in selected_display_weeks]
        members = sorted(t.name for t in store.conference_members(acting_conf))
        member_set = set(members)
        conf_games = [g for g in season_games if g.home_team in member_set or g.away_team in member_set]

        current_odd = []
        for week in selected_weeks:
            state = engine.conference_parity(store.copy_games(), int(season), week).get(acting_conf, "")
            if str(state).startswith("ODD"):
                current_odd.append(week)
        if current_odd:
            st.markdown(
                '<div class="outcome-note">Needs repair: '
                + ", ".join(f"Week {display_week(w)}" for w in current_odd)
                + ".</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Constraints & preferences", expanded=False):
            rules, protected_ids, avoid_ids, context = constraint_builder(
                db,
                prefix=f"confsolve_{season}_{acting_conf}",
                primary_team=members[0] if members else all_team_names[0],
                teams=members if members else all_team_names,
                games=conf_games,
            )

        if st.button("Solve conference problem", type="primary", use_container_width=True):
            run_store = store_with_locked(store, protected_ids)
            run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=12.0)
            intent = Intent(
                action="OPTIMIZE_NATIONAL",
                season=int(season),
                target_weeks=selected_weeks,
                conferences=[acting_conf],
                conference=acting_conf,
                all_conferences=False,
                preserve_fbs_conference_parity=True,
                max_additional_moves=60,
                avoid_game_ids=sorted(avoid_ids),
                coach_context=context,
                rules=rules + rules_for_schools(db, members),
                summary="Make selected conference weeks even",
            )
            with st.spinner("Looking across all affected schedules and minimizing the coordinated changes…"):
                plans = run_engine.optimize_national(intent)
            st.session_state[f"conf_plan_{acting_conf}"] = {
                "result": plans[0] if plans else None,
                "store": run_store,
                "rules": intent.rules,
                "context": context,
                "weeks": selected_weeks,
            }

        state = st.session_state.get(f"conf_plan_{acting_conf}")
        if state:
            plan = state.get("result")
            if plan and not bool((plan.metadata or {}).get("infeasible")):
                run_engine = AdvancedNonConferenceOptimizer(state["store"], time_limit_seconds=12.0)
                render_result(run_engine, plan, season=int(season), data_status=status_text, label="BEST COORDINATED PLAN")
                render_school_impacts(plan)
                affected = sorted({s for m in plan.moves for s in (m.home_team, m.away_team)})
                st.markdown(
                    f'<div class="outcome-note"><strong>{len(affected)} schools</strong> must approve. '
                    f'The platform sends one proposal and records each Yes/No/counterproposal.</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Send to all affected schools", type="primary", use_container_width=True):
                    objective = {
                        "type": "CONFERENCE_EVEN",
                        "conference": acting_conf,
                        "target_weeks": list(state["weeks"]),
                    }
                    payload = transaction_from_solution(
                        sol=plan,
                        proposer=acting_conf,
                        season=int(season),
                        data_status=status_text,
                        store=state["store"],
                        conference_policies=policies,
                        context=str(state.get("context", "")),
                        objective=objective,
                        rules=list(state.get("rules") or []),
                    )
                    tx_id = _tx_create(db, payload)
                    _tx_action(
            db,
                        tx_id,
                        actor=acting_conf,
                        action="SENT_TO_ALL_AFFECTED_SCHOOLS",
                        note=f"{len(affected)} schools asked for unanimous approval.",
                    )
                    st.success(f"One coordinated proposal sent to {len(affected)} schools.")
            elif state:
                st.error("No plan satisfies all selected conference outcomes and hard constraints.")

    with conf_transactions:
        relevant = [
            item["payload"] for item in _tx_items(db)
            if acting_conf in item["payload"].get("governing_conferences", [])
            or item["payload"].get("proposer") == acting_conf
        ]
        if not relevant:
            st.info("No transactions involve this conference yet.")
        else:
            for tx in relevant:
                tx_id = str(tx.get("transaction_id"))
                transaction_card(tx, acting_conf)
                status = str(tx.get("status", "PENDING")).upper()
                approvals = dict(tx.get("school_approvals", {}))
                if status == "PENDING":
                    pending = [s for s, v in approvals.items() if v == "PENDING"]
                    accepted = [s for s, v in approvals.items() if v == "ACCEPTED"]
                    st.caption(f"{len(accepted)} accepted · {len(pending)} pending")
                if status == "COMPLETED":
                    st.success("Unanimous school approval complete.")
                    with st.expander("Final confirmation", expanded=False):
                        text = confirmation_text(tx)
                        st.text_area("Approval record", value=text, height=200, disabled=True, key=f"conf_confirm_{tx_id}")
                if tx.get("suggestions"):
                    with st.expander("School counterproposals", expanded=True):
                        st.dataframe(pd.DataFrame(tx["suggestions"]), use_container_width=True, hide_index=True)

    with conf_governance:
        policy = conference_policy(db, acting_conf)
        enforce_parity = st.checkbox(
            "Do not allow a school transaction to create a new conference parity issue",
            value=bool(policy.get("enforce_no_new_parity", True)),
        )
        require_manual = st.checkbox(
            "Require conference approval after every affected school accepts",
            value=bool(policy.get("require_manual_approval", False)),
            help="Leave this off to make school-to-school transactions truly self-service.",
        )
        if st.button("Save governance", type="primary", use_container_width=True):
            db.put("conference_policy", acting_conf, {
                "conference": acting_conf,
                "enforce_no_new_parity": bool(enforce_parity),
                "require_manual_approval": bool(require_manual),
                "auto_complete_after_school_approvals": True,
            })
            st.success("Governance saved.")
        if not require_manual:
            st.success("Self-service mode is active: unanimous school approval completes an ordinary transaction.")

    with conf_schedules:
        rows = []
        members = sorted(t.name for t in store.conference_members(acting_conf))
        for team in members:
            games = sorted([g for g in store.games.values() if g.involves(team)], key=lambda g: g.week)
            row = {"School": team}
            for w in range(14):
                game = next((g for g in games if g.week == w), None)
                if game:
                    opp = game.away_team if game.home_team == team else game.home_team
                    site = game.site_for(team)
                    row[f"W{display_week(w)}"] = f"{'vs' if site == 'HOME' else '@' if site == 'AWAY' else 'N'} {opp}"
                else:
                    row[f"W{display_week(w)}"] = ""
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=650)

st.caption(
    "V7 milestone: solve the multi-school problem, send one coordinated proposal, automate counterproposals, "
    "collect unanimous school approval, and update the schedule without the conference office brokering every call."
)
