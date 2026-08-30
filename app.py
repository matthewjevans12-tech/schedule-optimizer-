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
    page_title="Schedule OS",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
  --ink:#121923;
  --muted:#697586;
  --subtle:#8A96A6;
  --canvas:#F5F7FA;
  --surface:#FFFFFF;
  --surface-2:#FAFBFC;
  --navy:#0B1F33;
  --navy-2:#102A43;
  --blue:#246BFD;
  --blue-soft:#EEF4FF;
  --line:#E5EAF0;
  --line-strong:#D7DEE8;
  --green:#16835D;
  --green-soft:#EAF8F2;
  --amber:#A56600;
  --amber-soft:#FFF5DF;
  --red:#C2413A;
  --red-soft:#FDEDEC;
  --shadow:0 10px 35px rgba(19,33,53,.07);
}
html,body,.stApp,[class*="css"]{
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif!important;
}
.stApp{background:var(--canvas)!important;color:var(--ink)!important}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent!important}
.block-container{
  max-width:1460px!important;
  padding:1.45rem 2.4rem 5rem!important;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--navy) 0%,#091A2B 100%)!important;
  border-right:1px solid rgba(255,255,255,.05)!important;
  min-width:280px!important;
  max-width:280px!important;
}
[data-testid="stSidebar"]>div:first-child{padding-top:1.1rem!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption{
  color:#CBD7E5!important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio>label{
  color:#8FA5BC!important;
  font-size:11px!important;
  font-weight:800!important;
  letter-spacing:.11em!important;
  text-transform:uppercase!important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:5px!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{
  display:flex!important;
  padding:10px 12px!important;
  border-radius:10px!important;
  color:#DDE7F1!important;
  font-size:14px!important;
  font-weight:650!important;
  letter-spacing:0!important;
  text-transform:none!important;
  transition:.15s ease!important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{
  background:rgba(255,255,255,.06)!important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked){
  background:rgba(76,135,255,.18)!important;
  color:#FFFFFF!important;
  box-shadow:inset 2px 0 0 #6E9BFF!important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] [data-testid="stMarkdownContainer"] p{
  color:inherit!important;
  font-size:14px!important;
  font-weight:650!important;
}
[data-testid="stSidebar"] div[data-baseweb="select"]>div{
  background:#122C45!important;
  color:#F5F8FC!important;
  border:1px solid #29445F!important;
  border-radius:10px!important;
  min-height:43px!important;
  box-shadow:none!important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] svg{fill:#AFC0D1!important}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.08)!important}
[data-testid="stSidebar"] .stButton>button{
  width:100%!important;
  background:#132E48!important;
  color:#E7EEF7!important;
  border:1px solid #2A4661!important;
}
.sidebar-brand{
  padding:4px 4px 20px;
}
.sidebar-mark{
  display:flex;align-items:center;gap:10px;color:#fff;font-size:18px;font-weight:760;letter-spacing:-.02em
}
.sidebar-logo{
  width:30px;height:30px;border-radius:9px;background:linear-gradient(145deg,#4D83FF,#235FDB);
  display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;font-weight:900;
  box-shadow:0 5px 16px rgba(36,107,253,.32)
}
.sidebar-sub{font-size:10px;color:#7890A8;margin-top:7px;letter-spacing:.13em;text-transform:uppercase;font-weight:800}
.sidebar-section{font-size:10px;color:#6F879F;letter-spacing:.13em;text-transform:uppercase;font-weight:800;margin:20px 3px 7px}
.app-header{
  display:flex;align-items:flex-start;justify-content:space-between;gap:20px;
  margin:2px 0 22px;padding-bottom:20px;border-bottom:1px solid var(--line);
}
.app-eyebrow{font-size:11px;font-weight:850;color:#7D8A99;text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px}
.app-title{font-size:30px;line-height:1.15;font-weight:750;color:var(--ink);letter-spacing:-.035em}
.app-subtitle{font-size:15px;color:var(--muted);margin-top:7px;line-height:1.5;max-width:800px}
.header-meta{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap;padding-top:2px}
.meta-chip{
  display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border:1px solid var(--line);
  background:var(--surface);border-radius:999px;font-size:12px;color:#536172;font-weight:650;white-space:nowrap
}
.meta-chip.good{background:var(--green-soft);border-color:#C9ECDF;color:var(--green)}
.meta-chip.warn{background:var(--amber-soft);border-color:#F0D7A5;color:var(--amber)}
.entity-bar{
  display:flex;align-items:center;gap:14px;background:var(--surface);border:1px solid var(--line);
  border-radius:16px;padding:14px 16px;margin-bottom:22px;box-shadow:0 4px 16px rgba(19,33,53,.035)
}
.entity-avatar{
  width:44px;height:44px;border-radius:13px;background:linear-gradient(145deg,#122B44,#183B5B);
  color:#fff;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:850;letter-spacing:.02em
}
.entity-name{font-size:19px;font-weight:730;color:var(--ink);letter-spacing:-.02em}
.entity-meta{font-size:13px;color:var(--muted);margin-top:2px}
.section-heading{font-size:20px;font-weight:720;color:var(--ink);letter-spacing:-.025em;margin:26px 0 5px}
.section-copy{font-size:14px;color:var(--muted);line-height:1.5;margin-bottom:14px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:10px 0 24px}
.kpi-card{
  background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:17px 18px;
  box-shadow:0 4px 16px rgba(19,33,53,.035)
}
.kpi-label{font-size:11px;text-transform:uppercase;letter-spacing:.09em;font-weight:800;color:#8995A4}
.kpi-value{font-size:27px;font-weight:760;letter-spacing:-.035em;color:var(--ink);margin-top:5px}
.kpi-detail{font-size:12px;color:var(--muted);margin-top:4px}
.panel{
  background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:20px;
  box-shadow:0 5px 20px rgba(19,33,53,.035);margin-bottom:16px
}
.panel-title{font-size:17px;font-weight:720;color:var(--ink);letter-spacing:-.02em}
.panel-sub{font-size:13px;color:var(--muted);margin-top:3px}
.schedule-shell{
  background:var(--surface);border:1px solid var(--line);border-radius:18px;overflow:hidden;
  box-shadow:0 5px 20px rgba(19,33,53,.035);margin:12px 0 18px
}
.schedule-top{
  display:flex;align-items:center;justify-content:space-between;padding:15px 17px;border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,#fff,#FCFDFE)
}
.schedule-title{font-size:15px;font-weight:720;color:var(--ink)}
.schedule-hint{font-size:12px;color:var(--muted)}
.schedule-scroll{overflow-x:auto;padding:13px 14px 16px}
.week-strip{display:grid;grid-template-columns:repeat(14,minmax(112px,1fr));gap:8px;min-width:1640px}
.week-card{
  position:relative;min-height:123px;border:1px solid var(--line);border-radius:13px;padding:11px 10px;
  background:var(--surface);display:flex;flex-direction:column
}
.week-card.open{background:#FBFCFD;border-style:dashed}
.week-card.pending{border-color:#F0D7A5;background:#FFFDF8}
.week-card.locked:after{
  content:"LOCKED";position:absolute;right:8px;top:8px;font-size:8px;font-weight:900;letter-spacing:.08em;color:#8995A4
}
.week-num{font-size:10px;font-weight:850;color:#8995A4;letter-spacing:.07em;text-transform:uppercase}
.week-opp{font-size:14px;font-weight:730;color:var(--ink);line-height:1.22;margin-top:12px}
.week-site{font-size:10px;font-weight:850;letter-spacing:.07em;text-transform:uppercase;margin-top:auto;padding-top:10px;color:#687687}
.week-site.home{color:var(--green)}.week-site.away{color:#586A82}.week-site.neutral{color:#7C5CC7}
.week-open{font-size:13px;color:#9AA5B3;margin:auto 0;font-weight:600}
.tba-band{padding:0 14px 14px;display:flex;gap:8px;flex-wrap:wrap}
.tba-chip{border:1px solid var(--line);background:#FAFBFC;border-radius:10px;padding:8px 10px;font-size:12px;color:#566474}
.year-block{margin:15px 0 22px}
.year-head{display:flex;align-items:center;justify-content:space-between;margin:0 2px 8px}
.year-title{font-size:17px;font-weight:730;color:var(--ink)}
.task-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 22px}
.task-card{
  background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px;min-height:116px;
  transition:.15s ease;box-shadow:0 3px 14px rgba(19,33,53,.025)
}
.task-card.active{border-color:#9FBBFF;background:#F8FAFF;box-shadow:0 0 0 3px rgba(36,107,253,.07)}
.task-icon{width:30px;height:30px;border-radius:9px;background:var(--blue-soft);color:var(--blue);display:flex;align-items:center;justify-content:center;font-weight:850}
.task-title{font-size:15px;font-weight:720;color:var(--ink);margin-top:11px}.task-sub{font-size:12px;color:var(--muted);line-height:1.4;margin-top:3px}
.decision-card-premium{
  background:var(--surface);border:1px solid #D9E2ED;border-radius:20px;box-shadow:var(--shadow);
  overflow:hidden;margin:18px 0
}
.decision-head{display:flex;justify-content:space-between;gap:20px;padding:21px 22px 17px;border-bottom:1px solid var(--line)}
.decision-kicker{font-size:10px;font-weight:900;color:var(--blue);letter-spacing:.12em;text-transform:uppercase}
.decision-number{font-size:31px;font-weight:790;letter-spacing:-.045em;color:var(--ink);margin-top:3px}
.decision-proof{font-size:12px;color:var(--muted);margin-top:4px}
.decision-badge{align-self:flex-start;padding:7px 10px;border-radius:999px;background:var(--green-soft);color:var(--green);font-size:11px;font-weight:800}
.decision-moves{padding:3px 22px}
.decision-move{display:grid;grid-template-columns:32px 1fr auto;gap:12px;align-items:center;padding:15px 0;border-bottom:1px solid #EDF0F4}
.decision-move:last-child{border-bottom:none}
.step-num{width:28px;height:28px;border-radius:8px;background:#F1F4F8;color:#667587;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800}
.step-game{font-size:15px;font-weight:720;color:var(--ink)}.step-from{font-size:12px;color:var(--muted);margin-top:3px}.step-to{font-size:15px;font-weight:760;color:var(--blue);white-space:nowrap}
.decision-validation{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line);background:#FBFCFD}
.validation-col{padding:17px 22px}.validation-col:first-child{border-right:1px solid var(--line)}
.validation-title{font-size:10px;font-weight:900;letter-spacing:.11em;text-transform:uppercase;color:#8A96A6;margin-bottom:9px}
.validation-item{font-size:13px;color:#536172;margin:7px 0;display:flex;gap:8px}.validation-item strong{color:var(--ink)}
.dot-good{color:var(--green)}.dot-warn{color:var(--amber)}
.tx-premium{
  background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:18px 19px;
  margin:11px 0;box-shadow:0 4px 18px rgba(19,33,53,.035)
}
.tx-row{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}
.tx-title{font-size:17px;font-weight:730;color:var(--ink)}.tx-meta{font-size:12px;color:var(--muted);margin-top:4px}
.status-badge{padding:6px 9px;border-radius:999px;font-size:10px;font-weight:900;letter-spacing:.06em;text-transform:uppercase}
.status-badge.pending{background:var(--amber-soft);color:var(--amber)}.status-badge.completed{background:var(--green-soft);color:var(--green)}
.status-badge.rejected,.status-badge.superseded{background:var(--red-soft);color:var(--red)}
.approval-line{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:13px 0 4px}
.approval-pill{font-size:11px;font-weight:700;padding:6px 8px;border:1px solid var(--line);border-radius:999px;color:#617082;background:#FAFBFC}
.approval-pill.accepted{background:var(--green-soft);border-color:#C9ECDF;color:var(--green)}
.approval-pill.pending{background:var(--amber-soft);border-color:#F0D7A5;color:var(--amber)}
.approval-pill.rejected,.approval-pill.changes_requested{background:var(--red-soft);border-color:#F2CAC6;color:var(--red)}
.viewer-impact{background:#F7F9FC;border:1px solid var(--line);border-radius:12px;padding:11px 12px;margin-top:13px}
.viewer-label{font-size:9px;font-weight:900;letter-spacing:.11em;text-transform:uppercase;color:#8A96A6}
.viewer-change{font-size:13px;color:#435264;margin-top:6px}.viewer-change strong{color:var(--ink)}
.market-premium{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px 17px;margin:10px 0}
.market-row{display:flex;justify-content:space-between;gap:15px}.market-title{font-size:15px;font-weight:730;color:var(--ink)}
.market-meta{font-size:12px;color:var(--muted);line-height:1.45;margin-top:4px}.market-badge{font-size:10px;font-weight:850;padding:5px 8px;border-radius:999px;background:var(--blue-soft);color:var(--blue);white-space:nowrap}
.activity-list{background:var(--surface);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.activity-row{display:grid;grid-template-columns:82px 1fr;gap:12px;padding:13px 15px;border-bottom:1px solid #EDF0F4}
.activity-row:last-child{border-bottom:none}.activity-time{font-size:11px;color:#8A96A6;font-weight:700}.activity-text{font-size:13px;color:#4D5C6D}.activity-text strong{color:var(--ink)}
.empty-state{background:var(--surface);border:1px dashed var(--line-strong);border-radius:18px;padding:40px;text-align:center}
.empty-title{font-size:18px;font-weight:730;color:var(--ink)}.empty-copy{font-size:13px;color:var(--muted);margin-top:6px}
.parity-strip{display:grid;grid-template-columns:repeat(14,minmax(0,1fr));gap:7px;margin:12px 0}
.parity-week{border:1px solid var(--line);border-radius:10px;background:var(--surface);padding:10px 7px;text-align:center}
.parity-week.even{border-color:#D1E9DF;background:#F5FBF8}.parity-week.odd{border-color:#F0D7A5;background:#FFFAEF}
.parity-num{font-size:10px;font-weight:850;color:#8793A2}.parity-state{font-size:10px;font-weight:850;margin-top:5px}.parity-week.even .parity-state{color:var(--green)}.parity-week.odd .parity-state{color:var(--amber)}
.stButton>button{
  min-height:43px!important;border-radius:10px!important;font-size:13px!important;font-weight:720!important;
  padding:0 16px!important;box-shadow:none!important;border:1px solid var(--line-strong)!important
}
.stButton>button[kind="primary"],.stButton>button[data-testid="baseButton-primary"]{
  background:var(--blue)!important;border-color:var(--blue)!important;color:#fff!important;
  box-shadow:0 4px 12px rgba(36,107,253,.20)!important
}
.stButton>button:hover{border-color:#AEB9C6!important}
.stButton>button[kind="primary"]:hover{background:#1E5FE5!important;border-color:#1E5FE5!important}
.stSelectbox label,.stRadio label,.stTextInput label,.stSlider label,.stMultiSelect label,.stTextArea label,.stCheckbox label,.stNumberInput label,.stFileUploader label{
  font-size:12px!important;color:#687687!important;font-weight:760!important;letter-spacing:.01em!important
}
div[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea,.stMultiSelect [data-baseweb="select"]>div,.stNumberInput input{
  background:var(--surface)!important;border:1px solid var(--line-strong)!important;border-radius:10px!important;min-height:45px!important;
  color:var(--ink)!important;box-shadow:none!important;font-size:13px!important
}
[data-testid="stExpander"]{
  border:1px solid var(--line)!important;border-radius:14px!important;background:var(--surface)!important;box-shadow:none!important;margin:10px 0!important
}
[data-testid="stExpander"] summary{font-size:13px!important;font-weight:700!important;color:#4D5C6D!important}
[data-testid="stMarkdownContainer"] p{font-size:14px;line-height:1.55;color:#5D6B7C}
small,.stCaption,[data-testid="stCaptionContainer"]{font-size:11px!important;color:#8491A0!important}
[data-testid="stAlert"]{border-radius:12px!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
.desktop-only-note{font-size:11px;color:#8995A4}
@media(max-width:1100px){
  .block-container{padding-left:1.2rem!important;padding-right:1.2rem!important}
  .kpi-grid,.task-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .decision-validation{grid-template-columns:1fr}.validation-col:first-child{border-right:none;border-bottom:1px solid var(--line)}
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
    affected = sorted({school for m in sol.moves for school in (m.home_team, m.away_team)})

    a4_moved = 0
    for move in sol.moves:
        game = engine.store.games.get(move.game_id)
        if not game:
            continue
        home = engine.store.teams.get(game.home_team)
        away = engine.store.teams.get(game.away_team)
        if home and away and home.is_a4 and away.is_a4:
            a4_moved += 1

    proof_text = "Minimum number of changes proven" if proven else "Best path found within the solve window"
    move_rows = []
    for i, move in enumerate(sol.moves, 1):
        move_rows.append(
            f'<div class="decision-move"><div class="step-num">{i}</div><div>'
            f'<div class="step-game">{move.away_team} @ {move.home_team}</div>'
            f'<div class="step-from">Current · Week {display_week(move.from_week)}</div></div>'
            f'<div class="step-to">Week {display_week(move.to_week)} →</div></div>'
        )

    science = [
        '<div class="validation-item"><span class="dot-good">●</span><span><strong>Hard rules</strong> satisfied</span></div>',
        (
            '<div class="validation-item"><span class="dot-good">●</span><span><strong>Conference parity</strong> protected</span></div>'
            if not new_odd
            else f'<div class="validation-item"><span class="dot-warn">●</span><span><strong>{len(new_odd)} parity issue(s)</strong> created</span></div>'
        ),
        (
            '<div class="validation-item"><span class="dot-good">●</span><span><strong>Minimum changes</strong> proven</span></div>'
            if proven
            else '<div class="validation-item"><span class="dot-warn">●</span><span><strong>Best path</strong> within time budget</span></div>'
        ),
    ]
    human = [
        f'<div class="validation-item"><span class="dot-good">●</span><span><strong>{len(affected)} school{"s" if len(affected) != 1 else ""}</strong> affected</span></div>',
        (
            '<div class="validation-item"><span class="dot-good">●</span><span><strong>No A4 game</strong> moved</span></div>'
            if a4_moved == 0
            else f'<div class="validation-item"><span class="dot-warn">●</span><span><strong>{a4_moved} A4 game{"s" if a4_moved != 1 else ""}</strong> moved</span></div>'
        ),
        (
            '<div class="validation-item"><span class="dot-good">●</span><span><strong>Coach / AD context</strong> attached</span></div>'
            if str(md.get("coach_context", "") or "").strip()
            else '<div class="validation-item"><span style="color:#A9B3BF">●</span><span>No additional human context attached</span></div>'
        ),
    ]

    st.markdown(
        f'<div class="decision-card-premium">'
        f'<div class="decision-head"><div><div class="decision-kicker">{label}</div>'
        f'<div class="decision-number">{len(sol.moves)} change{"s" if len(sol.moves) != 1 else ""}</div>'
        f'<div class="decision-proof">{proof_text} · {data_status}</div></div>'
        f'<div class="decision-badge">FEASIBLE</div></div>'
        f'<div class="decision-moves">{"".join(move_rows)}</div>'
        f'<div class="decision-validation">'
        f'<div class="validation-col"><div class="validation-title">Feasibility · Science</div>{"".join(science)}</div>'
        f'<div class="validation-col"><div class="validation-title">Human considerations · Judgment</div>{"".join(human)}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    if sol.warnings:
        with st.expander("Validation notes", expanded=False):
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


def _team_initials(name: str) -> str:
    parts = [p for p in str(name).replace("&", " ").split() if p]
    if not parts:
        return "CF"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _schedule_row_view(row: pd.Series, team: str) -> Tuple[str, str]:
    neutral = bool(row.get("neutral", False))
    if str(row["home_team"]) == team:
        return str(row["away_team"]), "Neutral" if neutral else "Home"
    return str(row["home_team"]), "Neutral" if neutral else "Away"


def render_schedule_strip(
    games_df: pd.DataFrame,
    team: str,
    year: int,
    *,
    authoritative: bool = False,
    title: Optional[str] = None,
):
    subset = games_df[
        (pd.to_numeric(games_df["season"], errors="coerce") == int(year))
        & ((games_df["home_team"] == team) | (games_df["away_team"] == team))
    ].copy()

    dated: Dict[int, pd.Series] = {}
    tba_rows: List[pd.Series] = []
    for _, row in subset.iterrows():
        week = _safe_internal_week(row.get("week"))
        if week is None:
            tba_rows.append(row)
        elif week not in dated:
            dated[week] = row

    cards = []
    for week in range(14):
        row = dated.get(week)
        if row is None:
            open_text = "Open" if authoritative else "No known game"
            cards.append(
                f'<div class="week-card open"><div class="week-num">Week {display_week(week)}</div>'
                f'<div class="week-open">{open_text}</div><div class="week-site">—</div></div>'
            )
            continue

        opp, site = _schedule_row_view(row, team)
        moveability = str(row.get("moveability", "") or "").upper()
        status = str(row.get("game_status", "") or "").upper()
        css = ""
        if moveability == "LOCKED":
            css = " locked"
        elif status in {"PENDING", "HOLD", "CONCEPT"}:
            css = " pending"
        site_class = site.lower()
        cards.append(
            f'<div class="week-card{css}"><div class="week-num">Week {display_week(week)}</div>'
            f'<div class="week-opp">{opp}</div><div class="week-site {site_class}">{site}</div></div>'
        )

    tba_html = ""
    if tba_rows:
        chips = []
        for row in tba_rows:
            opp, site = _schedule_row_view(row, team)
            chips.append(f'<span class="tba-chip">TBA · {opp} · {site}</span>')
        tba_html = f'<div class="tba-band">{"".join(chips)}</div>'

    st.markdown(
        f'<div class="schedule-shell"><div class="schedule-top">'
        f'<div class="schedule-title">{title or f"{team} · {year}"}</div>'
        f'<div class="schedule-hint">Weeks 1–14</div></div>'
        f'<div class="schedule-scroll"><div class="week-strip">{"".join(cards)}</div></div>'
        f'{tba_html}</div>',
        unsafe_allow_html=True,
    )


def render_all_years(games_df: pd.DataFrame, team: str, *, authoritative: bool = False):
    subset = games_df[(games_df["home_team"] == team) | (games_df["away_team"] == team)].copy()
    if subset.empty:
        st.markdown(
            '<div class="empty-state"><div class="empty-title">No future commitments loaded</div>'
            '<div class="empty-copy">Future games will appear here as schedule data is added.</div></div>',
            unsafe_allow_html=True,
        )
        return
    years = sorted(int(y) for y in subset["season"].dropna().unique())
    for year in years:
        st.markdown(
            f'<div class="year-block"><div class="year-head"><div class="year-title">{year}</div></div></div>',
            unsafe_allow_html=True,
        )
        render_schedule_strip(games_df, team, year, authoritative=authoritative, title=f"{team} · {year}")


def _tx_items(db: WorkspaceDB) -> List[Dict[str, object]]:
    """Read transactions without allowing persistence issues to crash the app shell."""
    try:
        return db.list("transaction")
    except Exception:
        return []


def _tx_get(db: WorkspaceDB, tx_id: str) -> Optional[Dict[str, object]]:
    try:
        return db.get("transaction", tx_id)
    except Exception:
        return None


def _tx_save(db: WorkspaceDB, tx_id: str, tx: Dict[str, object]) -> None:
    body = dict(tx)
    body["transaction_id"] = tx_id
    body["updated_at"] = datetime.now().isoformat()
    db.put("transaction", tx_id, body)


def _tx_create(db: WorkspaceDB, payload: Dict[str, object]) -> str:
    body = dict(payload)
    stamp = datetime.now()
    raw = json.dumps(body, sort_keys=True, default=str)
    tx_id = str(
        body.get("transaction_id")
        or f"tx_{stamp.strftime('%Y%m%d%H%M%S')}_{abs(hash(raw)) % 100000:05d}"
    )
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
        db,
        tx_id,
        actor=school,
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
        db,
        tx_id,
        actor=conference,
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
    _tx_action(
        db,
        tx_id,
        actor=actor,
        action=f"STATUS_{str(status).upper()}",
        note=note,
    )
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
    st.markdown('<div class="section-heading">Affected schools</div>', unsafe_allow_html=True)
    cols = st.columns(min(3, max(1, len(impacts))))
    for idx, school in enumerate(sorted(impacts)):
        with cols[idx % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{school}**")
                for text in impacts[school]:
                    st.caption(text)


def transaction_card(tx: Dict[str, object], viewer: str):
    status = str(tx.get("status", "PENDING")).upper()
    css = status.lower()
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

    approvals = dict(tx.get("school_approvals", {}))
    approval_html = []
    for school, astatus in approvals.items():
        approval_html.append(
            f'<span class="approval-pill {str(astatus).lower()}">{school} · {str(astatus).title()}</span>'
        )

    viewer_html = ""
    if viewer in approvals:
        own = [m for m in moves if viewer in {m.get("home_team"), m.get("away_team")}]
        own_add = [g for g in additions if viewer in {g.get("home_team"), g.get("away_team")}]
        changes = []
        for m in own:
            changes.append(
                f'<div class="viewer-change"><strong>{m["away_team"]} @ {m["home_team"]}</strong> · '
                f'Week {display_week(int(m["from_week"]))} → Week {display_week(int(m["to_week"]))}</div>'
            )
        for g in own_add:
            changes.append(
                f'<div class="viewer-change"><strong>Add {g["away_team"]} @ {g["home_team"]}</strong> · '
                f'Week {display_week(int(g["week"]))}</div>'
            )
        if changes:
            viewer_html = (
                f'<div class="viewer-impact"><div class="viewer-label">Your impact · {viewer}</div>'
                f'{"".join(changes)}</div>'
            )

    st.markdown(
        f'<div class="tx-premium"><div class="tx-row"><div>'
        f'<div class="tx-title">{title}</div>'
        f'<div class="tx-meta">{tx.get("season")} · Proposed by {tx.get("proposer")} · '
        f'{count} schedule action{"s" if count != 1 else ""}</div></div>'
        f'<span class="status-badge {css}">{status}</span></div>'
        f'<div class="approval-line">{"".join(approval_html)}</div>{viewer_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Full coordinated plan", expanded=False):
        for i, m in enumerate(moves, 1):
            st.write(
                f"{i}. {m['away_team']} @ {m['home_team']}: "
                f"Week {display_week(int(m['from_week']))} → Week {display_week(int(m['to_week']))}"
            )
        for g in additions:
            st.write(f"Add {g['away_team']} @ {g['home_team']} — Week {display_week(int(g['week']))}")
        confs = list(tx.get("governing_conferences", []))
        if confs:
            st.caption("Conference guardrails: " + ", ".join(confs))


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
    badge = (
        "Explicit match" if md.get("explicit_need")
        else "Early market" if str(md.get("market_liquidity")) == "HIGH"
        else "Available"
    )
    st.markdown(
        f'<div class="market-premium"><div class="market-row"><div>'
        f'<div class="market-title">{away} @ {home} · Week {display_week(week)}</div>'
        f'<div class="market-meta">{match.explanation}</div></div>'
        f'<span class="market-badge">{badge}</span></div></div>',
        unsafe_allow_html=True,
    )
    if impact["resolved_issues"]:
        st.caption(f"Resolves {len(impact['resolved_issues'])} modeled parity issue(s).")
    if impact["new_issues"]:
        st.warning(f"Creates {len(impact['new_issues'])} modeled parity issue(s).")
    if blocked:
        st.caption("Blocked by current conference guardrails.")
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



def render_app_header(
    *,
    eyebrow: str,
    title: str,
    subtitle: str,
    season: int,
    data_status: str,
    status_class: str,
    pending_count: int = 0,
):
    pending_chip = (
        f'<span class="meta-chip">{pending_count} pending action{"s" if pending_count != 1 else ""}</span>'
        if pending_count else ""
    )
    st.markdown(
        f'<div class="app-header"><div><div class="app-eyebrow">{eyebrow}</div>'
        f'<div class="app-title">{title}</div><div class="app-subtitle">{subtitle}</div></div>'
        f'<div class="header-meta"><span class="meta-chip">{season}</span>'
        f'<span class="meta-chip {status_class}">{data_status}</span>{pending_chip}</div></div>',
        unsafe_allow_html=True,
    )


def render_entity_bar(name: str, meta: str):
    st.markdown(
        f'<div class="entity-bar"><div class="entity-avatar">{_team_initials(name)}</div>'
        f'<div><div class="entity-name">{name}</div><div class="entity-meta">{meta}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_kpis(cards: List[Tuple[str, object, str]]):
    items = []
    for label, value, detail in cards:
        items.append(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-detail">{detail}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def render_empty(title: str, copy: str):
    st.markdown(
        f'<div class="empty-state"><div class="empty-title">{title}</div>'
        f'<div class="empty-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def transaction_activity_rows(transactions: List[Dict[str, object]], limit: int = 8) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    events: List[Tuple[str, str]] = []
    for tx in transactions:
        for item in tx.get("history", []):
            at = str(item.get("at", ""))
            actor = str(item.get("actor", "System"))
            action = str(item.get("action", "")).replace("_", " ").title()
            events.append((at, f"<strong>{actor}</strong> · {action}"))
    events.sort(key=lambda x: x[0], reverse=True)
    for at, text in events[:limit]:
        try:
            dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
            stamp = dt.strftime("%b %d · %H:%M")
        except Exception:
            stamp = "Recent"
        rows.append((stamp, text))
    return rows


def render_activity_feed(transactions: List[Dict[str, object]], limit: int = 8):
    rows = transaction_activity_rows(transactions, limit=limit)
    if not rows:
        render_empty("No recent activity", "Scheduling proposals and approvals will appear here.")
        return
    html = "".join(
        f'<div class="activity-row"><div class="activity-time">{stamp}</div>'
        f'<div class="activity-text">{text}</div></div>'
        for stamp, text in rows
    )
    st.markdown(f'<div class="activity-list">{html}</div>', unsafe_allow_html=True)


def render_parity_strip(engine: AdvancedNonConferenceOptimizer, conference: str, season: int):
    parity_html = []
    parity = {
        week: engine.conference_parity(engine.store.copy_games(), int(season), week).get(conference, "")
        for week in range(14)
    }
    for week in range(14):
        state = str(parity.get(week, ""))
        is_odd = state.startswith("ODD")
        css = "odd" if is_odd else "even"
        label = "ODD" if is_odd else "EVEN"
        parity_html.append(
            f'<div class="parity-week {css}"><div class="parity-num">W{display_week(week)}</div>'
            f'<div class="parity-state">{label}</div></div>'
        )
    st.markdown(f'<div class="parity-strip">{"".join(parity_html)}</div>', unsafe_allow_html=True)


def render_task_cards(active: str, cards: List[Tuple[str, str, str]]):
    html = []
    for title, subtitle, icon in cards:
        css = " active" if active == title else ""
        html.append(
            f'<div class="task-card{css}"><div class="task-icon">{icon}</div>'
            f'<div class="task-title">{title}</div><div class="task-sub">{subtitle}</div></div>'
        )
    st.markdown(f'<div class="task-grid">{"".join(html)}</div>', unsafe_allow_html=True)


def school_transaction_sets(db: WorkspaceDB, school: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    relevant = [
        item["payload"] for item in _tx_items(db)
        if school in item["payload"].get("affected_schools", [])
        or school == item["payload"].get("proposer")
    ]
    pending = [
        tx for tx in relevant
        if str(tx.get("status", "")).upper() == "PENDING"
        and dict(tx.get("school_approvals", {})).get(school) == "PENDING"
    ]
    return relevant, pending


# ------------------------ workspace / data ----------------------------

db = get_db()

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><div class="sidebar-mark"><div class="sidebar-logo">S</div>'
        '<div>Schedule OS</div></div><div class="sidebar-sub">College Football Scheduling</div></div>',
        unsafe_allow_html=True,
    )
    perspective = st.radio(
        "Workspace",
        ["School", "Conference"],
        key="desktop_workspace",
    )
    st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)
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
    snapshot = db.get("data_snapshot", "latest")
    with st.sidebar:
        uploaded = st.file_uploader("Upload authoritative workbook", type=["xlsx", "xlsm", "csv"])
        st.download_button(
            "Download import template",
            data=make_template_bytes(),
            file_name="college_football_schedule_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
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
        if report.ok:
            db.put("data_snapshot", "latest", {
                "teams": all_teams_df.to_dict("records"),
                "games": all_games_df.to_dict("records"),
                "slots": slots_df.to_dict("records"),
                "needs": needs_df.to_dict("records"),
                "source_name": uploaded.name,
                "saved_at": datetime.now().isoformat(),
            })
    elif snapshot:
        all_teams_df = pd.DataFrame(snapshot.get("teams", []))
        all_games_df = pd.DataFrame(snapshot.get("games", []))
        slots_df = pd.DataFrame(snapshot.get("slots", []))
        needs_df = pd.DataFrame(snapshot.get("needs", []))
    else:
        render_app_header(
            eyebrow="Schedule OS",
            title="Connect authoritative schedule data",
            subtitle="Upload the conference or school workbook to turn the optimizer into an operational scheduling workspace.",
            season=2028,
            data_status="No data loaded",
            status_class="warn",
        )
        render_empty(
            "Authoritative schedule required",
            "Use the left navigation to upload a workbook, or switch to Public prototype to explore the product.",
        )
        st.stop()

elif data_mode == "Public prototype":
    with st.spinner("Loading schedule intelligence…"):
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

with st.sidebar:
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
season_games = sorted(list(store.games.values()), key=lambda g: (g.week, g.home_team, g.away_team))
all_team_names = sorted(store.teams.keys())
authoritative = data_mode == "Authoritative upload" and report_ok

if perspective == "School":
    school_names = all_team_names
    default_school = school_names.index("Georgia") if "Georgia" in school_names else 0
    with st.sidebar:
        st.markdown('<div class="sidebar-section">Institution</div>', unsafe_allow_html=True)
        acting_school = st.selectbox("School", school_names, index=default_school)
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        navigation = st.radio(
            "Navigation",
            ["Overview", "Schedule", "Solve", "Needs", "Proposals"],
            label_visibility="collapsed",
            key="school_nav",
        )

    relevant_tx, pending_tx = school_transaction_sets(db, acting_school)
    team_obj = store.teams.get(acting_school)
    team_meta = (
        f"{team_obj.conference} · {team_obj.subdivision} · {season}"
        if team_obj else str(season)
    )
    render_app_header(
        eyebrow="School workspace",
        title=navigation,
        subtitle={
            "Overview": "Everything that requires your attention, in one place.",
            "Schedule": "One authoritative view of current and future non-conference commitments.",
            "Solve": "Define the outcome. The optimizer coordinates the affected schedules.",
            "Needs": "Publish what you need and surface compatible scheduling inventory.",
            "Proposals": "Review, negotiate and approve coordinated schedule changes.",
        }[navigation],
        season=int(season),
        data_status=status_text,
        status_class=status_class,
        pending_count=len(pending_tx),
    )
    render_entity_bar(acting_school, team_meta)

    if navigation == "Overview":
        current_commitments = len([g for g in store.games.values() if g.involves(acting_school)])
        current_needs = db_need_records(db, season=int(season), school=acting_school)
        rules = persistent_profile_rules(db, acting_school)
        render_kpis([
            ("Known commitments", current_commitments, f"{season} modeled schedule"),
            ("Open needs", len(current_needs), "buy game / A4 market"),
            ("Needs your decision", len(pending_tx), "pending proposals"),
            ("Scheduling rules", len(rules), "institutional profile"),
        ])

        if pending_tx:
            st.markdown('<div class="section-heading">Needs your attention</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">These proposals are waiting on your decision.</div>', unsafe_allow_html=True)
            for tx in pending_tx[:3]:
                transaction_card(tx, acting_school)

        st.markdown('<div class="section-heading">Active season</div>', unsafe_allow_html=True)
        render_schedule_strip(
            all_games_df, acting_school, int(season),
            authoritative=authoritative,
            title=f"{acting_school} · {season}",
        )

        left, right = st.columns([1.3, .7])
        with left:
            st.markdown('<div class="section-heading">Recent activity</div>', unsafe_allow_html=True)
            render_activity_feed(relevant_tx, limit=7)
        with right:
            st.markdown('<div class="section-heading">Institutional profile</div>', unsafe_allow_html=True)
            with st.container(border=True):
                if rules:
                    for rule in rules[:6]:
                        st.caption("• " + rule_summary(rule))
                else:
                    st.caption("No persistent scheduling rules saved yet.")

    elif navigation == "Schedule":
        st.markdown('<div class="section-heading">Active schedule</div>', unsafe_allow_html=True)
        render_schedule_strip(
            all_games_df, acting_school, int(season),
            authoritative=authoritative,
            title=f"{acting_school} · {season}",
        )
        st.markdown('<div class="section-heading">Future commitments</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">Every loaded year on one page. Scroll each season horizontally for the complete 14-week view.</div>', unsafe_allow_html=True)
        render_all_years(all_games_df, acting_school, authoritative=authoritative)

    elif navigation == "Solve":
        acting_conf = team_obj.conference if team_obj else ""
        preserve_parity = bool(
            conference_policy(db, acting_conf).get("enforce_no_new_parity", True)
        ) if acting_conf else False

        if "desktop_solve_mode" not in st.session_state:
            st.session_state["desktop_solve_mode"] = "Move a game"
        active_mode = st.session_state["desktop_solve_mode"]

        task_cards = [
            ("Move a game", "Rework an existing commitment", "↔"),
            ("Open a week", "Find the lowest-disruption path", "□"),
            ("Find a buy game", "Match compatible guarantee inventory", "$"),
            ("Find an A4 opponent", "Find compatible A4 inventory", "A4"),
        ]
        render_task_cards(active_mode, task_cards)

        bcols = st.columns(4)
        for i, (title, _, _) in enumerate(task_cards):
            with bcols[i]:
                if st.button(
                    "Select" if active_mode != title else "Selected",
                    key=f"select_task_{title}",
                    use_container_width=True,
                    disabled=active_mode == title,
                ):
                    st.session_state["desktop_solve_mode"] = title
                    st.rerun()

        st.markdown('<div class="section-heading">Configure outcome</div>', unsafe_allow_html=True)

        if active_mode == "Move a game":
            school_games = [
                g for g in season_games
                if g.involves(acting_school) and str(g.game_type).upper() != "CONFERENCE"
            ]
            if not school_games:
                render_empty("No movable games", "No dated non-conference commitments are available in the active season.")
            else:
                with st.container(border=True):
                    c1, c2 = st.columns([1.5, .7])
                    labels = {game_label(g): g for g in school_games}
                    with c1:
                        selected_game = labels[st.selectbox(
                            "Game", list(labels.keys()), key=f"premium_move_game_{acting_school}"
                        )]
                    with c2:
                        target_display = st.selectbox(
                            "Move to", list(range(1, 15)), index=int(selected_game.week),
                            key=f"premium_move_week_{acting_school}",
                        )
                    target_week = internal_week(target_display)

                    with st.expander("Constraints & preferences", expanded=False):
                        rules, protected_ids, avoid_ids, context = constraint_builder(
                            db,
                            prefix=f"premium_move_{season}_{acting_school}_{selected_game.game_id}",
                            primary_team=acting_school,
                            teams=all_team_names,
                            games=[g for g in season_games if g.game_id != selected_game.game_id],
                        )

                    if st.button("Find best path", type="primary", key=f"premium_solve_move_{acting_school}"):
                        run_store = store_with_locked(store, protected_ids)
                        run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=6.0)
                        intent = build_move_intent(
                            selected_game, target_week, rules, avoid_ids, context, preserve_parity
                        )
                        with st.spinner("Evaluating coordinated repair paths…"):
                            results = run_engine.solve_move_game(intent)
                        st.session_state[f"premium_result_{acting_school}"] = {
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

                state = st.session_state.get(f"premium_result_{acting_school}")
                if state:
                    sol = state.get("result")
                    run_store = store_with_locked(store, set(state.get("protected") or set()))
                    run_engine = AdvancedNonConferenceOptimizer(run_store, time_limit_seconds=6.0)
                    if sol:
                        render_result(run_engine, sol, season=int(season), data_status=status_text)
                        render_school_impacts(sol)
                        action_cols = st.columns([.7, .3])
                        with action_cols[0]:
                            if st.button(
                                "Send coordinated proposal",
                                type="primary",
                                use_container_width=True,
                                key=f"premium_send_move_{acting_school}",
                            ):
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
                                    db, tx_id, actor=acting_school,
                                    action="SENT_TO_AFFECTED_SCHOOLS",
                                    note="Unanimous school approval requested.",
                                )
                                st.success("Proposal sent.")
                        with action_cols[1]:
                            if st.button(
                                "Other options",
                                use_container_width=True,
                                key=f"premium_alts_{acting_school}",
                            ):
                                with st.spinner("Testing alternate tradeoffs…"):
                                    st.session_state[f"premium_alts_results_{acting_school}"] = run_engine.solve_move_game_alternatives(state["intent"], sol)
                        for alt in st.session_state.get(f"premium_alts_results_{acting_school}", []):
                            render_result(
                                run_engine, alt, season=int(season), data_status=status_text,
                                label=str((alt.metadata or {}).get("strategy_label", "ALTERNATIVE")).upper()
                            )
                    else:
                        st.error("No feasible path satisfies the current school and conference rules.")

        elif active_mode == "Open a week":
            with st.container(border=True):
                open_display = st.selectbox(
                    "Week that must be open", list(range(1, 15)), key=f"premium_open_week_{acting_school}"
                )
                open_week = internal_week(open_display)
                occupied = store.game_for_team_week(
                    store.copy_games(), acting_school, int(season), open_week
                )
                if occupied is None:
                    st.success(f"{acting_school} is already open in Week {open_display}.")
                elif str(occupied.game_type).upper() == "CONFERENCE":
                    st.error("That week contains a conference game.")
                else:
                    st.caption(f"Current conflict · {occupied.away_team} @ {occupied.home_team}")
                    rules = rules_for_schools(db, [acting_school])
                    if st.button("Find easiest path", type="primary", key=f"premium_open_solve_{acting_school}"):
                        with st.spinner("Searching the lowest-disruption relocation chain…"):
                            st.session_state[f"premium_open_result_{acting_school}"] = easiest_relocation(
                                store, occupied, rules=rules, preserve_parity=preserve_parity
                            )
            answer = st.session_state.get(f"premium_open_result_{acting_school}")
            if answer:
                destination, sol = answer
                render_result(engine, sol, season=int(season), data_status=status_text)
                render_school_impacts(sol)
                if st.button("Send coordinated proposal", type="primary", key=f"premium_send_open_{acting_school}"):
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
                    _tx_create(db, payload)
                    st.success("Proposal sent.")

        else:
            match_type = "BUY_GAME" if active_mode == "Find a buy game" else "A4"
            if match_type == "A4" and team_obj and not team_obj.is_a4:
                st.warning("This school is not classified as A4 in the current data.")
            else:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1,1,1])
                    with c1:
                        week_choice = st.selectbox(
                            "Week", ["Best available"] + list(range(1, 15)),
                            key=f"premium_market_week_{acting_school}_{match_type}",
                        )
                    target_week = None if week_choice == "Best available" else internal_week(int(week_choice))
                    with c2:
                        location = st.selectbox(
                            "Site preference",
                            ["HOME", "ANY", "AWAY"] if match_type == "A4" else ["HOME", "ANY"],
                            key=f"premium_market_location_{acting_school}_{match_type}",
                        )
                    max_guarantee = None
                    with c3:
                        if match_type == "BUY_GAME":
                            guarantee_value = st.number_input(
                                "Maximum guarantee", min_value=0, value=0, step=50000,
                                key=f"premium_market_guarantee_{acting_school}",
                            )
                            max_guarantee = None if guarantee_value == 0 else int(guarantee_value)
                        else:
                            st.caption("Weeks 1–4 receive a soft market preference.")
                    if st.button("Find best matches", type="primary", key=f"premium_find_market_{acting_school}_{match_type}"):
                        intent = Intent(
                            action="FIND_BUY_GAME" if match_type == "BUY_GAME" else "FIND_A4_GAME",
                            season=int(season), target_week=target_week, team_a=acting_school,
                            location=location, max_guarantee=max_guarantee,
                        )
                        st.session_state[f"premium_market_results_{acting_school}_{match_type}"] = engine.solve(intent)

                results = st.session_state.get(f"premium_market_results_{acting_school}_{match_type}", [])
                if results:
                    st.markdown('<div class="section-heading">Best matches</div>', unsafe_allow_html=True)
                    for idx, match in enumerate(results[:8]):
                        blocked, _ = market_result_card(
                            match, store=store, season=int(season), policies=policies
                        )
                        if not blocked and st.button(
                            "Propose matchup",
                            key=f"premium_propose_match_{acting_school}_{match_type}_{idx}",
                        ):
                            payload = transaction_from_match(
                                match=match, proposer=acting_school, season=int(season),
                                data_status=status_text, store=store,
                                conference_policies=policies, need_type=match_type,
                            )
                            _tx_create(db, payload)
                            st.success("Proposal sent.")

    elif navigation == "Needs":
        open_needs = db_need_records(db, school=acting_school)
        left, right = st.columns([1.25, .75])
        with left:
            st.markdown('<div class="section-heading">Open needs</div>', unsafe_allow_html=True)
            if open_needs:
                rows = [{
                    "Season": n["season"],
                    "Week": display_week(int(n["week"])),
                    "Need": n["need_type"],
                    "Site": n["location"],
                    "Min guarantee": n.get("min_guarantee"),
                    "Max guarantee": n.get("max_guarantee"),
                } for n in open_needs]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                render_empty("No open needs", "Publish a buy-game or A4 need when you are ready.")
        with right:
            st.markdown('<div class="section-heading">Publish need</div>', unsafe_allow_html=True)
            with st.container(border=True):
                need_type_label = st.selectbox(
                    "Need", ["FCS buy game", "A4 opponent"], key=f"premium_need_type_{acting_school}"
                )
                need_type = "FCS_BUY" if need_type_label == "FCS buy game" else "A4"
                need_year = st.selectbox(
                    "Season", available_years, index=available_years.index(season),
                    key=f"premium_need_year_{acting_school}",
                )
                need_weeks = st.multiselect(
                    "Acceptable weeks", list(range(1, 15)), default=[1,2,3,4],
                    key=f"premium_need_weeks_{acting_school}",
                )
                need_location = st.selectbox(
                    "Location", ["HOME","ANY","AWAY"], key=f"premium_need_location_{acting_school}"
                )
                min_g = max_g = None
                if need_type == "FCS_BUY":
                    min_value = st.number_input(
                        "Minimum guarantee", min_value=0, value=0, step=50000,
                        key=f"premium_need_min_{acting_school}",
                    )
                    max_value = st.number_input(
                        "Maximum guarantee", min_value=0, value=0, step=50000,
                        key=f"premium_need_max_{acting_school}",
                    )
                    min_g = None if min_value == 0 else int(min_value)
                    max_g = None if max_value == 0 else int(max_value)
                notes = st.text_input("Notes", key=f"premium_need_notes_{acting_school}")
                if st.button("Publish need", type="primary", use_container_width=True):
                    if not need_weeks:
                        st.warning("Choose at least one week.")
                    else:
                        save_school_need(
                            db, team=acting_school, season=int(need_year),
                            display_weeks=need_weeks, need_type=need_type,
                            location=need_location, min_guarantee=min_g,
                            max_guarantee=max_g, notes=notes,
                        )
                        st.success("Need published.")
                        st.rerun()

    elif navigation == "Proposals":
        if not relevant_tx:
            render_empty("You're all caught up", "No scheduling proposals involve this school.")
        else:
            pending = [tx for tx in relevant_tx if str(tx.get("status","")).upper() == "PENDING"]
            history = [tx for tx in relevant_tx if str(tx.get("status","")).upper() != "PENDING"]
            if pending:
                st.markdown('<div class="section-heading">Active proposals</div>', unsafe_allow_html=True)
            for tx in pending:
                tx_id = str(tx.get("transaction_id"))
                transaction_card(tx, acting_school)
                approvals = dict(tx.get("school_approvals", {}))
                my_status = approvals.get(acting_school)

                if my_status == "PENDING":
                    a1, a2, a3 = st.columns([1,1,1.2])
                    with a1:
                        if st.button("Accept", type="primary", use_container_width=True, key=f"premium_accept_{tx_id}_{acting_school}"):
                            updated = _tx_school_approval(db, tx_id, acting_school, "ACCEPTED")
                            if updated and updated.get("status") == "COMPLETED":
                                st.success("Unanimous approval complete.")
                            st.rerun()
                    with a2:
                        if st.button("Reject", use_container_width=True, key=f"premium_reject_toggle_{tx_id}_{acting_school}"):
                            st.session_state[f"premium_show_reject_{tx_id}_{acting_school}"] = True
                    with a3:
                        if st.button("Suggest another week", use_container_width=True, key=f"premium_counter_toggle_{tx_id}_{acting_school}"):
                            st.session_state[f"premium_show_counter_{tx_id}_{acting_school}"] = True

                    if st.session_state.get(f"premium_show_reject_{tx_id}_{acting_school}"):
                        with st.container(border=True):
                            reason = st.selectbox(
                                "Why doesn't it work?",
                                ["Coach preference","Contract issue","Travel issue","Game cannot move","Financial issue","Other"],
                                key=f"premium_reject_reason_{tx_id}_{acting_school}",
                            )
                            note = st.text_input("Detail", key=f"premium_reject_note_{tx_id}_{acting_school}")
                            if st.button("Confirm rejection", key=f"premium_confirm_reject_{tx_id}_{acting_school}"):
                                _tx_school_approval(db, tx_id, acting_school, "REJECTED", f"{reason}: {note}")
                                db.add_feedback(
                                    season=int(tx.get("season")), team=acting_school,
                                    game_id="", reason=reason, notes=note, payload=tx,
                                )
                                st.rerun()

                    if st.session_state.get(f"premium_show_counter_{tx_id}_{acting_school}"):
                        with st.container(border=True):
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
                                options[f"{g['away_team']} @ {g['home_team']} (new)"] = str(g["game_id"])
                            if options:
                                chosen = st.selectbox("Game", list(options.keys()), key=f"premium_counter_game_{tx_id}")
                                alt_display = st.selectbox("Suggested week", list(range(1,15)), key=f"premium_counter_week_{tx_id}")
                                alt_note = st.text_input("Why?", key=f"premium_counter_note_{tx_id}")
                                if st.button("Test counterproposal", type="primary", key=f"premium_counter_send_{tx_id}"):
                                    new_id, message = try_counterproposal(
                                        tx=tx, school=acting_school, game_id=options[chosen],
                                        requested_week=internal_week(alt_display), store=store,
                                        db=db, policies=policies, data_status=status_text,
                                    )
                                    if new_id:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)

                if str(tx.get("status","")).upper() == "COMPLETED":
                    with st.expander("Approval confirmation", expanded=False):
                        text = confirmation_text(tx)
                        st.text_area("Reply-all equivalent", value=text, height=180, disabled=True, key=f"premium_confirm_{tx_id}")

            if history:
                st.markdown('<div class="section-heading">History</div>', unsafe_allow_html=True)
                for tx in history[:20]:
                    transaction_card(tx, acting_school)

else:
    confs = engine.store.fbs_conferences()
    default_conf = confs.index("SEC") if "SEC" in confs else 0
    with st.sidebar:
        st.markdown('<div class="sidebar-section">Conference</div>', unsafe_allow_html=True)
        acting_conf = st.selectbox("Conference", confs, index=default_conf)
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        navigation = st.radio(
            "Navigation",
            ["Overview", "Solve", "Transactions", "Schools", "Rules"],
            label_visibility="collapsed",
            key="conference_nav",
        )

    members = sorted(t.name for t in store.conference_members(acting_conf))
    relevant_tx = [
        item["payload"] for item in _tx_items(db)
        if acting_conf in item["payload"].get("governing_conferences", [])
        or item["payload"].get("proposer") == acting_conf
    ]
    pending_conf_tx = [tx for tx in relevant_tx if str(tx.get("status","")).upper() == "PENDING"]
    render_app_header(
        eyebrow="Conference workspace",
        title=navigation,
        subtitle={
            "Overview":"Conference health, market activity and transactions requiring attention.",
            "Solve":"Coordinate the minimum schedule changes required to reach a conference outcome.",
            "Transactions":"Monitor school approvals, negotiations and completed transactions.",
            "Schools":"Review member schedules without opening five separate documents.",
            "Rules":"Set automated governance so ordinary transactions do not require brokerage.",
        }[navigation],
        season=int(season),
        data_status=status_text,
        status_class=status_class,
        pending_count=len(pending_conf_tx),
    )
    render_entity_bar(acting_conf, f"{len(members)} members · {season}")

    if navigation == "Overview":
        odd_weeks = []
        for week in range(14):
            state = engine.conference_parity(store.copy_games(), int(season), week).get(acting_conf, "")
            if str(state).startswith("ODD"):
                odd_weeks.append(week)
        conf_needs = [
            n for n in db_need_records(db, season=int(season))
            if n.get("team") in set(members)
        ]
        completed = len([tx for tx in relevant_tx if str(tx.get("status","")).upper() == "COMPLETED"])
        render_kpis([
            ("Member schools", len(members), "conference workspace"),
            ("Odd modeled weeks", len(odd_weeks), "conference availability"),
            ("Open school needs", len(conf_needs), "buy game / A4 market"),
            ("Pending transactions", len(pending_conf_tx), f"{completed} completed"),
        ])
        st.markdown('<div class="section-heading">Conference health</div>', unsafe_allow_html=True)
        render_parity_strip(engine, acting_conf, int(season))
        left, right = st.columns([1.2,.8])
        with left:
            st.markdown('<div class="section-heading">Recent activity</div>', unsafe_allow_html=True)
            render_activity_feed(relevant_tx, limit=8)
        with right:
            st.markdown('<div class="section-heading">Governance</div>', unsafe_allow_html=True)
            policy = conference_policy(db, acting_conf)
            with st.container(border=True):
                st.caption("Parity protection")
                st.markdown("**On**" if policy.get("enforce_no_new_parity", True) else "**Off**")
                st.caption("Manual conference approval")
                st.markdown("**Required**" if policy.get("require_manual_approval", False) else "**Not required**")

    elif navigation == "Solve":
        st.markdown('<div class="section-heading">Make selected weeks even</div>', unsafe_allow_html=True)
        with st.container(border=True):
            selected_display_weeks = st.multiselect(
                "Weeks that must be even", list(range(1,15)), default=[1,2,3]
            )
            selected_weeks = [internal_week(w) for w in selected_display_weeks]
            member_set = set(members)
            conf_games = [
                g for g in season_games
                if g.home_team in member_set or g.away_team in member_set
            ]
            with st.expander("Constraints & preferences", expanded=False):
                rules, protected_ids, avoid_ids, context = constraint_builder(
                    db,
                    prefix=f"premium_confsolve_{season}_{acting_conf}",
                    primary_team=members[0] if members else all_team_names[0],
                    teams=members if members else all_team_names,
                    games=conf_games,
                )
            if st.button("Find best coordinated plan", type="primary"):
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
                with st.spinner("Evaluating coordinated schedule paths…"):
                    plans = run_engine.optimize_national(intent)
                st.session_state[f"premium_conf_plan_{acting_conf}"] = {
                    "result": plans[0] if plans else None,
                    "store": run_store,
                    "rules": intent.rules,
                    "context": context,
                    "weeks": selected_weeks,
                }

        state = st.session_state.get(f"premium_conf_plan_{acting_conf}")
        if state:
            plan = state.get("result")
            if plan and not bool((plan.metadata or {}).get("infeasible")):
                run_engine = AdvancedNonConferenceOptimizer(state["store"], time_limit_seconds=12.0)
                render_result(
                    run_engine, plan, season=int(season),
                    data_status=status_text, label="BEST COORDINATED PLAN"
                )
                render_school_impacts(plan)
                if st.button("Send to all affected schools", type="primary", key=f"premium_send_conf_{acting_conf}"):
                    affected = sorted({s for m in plan.moves for s in (m.home_team, m.away_team)})
                    objective = {
                        "type":"CONFERENCE_EVEN",
                        "conference":acting_conf,
                        "target_weeks":list(state["weeks"]),
                    }
                    payload = transaction_from_solution(
                        sol=plan, proposer=acting_conf, season=int(season),
                        data_status=status_text, store=state["store"],
                        conference_policies=policies, context=str(state.get("context","")),
                        objective=objective, rules=list(state.get("rules") or []),
                    )
                    tx_id = _tx_create(db, payload)
                    _tx_action(
                        db, tx_id, actor=acting_conf,
                        action="SENT_TO_ALL_AFFECTED_SCHOOLS",
                        note=f"{len(affected)} schools asked for unanimous approval.",
                    )
                    st.success(f"Proposal sent to {len(affected)} schools.")
            elif state:
                st.error("No plan satisfies the selected conference outcome and hard constraints.")

    elif navigation == "Transactions":
        if not relevant_tx:
            render_empty("No transactions", "Coordinated proposals involving this conference will appear here.")
        else:
            active = [tx for tx in relevant_tx if str(tx.get("status","")).upper() == "PENDING"]
            finished = [tx for tx in relevant_tx if str(tx.get("status","")).upper() != "PENDING"]
            if active:
                st.markdown('<div class="section-heading">Active</div>', unsafe_allow_html=True)
                for tx in active:
                    transaction_card(tx, acting_conf)
            if finished:
                st.markdown('<div class="section-heading">Completed & history</div>', unsafe_allow_html=True)
                for tx in finished[:25]:
                    transaction_card(tx, acting_conf)

    elif navigation == "Schools":
        st.markdown('<div class="section-heading">Member schedule intelligence</div>', unsafe_allow_html=True)
        selected_member = st.selectbox("Inspect school", members, key=f"premium_member_{acting_conf}")
        render_schedule_strip(
            all_games_df, selected_member, int(season),
            authoritative=authoritative,
            title=f"{selected_member} · {season}",
        )
        rows = []
        for team in members:
            games = sorted([g for g in store.games.values() if g.involves(team)], key=lambda g:g.week)
            row = {"School":team}
            for w in range(14):
                game = next((g for g in games if g.week == w), None)
                if game:
                    opp = game.away_team if game.home_team == team else game.home_team
                    site = game.site_for(team)
                    row[f"W{display_week(w)}"] = f"{'vs' if site=='HOME' else '@' if site=='AWAY' else 'N'} {opp}"
                else:
                    row[f"W{display_week(w)}"] = ""
            rows.append(row)
        st.markdown('<div class="section-heading">Conference matrix</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=600)

    elif navigation == "Rules":
        policy = conference_policy(db, acting_conf)
        left, right = st.columns([.9,1.1])
        with left:
            st.markdown('<div class="section-heading">Governance</div>', unsafe_allow_html=True)
            with st.container(border=True):
                enforce_parity = st.checkbox(
                    "Prevent new conference parity issues",
                    value=bool(policy.get("enforce_no_new_parity", True)),
                )
                require_manual = st.checkbox(
                    "Require conference approval after unanimous school approval",
                    value=bool(policy.get("require_manual_approval", False)),
                )
                if st.button("Save governance", type="primary", use_container_width=True):
                    db.put("conference_policy", acting_conf, {
                        "conference":acting_conf,
                        "enforce_no_new_parity":bool(enforce_parity),
                        "require_manual_approval":bool(require_manual),
                        "auto_complete_after_school_approvals":True,
                    })
                    st.success("Governance saved.")
        with right:
            st.markdown('<div class="section-heading">School profiles</div>', unsafe_allow_html=True)
            profile_rows = []
            for school in members:
                school_rules = persistent_profile_rules(db, school)
                profile_rows.append({
                    "School":school,
                    "Saved rules":len(school_rules),
                    "Profile":"Configured" if school_rules else "Not configured",
                })
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

st.markdown(
    '<div class="desktop-only-note" style="margin-top:28px;padding-top:14px;border-top:1px solid #E5EAF0">'
    'Schedule OS · Desktop pilot · Use authoritative schedule data and durable Postgres before operational deployment.'
    '</div>',
    unsafe_allow_html=True,
)
