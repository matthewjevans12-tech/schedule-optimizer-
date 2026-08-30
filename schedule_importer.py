from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd


GAME_COLUMNS = [
    "game_id", "season", "week", "date", "home_team", "away_team",
    "neutral", "campus_home_team", "game_status", "moveability",
    "game_type", "guarantee", "contract_link", "earliest_week",
    "latest_week", "source", "last_verified", "confidence", "notes",
]
TEAM_COLUMNS = ["name", "subdivision", "conference", "is_a4", "parity_managed"]
SLOT_COLUMNS = ["team", "season", "week", "status", "location"]
NEED_COLUMNS = ["team", "season", "week", "need_type", "location", "min_guarantee", "max_guarantee", "status", "notes"]

TRUE = {"1", "true", "yes", "y", "x"}
FALSE = {"0", "false", "no", "n", ""}


@dataclass
class ImportReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return default
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    return default


def _clean_text(v, default="") -> str:
    if pd.isna(v):
        return default
    return str(v).strip()


def _normalize_week(
    series: pd.Series,
    report: ImportReport,
    sheet: str,
    *,
    allow_blank: bool = False,
) -> pd.Series:
    """Convert user-facing Weeks 1-14 to nullable internal Weeks 0-13.

    Future schedule commitments are allowed to be Week TBA in the Games sheet.
    TBA games remain visible in schedule intelligence but are excluded from
    optimization until a real week is supplied.
    """
    original = series.copy()
    blank = original.isna() | original.astype(str).str.strip().isin({"", "nan", "None", "TBA", "tba"})
    raw = pd.to_numeric(original.where(~blank), errors="coerce")
    nonnumeric = (~blank) & raw.isna()
    if nonnumeric.any():
        report.errors.append(f"{sheet}: {int(nonnumeric.sum())} row(s) have a non-numeric week.")
    if blank.any() and not allow_blank:
        report.errors.append(f"{sheet}: {int(blank.sum())} row(s) are missing a week.")
    outside = raw.notna() & ((raw < 1) | (raw > 14))
    if outside.any():
        report.errors.append(f"{sheet}: week values must be 1-14.")
    converted = raw - 1
    return converted.astype("Int64")


def load_schedule_upload(
    raw_bytes: bytes,
    filename: str,
    public_teams: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ImportReport]:
    report = ImportReport()
    suffix = filename.lower().split(".")[-1]

    if suffix in {"xlsx", "xlsm"}:
        book = pd.ExcelFile(io.BytesIO(raw_bytes))
        names = {s.lower(): s for s in book.sheet_names}
        if "games" not in names:
            report.errors.append("Excel file must contain a Games sheet.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), report
        games = pd.read_excel(book, sheet_name=names["games"])
        teams = pd.read_excel(book, sheet_name=names["teams"]) if "teams" in names else pd.DataFrame()
        slots = pd.read_excel(book, sheet_name=names["slots"]) if "slots" in names else pd.DataFrame()
        needs = pd.read_excel(book, sheet_name=names["needs"]) if "needs" in names else pd.DataFrame()
    elif suffix == "csv":
        games = pd.read_csv(io.BytesIO(raw_bytes))
        teams = pd.DataFrame()
        slots = pd.DataFrame()
        needs = pd.DataFrame()
        report.warnings.append(
            "CSV contains Games only. Team metadata will be matched to public metadata where possible; Excel is preferred."
        )
    else:
        report.errors.append("Upload an .xlsx or .csv file.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), report

    games.columns = [str(c).strip().lower() for c in games.columns]
    required_games = {
        "season", "week", "home_team", "away_team",
        "neutral", "game_status", "moveability", "game_type",
    }
    missing = required_games - set(games.columns)
    if missing:
        report.errors.append("Games is missing required column(s): " + ", ".join(sorted(missing)))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), report

    for col in GAME_COLUMNS:
        if col not in games.columns:
            games[col] = ""

    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    if games["season"].isna().any():
        report.errors.append("Games: every row needs a numeric season.")
    games["season"] = games["season"].fillna(0).astype(int)
    games["week"] = _normalize_week(games["week"], report, "Games", allow_blank=True)
    games["home_team"] = games["home_team"].map(_clean_text)
    games["away_team"] = games["away_team"].map(_clean_text)
    games["neutral"] = games["neutral"].map(_bool)
    games["campus_home_team"] = games["campus_home_team"].map(_clean_text)
    games["game_status"] = games["game_status"].map(lambda v: _clean_text(v, "CONTRACTED").upper())
    games["moveability"] = games["moveability"].map(lambda v: _clean_text(v, "UNKNOWN").upper())
    games["game_type"] = games["game_type"].map(lambda v: _clean_text(v, "NONCONFERENCE").upper())
    games["confidence"] = games["confidence"].map(lambda v: _clean_text(v, "AUTHORITATIVE").upper())
    games["source"] = games["source"].map(lambda v: _clean_text(v, "Administrator import"))
    games["game_id"] = games["game_id"].map(_clean_text)
    games["notes"] = games["notes"].map(_clean_text)
    games["contract_link"] = games["contract_link"].map(_clean_text)
    games["last_verified"] = games["last_verified"].map(_clean_text)
    games["date"] = games["date"].map(_clean_text)

    for col in ["earliest_week", "latest_week"]:
        converted = []
        for value in games[col]:
            if pd.isna(value) or str(value).strip() == "":
                converted.append("")
                continue
            try:
                display = int(float(value))
                if display < 1 or display > 14:
                    report.errors.append(f"Games: {col} values must be 1-14 when supplied.")
                    converted.append("")
                else:
                    converted.append(display - 1)
            except Exception:
                report.errors.append(f"Games: {col} contains a non-numeric value.")
                converted.append("")
        games[col] = converted

    unknown_move = games["moveability"] == "UNKNOWN"
    if unknown_move.any():
        report.warnings.append(
            f"Games: {int(unknown_move.sum())} game(s) have UNKNOWN moveability and will be treated as locked."
        )

    tba_games = games["week"].isna()
    if tba_games.any():
        report.info.append(
            f"{int(tba_games.sum())} game(s) are Week TBA. They will appear in schedules but remain outside the optimizer until dated."
        )

    for i, row in games.iterrows():
        if not row["game_id"]:
            games.at[i, "game_id"] = f"import_{row['season']}_{i+1}"
        if row["neutral"]:
            games.at[i, "campus_home_team"] = ""
        elif not row["campus_home_team"]:
            games.at[i, "campus_home_team"] = row["home_team"]

    duplicates = games.duplicated(subset=["season", "home_team", "away_team"], keep=False)
    if duplicates.any():
        report.warnings.append(f"Games: {int(duplicates.sum())} row(s) are part of duplicate home/away/year pairs.")

    valid_moveability = {"MOVABLE", "FLEXIBLE", "LOCKED", "UNKNOWN"}
    bad_move = ~games["moveability"].isin(valid_moveability)
    if bad_move.any():
        report.errors.append("Games: moveability must be MOVABLE, FLEXIBLE, LOCKED, or UNKNOWN.")

    valid_status = {"CONTRACTED", "PENDING", "HOLD", "CONCEPT"}
    bad_status = ~games["game_status"].isin(valid_status)
    if bad_status.any():
        report.errors.append("Games: game_status must be CONTRACTED, PENDING, HOLD, or CONCEPT.")

    # Team metadata.
    if len(teams):
        teams.columns = [str(c).strip().lower() for c in teams.columns]
        if "name" not in teams.columns:
            report.errors.append("Teams sheet must include name.")
        for col in TEAM_COLUMNS:
            if col not in teams.columns:
                teams[col] = ""
        teams["name"] = teams["name"].map(_clean_text)
        teams["subdivision"] = teams["subdivision"].map(lambda v: _clean_text(v, "FBS").upper())
        teams["conference"] = teams["conference"].map(lambda v: _clean_text(v, "Independent"))
        teams["is_a4"] = teams["is_a4"].map(_bool)
        teams["parity_managed"] = teams["parity_managed"].map(lambda v: _bool(v, True))
    else:
        names = sorted(set(games["home_team"]) | set(games["away_team"]))
        lookup: Dict[str, dict] = {}
        if public_teams is not None and len(public_teams):
            for _, r in public_teams.iterrows():
                lookup[str(r["name"])] = r.to_dict()
        rows = []
        unknown = []
        for name in names:
            meta = lookup.get(name)
            if meta:
                rows.append({
                    "name": name,
                    "subdivision": str(meta.get("subdivision", "FBS")),
                    "conference": str(meta.get("conference", "Independent")),
                    "is_a4": bool(meta.get("is_a4", False)),
                    "parity_managed": bool(meta.get("parity_managed", True)),
                })
            else:
                unknown.append(name)
                rows.append({
                    "name": name,
                    "subdivision": "FCS",
                    "conference": "Unknown",
                    "is_a4": False,
                    "parity_managed": False,
                })
        teams = pd.DataFrame(rows)
        if unknown:
            report.warnings.append(
                "Team metadata was not supplied for: " + ", ".join(unknown[:12]) +
                ("…" if len(unknown) > 12 else "") + ". They were provisionally classified as FCS/Unknown."
            )

    team_names = set(teams["name"].astype(str))
    missing_teams = sorted((set(games["home_team"]) | set(games["away_team"])) - team_names)
    if missing_teams:
        report.errors.append("Games reference team(s) missing from Teams: " + ", ".join(missing_teams[:20]))

    # Slots are optional. They represent external blockers, not games already present in Games.
    if len(slots):
        slots.columns = [str(c).strip().lower() for c in slots.columns]
        required = {"team", "season", "week", "status"}
        missing_slots = required - set(slots.columns)
        if missing_slots:
            report.errors.append("Slots is missing required column(s): " + ", ".join(sorted(missing_slots)))
        else:
            if "location" not in slots.columns:
                slots["location"] = "ANY"
            slots["team"] = slots["team"].map(_clean_text)
            slots["season"] = pd.to_numeric(slots["season"], errors="coerce").fillna(0).astype(int)
            slots["week"] = _normalize_week(slots["week"], report, "Slots", allow_blank=False)
            slots["status"] = slots["status"].map(lambda v: _clean_text(v, "OPEN").upper())
            slots["location"] = slots["location"].map(lambda v: _clean_text(v, "ANY").upper())
    else:
        slots = pd.DataFrame(columns=SLOT_COLUMNS)

    # School scheduling needs are optional, but they dramatically improve matching.
    if len(needs):
        needs.columns = [str(c).strip().lower() for c in needs.columns]
        required_needs = {"team", "season", "week", "need_type", "location"}
        missing_needs = required_needs - set(needs.columns)
        if missing_needs:
            report.errors.append("Needs is missing required column(s): " + ", ".join(sorted(missing_needs)))
        else:
            for col in NEED_COLUMNS:
                if col not in needs.columns:
                    needs[col] = ""
            needs["team"] = needs["team"].map(_clean_text)
            needs["season"] = pd.to_numeric(needs["season"], errors="coerce").fillna(0).astype(int)
            needs["week"] = _normalize_week(needs["week"], report, "Needs", allow_blank=False)
            needs["need_type"] = needs["need_type"].map(lambda v: _clean_text(v).upper())
            needs["location"] = needs["location"].map(lambda v: _clean_text(v, "ANY").upper())
            needs["status"] = needs["status"].map(lambda v: _clean_text(v, "OPEN").upper())
            needs["notes"] = needs["notes"].map(_clean_text)
            for col in ["min_guarantee", "max_guarantee"]:
                needs[col] = pd.to_numeric(needs[col], errors="coerce")
    else:
        needs = pd.DataFrame(columns=NEED_COLUMNS)

    report.info.append(f"{len(games)} games loaded.")
    report.info.append(f"{len(teams)} teams loaded.")
    if len(slots):
        report.info.append(f"{len(slots)} explicit slot records loaded.")
    else:
        report.info.append("No Slots sheet supplied; unmodeled weeks default to OPEN.")
    if len(needs):
        report.info.append(f"{len(needs)} explicit school need record(s) loaded.")
    return teams[TEAM_COLUMNS].copy(), games[GAME_COLUMNS].copy(), slots.copy(), needs[NEED_COLUMNS].copy(), report


def make_template_bytes() -> bytes:
    teams = pd.DataFrame([
        {"name": "Florida", "subdivision": "FBS", "conference": "SEC", "is_a4": True, "parity_managed": True},
        {"name": "Georgia", "subdivision": "FBS", "conference": "SEC", "is_a4": True, "parity_managed": True},
        {"name": "Furman", "subdivision": "FCS", "conference": "SoCon", "is_a4": False, "parity_managed": False},
    ])
    games = pd.DataFrame([
        {
            "game_id": "2028_florida_furman",
            "season": 2028,
            "week": 2,
            "date": "2028-09-09",
            "home_team": "Florida",
            "away_team": "Furman",
            "neutral": False,
            "campus_home_team": "Florida",
            "game_status": "CONTRACTED",
            "moveability": "MOVABLE",
            "game_type": "FCS_GUARANTEE",
            "guarantee": 750000,
            "contract_link": "",
            "earliest_week": 1,
            "latest_week": 5,
            "source": "Conference office",
            "last_verified": str(date.today()),
            "confidence": "AUTHORITATIVE",
            "notes": "",
        },
        {
            "game_id": "2028_florida_georgia",
            "season": 2028,
            "week": 10,
            "date": "2028-11-04",
            "home_team": "Florida",
            "away_team": "Georgia",
            "neutral": True,
            "campus_home_team": "",
            "game_status": "CONTRACTED",
            "moveability": "LOCKED",
            "game_type": "CONFERENCE",
            "guarantee": "",
            "contract_link": "",
            "earliest_week": 10,
            "latest_week": 10,
            "source": "Conference office",
            "last_verified": str(date.today()),
            "confidence": "AUTHORITATIVE",
            "notes": "Neutral-site example; designated home does not count as campus home.",
        },
    ])
    slots = pd.DataFrame([
        {"team": "Florida", "season": 2028, "week": 6, "status": "BLOCKED", "location": "ANY"},
    ])
    needs = pd.DataFrame([
        {
            "team": "Florida", "season": 2028, "week": 3,
            "need_type": "A4", "location": "HOME",
            "min_guarantee": "", "max_guarantee": "",
            "status": "OPEN", "notes": "Example A4 need."
        },
        {
            "team": "Furman", "season": 2028, "week": 2,
            "need_type": "FCS_BUY", "location": "AWAY",
            "min_guarantee": 650000, "max_guarantee": "",
            "status": "OPEN", "notes": "Example guarantee-game need."
        },
    ])
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        teams.to_excel(writer, sheet_name="Teams", index=False)
        games.to_excel(writer, sheet_name="Games", index=False)
        slots.to_excel(writer, sheet_name="Slots", index=False)
        needs.to_excel(writer, sheet_name="Needs", index=False)
    return bio.getvalue()
