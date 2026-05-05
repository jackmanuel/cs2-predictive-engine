import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_TEAM_RANK, HLTV_MATCHES_FILE
from processing.clean import detect_is_lan, get_showmatch_exclusion_reason, normalize_format, parse_rank


FORFEIT_TEXT_PATTERNS = [
    r"\bforfeit(?:ed|s)?\b",
    r"\bdefault win\b",
    r"\bdisqualif(?:ied|ication)\b",
    r"\bunable to field\b",
    r"\bfailed to show up\b",
    r"\bwithdraw(?:n|s)?\b",
]

FORFEIT_TEAM_PATTERNS = [
    r"\*\*\s*(?P<team>.+?)\s+forfeit(?:ed|s)?\b",
    r"\*\*\s*(?P<team>.+?)\s+have been disqualified\b",
    r"\*\*\s*(?P<team>.+?)\s+has been disqualified\b",
    r"\*\*\s*(?P<team>.+?)\s+are disqualified\b",
    r"\*\*\s*(?P<team>.+?)\s+is disqualified\b",
    r"\*\*\s*(?P<team>.+?)\s+failed to show up\b",
    r"\*\*\s*(?P<team>.+?)\s+withdraw(?:n|s)?\s+from the (?:event|tournament)\b",
]

REGION_KEYWORDS = [
    ("North America", ["north america", " na ", "dust2.us", "esea advanced na"]),
    ("South America", ["south america", "latin america", "brazil", "brasil", "latam"]),
    ("Europe", ["europe", "eu ", "european", "dach", "benelux", "balkan", "ukic", "united21"]),
    ("CIS", ["cis", "russia", "russian", "ukraine", "kazakhstan"]),
    ("Asia", ["asia", "asian", "china", "chinese", "mongolia", "india", "nodwin"]),
    ("Oceania", ["oceania", "australia", "anz"]),
    ("Middle East", ["middle east", "mena", "arabia", "uae"]),
]

ORGANIZER_PREFIXES = [
    "BLAST",
    "ESL",
    "IEM",
    "PGL",
    "CCT",
    "YaLLa",
    "Thunderpick",
    "ESEA",
    "EPL",
    "European Pro League",
    "United21",
    "Dust2.us",
    "NODWIN",
    "StarLadder",
    "MESA",
    "RES",
    "Fragadelphia",
    "Exort",
    "LORGAR",
    "A1",
    "Galaxy Battle",
]

EXTRA_NONSTANDARD_PATTERNS = [
    r"\b1\s*vs\s*1\b",
    r"\b1v1\b",
]


def normalize_team(name: str | None) -> str:
    return str(name or "").strip()


def clean_event_name(event: str | None) -> str:
    return str(event or "Unknown").strip() or "Unknown"


def infer_organizer(event: str | None) -> str:
    event_name = clean_event_name(event)
    low = event_name.lower()

    for prefix in ORGANIZER_PREFIXES:
        if low.startswith(prefix.lower()):
            return prefix

    season_split = re.split(r"\s+(?:Season|Series|Cup|League|Masters|Open|Closed|Qualifier)\b", event_name, maxsplit=1)
    candidate = season_split[0].strip(" :-")
    return candidate if candidate else event_name


def infer_region(match: dict[str, Any]) -> str:
    event = f" {clean_event_name(match.get('event')).lower()} "
    info = f" {' '.join(str(line) for line in match.get('match_info', [])).lower()} "
    haystack = event + info

    for region, keywords in REGION_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return region

    if detect_is_lan(match.get("match_info", [])):
        return "LAN/International"

    return "Unknown"


def find_forfeit_notes(match: dict[str, Any]) -> list[str]:
    notes = []
    for line in match.get("match_info", []):
        text = str(line).strip()
        low = text.lower()
        if any(re.search(pattern, low) for pattern in FORFEIT_TEXT_PATTERNS):
            notes.append(text)
    return notes


def is_nonstandard_match(match: dict[str, Any]) -> bool:
    if get_showmatch_exclusion_reason(match):
        return True

    haystack = " ".join(
        [
            clean_event_name(match.get("event")),
            str(match.get("format", "")),
            " ".join(str(line) for line in match.get("match_info", [])),
        ]
    ).lower()
    return any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in EXTRA_NONSTANDARD_PATTERNS)


def is_default_map(map_data: dict[str, Any]) -> bool:
    return str(map_data.get("map_name", "")).strip().lower() in {"default", "forfeit"}


def default_map_forfeit_team(match: dict[str, Any], map_data: dict[str, Any]) -> str | None:
    if not is_default_map(map_data):
        return None

    team1 = normalize_team(match.get("team1")) or normalize_team(map_data.get("team1"))
    team2 = normalize_team(match.get("team2")) or normalize_team(map_data.get("team2"))
    score1 = str(map_data.get("team1_score", "")).strip()
    score2 = str(map_data.get("team2_score", "")).strip()

    if score1 == "0" and score2 == "1":
        return team1
    if score1 == "1" and score2 == "0":
        return team2
    return None


def default_map_winner(match: dict[str, Any], map_data: dict[str, Any]) -> str | None:
    if not is_default_map(map_data):
        return None

    team1 = normalize_team(match.get("team1")) or normalize_team(map_data.get("team1"))
    team2 = normalize_team(match.get("team2")) or normalize_team(map_data.get("team2"))
    score1 = str(map_data.get("team1_score", "")).strip()
    score2 = str(map_data.get("team2_score", "")).strip()

    if score1 == "1" and score2 == "0":
        return team1
    if score1 == "0" and score2 == "1":
        return team2
    return None


def note_forfeit_team(match: dict[str, Any], notes: list[str]) -> str | None:
    teams = [normalize_team(match.get("team1")), normalize_team(match.get("team2"))]
    for note in notes:
        for pattern in FORFEIT_TEAM_PATTERNS:
            found = re.search(pattern, note, flags=re.IGNORECASE)
            if not found:
                continue
            candidate = found.group("team").strip(" *.")
            for team in teams:
                if candidate.lower() == team.lower():
                    return team

    low_notes = " ".join(notes).lower()
    for team in teams:
        if team and team.lower() in low_notes:
            return team
    return None


def rank_for_team(match: dict[str, Any], team: str | None, rank_type: str) -> int:
    if not team:
        return DEFAULT_TEAM_RANK

    ranks = match.get("team_ranks") or {}
    for name, data in ranks.items():
        if str(name).strip().lower() == team.lower():
            if rank_type == "vrs":
                return parse_rank((data or {}).get("vrs_before_rank"))
            return parse_rank((data or {}).get("world_rank"))
    return DEFAULT_TEAM_RANK


def rank_bucket(rank: int) -> str:
    if rank <= 10:
        return "1-10"
    if rank <= 30:
        return "11-30"
    if rank <= 50:
        return "31-50"
    if rank <= 100:
        return "51-100"
    if rank <= 200:
        return "101-200"
    if rank < DEFAULT_TEAM_RANK:
        return "201+"
    return "Unranked/unknown"


def classify_rank_side(forfeit_rank: int, opponent_rank: int) -> str:
    if forfeit_rank == DEFAULT_TEAM_RANK and opponent_rank == DEFAULT_TEAM_RANK:
        return "unknown"
    if forfeit_rank < opponent_rank:
        return "higher-ranked forfeit"
    if forfeit_rank > opponent_rank:
        return "lower-ranked forfeit"
    return "same rank"


def reason_category(notes: list[str]) -> str:
    low = " ".join(notes).lower()
    if "scheduling conflict" in low:
        return "scheduling conflict"
    if "eligible roster" in low or "complete lineup" in low or "unable to field" in low:
        return "roster eligibility/lineup"
    if "integrity" in low or "disqualified" in low or "disqualification" in low:
        return "integrity/disqualification"
    if "technical" in low or "power" in low or "connection" in low:
        return "technical"
    if "failed to show up" in low:
        return "no-show"
    if any(
        phrase in low
        for phrase in (
            "withdraw from the event",
            "withdraws from the event",
            "withdrawn from the event",
            "withdraw from the tournament",
            "withdraws from the tournament",
            "withdrawn from the tournament",
        )
    ):
        return "event withdrawal"
    if notes:
        return "other stated reason"
    return "default map only"


def build_match_rows(matches: list[dict[str, Any]], rank_type: str) -> pd.DataFrame:
    rows = []
    for match in matches:
        notes = find_forfeit_notes(match)
        default_maps = [m for m in match.get("hltv_maps", []) if is_default_map(m)]
        match_format = normalize_format(match.get("format", "unknown"))
        has_forfeit = match_format == "def" or bool(default_maps) or bool(notes)

        default_forfeit_teams = [team for team in (default_map_forfeit_team(match, m) for m in default_maps) if team]
        default_winners = [team for team in (default_map_winner(match, m) for m in default_maps) if team]
        forfeit_team = Counter(default_forfeit_teams).most_common(1)[0][0] if default_forfeit_teams else note_forfeit_team(match, notes)
        default_winner = Counter(default_winners).most_common(1)[0][0] if default_winners else None

        team1 = normalize_team(match.get("team1"))
        team2 = normalize_team(match.get("team2"))
        opponent = None
        if forfeit_team:
            opponent = team2 if forfeit_team.lower() == team1.lower() else team1

        forfeit_rank = rank_for_team(match, forfeit_team, rank_type)
        opponent_rank = rank_for_team(match, opponent, rank_type)
        event = clean_event_name(match.get("event"))
        date = pd.to_datetime(match.get("date"), errors="coerce")

        rows.append(
            {
                "date": date,
                "url": match.get("url", ""),
                "event": event,
                "organizer": infer_organizer(event),
                "region": infer_region(match),
                "is_lan": detect_is_lan(match.get("match_info", [])),
                "format": match_format,
                "team1": team1,
                "team2": team2,
                "team1_rank": rank_for_team(match, team1, rank_type),
                "team2_rank": rank_for_team(match, team2, rank_type),
                "has_forfeit": has_forfeit,
                "is_full_default": match_format == "def",
                "has_partial_default_map": bool(default_maps) and match_format != "def",
                "default_map_count": len(default_maps),
                "forfeit_team": forfeit_team or "Unknown",
                "default_winner": default_winner or "Unknown",
                "forfeit_team_rank": forfeit_rank,
                "opponent_rank": opponent_rank,
                "forfeit_rank_bucket": rank_bucket(forfeit_rank),
                "rank_side": classify_rank_side(forfeit_rank, opponent_rank),
                "reason": reason_category(notes),
                "notes": " | ".join(notes),
                "has_time_of_day": False,
            }
        )

    return pd.DataFrame(rows)


def pct(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def fmt_pct(numerator: int | float, denominator: int | float) -> str:
    return f"{pct(numerator, denominator):5.2f}%"


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> None:
    if df.empty:
        print("No rows.")
        return

    out = df[columns].copy()
    if limit:
        out = out.head(limit)
    print(out.to_string(index=False))


def rate_table(
    df: pd.DataFrame,
    group_col: str,
    *,
    min_matches: int = 1,
    limit: int = 20,
    label: str | None = None,
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(matches=("has_forfeit", "size"), forfeits=("has_forfeit", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["matches"] >= min_matches].copy()
    grouped["forfeit_rate"] = grouped["forfeits"] / grouped["matches"]
    grouped = grouped.sort_values(["forfeit_rate", "forfeits", "matches"], ascending=[False, False, False])
    grouped["forfeit_rate"] = grouped["forfeit_rate"].map(lambda value: f"{value * 100:5.2f}%")
    if label:
        grouped = grouped.rename(columns={group_col: label})
    return grouped.head(limit)


def team_rate_table(df: pd.DataFrame, min_matches: int, limit: int) -> pd.DataFrame:
    appearances = []
    for side in ("team1", "team2"):
        side_df = df[[side, "has_forfeit", "forfeit_team"]].copy()
        side_df = side_df.rename(columns={side: "team"})
        side_df["team_forfeited"] = side_df.apply(
            lambda row: row["has_forfeit"] and str(row["team"]).lower() == str(row["forfeit_team"]).lower(),
            axis=1,
        )
        appearances.append(side_df[["team", "team_forfeited"]])

    team_df = pd.concat(appearances, ignore_index=True)
    grouped = (
        team_df.groupby("team", dropna=False)
        .agg(matches=("team_forfeited", "size"), team_forfeits=("team_forfeited", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["matches"] >= min_matches].copy()
    grouped["team_forfeit_rate"] = grouped["team_forfeits"] / grouped["matches"]
    grouped = grouped.sort_values(["team_forfeit_rate", "team_forfeits", "matches"], ascending=[False, False, False])
    grouped["team_forfeit_rate"] = grouped["team_forfeit_rate"].map(lambda value: f"{value * 100:5.2f}%")
    return grouped.head(limit)


def print_trend(df: pd.DataFrame, cadence: str) -> None:
    trend = df.dropna(subset=["date"]).copy()
    if trend.empty:
        print("No dated rows.")
        return

    trend["period"] = trend["date"].dt.to_period(cadence).astype(str)
    grouped = (
        trend.groupby("period")
        .agg(matches=("has_forfeit", "size"), forfeits=("has_forfeit", "sum"))
        .reset_index()
    )
    grouped["forfeit_rate"] = grouped.apply(lambda row: fmt_pct(row["forfeits"], row["matches"]), axis=1)
    print_table(grouped, ["period", "matches", "forfeits", "forfeit_rate"])


def print_forfeit_rate_regression(df: pd.DataFrame, cadence: str) -> None:
    trend = df.dropna(subset=["date"]).copy()
    if trend.empty:
        print("No dated rows for regression.")
        return

    trend["period_start"] = trend["date"].dt.to_period(cadence).dt.start_time
    grouped = (
        trend.groupby("period_start")
        .agg(matches=("has_forfeit", "size"), forfeits=("has_forfeit", "sum"))
        .reset_index()
        .sort_values("period_start")
    )
    grouped = grouped[grouped["matches"] > 0].copy()
    if len(grouped) < 3:
        print("Not enough time buckets for regression.")
        return

    grouped["forfeit_rate"] = grouped["forfeits"] / grouped["matches"]
    grouped["days_since_start"] = (grouped["period_start"] - grouped["period_start"].min()).dt.days

    result = stats.linregress(grouped["days_since_start"], grouped["forfeit_rate"])
    slope_per_day = result.slope
    slope_per_30_days = slope_per_day * 30
    r_squared = result.rvalue ** 2
    significant = "yes" if result.pvalue < 0.05 else "no"
    start_date = grouped["period_start"].min().strftime("%Y-%m-%d")
    end_date = grouped["period_start"].max().strftime("%Y-%m-%d")

    print()
    print("Linear Regression Trend Test")
    print("----------------------------")
    print(f"Date span: {start_date} to {end_date}")
    print(f"Time buckets: {len(grouped)} ({cadence})")
    print(f"Slope: {slope_per_30_days * 100:+.2f} percentage points per 30 days")
    print(f"Intercept: {result.intercept * 100:.2f}%")
    print(f"R-squared: {r_squared:.3f}")
    print(f"p-value: {result.pvalue:.4f}")
    print(f"Statistically significant at alpha=0.05: {significant}")
    print("Note: this is ordinary least squares on bucketed forfeit rates; it is exploratory and does not prove causality.")


def print_recent_forfeits(df: pd.DataFrame, limit: int) -> None:
    recent = df[df["has_forfeit"]].sort_values("date", ascending=False).head(limit).copy()
    if recent.empty:
        print("No forfeits detected.")
        return

    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    print_table(
        recent,
        [
            "date",
            "forfeit_team",
            "default_winner",
            "event",
            "region",
            "format",
            "reason",
            "notes",
        ],
    )


def print_forfeit_analysis(args: argparse.Namespace) -> None:
    with Path(args.input).open("r", encoding="utf-8") as f:
        matches = json.load(f)

    excluded_nonstandard = 0
    if not args.include_nonstandard:
        standard_matches = []
        for match in matches:
            if is_nonstandard_match(match):
                excluded_nonstandard += 1
                continue
            standard_matches.append(match)
        matches = standard_matches

    df = build_match_rows(matches, args.rank_type)
    total_matches = len(df)
    forfeits = int(df["has_forfeit"].sum())
    full_defaults = int(df["is_full_default"].sum())
    partial_defaults = int(df["has_partial_default_map"].sum())
    online = df[~df["is_lan"]]
    lan = df[df["is_lan"]]

    print_header("Forfeit Analysis")
    print(f"Source: {Path(args.input)}")
    print(f"Rank basis: {args.rank_type}")
    print(f"Non-standard/showmatch matches excluded: {excluded_nonstandard:,}")
    print(f"Matches: {total_matches:,}")
    print(f"Forfeit/default affected matches: {forfeits:,} ({fmt_pct(forfeits, total_matches)})")
    print(f"Full default matches: {full_defaults:,} ({fmt_pct(full_defaults, total_matches)})")
    print(f"Partial default map matches: {partial_defaults:,} ({fmt_pct(partial_defaults, total_matches)})")
    print(f"Default/forfeit map rows: {int(df['default_map_count'].sum()):,}")
    print(f"Online forfeit rate: {int(online['has_forfeit'].sum()):,}/{len(online):,} ({fmt_pct(online['has_forfeit'].sum(), len(online))})")
    print(f"LAN forfeit rate: {int(lan['has_forfeit'].sum()):,}/{len(lan):,} ({fmt_pct(lan['has_forfeit'].sum(), len(lan))})")
    print("Match start time-of-day: unavailable in current raw HLTV data; saved dates are day-level only.")

    print_header("Trend")
    print_trend(df, args.cadence)
    print_forfeit_rate_regression(df, args.cadence)

    print_header("Online vs LAN")
    print_table(rate_table(df, "is_lan", label="is_lan"), ["is_lan", "matches", "forfeits", "forfeit_rate"])

    print_header("By Region")
    print_table(rate_table(df, "region", min_matches=args.min_group_matches, limit=args.limit), ["region", "matches", "forfeits", "forfeit_rate"])

    print_header("By TO/Event Family")
    print_table(
        rate_table(df, "organizer", min_matches=args.min_group_matches, limit=args.limit, label="TO/event_family"),
        ["TO/event_family", "matches", "forfeits", "forfeit_rate"],
    )

    print_header("By Event")
    print_table(rate_table(df, "event", min_matches=args.min_group_matches, limit=args.limit), ["event", "matches", "forfeits", "forfeit_rate"])

    print_header("By Forfeiting Team Rank")
    forfeits_df = df[df["has_forfeit"]].copy()
    rank_buckets = (
        forfeits_df.groupby("forfeit_rank_bucket")
        .agg(forfeits=("has_forfeit", "size"))
        .reset_index()
        .sort_values("forfeits", ascending=False)
    )
    rank_buckets["share_of_forfeits"] = rank_buckets["forfeits"].map(lambda value: fmt_pct(value, forfeits))
    print_table(rank_buckets, ["forfeit_rank_bucket", "forfeits", "share_of_forfeits"], limit=args.limit)

    print_header("Rank Side Of Forfeiting Team")
    rank_side = (
        forfeits_df.groupby("rank_side")
        .agg(forfeits=("has_forfeit", "size"))
        .reset_index()
        .sort_values("forfeits", ascending=False)
    )
    rank_side["share_of_forfeits"] = rank_side["forfeits"].map(lambda value: fmt_pct(value, forfeits))
    print_table(rank_side, ["rank_side", "forfeits", "share_of_forfeits"])

    print_header("Forfeit Reasons")
    reasons = (
        forfeits_df.groupby("reason")
        .agg(forfeits=("has_forfeit", "size"))
        .reset_index()
        .sort_values("forfeits", ascending=False)
    )
    reasons["share_of_forfeits"] = reasons["forfeits"].map(lambda value: fmt_pct(value, forfeits))
    print_table(reasons, ["reason", "forfeits", "share_of_forfeits"])

    print_header("Teams With Most Forfeits")
    team_counts = (
        forfeits_df.groupby("forfeit_team")
        .agg(team_forfeits=("has_forfeit", "size"))
        .reset_index()
        .sort_values("team_forfeits", ascending=False)
    )
    print_table(team_counts, ["forfeit_team", "team_forfeits"], limit=args.limit)

    print_header("Team Forfeit Rate")
    print_table(team_rate_table(df, args.min_team_matches, args.limit), ["team", "matches", "team_forfeits", "team_forfeit_rate"])

    print_header("Recent Forfeits")
    print_recent_forfeits(df, args.recent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal report for HLTV forfeit/default match analysis.")
    parser.add_argument("--input", default=str(HLTV_MATCHES_FILE), help="Raw HLTV matches JSON path.")
    parser.add_argument("--rank-type", choices=["world", "vrs"], default="world", help="Rank source for team-rank breakdowns.")
    parser.add_argument("--cadence", choices=["D", "W", "M"], default="W", help="Trend grouping: D=daily, W=weekly, M=monthly.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows per table.")
    parser.add_argument("--recent", type=int, default=15, help="Recent forfeit rows to print.")
    parser.add_argument("--min-group-matches", type=int, default=10, help="Minimum matches for event/region/TO rate tables.")
    parser.add_argument("--min-team-matches", type=int, default=5, help="Minimum appearances for team-rate table.")
    parser.add_argument("--include-nonstandard", action="store_true", help="Include showmatches and other non-standard match formats.")
    return parser.parse_args()


if __name__ == "__main__":
    print_forfeit_analysis(parse_args())
