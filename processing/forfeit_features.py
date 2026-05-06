import json
import math
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_TEAM_RANK, HLTV_MATCHES_FILE
from evaluation.forfeit_analysis import (
    build_match_rows,
    clean_event_name,
    infer_organizer,
    infer_region,
    is_nonstandard_match,
    rank_bucket,
)
from processing.clean import detect_is_lan, normalize_format, normalize_name, parse_rank


FORFEIT_TARGET_COL = "forfeit_target"
FORFEIT_PRIOR = 0.097
RATE_PRIOR_WEIGHT = 20.0
RECENT_WINDOWS_DAYS = (30, 90)

NUMERIC_FORFEIT_FEATURES = [
    "is_lan",
    "best_of",
    "days_since_start",
    "team1_rank",
    "team2_rank",
    "min_rank",
    "max_rank",
    "avg_log_rank",
    "abs_log_rank_diff",
    "either_unranked",
    "both_unranked",
    "global_forfeit_rate",
    "surface_forfeit_rate",
    "region_forfeit_rate",
    "organizer_forfeit_rate",
    "event_forfeit_rate",
    "team1_match_forfeit_rate",
    "team2_match_forfeit_rate",
    "team_avg_match_forfeit_rate",
    "team_max_match_forfeit_rate",
    "team1_own_forfeit_rate",
    "team2_own_forfeit_rate",
    "team_avg_own_forfeit_rate",
    "team_max_own_forfeit_rate",
    "team1_match_count_log",
    "team2_match_count_log",
    "organizer_match_count_log",
    "event_match_count_log",
    "recent_30d_forfeit_rate",
    "recent_90d_forfeit_rate",
    "recent_surface_30d_forfeit_rate",
    "recent_surface_90d_forfeit_rate",
]

CATEGORICAL_FORFEIT_FEATURES = [
    "region",
    "format",
    "team1_rank_bucket",
    "team2_rank_bucket",
    "rank_pair_bucket",
]

FORFEIT_MODEL_FEATURES = NUMERIC_FORFEIT_FEATURES + CATEGORICAL_FORFEIT_FEATURES


@dataclass
class RunningRate:
    successes: int = 0
    count: int = 0

    def rate(self, prior: float = FORFEIT_PRIOR, prior_weight: float = RATE_PRIOR_WEIGHT) -> float:
        return (self.successes + prior * prior_weight) / (self.count + prior_weight)

    def count_log(self) -> float:
        return math.log1p(self.count)

    def update(self, success: bool) -> None:
        self.successes += int(bool(success))
        self.count += 1


@dataclass
class RecentWindow:
    rows: deque = field(default_factory=deque)

    def rate_before(self, current_date: pd.Timestamp, days: int, surface: bool | None = None) -> float:
        cutoff = current_date - pd.Timedelta(days=days)
        while self.rows and self.rows[0][0] < cutoff:
            self.rows.popleft()

        eligible = [
            target
            for date, target, is_lan in self.rows
            if date < current_date and (surface is None or bool(is_lan) == bool(surface))
        ]
        if not eligible:
            return FORFEIT_PRIOR
        return sum(eligible) / len(eligible)

    def update(self, date: pd.Timestamp, target: bool, is_lan: bool) -> None:
        self.rows.append((date, int(bool(target)), bool(is_lan)))


@dataclass
class ForfeitHistoryState:
    start_date: pd.Timestamp
    global_rate: RunningRate = field(default_factory=RunningRate)
    surface_rates: dict[bool, RunningRate] = field(default_factory=lambda: defaultdict(RunningRate))
    region_rates: dict[str, RunningRate] = field(default_factory=lambda: defaultdict(RunningRate))
    organizer_rates: dict[str, RunningRate] = field(default_factory=lambda: defaultdict(RunningRate))
    event_rates: dict[str, RunningRate] = field(default_factory=lambda: defaultdict(RunningRate))
    team_match_rates: dict[str, RunningRate] = field(default_factory=lambda: defaultdict(RunningRate))
    team_own_forfeit_rates: dict[str, RunningRate] = field(default_factory=lambda: defaultdict(RunningRate))
    recent: RecentWindow = field(default_factory=RecentWindow)


def load_raw_hltv_matches(input_path: str | Path = HLTV_MATCHES_FILE) -> list[dict[str, Any]]:
    with Path(input_path).open("r", encoding="utf-8") as f:
        matches = json.load(f)
    return matches if isinstance(matches, list) else [matches]


def build_forfeit_base_rows(
    matches: list[dict[str, Any]] | None = None,
    *,
    input_path: str | Path = HLTV_MATCHES_FILE,
    include_nonstandard: bool = False,
    rank_type: str = "world",
) -> pd.DataFrame:
    """Builds one row per historical match with the settlement-forfeit target."""
    if matches is None:
        matches = load_raw_hltv_matches(input_path)

    valid_matches = []
    for match in matches:
        if not include_nonstandard and is_nonstandard_match(match):
            continue
        if not match.get("team1") or not match.get("team2"):
            continue
        if pd.isna(pd.to_datetime(match.get("date"), errors="coerce", utc=True)):
            continue
        valid_matches.append(match)

    df = build_match_rows(valid_matches, rank_type=rank_type)
    if df.empty:
        return df

    df = df.rename(
        columns={
            "url": "match_id",
            "format": "match_format",
            "has_forfeit": FORFEIT_TARGET_COL,
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "team1", "team2", "match_id"]).copy()
    df = df.drop_duplicates(subset=["match_id"], keep="last")
    df["team1_id"] = df["team1"].map(normalize_name)
    df["team2_id"] = df["team2"].map(normalize_name)
    df["event"] = df["event"].map(clean_event_name)
    df["organizer"] = df["organizer"].fillna("Unknown").astype(str)
    df["region"] = df["region"].fillna("Unknown").astype(str)
    df["match_format"] = df["match_format"].fillna("unknown").astype(str)
    df["settlement_match_format"] = df["match_format"]
    # HLTV stores full defaults as result format "def", which is only known
    # after settlement. Use a conservative scheduled-format fallback so the
    # classifier cannot learn the target from the format field.
    df.loc[df["match_format"] == "def", "match_format"] = "bo3"
    df[FORFEIT_TARGET_COL] = df[FORFEIT_TARGET_COL].astype(int)
    df["team1_rank_bucket"] = df["team1_rank"].map(rank_bucket)
    df["team2_rank_bucket"] = df["team2_rank"].map(rank_bucket)
    df["rank_pair_bucket"] = df.apply(
        lambda row: " / ".join(sorted([row["team1_rank_bucket"], row["team2_rank_bucket"]])),
        axis=1,
    )
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def _safe_rank(value: Any) -> int:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TEAM_RANK
    return max(rank, 1)


def _best_of(match_format: str) -> int:
    if "bo5" in match_format:
        return 5
    if "bo3" in match_format:
        return 3
    if "bo1" in match_format:
        return 1
    return 0


def _own_forfeit(row: pd.Series, team_name: str, team_id: str) -> bool:
    forfeit_team = str(row.get("forfeit_team", "") or "")
    if not forfeit_team or forfeit_team.lower() == "unknown":
        return False
    return forfeit_team.lower() == str(team_name).lower() or normalize_name(forfeit_team) == team_id


def _feature_row(row: pd.Series, state: ForfeitHistoryState) -> dict[str, Any]:
    date = pd.to_datetime(row["date"], utc=True)
    is_lan = bool(row.get("is_lan", False))
    team1_id = normalize_name(row.get("team1_id") or row.get("team1"))
    team2_id = normalize_name(row.get("team2_id") or row.get("team2"))
    event = clean_event_name(row.get("event"))
    organizer = str(row.get("organizer") or infer_organizer(event))
    region = str(row.get("region") or "Unknown")
    match_format = str(row.get("match_format") or "unknown")
    feature_format = "unknown" if match_format == "def" else match_format

    team1_rank = _safe_rank(row.get("team1_rank", DEFAULT_TEAM_RANK))
    team2_rank = _safe_rank(row.get("team2_rank", DEFAULT_TEAM_RANK))
    log1 = math.log(team1_rank)
    log2 = math.log(team2_rank)
    team1_match = state.team_match_rates[team1_id]
    team2_match = state.team_match_rates[team2_id]
    team1_own = state.team_own_forfeit_rates[team1_id]
    team2_own = state.team_own_forfeit_rates[team2_id]

    recent_rates = {}
    for days in RECENT_WINDOWS_DAYS:
        recent_rates[f"recent_{days}d_forfeit_rate"] = state.recent.rate_before(date, days)
        recent_rates[f"recent_surface_{days}d_forfeit_rate"] = state.recent.rate_before(date, days, surface=is_lan)

    features = {
        "match_id": row.get("match_id", ""),
        "date": date,
        "team1": row.get("team1", ""),
        "team2": row.get("team2", ""),
        "team1_id": team1_id,
        "team2_id": team2_id,
        "event": event,
        "organizer": organizer,
        "region": region,
        "format": feature_format,
        "is_lan": int(is_lan),
        "best_of": _best_of(feature_format),
        "days_since_start": max((date - state.start_date).days, 0),
        "team1_rank": team1_rank,
        "team2_rank": team2_rank,
        "min_rank": min(team1_rank, team2_rank),
        "max_rank": max(team1_rank, team2_rank),
        "avg_log_rank": (log1 + log2) / 2,
        "abs_log_rank_diff": abs(log1 - log2),
        "either_unranked": int(team1_rank >= DEFAULT_TEAM_RANK or team2_rank >= DEFAULT_TEAM_RANK),
        "both_unranked": int(team1_rank >= DEFAULT_TEAM_RANK and team2_rank >= DEFAULT_TEAM_RANK),
        "global_forfeit_rate": state.global_rate.rate(),
        "surface_forfeit_rate": state.surface_rates[is_lan].rate(),
        "region_forfeit_rate": state.region_rates[region].rate(),
        "organizer_forfeit_rate": state.organizer_rates[organizer].rate(),
        "event_forfeit_rate": state.event_rates[event].rate(),
        "team1_match_forfeit_rate": team1_match.rate(),
        "team2_match_forfeit_rate": team2_match.rate(),
        "team_avg_match_forfeit_rate": (team1_match.rate() + team2_match.rate()) / 2,
        "team_max_match_forfeit_rate": max(team1_match.rate(), team2_match.rate()),
        "team1_own_forfeit_rate": team1_own.rate(),
        "team2_own_forfeit_rate": team2_own.rate(),
        "team_avg_own_forfeit_rate": (team1_own.rate() + team2_own.rate()) / 2,
        "team_max_own_forfeit_rate": max(team1_own.rate(), team2_own.rate()),
        "team1_match_count_log": team1_match.count_log(),
        "team2_match_count_log": team2_match.count_log(),
        "organizer_match_count_log": state.organizer_rates[organizer].count_log(),
        "event_match_count_log": state.event_rates[event].count_log(),
        "team1_rank_bucket": rank_bucket(team1_rank),
        "team2_rank_bucket": rank_bucket(team2_rank),
        "rank_pair_bucket": " / ".join(sorted([rank_bucket(team1_rank), rank_bucket(team2_rank)])),
    }
    features.update(recent_rates)

    if FORFEIT_TARGET_COL in row:
        features[FORFEIT_TARGET_COL] = int(row[FORFEIT_TARGET_COL])
    return features


def _update_state(row: pd.Series, state: ForfeitHistoryState) -> None:
    target = bool(row[FORFEIT_TARGET_COL])
    date = pd.to_datetime(row["date"], utc=True)
    is_lan = bool(row.get("is_lan", False))
    region = str(row.get("region") or "Unknown")
    event = clean_event_name(row.get("event"))
    organizer = str(row.get("organizer") or infer_organizer(event))
    team1_id = normalize_name(row.get("team1_id") or row.get("team1"))
    team2_id = normalize_name(row.get("team2_id") or row.get("team2"))

    state.global_rate.update(target)
    state.surface_rates[is_lan].update(target)
    state.region_rates[region].update(target)
    state.organizer_rates[organizer].update(target)
    state.event_rates[event].update(target)
    state.team_match_rates[team1_id].update(target)
    state.team_match_rates[team2_id].update(target)
    state.team_own_forfeit_rates[team1_id].update(_own_forfeit(row, row.get("team1", ""), team1_id))
    state.team_own_forfeit_rates[team2_id].update(_own_forfeit(row, row.get("team2", ""), team2_id))
    state.recent.update(date, target, is_lan)


def build_forfeit_feature_frame(base_df: pd.DataFrame) -> pd.DataFrame:
    """Creates leakage-safe historical-rate features using only prior dates."""
    if base_df.empty:
        return base_df

    df = base_df.sort_values(["date", "match_id"]).reset_index(drop=True)
    state = ForfeitHistoryState(start_date=pd.to_datetime(df["date"].min(), utc=True))
    feature_rows = []

    for _, day_df in df.groupby(df["date"].dt.normalize(), sort=True):
        for _, row in day_df.iterrows():
            feature_rows.append(_feature_row(row, state))
        for _, row in day_df.iterrows():
            _update_state(row, state)

    return pd.DataFrame(feature_rows)


def build_full_history_state(base_df: pd.DataFrame) -> ForfeitHistoryState:
    df = base_df.sort_values(["date", "match_id"]).reset_index(drop=True)
    state = ForfeitHistoryState(start_date=pd.to_datetime(df["date"].min(), utc=True))
    for _, row in df.iterrows():
        _update_state(row, state)
    return state


def row_from_match(match: dict[str, Any], *, default_date: datetime | None = None) -> pd.Series:
    default_date = default_date or datetime.now(timezone.utc)
    event = clean_event_name(match.get("event"))
    match_format = normalize_format(match.get("format", "unknown"))
    match_info = match.get("match_info", []) or []
    is_lan = bool(match.get("is_lan", detect_is_lan(match_info)))
    team1 = str(match.get("team1") or match.get("team_a") or "").strip()
    team2 = str(match.get("team2") or match.get("team_b") or "").strip()
    date = pd.to_datetime(match.get("date", default_date), errors="coerce", utc=True)
    if pd.isna(date):
        date = pd.to_datetime(default_date, utc=True)

    ranks = match.get("team_ranks") or {}

    def rank_for(team: str, explicit_key: str) -> int:
        explicit = match.get(explicit_key)
        if explicit is not None:
            return _safe_rank(explicit)
        for name, data in ranks.items():
            if normalize_name(name) == normalize_name(team):
                return parse_rank((data or {}).get("world_rank"))
        return DEFAULT_TEAM_RANK

    return pd.Series(
        {
            "match_id": match.get("url") or match.get("match_id") or f"manual_{team1}_{team2}_{date}",
            "date": date,
            "team1": team1,
            "team2": team2,
            "team1_id": normalize_name(team1),
            "team2_id": normalize_name(team2),
            "event": event,
            "organizer": match.get("organizer") or infer_organizer(event),
            "region": match.get("region") or infer_region(match),
            "match_format": match_format,
            "is_lan": is_lan,
            "team1_rank": rank_for(team1, "team1_rank"),
            "team2_rank": rank_for(team2, "team2_rank"),
        }
    )


def features_for_match(match: dict[str, Any], state: ForfeitHistoryState) -> pd.DataFrame:
    row = row_from_match(match)
    return pd.DataFrame([_feature_row(row, state)])
