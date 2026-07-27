"""Walk-forward backtests for map-veto ban weighting models.

The backtest replays parsed HLTV vetoes chronologically. For each historical ban
action it builds probabilities from matches that happened strictly earlier, then
scores the map that was actually banned.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HLTV_MATCHES_FILE
from model.veto_sim import MAP_POOL
from processing.clean import get_invalid_veto_exclusion_reason, normalize_format, normalize_name
from processing.map_pool import canonical_map_name, map_pool_eras_with_bounds


ACTION_RE = re.compile(
    r"^\s*(?P<index>\d+)\.\s*(?P<team>.*?)\s+(?P<action>removed|picked)\s+"
    r"(?P<map>.+?)\s*$",
    re.IGNORECASE,
)
DECIDER_RE = re.compile(
    r"^\s*(?P<index>\d+)\.\s*(?P<map>.+?)"
    r"\s+was\s+left\s+over\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MapPoolEra:
    name: str
    start: datetime | None
    end: datetime | None
    maps: tuple[str, ...]
    events: tuple[str, ...] = ()

    def contains(self, date: datetime) -> bool:
        if self.start and date < self.start:
            return False
        if self.end and date >= self.end:
            return False
        return True


@dataclass(frozen=True)
class VetoAction:
    match_id: str
    date: datetime
    match_format: str
    team_id: str | None
    opponent_id: str | None
    action_type: str
    map_name: str
    global_action_index: int
    team_ban_index: int | None
    pool_before: tuple[str, ...]
    era_name: str


@dataclass(frozen=True)
class MatchVeto:
    match_id: str
    date: datetime
    match_format: str
    team_a_id: str
    team_b_id: str
    actions: tuple[VetoAction, ...]
    played_maps: tuple[str, ...]
    map_winners: tuple[tuple[str, str], ...]
    picked_maps: tuple[tuple[str, str], ...]
    winner_id: str | None


@dataclass(frozen=True)
class FeatureRow:
    actual_map: str
    pool: tuple[str, ...]
    match_format: str
    team_ban_index: int
    prior_team_bans: int
    signals: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: dict[str, float]


DEFAULT_ERAS = tuple(
    MapPoolEra(name=era.name, start=era.effective_from, end=end, maps=era.maps)
    for era, end in map_pool_eras_with_bounds()
)

PRESET_MODELS = [
    ModelSpec("uniform", {}),
    ModelSpec("current_like", {"current_like": 1.0}),
    ModelSpec(
        "grid_best",
        {
            "slot": 0.60,
            "eventual": 0.25,
            "team_ban": 0.15,
        },
    ),
    ModelSpec(
        "grid_best_locked",
        {
            "slot": 0.60,
            "eventual": 0.25,
            "team_ban": 0.15,
            "_lock_probability": 0.90,
            "_lock_min_sample": 10,
            "_lock_min_rate": 0.75,
        },
    ),
    ModelSpec(
        "grid_best_shared_lock",
        {
            "slot": 0.60,
            "eventual": 0.25,
            "team_ban": 0.15,
            "_lock_probability": 0.90,
            "_lock_min_sample": 10,
            "_lock_min_rate": 0.75,
            "_shared_lock_min_rate": 0.75,
            "_shared_lock_min_sample": 10,
        },
    ),
    ModelSpec(
        "ban_history_balanced",
        {
            "slot": 0.45,
            "eventual": 0.25,
            "team_ban": 0.15,
            "global": 0.05,
            "avoidance": 0.05,
            "threat": 0.05,
        },
    ),
    ModelSpec(
        "ban_history_slot",
        {
            "slot": 0.60,
            "eventual": 0.15,
            "team_ban": 0.10,
            "global": 0.05,
            "avoidance": 0.05,
            "threat": 0.05,
        },
    ),
    ModelSpec(
        "ban_history_eventual",
        {
            "slot": 0.30,
            "eventual": 0.35,
            "team_ban": 0.15,
            "global": 0.05,
            "avoidance": 0.10,
            "threat": 0.05,
        },
    ),
]


def parse_date(raw_date: object) -> datetime | None:
    if raw_date is None:
        return None
    text = str(raw_date).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_optional_date(raw_date: str | None) -> datetime | None:
    if not raw_date:
        return None
    parsed = parse_date(raw_date)
    if parsed is None:
        raise ValueError(f"Invalid date: {raw_date}")
    return parsed


def canonical_map(raw_map: str) -> str | None:
    return canonical_map_name(raw_map)


def load_map_pool_eras(path: Path | None) -> tuple[MapPoolEra, ...]:
    if path is None:
        return DEFAULT_ERAS

    payload = json.loads(path.read_text(encoding="utf-8"))
    eras = []
    for item in payload:
        maps = tuple(canonical_map(map_name) or str(map_name).strip() for map_name in item["maps"])
        eras.append(
            MapPoolEra(
                name=str(item.get("name") or f"era_{len(eras) + 1}"),
                start=parse_optional_date(item.get("start")),
                end=parse_optional_date(item.get("end")),
                maps=maps,
                events=tuple(str(event).strip() for event in item.get("events", []) if str(event).strip()),
            )
        )
    if not eras:
        raise ValueError(f"No map-pool eras found in {path}")
    return tuple(eras)


def era_for_date(date: datetime, eras: Iterable[MapPoolEra]) -> MapPoolEra | None:
    matches = [era for era in eras if era.contains(date)]
    if not matches:
        return None
    return max(matches, key=lambda era: era.start or datetime.min.replace(tzinfo=timezone.utc))


def event_key(record: dict) -> str:
    return str(record.get("event") or "").strip().casefold()


def era_named(name: object, eras: Iterable[MapPoolEra]) -> MapPoolEra | None:
    key = str(name or "").strip().casefold()
    return next((era for era in eras if era.name.casefold() == key), None)


def veto_maps_for_record(record: dict) -> frozenset[str]:
    """Extracts all registered map names that appear in an HLTV veto."""
    maps = set()
    for line in record.get("hltv_vetoes") or []:
        text = str(line).strip()
        match = ACTION_RE.match(text) or DECIDER_RE.match(text)
        if not match:
            continue
        map_name = canonical_map(match.group("map"))
        if map_name:
            maps.add(map_name)
    return frozenset(maps)


def era_from_veto(record: dict, eras: tuple[MapPoolEra, ...]) -> MapPoolEra | None:
    """Returns an era when the maps named in a veto identify one pool."""
    observed = veto_maps_for_record(record)
    if not observed:
        return None
    candidates = [era for era in eras if observed.issubset(era.maps)]
    if len(candidates) == 1:
        return candidates[0]

    distinct_pools = {frozenset(era.maps) for era in candidates}
    if len(distinct_pools) != 1:
        return None

    date = parse_date(record.get("date"))
    return era_for_date(date, candidates) if date else candidates[-1]


def explicit_era_for_record(record: dict, eras: tuple[MapPoolEra, ...]) -> MapPoolEra | None:
    """Resolves match metadata or configured event overrides before inference."""
    explicit_name = record.get("map_pool_era")
    if explicit_name:
        return era_named(explicit_name, eras)

    explicit_pool = record.get("map_pool")
    if isinstance(explicit_pool, (list, tuple)):
        maps = tuple(canonical_map(map_name) or str(map_name).strip() for map_name in explicit_pool)
        exact = next((era for era in eras if frozenset(era.maps) == frozenset(maps)), None)
        if exact:
            return exact
        if maps:
            return MapPoolEra(name="explicit_match_pool", start=None, end=None, maps=maps)

    key = event_key(record)
    if key:
        for era in eras:
            if key in {event.casefold() for event in era.events}:
                return era
    return None


def infer_event_era_hints(records: Iterable[dict], eras: tuple[MapPoolEra, ...]) -> dict[str, str]:
    """Learns unambiguous event pools from matches with distinctive vetoes."""
    evidence: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = event_key(record)
        if not key:
            continue
        era = explicit_era_for_record(record, eras) or era_from_veto(record, eras)
        if era and era_named(era.name, eras):
            evidence[key].add(era.name)
    return {key: next(iter(names)) for key, names in evidence.items() if len(names) == 1}


def resolve_match_era(
    record: dict,
    date: datetime,
    eras: tuple[MapPoolEra, ...],
    event_era_hints: dict[str, str] | None = None,
) -> MapPoolEra | None:
    """Resolves the actual match pool, using the effective date only as fallback."""
    explicit = explicit_era_for_record(record, eras)
    if explicit:
        return explicit

    inferred = era_from_veto(record, eras)
    if inferred:
        return inferred

    hint_name = (event_era_hints or {}).get(event_key(record))
    hinted = era_named(hint_name, eras)
    if hinted:
        return hinted

    return era_for_date(date, eras)


def match_id_from_record(record: dict) -> str:
    url = str(record.get("url") or record.get("match_id") or "")
    if "/matches/" in url:
        return url.split("/matches/", 1)[1].split("/", 1)[0]
    return url


def parse_veto_match(
    record: dict,
    eras: tuple[MapPoolEra, ...],
    event_era_hints: dict[str, str] | None = None,
) -> MatchVeto | None:
    if get_invalid_veto_exclusion_reason(record):
        return None

    date = parse_date(record.get("date"))
    if date is None:
        return None
    era = resolve_match_era(record, date, eras, event_era_hints)
    if era is None:
        return None

    team_a = normalize_name(record.get("team1"))
    team_b = normalize_name(record.get("team2"))
    if not team_a or not team_b:
        return None

    pool = list(era.maps)
    action_rows: list[VetoAction] = []
    team_ban_counts: dict[str, int] = defaultdict(int)
    match_format = normalize_format(record.get("format"))
    if match_format not in {"bo1", "bo3", "bo5"}:
        return None
    match_id = str(record.get("url") or match_id_from_record(record))

    for line in record.get("hltv_vetoes") or []:
        text = str(line).strip()
        action_match = ACTION_RE.match(text)
        decider_match = DECIDER_RE.match(text)
        if action_match:
            map_name = canonical_map(action_match.group("map"))
            if map_name is None or map_name not in pool:
                continue
            team_id = normalize_name(action_match.group("team"))
            if team_id not in {team_a, team_b}:
                continue
            opponent_id = team_b if team_id == team_a else team_a
            action_type = "ban" if action_match.group("action").lower() == "removed" else "pick"
            team_ban_index = None
            if action_type == "ban":
                team_ban_counts[team_id] += 1
                team_ban_index = team_ban_counts[team_id]

            action_rows.append(
                VetoAction(
                    match_id=match_id,
                    date=date,
                    match_format=match_format,
                    team_id=team_id,
                    opponent_id=opponent_id,
                    action_type=action_type,
                    map_name=map_name,
                    global_action_index=int(action_match.group("index")),
                    team_ban_index=team_ban_index,
                    pool_before=tuple(pool),
                    era_name=era.name,
                )
            )
            pool.remove(map_name)
        elif decider_match:
            map_name = canonical_map(decider_match.group("map"))
            if map_name is None or map_name not in pool:
                continue
            action_rows.append(
                VetoAction(
                    match_id=match_id,
                    date=date,
                    match_format=match_format,
                    team_id=None,
                    opponent_id=None,
                    action_type="decider",
                    map_name=map_name,
                    global_action_index=int(decider_match.group("index")),
                    team_ban_index=None,
                    pool_before=tuple(pool),
                    era_name=era.name,
                )
            )
            pool.remove(map_name)

    if not action_rows:
        return None

    played_maps = []
    map_winners = []
    picked_maps = []
    for map_data in record.get("hltv_maps") or []:
        map_name = canonical_map(map_data.get("map_name"))
        if map_name is None:
            continue

        played_maps.append(map_name)

        picker = normalize_name(map_data.get("picker"))
        if picker:
            picked_maps.append((picker, map_name))

        winner = map_winner(record, map_data)
        if winner:
            map_winners.append((map_name, winner))

    winner_id = normalize_name(record.get("winner")) or None
    return MatchVeto(
        match_id=match_id,
        date=date,
        match_format=match_format,
        team_a_id=team_a,
        team_b_id=team_b,
        actions=tuple(action_rows),
        played_maps=tuple(played_maps),
        map_winners=tuple(map_winners),
        picked_maps=tuple(picked_maps),
        winner_id=winner_id,
    )


def map_winner(record: dict, map_data: dict) -> str | None:
    score1 = parse_score(map_data.get("team1_score"))
    score2 = parse_score(map_data.get("team2_score"))
    if score1 is None or score2 is None or score1 == score2:
        return None

    raw_winner = normalize_name(map_data.get("team1") if score1 > score2 else map_data.get("team2"))
    team_a = normalize_name(record.get("team1"))
    team_b = normalize_name(record.get("team2"))
    if raw_winner == team_a:
        return team_a
    if raw_winner == team_b:
        return team_b
    if raw_winner in team_a or team_a in raw_winner:
        return team_a
    if raw_winner in team_b or team_b in raw_winner:
        return team_b
    return None


def parse_score(raw_score: object) -> int | None:
    try:
        return int(str(raw_score).strip())
    except (TypeError, ValueError):
        return None


class VetoHistory:
    def __init__(self, window_days: int | None = 90) -> None:
        self.window_days = window_days
        self.ban_events = deque()
        self.pick_events = deque()
        self.eventual_bans = deque()
        self.team_series = deque()
        self.map_plays = deque()

        self.global_ban_counts = Counter()
        self.global_slot_counts = Counter()
        self.team_ban_counts = Counter()
        self.team_slot_counts = Counter()
        self.team_ban_totals = Counter()
        self.team_eventual_counts = Counter()
        self.team_series_totals = Counter()
        self.team_pickable_totals = Counter()
        self.team_play_counts = Counter()
        self.team_win_counts = Counter()
        self.team_pick_counts = Counter()

    def _fresh(self, date: datetime, event_date: datetime) -> bool:
        if self.window_days is None:
            return event_date < date
        age_days = (date - event_date).days
        return 0 <= age_days <= self.window_days

    def build_row(self, action: VetoAction) -> FeatureRow:
        assert action.team_id is not None
        assert action.opponent_id is not None
        assert action.team_ban_index is not None

        self.prune(action.date)

        pool = action.pool_before
        team = action.team_id
        opponent = action.opponent_id
        fmt = action.match_format
        slot = action.team_ban_index
        era = action.era_name

        prior_team_bans = self.team_ban_totals[(era, team)]
        prior_team_series = self.team_series_totals[(era, team)]
        prior_pickable_series = self.team_pickable_totals[(era, team)]

        signals: dict[str, dict[str, float]] = {}
        slot_total = sum(self.team_slot_counts[(era, team, fmt, slot, map_name)] for map_name in pool)
        first_slot_total = sum(self.team_slot_counts[(era, team, fmt, 1, map_name)] for map_name in pool)
        opponent_first_slot_total = sum(self.team_slot_counts[(era, opponent, fmt, 1, map_name)] for map_name in pool)
        eventual_total = sum(self.team_eventual_counts[(era, team, fmt, map_name)] for map_name in pool)
        team_ban_total = sum(self.team_ban_counts[(era, team, map_name)] for map_name in pool)
        global_slot_total = sum(self.global_slot_counts[(era, fmt, slot, map_name)] for map_name in pool)
        for map_name in pool:
            opponent_win_rate = smoothed_rate(
                self.team_win_counts[(era, opponent, map_name)],
                self.team_play_counts[(era, opponent, map_name)],
                0.5,
                4.0,
            )
            own_win_rate = smoothed_rate(
                self.team_win_counts[(era, team, map_name)],
                self.team_play_counts[(era, team, map_name)],
                0.5,
                4.0,
            )
            own_play_rate = smoothed_rate(
                self.team_play_counts[(era, team, map_name)],
                prior_team_series,
                0.0,
                0.0,
            )
            own_pick_rate = smoothed_rate(
                self.team_pick_counts[(era, team, map_name)],
                prior_pickable_series,
                0.0,
                0.0,
            )
            signals[map_name] = {
                "slot": smoothed_probability(
                    self.team_slot_counts[(era, team, fmt, slot, map_name)], slot_total, pool, map_name, 4.0
                ),
                "eventual": smoothed_probability(
                    self.team_eventual_counts[(era, team, fmt, map_name)], eventual_total, pool, map_name, 4.0
                ),
                "team_ban": smoothed_probability(
                    self.team_ban_counts[(era, team, map_name)], team_ban_total, pool, map_name, 7.0
                ),
                "global": smoothed_probability(
                    self.global_slot_counts[(era, fmt, slot, map_name)], global_slot_total, pool, map_name, 5.0
                ),
                "avoidance": 0.65 * (1.0 - own_play_rate) + 0.35 * (1.0 - own_pick_rate),
                "threat": opponent_win_rate,
                "own_loss": 1.0 - own_win_rate,
                "own_play_rate": own_play_rate,
                "own_pick_rate": own_pick_rate,
                "raw_first_slot_rate": (
                    self.team_slot_counts[(era, team, fmt, 1, map_name)] / first_slot_total
                    if first_slot_total
                    else 0.0
                ),
                "raw_first_slot_sample": float(first_slot_total),
                "opponent_raw_first_slot_rate": (
                    self.team_slot_counts[(era, opponent, fmt, 1, map_name)] / opponent_first_slot_total
                    if opponent_first_slot_total
                    else 0.0
                ),
                "opponent_raw_first_slot_sample": float(opponent_first_slot_total),
            }

        return FeatureRow(
            actual_map=action.map_name,
            pool=pool,
            match_format=fmt,
            team_ban_index=slot,
            prior_team_bans=prior_team_bans,
            signals=signals,
        )

    def update(self, match: MatchVeto) -> None:
        self.prune(match.date)

        seen_teams = {match.team_a_id, match.team_b_id}
        for team_id in seen_teams:
            series_event = (match.date, match.actions[0].era_name, team_id, match.match_format)
            self.team_series.append(series_event)
            self.team_series_totals[(match.actions[0].era_name, team_id)] += 1
            if match.match_format != "bo1":
                self.team_pickable_totals[(match.actions[0].era_name, team_id)] += 1

        ban_by_team: dict[str, set[str]] = defaultdict(set)
        for action in match.actions:
            if action.action_type == "ban":
                self.ban_events.append(action)
                self.global_ban_counts[(action.era_name, action.map_name)] += 1
                self.global_slot_counts[(action.era_name, action.match_format, action.team_ban_index, action.map_name)] += 1
                if action.team_id is not None:
                    self.team_ban_counts[(action.era_name, action.team_id, action.map_name)] += 1
                    self.team_slot_counts[
                        (action.era_name, action.team_id, action.match_format, action.team_ban_index, action.map_name)
                    ] += 1
                    self.team_ban_totals[(action.era_name, action.team_id)] += 1
                if action.team_id:
                    ban_by_team[action.team_id].add(action.map_name)
            elif action.action_type == "pick":
                self.pick_events.append(action)
                if action.team_id is not None:
                    self.team_pick_counts[(action.era_name, action.team_id, action.map_name)] += 1

        for team_id, maps in ban_by_team.items():
            for map_name in maps:
                event = (match.date, team_id, match.match_format, match.actions[0].era_name, map_name, match.match_id)
                self.eventual_bans.append(event)
                self.team_eventual_counts[(match.actions[0].era_name, team_id, match.match_format, map_name)] += 1

        winner_lookup = dict(match.map_winners)
        for map_name in match.played_maps:
            for team_id in seen_teams:
                won = bool(winner_lookup.get(map_name) == team_id)
                play_event = (match.date, match.actions[0].era_name, team_id, map_name, won)
                self.map_plays.append(play_event)
                self.team_play_counts[(match.actions[0].era_name, team_id, map_name)] += 1
                if won:
                    self.team_win_counts[(match.actions[0].era_name, team_id, map_name)] += 1

    def prune(self, date: datetime) -> None:
        if self.window_days is None:
            return
        while self.ban_events and not self._fresh(date, self.ban_events[0].date):
            action = self.ban_events.popleft()
            self.global_ban_counts[(action.era_name, action.map_name)] -= 1
            self.global_slot_counts[(action.era_name, action.match_format, action.team_ban_index, action.map_name)] -= 1
            if action.team_id is not None:
                self.team_ban_counts[(action.era_name, action.team_id, action.map_name)] -= 1
                self.team_slot_counts[
                    (action.era_name, action.team_id, action.match_format, action.team_ban_index, action.map_name)
                ] -= 1
                self.team_ban_totals[(action.era_name, action.team_id)] -= 1

        while self.pick_events and not self._fresh(date, self.pick_events[0].date):
            action = self.pick_events.popleft()
            if action.team_id is not None:
                self.team_pick_counts[(action.era_name, action.team_id, action.map_name)] -= 1

        while self.eventual_bans and not self._fresh(date, self.eventual_bans[0][0]):
            event_date, team_id, match_format, era_name, map_name, _match_id = self.eventual_bans.popleft()
            self.team_eventual_counts[(era_name, team_id, match_format, map_name)] -= 1

        while self.team_series and not self._fresh(date, self.team_series[0][0]):
            event_date, era_name, team_id, match_format = self.team_series.popleft()
            self.team_series_totals[(era_name, team_id)] -= 1
            if match_format != "bo1":
                self.team_pickable_totals[(era_name, team_id)] -= 1

        while self.map_plays and not self._fresh(date, self.map_plays[0][0]):
            event_date, era_name, team_id, map_name, won = self.map_plays.popleft()
            self.team_play_counts[(era_name, team_id, map_name)] -= 1
            if won:
                self.team_win_counts[(era_name, team_id, map_name)] -= 1


def smoothed_rate(successes: int, total: int, prior: float, alpha: float) -> float:
    if total <= 0 and alpha <= 0:
        return prior
    return (successes + prior * alpha) / (total + alpha)


def smoothed_probability(count: int, total: int, pool: tuple[str, ...], map_name: str, alpha: float) -> float:
    prior = 1.0 / max(len(pool), 1)
    return (count + prior * alpha) / (total + alpha)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    cleaned = {map_name: max(float(score), 1e-12) for map_name, score in scores.items()}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / max(len(cleaned), 1)
        return {map_name: uniform for map_name in cleaned}
    return {map_name: score / total for map_name, score in cleaned.items()}


def predict_probabilities(row: FeatureRow, model: ModelSpec) -> dict[str, float]:
    if model.name == "uniform":
        return {map_name: 1.0 / len(row.pool) for map_name in row.pool}

    if model.name == "current_like":
        permabans = [
            map_name
            for map_name in row.pool
            if row.signals[map_name]["own_play_rate"] < 0.05
            and row.signals[map_name]["own_pick_rate"] < 0.01
        ]
        if row.team_ban_index == 1 and permabans:
            return normalize_scores(
                {
                    map_name: row.signals[map_name]["threat"] if map_name in permabans else 0.0001
                    for map_name in row.pool
                }
            )
        return normalize_scores(
            {
                map_name: row.signals[map_name]["threat"] + row.signals[map_name]["own_loss"]
                for map_name in row.pool
            }
        )

    scoring_weights = {key: value for key, value in model.weights.items() if not key.startswith("_")}
    scores = {}
    for map_name in row.pool:
        signal = row.signals[map_name]
        scores[map_name] = sum(weight * signal[key] for key, weight in scoring_weights.items())
    probabilities = normalize_scores(scores)
    return apply_lock_adjustment(row, model, probabilities)


def apply_lock_adjustment(row: FeatureRow, model: ModelSpec, probabilities: dict[str, float]) -> dict[str, float]:
    lock_probability = model.weights.get("_lock_probability")
    if not lock_probability or row.team_ban_index != 1:
        return probabilities

    min_sample = model.weights.get("_lock_min_sample", 10)
    min_rate = model.weights.get("_lock_min_rate", 0.75)
    lock_candidates = [
        (
            map_name,
            row.signals[map_name]["raw_first_slot_rate"],
            row.signals[map_name]["raw_first_slot_sample"],
        )
        for map_name in row.pool
    ]
    locked_map, locked_rate, locked_sample = max(lock_candidates, key=lambda item: item[1])
    if locked_sample < min_sample or locked_rate < min_rate:
        return probabilities

    shared_min_rate = model.weights.get("_shared_lock_min_rate")
    shared_min_sample = model.weights.get("_shared_lock_min_sample", min_sample)
    if shared_min_rate is not None:
        opponent_rate = row.signals[locked_map]["opponent_raw_first_slot_rate"]
        opponent_sample = row.signals[locked_map]["opponent_raw_first_slot_sample"]
        if opponent_sample >= shared_min_sample and opponent_rate >= shared_min_rate:
            return probabilities

    residual = 1.0 - lock_probability
    other_total = sum(prob for map_name, prob in probabilities.items() if map_name != locked_map)
    adjusted = {}
    for map_name, probability in probabilities.items():
        if map_name == locked_map:
            adjusted[map_name] = lock_probability
        elif other_total > 0:
            adjusted[map_name] = residual * probability / other_total
        else:
            adjusted[map_name] = residual / max(len(probabilities) - 1, 1)
    return adjusted


def score_rows(rows: list[FeatureRow], models: list[ModelSpec]) -> list[dict[str, object]]:
    results = []
    for model in models:
        buckets: dict[tuple[str, str], list[tuple[float, int, int, float]]] = defaultdict(list)
        overall: list[tuple[float, int, int, float]] = []
        for row in rows:
            probs = predict_probabilities(row, model)
            actual_prob = max(probs.get(row.actual_map, 1e-12), 1e-12)
            ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            rank = next((index + 1 for index, (map_name, _) in enumerate(ranked) if map_name == row.actual_map), len(ranked))
            metric_row = (-math.log(actual_prob), int(rank == 1), int(rank <= 2), actual_prob)
            overall.append(metric_row)
            slot_label = "first_ban" if row.team_ban_index == 1 else "later_ban"
            buckets[(row.match_format, slot_label)].append(metric_row)

        results.append(summary_record(model, "all", "all", overall))
        for (fmt, slot_label), values in sorted(buckets.items()):
            results.append(summary_record(model, fmt, slot_label, values))
    return results


def summary_record(model: ModelSpec, match_format: str, slot_label: str, values: list[tuple[float, int, int, float]]) -> dict[str, object]:
    rows = len(values)
    if rows == 0:
        return {
            "model": model.name,
            "weights": format_weights(model.weights),
            "format": match_format,
            "slot": slot_label,
            "rows": 0,
            "log_loss": None,
            "top1": None,
            "top2": None,
            "mean_actual_prob": None,
        }
    return {
        "model": model.name,
        "weights": format_weights(model.weights),
        "format": match_format,
        "slot": slot_label,
        "rows": rows,
        "log_loss": sum(value[0] for value in values) / rows,
        "top1": sum(value[1] for value in values) / rows,
        "top2": sum(value[2] for value in values) / rows,
        "mean_actual_prob": sum(value[3] for value in values) / rows,
    }


def format_weights(weights: dict[str, float]) -> str:
    if not weights:
        return ""
    return ",".join(f"{key}={value:g}" for key, value in weights.items() if value > 1e-9)


def build_grid_models() -> list[ModelSpec]:
    models = []
    slot_values = [0.0, 0.35, 0.6]
    eventual_values = [0.0, 0.25, 0.5]
    team_ban_values = [0.0, 0.15, 0.3]
    avoidance_values = [0.0, 0.1]
    threat_values = [0.0, 0.1]
    index = 1
    for slot in slot_values:
        for eventual in eventual_values:
            for team_ban in team_ban_values:
                for avoidance in avoidance_values:
                    for threat in threat_values:
                        global_weight = 1.0 - slot - eventual - team_ban - avoidance - threat
                        if global_weight < -1e-9:
                            continue
                        weights = {
                            "slot": slot,
                            "eventual": eventual,
                            "team_ban": team_ban,
                            "global": max(global_weight, 0.0),
                            "avoidance": avoidance,
                            "threat": threat,
                        }
                        if sum(weights.values()) <= 0:
                            continue
                        models.append(ModelSpec(f"grid_{index:04d}", weights))
                        index += 1
    return models


def load_matches(raw_path: Path, eras: tuple[MapPoolEra, ...]) -> list[MatchVeto]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    event_era_hints = infer_event_era_hints(payload, eras)
    matches = [parse_veto_match(record, eras, event_era_hints) for record in payload]
    return sorted((match for match in matches if match is not None), key=lambda match: (match.date, match.match_id))


def build_feature_rows(matches: list[MatchVeto], window_days: int | None, min_prior_team_bans: int) -> list[FeatureRow]:
    history = VetoHistory(window_days=window_days)
    rows: list[FeatureRow] = []
    for match in matches:
        for action in match.actions:
            if action.action_type != "ban":
                continue
            row = history.build_row(action)
            if row.prior_team_bans >= min_prior_team_bans:
                rows.append(row)
        history.update(match)
    return rows


def print_table(records: list[dict[str, object]], limit: int | None = None, include_weights: bool = False) -> None:
    display = records[:limit] if limit else records
    headers = ["model", "format", "slot", "rows", "log_loss", "top1", "top2", "mean_actual_prob"]
    if include_weights:
        headers.append("weights")
    widths = {header: len(header) for header in headers}
    rendered_rows = []
    for record in display:
        rendered = {}
        for header in headers:
            value = record[header]
            if isinstance(value, float):
                rendered[header] = f"{value:.4f}"
            else:
                rendered[header] = "" if value is None else str(value)
            widths[header] = max(widths[header], len(rendered[header]))
        rendered_rows.append(rendered)

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rendered_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "weights", "format", "slot", "rows", "log_loss", "top1", "top2", "mean_actual_prob"],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest historical CS2 veto ban-weighting models.")
    parser.add_argument("--raw", type=Path, default=HLTV_MATCHES_FILE, help="Raw HLTV match JSON path.")
    parser.add_argument("--map-pool-eras", type=Path, help="Optional JSON file with map-pool era definitions.")
    parser.add_argument("--window-days", type=int, default=90, help="Historical lookback window. Use 0 for all history.")
    parser.add_argument("--min-prior-team-bans", type=int, default=0, help="Exclude rows before this many prior team bans.")
    parser.add_argument("--grid-search", action="store_true", help="Evaluate a coarse grid of ban-history weights.")
    parser.add_argument("--top", type=int, default=12, help="Rows to print from the sorted overall summary.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    eras = load_map_pool_eras(args.map_pool_eras)
    window_days = None if args.window_days == 0 else args.window_days
    matches = load_matches(args.raw, eras)
    rows = build_feature_rows(matches, window_days=window_days, min_prior_team_bans=args.min_prior_team_bans)

    models = PRESET_MODELS + (build_grid_models() if args.grid_search else [])
    records = score_rows(rows, models)
    overall = sorted(
        (record for record in records if record["format"] == "all" and record["slot"] == "all"),
        key=lambda record: float(record["log_loss"] or 999),
    )
    splits = [
        record
        for record in records
        if record["model"] in {item["model"] for item in overall[: min(5, len(overall))]}
        and not (record["format"] == "all" and record["slot"] == "all")
    ]

    print(f"Parsed matches: {len(matches):,}")
    print(f"Scored ban rows: {len(rows):,}")
    print(f"Map-pool eras: {', '.join(era.name for era in eras)}")
    print("\nOverall model ranking")
    print_table(overall, limit=args.top, include_weights=True)
    print("\nSplits for top models")
    print_table(sorted(splits, key=lambda record: (str(record["model"]), str(record["format"]), str(record["slot"]))))

    if args.output:
        write_csv(records, args.output)
        print(f"\nWrote CSV: {args.output}")


if __name__ == "__main__":
    main()
