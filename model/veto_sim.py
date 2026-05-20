import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import random
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

# Ensure project root is in path for config and historical data access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import PROCESSED_DIR, DATA_DIR, HLTV_MATCHES_FILE, VETO_WINDOW_DAYS, MC_ITERATIONS
except ImportError:
    PROCESSED_DIR = Path("data/processed")
    DATA_DIR = Path("data")
    HLTV_MATCHES_FILE = DATA_DIR / "raw" / "hltv_matches.json"
    VETO_WINDOW_DAYS = 90
    MC_ITERATIONS = 10000

from processing.clean import get_invalid_veto_exclusion_reason, normalize_format, normalize_name

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Standard Active Duty Pool for CS2 (January 2026 onwards)
MAP_POOL = ["Mirage", "Ancient", "Dust2", "Nuke", "Inferno", "Anubis", "Overpass"]

SIMULATIONS = MC_ITERATIONS
TOP_SEQUENCES_DISPLAY = 5
BAN_SLOT_WEIGHT = 0.60
EVENTUAL_BAN_WEIGHT = 0.25
TEAM_BAN_WEIGHT = 0.15
LOCKED_FIRST_BAN_MIN_SAMPLE = 10
LOCKED_FIRST_BAN_RATE = 0.75
LOCKED_FIRST_BAN_PROBABILITY = 0.90
SHARED_LOCKED_FIRST_BAN_MIN_SAMPLE = 10
SHARED_LOCKED_FIRST_BAN_RATE = 0.75

ACTION_RE = re.compile(
    r"^\s*\d+\.\s*(?P<team>.*?)\s+(?P<action>removed|picked)\s+"
    r"(?P<map>Mirage|Ancient|Dust2|Nuke|Inferno|Anubis|Overpass)\s*$",
    re.IGNORECASE,
)
_RAW_MATCH_CACHE = {"signature": None, "matches": None}

def load_data():
    """Loads the processed match historical data."""
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean maps not found at {clean_path}. Please run the data ingestion pipeline first.")
    return pd.read_parquet(clean_path)

def parse_date(raw_date):
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

def canonical_map(raw_map: str) -> str | None:
    for map_name in MAP_POOL:
        if map_name.lower() == str(raw_map or "").strip().lower():
            return map_name
    return None

def empty_ban_history() -> dict:
    return {
        "slot_counts": defaultdict(Counter),
        "eventual_counts": defaultdict(Counter),
        "team_ban_counts": Counter(),
        "slot_totals": Counter(),
        "eventual_totals": Counter(),
        "team_ban_total": 0,
        "series": 0,
    }

def load_raw_matches(raw_path: Path = HLTV_MATCHES_FILE) -> list:
    if not raw_path.exists():
        return []

    try:
        stat = raw_path.stat()
        signature = (str(raw_path), stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        logger.warning(f"Warning: Could not stat raw veto history: {exc}")
        return []

    if _RAW_MATCH_CACHE["signature"] == signature and _RAW_MATCH_CACHE["matches"] is not None:
        return _RAW_MATCH_CACHE["matches"]

    try:
        matches = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Warning: Could not load raw veto history: {exc}")
        return []

    _RAW_MATCH_CACHE["signature"] = signature
    _RAW_MATCH_CACHE["matches"] = matches
    return matches

def load_team_ban_history(team_id: str, cutoff=None, raw_path: Path = HLTV_MATCHES_FILE) -> dict:
    """Loads explicit HLTV ban history for a team from raw match veto text."""
    history = empty_ban_history()
    matches = load_raw_matches(raw_path)

    for match in matches:
        if get_invalid_veto_exclusion_reason(match):
            continue

        match_date = parse_date(match.get("date"))
        if match_date is None or (cutoff is not None and match_date < cutoff):
            continue

        teams = {normalize_name(match.get("team1")), normalize_name(match.get("team2"))}
        if team_id not in teams:
            continue

        match_format = normalize_format(match.get("format"))
        if match_format not in {"bo1", "bo3", "bo5"}:
            continue

        team_ban_index = 0
        eventual_maps = set()
        for line in match.get("hltv_vetoes", []):
            parsed = ACTION_RE.match(str(line).strip())
            if not parsed:
                continue
            action_team = normalize_name(parsed.group("team"))
            if action_team != team_id or parsed.group("action").lower() != "removed":
                continue

            map_name = canonical_map(parsed.group("map"))
            if map_name is None:
                continue

            team_ban_index += 1
            slot_key = (match_format, team_ban_index)
            history["slot_counts"][slot_key][map_name] += 1
            history["slot_totals"][slot_key] += 1
            history["team_ban_counts"][map_name] += 1
            history["team_ban_total"] += 1
            eventual_maps.add(map_name)

        if team_ban_index > 0:
            history["series"] += 1
            for map_name in eventual_maps:
                history["eventual_counts"][match_format][map_name] += 1
                history["eventual_totals"][match_format] += 1

    return history

def smoothed_probability(count: int, total: int, pool_size: int, alpha: float) -> float:
    prior = 1.0 / max(pool_size, 1)
    return (count + prior * alpha) / (total + alpha)

def raw_first_ban_rate(stats: dict, map_name: str, series_format: str) -> tuple[float, int]:
    slot_key = f"{series_format}:1"
    total = int(stats.get('metadata', {}).get('ban_slot_totals', {}).get(slot_key, 0))
    if total <= 0:
        return 0.0, 0
    count = int(stats[map_name].get('ban_slot_counts', {}).get(slot_key, 0))
    return count / total, total

def get_team_stats(team_id: str, df: pd.DataFrame, days: int = VETO_WINDOW_DAYS) -> Dict[str, Dict]:
    """
    Calculates the necessary heuristics for a team on each map.
    Returns win_rate, pick_rate, play_rate, and loss_rate.
    Uses a rolling window (default 90 days) for more representative statistics.
    """
    # Filter to recent data for more representative statistics
    if days:
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
        df = df[df['date'] >= cutoff]
    else:
        cutoff = None
    
    # Filter for all maps involving this team
    team_df = df[(df['team_a_id'] == team_id) | (df['team_b_id'] == team_id)]
    
    # Total unique series played by the team
    total_series = team_df['match_id'].nunique()
    
    # Filter for series where teams actually pick maps (exclude BO1s)
    pickable_series = team_df[team_df['match_format'] != 'bo1']['match_id'].nunique()
    
    if total_series == 0:
        logger.warning(f"Warning: No historical data found for team '{team_id}'. Using default probabilities.")

    raw_cutoff = cutoff.to_pydatetime() if cutoff is not None else None
    ban_history = load_team_ban_history(team_id, cutoff=raw_cutoff)
    
    stats = {}
    for m in MAP_POOL:
        # Filter matches for this specific map
        map_df = team_df[team_df['map_name'] == m]
        matches_on_map = len(map_df)
        
        # 1. Win Rate (Maps won / Maps played)
        wins = len(map_df[map_df['winner_id'] == team_id])
        win_rate = wins / matches_on_map if matches_on_map > 0 else 0.5
        
        # 2. Pick Rate (Times the map was specifically picked by this team / Total Pickable series)
        # We apply Laplace Smoothing (+1/+len(MAP_POOL)) to ensure every map has a baseline weight.
        picks = len(map_df[
            ((map_df['team_a_id'] == team_id) & map_df['team_a_picked']) |
            ((map_df['team_b_id'] == team_id) & map_df['team_b_picked'])
        ])
        pick_rate = (picks + 1) / (pickable_series + len(MAP_POOL))
        
        # 3. Play Rate (Maps played / Total series) - used for Permaban Override
        play_rate = matches_on_map / total_series if total_series > 0 else 0.0
        
        stats[m] = {
            'win_rate': win_rate,
            'pick_rate': pick_rate,
            'play_rate': play_rate,
            'loss_rate': 1.0 - win_rate,
            'sample_size': matches_on_map,
            'picks': picks,
            'ban_slot_counts': {
                f"{fmt}:{slot}": int(counts.get(m, 0))
                for (fmt, slot), counts in ban_history["slot_counts"].items()
            },
            'eventual_ban_counts': {
                fmt: int(counts.get(m, 0))
                for fmt, counts in ban_history["eventual_counts"].items()
            },
            'team_ban_count': int(ban_history["team_ban_counts"].get(m, 0)),
        }
    stats['metadata'] = {
        'total_series': total_series,
        'pickable_series': pickable_series,
        'ban_series': int(ban_history["series"]),
        'ban_slot_totals': {
            f"{fmt}:{slot}": int(total)
            for (fmt, slot), total in ban_history["slot_totals"].items()
        },
        'eventual_ban_totals': {
            fmt: int(total)
            for fmt, total in ban_history["eventual_totals"].items()
        },
        'team_ban_total': int(ban_history["team_ban_total"]),
    }
    return stats

def print_team_summary(team_id: str, stats: dict):
    """Prints a formatted table of historical statistics for a team."""
    meta = stats['metadata']
    print(f"\n HISTORICAL STATS: {team_id}")
    print(f" Total Series: {meta['total_series']} | Pickable (BO3/BO5): {meta['pickable_series']} | Veto Series: {meta.get('ban_series', 0)}")
    print(f"{'Map Name':12} | {'Win Rate':15} | {'Pick Rate':15} | {'First Bans':15} | {'Times Played':12}")
    print("-" * 85)
    
    for m in MAP_POOL:
        s = stats[m]
        # Win Rate (Wins / Matches Played on Map)
        wins = int(round(s['win_rate'] * s['sample_size'])) if s['sample_size'] > 0 else 0
        wr_val = (s['win_rate'] * 100) if s['sample_size'] > 0 else 0.0
        wr_str = f"{wr_val:5.1f}% ({wins}/{s['sample_size']})"
        
        # Pick Rate (Picks / Pickable Series)
        pks = s['picks']
        pr = (pks / meta['pickable_series'] * 100) if meta['pickable_series'] > 0 else 0.0
        pr_str = f"{pr:5.1f}% ({pks}/{meta['pickable_series']})"
        first_bans = sum(count for key, count in s.get('ban_slot_counts', {}).items() if key.endswith(":1"))
        first_total = sum(total for key, total in meta.get('ban_slot_totals', {}).items() if key.endswith(":1"))
        first_rate = (first_bans / first_total * 100) if first_total > 0 else 0.0
        fb_str = f"{first_rate:5.1f}% ({first_bans}/{first_total})"
        
        print(f"{m:12} | {wr_str:15} | {pr_str:15} | {fb_str:15} | {s['sample_size']:^12}")
    print("-" * 85)

def get_ban_weight(current_stats, opponent_stats, pool, is_first_ban=False, series_format="bo3", team_ban_index=1):
    """
    Calculates ban weights from explicit historical veto behaviour.

    The weights come from the veto backtest's best coarse-grid model:
    format/slot ban rate, eventual ban rate within that format, and overall
    team ban rate. BO1 later bans are therefore credited as evidence that a map
    is a real avoid target even when it was not removed at the first chance.
    """
    meta = current_stats.get('metadata', {})
    slot_key = f"{series_format}:{team_ban_index}"
    slot_total = int(meta.get('ban_slot_totals', {}).get(slot_key, 0))
    eventual_total = int(meta.get('eventual_ban_totals', {}).get(series_format, 0))
    team_ban_total = int(meta.get('team_ban_total', 0))
    pool_size = len(pool)

    weights = []
    for m in pool:
        map_stats = current_stats[m]
        slot_count = int(map_stats.get('ban_slot_counts', {}).get(slot_key, 0))
        eventual_count = int(map_stats.get('eventual_ban_counts', {}).get(series_format, 0))
        team_ban_count = int(map_stats.get('team_ban_count', 0))
        slot_prob = smoothed_probability(slot_count, slot_total, pool_size, alpha=4.0)
        eventual_prob = smoothed_probability(eventual_count, eventual_total, pool_size, alpha=4.0)
        team_ban_prob = smoothed_probability(team_ban_count, team_ban_total, pool_size, alpha=7.0)
        w = (
            BAN_SLOT_WEIGHT * slot_prob
            + EVENTUAL_BAN_WEIGHT * eventual_prob
            + TEAM_BAN_WEIGHT * team_ban_prob
        )
        weights.append(max(w, 1e-6))

    if team_ban_index == 1 and slot_total >= LOCKED_FIRST_BAN_MIN_SAMPLE:
        raw_slot_rates = {
            m: int(current_stats[m].get('ban_slot_counts', {}).get(slot_key, 0)) / slot_total
            for m in pool
        }
        locked_map, locked_rate = max(raw_slot_rates.items(), key=lambda item: item[1])
        if locked_rate >= LOCKED_FIRST_BAN_RATE:
            opponent_rate, opponent_sample = raw_first_ban_rate(opponent_stats, locked_map, series_format)
            if (
                opponent_sample >= SHARED_LOCKED_FIRST_BAN_MIN_SAMPLE
                and opponent_rate >= SHARED_LOCKED_FIRST_BAN_RATE
            ):
                return weights

            locked_index = pool.index(locked_map)
            residual = 1.0 - LOCKED_FIRST_BAN_PROBABILITY
            other_total = sum(weight for index, weight in enumerate(weights) if index != locked_index)
            locked_weights = []
            for index, weight in enumerate(weights):
                if index == locked_index:
                    locked_weights.append(LOCKED_FIRST_BAN_PROBABILITY)
                elif other_total > 0:
                    locked_weights.append(residual * weight / other_total)
                else:
                    locked_weights.append(residual / max(pool_size - 1, 1))
            return locked_weights
    return weights

def get_pick_weight(current_stats, pool):
    """
    Calculates weights for the Pick Phase: own_win_rate * own_pick_rate.
    """
    weights = []
    for m in pool:
        w = current_stats[m]['win_rate'] * current_stats[m]['pick_rate']
        weights.append(max(w, 1e-6))
    return weights

def simulate_veto(stats_a: dict, stats_b: dict, series_format: str = "bo3") -> List[str]:
    """
    Simulates a Best-of-1, Best-of-3, or Best-of-5 veto sequence.
    """
    pool = MAP_POOL.copy()
    played_maps = []
    
    # Define the sequence of moves (Type, Team, isFirstBan)
    if series_format == "bo1":
        # User specified: A ban, A ban, B ban, B ban, B ban, A ban
        steps = [
            ("ban", "a", True), ("ban", "a", True),
            ("ban", "b", True), ("ban", "b", True), ("ban", "b", True),
            ("ban", "a", False)
        ]
    elif series_format == "bo5":
        # User specified: A ban, B ban, A pick, B pick, etc.
        steps = [
            ("ban", "a", True), ("ban", "b", True),
            ("pick", "a"), ("pick", "b"),
            ("pick", "a"), ("pick", "b")
        ]
    else: # Default BO3
        steps = [
            ("ban", "a", True), ("ban", "b", True),
            ("pick", "a"), ("pick", "b"),
            ("ban", "a", False), ("ban", "b", False)
        ]

    ban_counts = {"a": 0, "b": 0}
    for step in steps:
        move_type = step[0]
        team = step[1]
        
        if move_type == "ban":
            current = stats_a if team == "a" else stats_b
            opponent = stats_b if team == "a" else stats_a
            ban_counts[team] += 1
            team_ban_index = ban_counts[team]
            w = get_ban_weight(
                current,
                opponent,
                pool,
                is_first_ban=team_ban_index == 1,
                series_format=series_format,
                team_ban_index=team_ban_index,
            )
            m = random.choices(pool, weights=w, k=1)[0]
            pool.remove(m)
        else: # pick
            current = stats_a if team == "a" else stats_b
            w = get_pick_weight(current, pool)
            m = random.choices(pool, weights=w, k=1)[0]
            played_maps.append(m)
            pool.remove(m)
            
    # Remaining map is the decider (for all formats)
    played_maps.append(pool[0])
    return played_maps

def run_simulations(stats_a: dict, stats_b: dict, iters: int = MC_ITERATIONS, series_format: str = "bo3", starts_veto: str = None):
    """
    Runs multiple Monte Carlo simulations of the map veto process.
    Returns sequence_counts (dict) and map_counts (dict).
    """
    map_counts = {m: 0 for m in MAP_POOL}
    sequence_counts = {}
    
    # Handle Starts Veto Logic
    start_override = None
    if starts_veto:
        if starts_veto.lower() in ["a", "team_a"]:
            start_override = "a"
        elif starts_veto.lower() in ["b", "team_b"]:
            start_override = "b"

    # Run the simulation loop
    for _ in range(iters):
        # Determine who starts for this iteration
        if start_override == "a":
            first, second = stats_a, stats_b
        elif start_override == "b":
            first, second = stats_b, stats_a
        else:
            # Random selection
            if random.random() < 0.5:
                first, second = stats_a, stats_b
            else:
                first, second = stats_b, stats_a

        played = simulate_veto(first, second, series_format)
        for m in played:
            map_counts[m] += 1
        
        seq_str = ",".join(played)
        sequence_counts[seq_str] = sequence_counts.get(seq_str, 0) + 1
        
    return sequence_counts, map_counts

def main():

    parser = argparse.ArgumentParser(description="Monte Carlo Map Veto Simulation for CS2")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    parser.add_argument("--iters", type=int, default=SIMULATIONS, help="Number of iterations")
    parser.add_argument("--format", choices=["bo1", "bo3", "bo5"], default="bo3", help="Series format")
    parser.add_argument("--starts-veto", help="Specific team name that starts the veto (defaults to 50/50 random)")
    args = parser.parse_args()
    
    # Load team mappings for name normalization
    t_a_id = normalize_name(args.team_a)
    t_b_id = normalize_name(args.team_b)
    
    # Handle Starts Veto Logic
    start_override = None
    if args.starts_veto:
        start_id = normalize_name(args.starts_veto)
        if start_id == t_a_id:
            start_override = "a"
        elif start_id == t_b_id:
            start_override = "b"
        else:
            logger.warning(f"Warning: --starts-veto team '{args.starts_veto}' matches neither '{t_a_id}' nor '{t_b_id}'. Using 50/50.")

    try:
        df = load_data()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Pre-calculate stats for both teams
    stats_a = get_team_stats(t_a_id, df)
    stats_b = get_team_stats(t_b_id, df)
    
    # Print Team Summaries
    print_team_summary(t_a_id, stats_a)
    print_team_summary(t_b_id, stats_b)
    
    print(f"\n" + "="*60)
    print(f" MONTE CARLO VETO SIMULATION: {t_a_id} vs {t_b_id} ({args.format.upper()})")
    
    if start_override:
        starter = t_a_id if start_override == "a" else t_b_id
        print(f" Starting Team: {starter} (Forced)")
    else:
        print(f" Starting Team: 50/50 Randomized")
    print(f" Iterations: {args.iters:,}")
    print("="*60)
    
    # Run simulation
    sequence_counts, map_counts = run_simulations(
        stats_a, stats_b, iters=args.iters, series_format=args.format, starts_veto=start_override
    )
            
    # Sort results by probability
    sorted_results = sorted(map_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nPROBABILITY OF MAP APPEARING IN SERIES:")
    print(f"{'Map Name':12} | {'Probability':12} | {'Distribution'}")
    print("-" * 60)
    
    for m, count in sorted_results:
        prob = (count / args.iters) * 100
        bar_len = int(prob / 2)
        bar = "#" * bar_len
        print(f"{m:12} | {prob:10.2f}% | {bar}")
    
    print("-" * 60)
    
    # Display Top Sequences
    print(f"\nMOST LIKELY MAP SEQUENCES (TOP {TOP_SEQUENCES_DISPLAY}):")
    sorted_seqs = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (seq, count) in enumerate(sorted_seqs[:TOP_SEQUENCES_DISPLAY]):
        prob = (count / args.iters) * 100
        print(f" {i+1}. {prob:5.1f}% | {seq}")

    print(f"\nHeuristics: Ban (explicit veto history: slot {BAN_SLOT_WEIGHT:.0%}, eventual {EVENTUAL_BAN_WEIGHT:.0%}, team {TEAM_BAN_WEIGHT:.0%}; shared-aware locked first ban at {LOCKED_FIRST_BAN_RATE:.0%}+), Pick (Own Wr * Own Pr)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
