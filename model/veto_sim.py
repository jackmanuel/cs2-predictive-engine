import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

# Ensure project root is in path for config and historical data access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import PROCESSED_DIR, DATA_DIR
except ImportError:
    # Fallback if not running from project structure
    PROCESSED_DIR = Path("data/processed")
    DATA_DIR = Path("data")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Standard Active Duty Pool for CS2 (April 2024 onwards)
# Note: Vertigo may be missing from some historical datasets but is part of the active pool.
MAP_POOL = ["Mirage", "Ancient", "Dust2", "Nuke", "Inferno", "Anubis", "Overpass"]

SIMULATIONS = 10000
TOP_SEQUENCES_DISPLAY = 5
PERMABAN_THRESHOLD = 0.05

def load_data():
    """Loads the processed match historical data."""
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean maps not found at {clean_path}. Please run the data ingestion pipeline first.")
    return pd.read_parquet(clean_path)

def normalize_name(name: str, mappings: dict) -> str:
    """Normalizes team names using the standard project mapping."""
    if not name: return ""
    name_strip = name.strip()
    if name_strip in mappings:
        return mappings[name_strip].upper().strip()
    return name_strip.upper()

def get_team_stats(team_id: str, df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculates the necessary heuristics for a team on each map.
    Returns win_rate, pick_rate, play_rate, and loss_rate.
    """
    # Filter for all maps involving this team
    team_df = df[(df['team_a_id'] == team_id) | (df['team_b_id'] == team_id)]
    
    # Total unique series played by the team
    total_series = team_df['match_id'].nunique()
    
    # Filter for series where teams actually pick maps (exclude BO1s)
    pickable_series = team_df[team_df['match_format'] != 'bo1']['match_id'].nunique()
    
    if total_series == 0:
        logger.warning(f"Warning: No historical data found for team '{team_id}'. Using default probabilities.")
    
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
            'picks': picks
        }
    stats['metadata'] = {
        'total_series': total_series,
        'pickable_series': pickable_series
    }
    return stats

def print_team_summary(team_id: str, stats: dict):
    """Prints a formatted table of historical statistics for a team."""
    meta = stats['metadata']
    print(f"\n HISTORICAL STATS: {team_id}")
    print(f" Total Series: {meta['total_series']} | Pickable (BO3/BO5): {meta['pickable_series']}")
    print(f"{'Map Name':12} | {'Win Rate':15} | {'Pick Rate':15} | {'Times Played':12}")
    print("-" * 65)
    
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
        
        print(f"{m:12} | {wr_str:15} | {pr_str:15} | {s['sample_size']:^12}")
    print("-" * 65)

def get_ban_weight(current_stats, opponent_stats, pool, is_first_ban=False):
    """
    Calculates weights for the Ban Phase: opponent_win_rate + own_loss_rate.
    Applies Permaban Override with Opponent Threat Tiebreaker for maps with low play/pick rates.
    """
    # Identify maps that meet the permaban criteria (< 0.05 play rate and < 0.01 pick rate)
    pickable = current_stats['metadata']['pickable_series']
    permabans = []
    for m in pool:
        play_rate = current_stats[m]['play_rate']
        picks = current_stats[m]['picks']
        raw_pick_rate = picks / pickable if pickable > 0 else 0.0
        
        if play_rate < PERMABAN_THRESHOLD and raw_pick_rate < 0.01:
            permabans.append(m)
    
    if is_first_ban and permabans:
        # Opponent Threat Tiebreaker with Shared Permaban Bluffing
        weights = []
        for m in pool:
            if m in permabans:
                # Check for Shared Permaban (Bluffing Logic)
                # If the opponent also has a play rate < 0.05 on this map, we refuse to ban it,
                # assuming they will be forced to ban it themselves.
                if opponent_stats[m]['play_rate'] < PERMABAN_THRESHOLD:
                    w = 0.0001
                else:
                    # Threat Tiebreaker: prioritize banning maps the opponent is statistically proficient at
                    w = max(opponent_stats[m]['win_rate'], 0.01)
                weights.append(w)
            else:
                # Maps not in the permaban list continue to get negligible weight during this override phase
                weights.append(0.0001)
        return weights
    
    # Standard heuristic: opponent_map_win_rate + own_map_loss_rate
    weights = []
    for m in pool:
        w = opponent_stats[m]['win_rate'] + current_stats[m]['loss_rate']
        weights.append(max(w, 1e-6)) # Avoid zero weights for random.choices
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

    for step in steps:
        move_type = step[0]
        team = step[1]
        
        if move_type == "ban":
            is_first = step[2]
            current = stats_a if team == "a" else stats_b
            opponent = stats_b if team == "a" else stats_a
            w = get_ban_weight(current, opponent, pool, is_first_ban=is_first)
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

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Map Veto Simulation for CS2")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    parser.add_argument("--iters", type=int, default=SIMULATIONS, help="Number of iterations")
    parser.add_argument("--format", choices=["bo1", "bo3", "bo5"], default="bo3", help="Series format")
    parser.add_argument("--starts-veto", help="Specific team name that starts the veto (defaults to 50/50 random)")
    args = parser.parse_args()
    
    # Load team mappings for name normalization
    mapping_file = DATA_DIR / "team_mappings.json"
    mappings = {}
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            
    t_a_id = normalize_name(args.team_a, mappings)
    t_b_id = normalize_name(args.team_b, mappings)
    
    # Handle Starts Veto Logic
    start_override = None
    if args.starts_veto:
        start_id = normalize_name(args.starts_veto, mappings)
        if start_id == t_a_id:
            start_override = "a"
        elif start_id == t_b_id:
            start_override = "b"
        else:
            logger.warning(f"Warning: --start-veto team '{args.start_veto}' matches neither '{t_a_id}' nor '{t_b_id}'. Using 50/50.")

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
    
    map_counts = {m: 0 for m in MAP_POOL}
    sequence_counts = {}
    
    # Run the simulation loop
    for _ in range(args.iters):
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

        played = simulate_veto(first, second, args.format)
        for m in played:
            map_counts[m] += 1
        
        seq_str = ",".join(played)
        sequence_counts[seq_str] = sequence_counts.get(seq_str, 0) + 1
            
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

    print(f"\nHeuristics: Ban (Bluffing / Threat / Opp Wr + Own Lr), Pick (Own Wr * Own Pr), Permaban (<{PERMABAN_THRESHOLD*100:.0f}% play & <1% pick)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
