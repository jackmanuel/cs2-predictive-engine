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
MAP_POOL = ["Mirage", "Ancient", "Dust2", "Nuke", "Inferno", "Anubis", "Vertigo"]

SIMULATIONS = 10000

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
        
        # 2. Pick Rate (Times the map was specifically picked by this team / Total series)
        # Assuming 'team_a_picked' is True if team_a_id picked it.
        picks = len(map_df[
            ((map_df['team_a_id'] == team_id) & map_df['team_a_picked']) |
            ((map_df['team_b_id'] == team_id) & map_df['team_b_picked'])
        ])
        pick_rate = picks / total_series if total_series > 0 else 0.0
        
        # 3. Play Rate (Maps played / Total series) - used for Permaban Override
        play_rate = matches_on_map / total_series if total_series > 0 else 0.0
        
        stats[m] = {
            'win_rate': win_rate,
            'pick_rate': pick_rate,
            'play_rate': play_rate,
            'loss_rate': 1.0 - win_rate
        }
    return stats

def get_ban_weight(current_stats, opponent_stats, pool, is_first_ban=False):
    """
    Calculates weights for the Ban Phase: opponent_win_rate + own_loss_rate.
    Applies Permaban Override (99.9% probability) for maps with < 5% play rate.
    """
    # Identify maps that meet the permaban criteria (< 5% play rate)
    permabans = [m for m in pool if current_stats[m]['play_rate'] < 0.05]
    
    if is_first_ban and permabans:
        # If permabans exist, they collectively take 99.9% of the probability.
        # We distribute 1.0 weight to permabans and a negligible amount to others.
        return [1.0 if m in permabans else (0.001 / len(pool)) for m in pool]
    
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
    args = parser.parse_args()
    
    # Load team mappings for name normalization
    mapping_file = DATA_DIR / "team_mappings.json"
    mappings = {}
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            
    t_a_id = normalize_name(args.team_a, mappings)
    t_b_id = normalize_name(args.team_b, mappings)
    
    try:
        df = load_data()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Pre-calculate stats for both teams
    stats_a = get_team_stats(t_a_id, df)
    stats_b = get_team_stats(t_b_id, df)
    
    print(f"\n" + "="*60)
    print(f" MONTE CARLO VETO SIMULATION: {t_a_id} vs {t_b_id} ({args.format.upper()})")
    print(f" Iterations: {args.iters:,}")
    print("="*60)
    
    map_counts = {m: 0 for m in MAP_POOL}
    
    # Run the simulation loop
    for _ in range(args.iters):
        played = simulate_veto(stats_a, stats_b, args.format)
        for m in played:
            map_counts[m] += 1
            
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
    print("Heuristics: Ban (Opp Wr + Own Lr), Pick (Own Wr * Own Pr), Permaban (<5% play)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
