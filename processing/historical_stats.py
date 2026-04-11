import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.warning("DEPRECATION: processing.historical_stats is deprecated. Long-term winrates are calculated from the canonical HLTV dataset in processing.features.")

MAPPING_FILE = DATA_DIR / "team_mappings.json"

def load_mappings() -> dict:
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def normalize_name(name: str, mappings: dict) -> str:
    if not name: return ""
    name_strip = name.strip()
    if name_strip in mappings:
        return mappings[name_strip].upper().strip()
    return name_strip.upper()

def compute_historical_stats():
    """Calculates long-term winrates from raw PandaScore data for baseline features."""
    mappings = load_mappings()
    
    # We'll use the large PandaScore file for historical context
    pandascore_path = RAW_DIR / "matches_20260409.json"
    if not pandascore_path.exists():
        logger.warning(f"PandaScore base file not found at {pandascore_path}. Skipping historical stats.")
        return

    logger.info(f"Loading PandaScore history from {pandascore_path}...")
    with open(pandascore_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    match_records = []
    for m in data:
        if m.get("status") != "finished" or m.get("forfeit"):
            continue
            
        opponents = m.get("opponents", [])
        if len(opponents) != 2:
            continue
            
        t_a = opponents[0]["opponent"]["name"]
        t_b = opponents[1]["opponent"]["name"]
        winner_id = m.get("winner_id")
        t_a_id = opponents[0]["opponent"]["id"]
        t_b_id = opponents[1]["opponent"]["id"]
        
        # Determine winner by ID
        win_a = 1 if winner_id == t_a_id else 0
        win_b = 1 if winner_id == t_b_id else 0
        
        match_records.append({
            "date": pd.to_datetime(m.get("begin_at")),
            "team_a": normalize_name(t_a, mappings),
            "win_a": win_a,
            "team_b": normalize_name(t_b, mappings),
            "win_b": win_b
        })
        
    df = pd.DataFrame(match_records).sort_values("date")
    
    # Calculate lifetime winrates (up to that point) for baseline
    # Since we want this for THE CURRENT PIPELINE, we'll actually just export
    # a simple 'Team -> Overall WR' or 'Team -> Rolling WR' that we can join.
    # For now, let's just make a table of Team -> [Total Matches, Total Wins] 
    # as a snapshot of the historical PandaScore era.
    
    stats = {}
    for _, row in df.iterrows():
        for prefix in ['a', 'b']:
            team = row[f'team_{prefix}']
            win = row[f'win_{prefix}']
            if team not in stats:
                stats[team] = {"matches": 0, "wins": 0}
            stats[team]["matches"] += 1
            stats[team]["wins"] += win
            
    stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index()
    stats_df.columns = ['team_name', 'pcore_matches', 'pcore_wins']
    stats_df['pcore_wr'] = stats_df['pcore_wins'] / stats_df['pcore_matches']
    
    out_path = PROCESSED_DIR / "historical_pandascore_stats.parquet"
    stats_df.to_parquet(out_path, index=False)
    logger.info(f"Historical PandaScore stats saved to {out_path} ({len(stats_df)} teams).")

if __name__ == "__main__":
    compute_historical_stats()
