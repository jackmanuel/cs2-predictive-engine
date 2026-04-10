import os
import sys
import logging
import re
import json
from pathlib import Path
import pandas as pd

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    if not name: return ""
    return name.upper().strip()

def load_raw_maps() -> pd.DataFrame:
    """Loads all raw JSON match files and explodes them into individual map rows."""
    all_maps = []
    
    # We prioritize augmented files as they contain the HLTV map names
    # but we can look at all matches_*.json files.
    for file_path in RAW_DIR.glob("matches_*.json"):
        logger.info(f"Loading {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            matches_data = json.load(f)
            
        for match in matches_data:
            # Basic validation
            if match.get("status") != "finished":
                continue
                
            opponents = match.get("opponents", [])
            if not opponents or len(opponents) != 2:
                continue
                
            team_a = opponents[0].get("opponent", {})
            team_b = opponents[1].get("opponent", {})
            if not team_a or not team_b:
                continue
                
            team_a_id = team_a.get("id")
            team_b_id = team_b.get("id")
            team_a_name = team_a.get("name")
            team_b_name = team_b.get("name")
            
            match_id = match.get("id")
            match_date = pd.to_datetime(match.get("begin_at"))
            
            # Check if we have HLTV augmented data
            hltv_maps = match.get("hltv_maps", [])
            
            if not hltv_maps:
                # If no HLTV data, we can't reliably get map names like "Mirage"
                # For now, we skip these as our new model requires map-specific knowledge.
                logger.debug(f"Skipping match {match_id} (no HLTV map data)")
                continue

            for m_idx, m_data in enumerate(hltv_maps):
                map_name = m_data.get("map_name")
                if not map_name or map_name.lower() == "tbd":
                    continue
                
                # Check if this map was actually played (score exists and is not '-')
                s1_raw = str(m_data.get("team1_score", ""))
                s2_raw = str(m_data.get("team2_score", ""))
                
                if s1_raw == "-" or s2_raw == "-" or not s1_raw or not s2_raw:
                    # Map was likely vetoed or not played (common in Bo3 that ends 2-0)
                    continue
                
                try:
                    score1 = int(s1_raw)
                    score2 = int(s2_raw)
                except ValueError:
                    continue

                # Determine which HLTV team is which PandaScore team
                # HLTV team1 is m_data['team1']
                h_t1 = normalize_name(m_data.get("team1", ""))
                h_t2 = normalize_name(m_data.get("team2", ""))
                
                p_t1 = normalize_name(team_a_name)
                p_t2 = normalize_name(team_b_name)
                
                # Simple name matching logic (similar to fetch_hltv_features)
                is_swapped = False
                if (h_t1 in p_t2 or p_t2 in h_t1) and (h_t2 in p_t1 or p_t1 in h_t2):
                    is_swapped = True
                elif (h_t1 in p_t1 or p_t1 in h_t1) and (h_t2 in p_t2 or p_t2 in h_t2):
                    is_swapped = False
                else:
                    # If we can't be sure about the mapping, skip to avoid bad labels
                    logger.warning(f"Ambiguous team mapping in match {match_id}: {h_t1}/{h_t2} vs {p_t1}/{p_t2}")
                    continue

                # Assign scores to PandaScore IDs
                # If not swapped: PandaScore TeamA = HLTV Team1, PandaScore TeamB = HLTV Team2
                # If swapped: PandaScore TeamA = HLTV Team2, PandaScore TeamB = HLTV Team1
                if not is_swapped:
                    score_a = score1
                    score_b = score2
                else:
                    score_a = score2
                    score_b = score1
                
                # Winner for this specific map
                map_winner_id = team_a_id if score_a > score_b else team_b_id
                
                # Picker info
                picker_name = m_data.get("picker")
                is_team_a_picker = False
                is_team_b_picker = False
                if picker_name:
                    n_picker = normalize_name(picker_name)
                    if n_picker in p_t1 or p_t1 in n_picker:
                        is_team_a_picker = True
                    elif n_picker in p_t2 or p_t2 in n_picker:
                        is_team_b_picker = True

                row = {
                    "match_id": match_id,
                    "map_name": map_name,
                    "date": match_date,
                    "team_a_id": team_a_id,
                    "team_a_name": team_a_name,
                    "team_b_id": team_b_id,
                    "team_b_name": team_b_name,
                    "score_a": score_a,
                    "score_b": score_b,
                    "winner_id": map_winner_id,
                    "team_a_picked": is_team_a_picker,
                    "team_b_picked": is_team_b_picker,
                    "best_of": match.get("number_of_games", 1),
                    "tournament_name": match.get("tournament", {}).get("name")
                }
                all_maps.append(row)
                
    df = pd.DataFrame(all_maps)
    if df.empty:
        return df
        
    df = df.sort_values("date").reset_index(drop=True)
    return df

def clean_data():
    """Pipeline to clean raw augmented matches and save map-level Parquet."""
    logger.info("Starting map-level data cleaning...")
    df = load_raw_maps()
    
    if not df.empty:
        out_path = PROCESSED_DIR / "clean_maps.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Cleaned map data saved to {out_path} with {len(df)} rows.")
    else:
        logger.warning("No valid map data found. Ensure you have run fetch_hltv_features.py first.")

if __name__ == "__main__":
    clean_data()
