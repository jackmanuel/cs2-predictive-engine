import json
import logging
from pathlib import Path
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

def load_raw_matches() -> pd.DataFrame:
    """Loads all raw JSON match files and flattens them into a pandas DataFrame."""
    all_matches = []
    
    # Iterate over all JSON files in raw dir
    for file_path in RAW_DIR.glob("matches_*.json"):
        logger.info(f"Loading {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            matches_data = json.load(f)
            
        for match in matches_data:
            # We only care about finished matches with exactly 2 opponents that are complete
            if match.get("status") != "finished":
                continue
                
            opponents = match.get("opponents", [])
            # Some matches are missing opponent data or aren't standard 1v1 teams (e.g. FFA)
            if not opponents or len(opponents) != 2:
                continue
                
            # Usually index 0 is Team A, index 1 is Team B
            team_a = opponents[0].get("opponent", {})
            team_b = opponents[1].get("opponent", {})
            
            if not team_a or not team_b:
                continue
                
            team_a_id = team_a.get("id")
            team_b_id = team_b.get("id")
            
            winner_id = match.get("winner_id")
            if winner_id is None:
                continue
                
            # Extract scores
            score_a = 0
            score_b = 0
            for result in match.get("results", []):
                if result.get("team_id") == team_a_id:
                    score_a = result.get("score", 0)
                elif result.get("team_id") == team_b_id:
                    score_b = result.get("score", 0)
            
            row = {
                "match_id": match.get("id"),
                "date": pd.to_datetime(match.get("begin_at")),
                "team_a_id": team_a_id,
                "team_a_name": team_a.get("name"),
                "team_b_id": team_b_id,
                "team_b_name": team_b.get("name"),
                "winner_id": winner_id,
                "score_a": score_a,
                "score_b": score_b,
                "best_of": match.get("number_of_games", 1),
                "tournament_name": match.get("tournament", {}).get("name"),
                "serie_name": match.get("serie", {}).get("name")
            }
            all_matches.append(row)
            
    df = pd.DataFrame(all_matches)
    
    if df.empty:
        logger.warning("No valid matches found to clean.")
        return df
        
    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)
    return df

def clean_data():
    """Pipeline to clean raw matches and save as Parquet."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    logger.info("Starting data cleaning...")
    df = load_raw_matches()
    
    if not df.empty:
        out_path = PROCESSED_DIR / "clean_matches.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Cleaned data saved to {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    clean_data()
