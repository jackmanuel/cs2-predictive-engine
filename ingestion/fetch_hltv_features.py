import os
import sys
import json
import logging
import random
import time
import argparse
from typing import List, Dict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.hltv_client import HLTVClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAPPING_FILE = "data/team_mappings.json"

def load_mappings() -> Dict[str, str]:
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
    return {}

def normalize_name(name: str, mappings: Dict[str, str]) -> str:
    """Normalizes team names using the loaded mappings or defaults to upper case."""
    if not name:
        return ""
    if name in mappings:
        return mappings[name].upper()
    return name.upper().strip()

def match_games(pandascore_match: Dict, hltv_results: List[Dict], mappings: Dict[str, str]) -> str:
    """
    Attempts to cross-reference a PandaScore match against the pulled HLTV results list.
    Enforces team name equivalence and a +/- 24h date window.
    """
    if "opponents" not in pandascore_match or len(pandascore_match["opponents"]) < 2:
        return None
        
    p_team1 = pandascore_match["opponents"][0]["opponent"]["name"]
    p_team2 = pandascore_match["opponents"][1]["opponent"]["name"]
    # PandaScore timestamp: "2026-04-09T08:00:00Z"
    p_date_str = pandascore_match.get("begin_at", "").split("T")[0]
    
    n_p1 = normalize_name(p_team1, mappings)
    n_p2 = normalize_name(p_team2, mappings)
    
    for hltv in hltv_results:
        n_h1 = hltv['team1'].upper().strip()
        n_h2 = hltv['team2'].upper().strip()
        h_date = hltv.get('date') # "YYYY-MM-DD"
        
        # Team match
        if (n_p1 == n_h1 and n_p2 == n_h2) or (n_p1 == n_h2 and n_p2 == n_h1):
             # Date match (+/- 1 day to account for timezone shifts between APIs)
             if p_date_str and h_date:
                 p_dt = datetime.strptime(p_date_str, "%Y-%m-%d")
                 h_dt = datetime.strptime(h_date, "%Y-%m-%d")
                 diff_days = abs((p_dt - h_dt).days)
                 if diff_days <= 1:
                     return hltv['url']
             
    return None

def main():
    parser = argparse.ArgumentParser(description="Bulk augment PandaScore matches with HLTV data.")
    parser.add_argument("--count", type=int, default=50, help="Number of new matches to process in this run.")
    parser.add_argument("--pages", type=int, default=3, help="How many HLTV results pages to scan for mappings (1 page = 100 matches).")
    args = parser.parse_args()

    target_data_file = "data/raw/matches_20260409.json"
    output_data_file = "data/raw/matches_20260409_augmented.json"
    
    if not os.path.exists(target_data_file):
        logger.error(f"Cannot find data file {target_data_file}")
        return
        
    with open(target_data_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    # Load existing augmented data to skip repeats
    augmented_data = []
    processed_ids = set()
    if os.path.exists(output_data_file):
        with open(output_data_file, 'r', encoding='utf-8') as f:
            augmented_data = json.load(f)
            processed_ids = {m.get("id") for m in augmented_data}
        logger.info(f"Loaded {len(processed_ids)} already augmented matches.")

    # Sort matches oldest-to-newest and filter for un-processed ones
    # We slice based on the --count argument
    to_process = [m for m in matches if m.get("id") not in processed_ids]
    # We'll take the newest ones first for our batch
    to_process = to_process[::-1][:args.count]

    if not to_process:
        logger.info("No new matches found to augment. Training set is already up to date.")
        return

    logger.info(f"Preparing to augment {len(to_process)} matches...")
    
    mappings = load_mappings()
    
    hltv = HLTVClient()
    
    try:
        logger.info(f"Fetching {args.pages} pages of recent HLTV results for mapping...")
        hltv_results = hltv.fetch_recent_results(pages=args.pages)
        logger.info(f"Loaded repository of {len(hltv_results)} historical HLTV matches.")
        
        for i, match in enumerate(to_process):
            p_name = match.get("name", "Unknown Match")
            logger.info(f"[{i+1}/{len(to_process)}] Processing PandaScore Match: {p_name}")
            
            hltv_url = match_games(match, hltv_results, mappings)
            
            if hltv_url:
                 logger.info(f"  -> Successfully mapped to HLTV: {hltv_url}")
                 details = hltv.fetch_match_details(hltv_url)
                 
                 for map_obj in details['maps']:
                     s_url = map_obj.get('stats_url')
                     if s_url:
                         map_obj['round_history'] = hltv.fetch_round_history(s_url)
                     else:
                         map_obj['round_history'] = []
                 
                 match['hltv_vetoes'] = details['vetoes']
                 match['hltv_maps'] = details['maps']
                 match['hltv_url'] = hltv_url
                 logger.info(f"  -> Extracted {len(details['maps'])} map results.")
            else:
                 logger.info("  -> No HLTV map found.")
                 # Mark as attempted so we don't try again immediately
                 match['hltv_url'] = None
                 match['hltv_maps'] = []
            
            # Add to the running dataset
            augmented_data.append(match)
            
            # Incremental save
            with open(output_data_file, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, indent=2)
            
            # --- Anti-Scraping Logic ---
            # Every 10 matches, take a long break (human-like)
            if (i + 1) % 10 == 0 and (i + 1) < len(to_process):
                break_time = random.uniform(60.0, 120.0)
                logger.info(f"Take a breather... pausing for {break_time:.1f} seconds to avoid rate detection.")
                time.sleep(break_time)
             
        logger.info(f"Batch complete. Augmented dataset contains {len(augmented_data)} total matches.")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        hltv.stop()

if __name__ == "__main__":
    main()
