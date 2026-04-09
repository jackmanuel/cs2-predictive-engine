import os
import sys
import json
import logging
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.hltv_client import HLTVClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Hardcoded mapping from PandaScore full names to HLTV abbreviations
TEAM_NAME_MAPPING = {
    "Natus Vincere": "NAVI",
    "mousesports": "MOUZ",
    "Ninjas in Pyjamas": "NIP",
    "Virtus.pro": "VP",
    "Eternal Fire": "EF",
    "BIG": "BIG",
    "FUT esports": "FUT",
    "FUT Academy": "FUT.A",
    "G2 Esports": "G2",
    "Team Liquid": "Liquid",
    "Team Vitality": "Vitality",
    "FaZe Clan": "FaZe",
    "Astralis": "Astralis",
    "Complexity Gaming": "Complexity",
    "ENCE": "ENCE",
    "Cloud9": "Cloud9",
    "Heroic": "Heroic",
    "Wildcard": "Wildcard",
    "PARIVISION": "PARIVISION",
    "Imperial": "Imperial",
    "Imperial Academy": "Imperial.A",
    "Misa Esports": "MISA",
    "REGAIN": "REGAIN",
    "InControl": "InC"
}

def normalize_name(name: str) -> str:
    """Normalizes team names using the dictionary, or defaults to upper case acronym comparison."""
    if not name:
        return ""
    if name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[name].upper()
    return name.upper().strip()

def match_games(pandascore_match: Dict, hltv_results: List[Dict]) -> str:
    """
    Attempts to cross-reference a PandaScore match against the pulled HLTV results list.
    Returns the HLTV 'url' if matched, else None.
    """
    if "opponents" not in pandascore_match or len(pandascore_match["opponents"]) < 2:
        return None
        
    p_team1 = pandascore_match["opponents"][0]["opponent"]["name"]
    p_team2 = pandascore_match["opponents"][1]["opponent"]["name"]
    
    n_p1 = normalize_name(p_team1)
    n_p2 = normalize_name(p_team2)
    
    for hltv in hltv_results:
        n_h1 = normalize_name(hltv['team1'])
        n_h2 = normalize_name(hltv['team2'])
        
        # Check permutations (Team A vs Team B) or (Team B vs Team A)
        if (n_p1 in n_h1 or n_h1 in n_p1) and (n_p2 in n_h2 or n_h2 in n_p2):
             return hltv['url']
        if (n_p1 in n_h2 or n_h2 in n_p1) and (n_p2 in n_h1 or n_h1 in n_p2):
             return hltv['url']
             
    return None

def main():
    target_data_file = "data/raw/matches_20260409.json"
    output_data_file = "data/raw/matches_20260409_augmented.json"
    
    if not os.path.exists(target_data_file):
        logger.error(f"Cannot find data file {target_data_file}")
        return
        
    with open(target_data_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)
        
    # The user noted newest matches are at the bottom. We'll reverse the list and slice the last 5.
    matches_to_process = matches[::-1][:5]
    
    hltv = HLTVClient()
    
    try:
        logger.info("Fetching 1 page of recent HLTV results for mapping...")
        hltv_results = hltv.fetch_recent_results(pages=1)
        logger.info(f"Loaded {len(hltv_results)} historical HLTV matches.")
        
        for i, match in enumerate(matches_to_process):
            p_name = match.get("name", "Unknown Match")
            logger.info(f"[{i+1}/{len(matches_to_process)}] Processing PandaScore Match: {p_name}")
            
            hltv_url = match_games(match, hltv_results)
            
            if hltv_url:
                 logger.info(f"  -> Successfully mapped to HLTV: {hltv_url}")
                 details = hltv.fetch_match_details(hltv_url)
                 
                 # Optional: Deep scrape round history for each map
                 # To be safe and respectful of the user's preference for speed, we'll do it by default here 
                 # as we are only processing a small batch.
                 for map_obj in details['maps']:
                     s_url = map_obj.get('stats_url')
                     if s_url:
                         map_obj['round_history'] = hltv.fetch_round_history(s_url)
                     else:
                         map_obj['round_history'] = []
                 
                 # Append straight into the data structure
                 match['hltv_vetoes'] = details['vetoes']
                 match['hltv_maps'] = details['maps']
                 match['hltv_url'] = hltv_url
                 
                 logger.info(f"  -> Extracted {len(details['vetoes'])} veto records and {len(details['maps'])} map results.")
            else:
                 logger.info("  -> No HLTV map found in the recent results page.")
                 match['hltv_vetoes'] = []
                 match['hltv_maps'] = []
                 match['hltv_url'] = None
            
            # Save augmented data after each match
            with open(output_data_file, 'w', encoding='utf-8') as f:
                json.dump(matches_to_process, f, indent=2)
            logger.debug("Incremental save complete.")
             
        logger.info(f"Augmented dataset saved to {output_data_file}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        hltv.stop()

if __name__ == "__main__":
    main()
