import os
import sys
import logging
import re
import json
from pathlib import Path
import pandas as pd

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR, DATA_DIR, DEFAULT_TEAM_RANK, HLTV_MATCHES_FILE, TEAM_MAPPINGS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAPPING_FILE = TEAM_MAPPINGS_FILE

def load_mappings() -> dict:
    if MAPPING_FILE.exists():
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
    return {}

def normalize_name(name: str, mappings: dict) -> str:
    if not name: return ""
    name_strip = name.strip()
    if name_strip in mappings:
        return mappings[name_strip].upper().strip()
    return name_strip.upper()

def parse_rank(rank_str: str) -> int:
    """Parses HLTV rank string like '#42' or 'Unranked' to integer."""
    if not rank_str or "unranked" in str(rank_str).lower():
        return DEFAULT_TEAM_RANK
    try:
        # Extract digits
        match = re.search(r'\d+', str(rank_str))
        if match:
            return int(match.group())
        return DEFAULT_TEAM_RANK
    except:
        return DEFAULT_TEAM_RANK

def normalize_format(fmt: str) -> str:
    """Normalises format strings like 'bo3' or 'best_of_3' or 'def'."""
    fmt = str(fmt).lower().strip()
    if "bo3" in fmt or "best of 3" in fmt: return "bo3"
    if "bo5" in fmt or "best of 5" in fmt: return "bo5"
    
    # Specifically flag default (forfeit) wins so they can be excluded
    if "def" in fmt or "default" in fmt: return "def"
    
    # Map abbreviations for BO1s on HLTV results page
    # Covers current active duty and common pool maps
    map_abbreviations = ["mrg", "anc", "inf", "nuke", "anb", "d2", "vtg", "ovp", "trn", "cbl", "cch"]
    if "bo1" in fmt or "best of 1" in fmt or fmt in map_abbreviations:
        return "bo1"
        
    return "unknown"

def process_hltv_map_data(m_data, team_a_name, team_b_name, team_a_id, team_b_id, match_id, match_date, mappings, ranks=None, match_format="unknown"):
    """Common logic to extract a map row from HLTV map object."""
    # ... (no changes here, just showing context)
    map_name = m_data.get("map_name")
    if not map_name or map_name.lower() == "tbd":
        return None
    
    # Check if this map was actually played
    s1_raw = str(m_data.get("team1_score", ""))
    s2_raw = str(m_data.get("team2_score", ""))
    
    if s1_raw == "-" or s2_raw == "-" or not s1_raw or not s2_raw:
        return None
    
    try:
        score1 = int(s1_raw)
        score2 = int(s2_raw)
    except ValueError:
        return None

    # Normalise names for comparison
    h_t1 = normalize_name(m_data.get("team1", ""), mappings)
    h_t2 = normalize_name(m_data.get("team2", ""), mappings)
    
    p_t1 = team_a_id 
    p_t2 = team_b_id
    
    is_swapped = False
    if (h_t1 == p_t2) and (h_t2 == p_t1):
        is_swapped = True
    elif (h_t1 == p_t1) and (h_t2 == p_t2):
        is_swapped = False
    else:
        # Partial match fallback
        if (h_t1 in p_t2 or p_t2 in h_t1) and (h_t2 in p_t1 or p_t1 in h_t2):
            is_swapped = True
        elif (h_t1 in p_t1 or p_t1 in h_t1) and (h_t2 in p_t2 or p_t2 in h_t2):
            is_swapped = False
        else:
            return None

    if not is_swapped:
        score_a, score_b = score1, score2
    else:
        score_a, score_b = score2, score1
    
    winner_id = team_a_id if score_a > score_b else team_b_id
    
    # Extract Ranks
    r_a_world = DEFAULT_TEAM_RANK
    r_a_vrs = DEFAULT_TEAM_RANK
    r_b_world = DEFAULT_TEAM_RANK
    r_b_vrs = DEFAULT_TEAM_RANK

    if ranks:
        for name, r_data in ranks.items():
            n = normalize_name(name, mappings)
            if n == team_a_id:
                r_a_world = parse_rank(r_data.get("world_rank"))
                r_a_vrs = parse_rank(r_data.get("vrs_before_rank"))
            elif n == team_b_id:
                r_b_world = parse_rank(r_data.get("world_rank"))
                r_b_vrs = parse_rank(r_data.get("vrs_before_rank"))

    picker_name = m_data.get("picker")
    is_team_a_picker = False
    is_team_b_picker = False
    if picker_name:
        n_picker = normalize_name(picker_name, mappings)
        if n_picker == team_a_id:
            is_team_a_picker = True
        elif n_picker == team_b_id:
            is_team_b_picker = True

    return {
        "match_id": match_id,
        "map_name": map_name,
        "date": match_date,
        "team_a_id": team_a_id,
        "team_a_name": team_a_name,
        "team_b_id": team_b_id,
        "team_b_name": team_b_name,
        "score_a": score_a,
        "score_b": score_b,
        "winner_id": winner_id,
        "team_a_picked": is_team_a_picker,
        "team_b_picked": is_team_b_picker,
        "team_a_world_rank": r_a_world,
        "team_a_vrs_rank": r_a_vrs,
        "team_b_world_rank": r_b_world,
        "team_b_vrs_rank": r_b_vrs,
        "match_format": match_format,
        "is_forfeit": map_name.lower() in ["default", "forfeit"]
    }

def load_raw_maps() -> pd.DataFrame:
    """Loads raw HLTV match JSON and explodes them into map rows."""
    all_maps = []
    mappings = load_mappings()
    
    # Pure HLTV files (Canonical Source)
    hltv_pure_path = HLTV_MATCHES_FILE
    if hltv_pure_path.exists():
        with open(hltv_pure_path, "r", encoding="utf-8") as f:
            hltv_data = json.load(f)
            
        for match in hltv_data:
            t1 = match.get("team1")
            t2 = match.get("team2")
            m_date = pd.to_datetime(match.get("date"), utc=True)
            m_id = match.get("url", "hltv_" + str(hash(t1 + t2 + str(m_date))))
            t1_id = normalize_name(t1, mappings)
            t2_id = normalize_name(t2, mappings)
            ranks = match.get("team_ranks")

            # HLTV format is already in the match object
            m_format = normalize_format(match.get("format", "unknown"))

            for m_data in match.get("hltv_maps", []):
                row = process_hltv_map_data(m_data, t1, t2, t1_id, t2_id, m_id, m_date, mappings, ranks, match_format=m_format)
                if row: all_maps.append(row)
                
    df = pd.DataFrame(all_maps)
    if df.empty: return df
    return df.sort_values("date").reset_index(drop=True)

def clean_data():
    logger.info("Starting map-level data cleaning with rank features...")
    df = load_raw_maps()
    if not df.empty:
        initial_count = len(df)
        
        # 1. Flag matches that contain a forfeited map
        forfeit_match_ids = df[df["is_forfeit"] == True]["match_id"].unique()
        df["match_has_forfeit"] = df["match_id"].isin(forfeit_match_ids)
        
        # 2. Exclude the "Default" map rows themselves as they contain no comparative gameplay data
        df = df[df["is_forfeit"] == False]
        
        # 3. Exclude matches where the ENTIRE series was marked as 'def'
        df = df[df["match_format"] != "def"]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Excluded {removed} maps from default/forfeit wins.")

        out_path = PROCESSED_DIR / "clean_maps.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Cleaned map data saved to {out_path} with {len(df)} rows.")
    else:
        logger.warning("No valid map data found in RAW_DIR.")

if __name__ == "__main__":
    clean_data()
