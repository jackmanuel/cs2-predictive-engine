import os
import sys
import logging
import re
import json
from typing import List
import pandas as pd

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, DEFAULT_TEAM_RANK, HLTV_MATCHES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SHOWMATCH_KEYWORDS = [
    "showmatch",
    "show match",
    "all-star",
    "all star",
    "charity match",
    "streamer",
    "content creator",
    "wingman",
    "bombsite a only",
    "bombsite b only",
    "1v1",
    "2v2",
    "3v3",
    "4v4",
    "1 vs 1",
    "2 vs 2",
    "3 vs 3",
    "4 vs 4"
]

MAX_STANDARD_PLAYERS = 10

SUBSTITUTION_KEYWORDS = [
    "substitute",
    "substitutes",
    "substituted",
    "stand-in",
    "stand in",
]

TEAM_REPLACEMENT_KEYWORDS = [
    "withdraw",
    "withdrawn",
    "disqualified",
    "replaced by",
    "replaces",
]

MID_MAP_SUB_PATTERNS = [
    r"\bduring map\b",
    r"\bmid[- ]?map\b",
    r"\bafter round\b",
    r"\bafter the \d+(st|nd|rd|th)? round\b",
    r"\bround interval\b",
    r"\bfor the remainder\b",
]

MAP_SCOPE_PATTERNS = [
    r"\bon the \d+(st|nd|rd|th)? map\b",
    r"\bmap \d+\b",
    r"\b\d+(st|nd|rd|th)? map\b",
    r"\b\d+(st|nd|rd|th)? and \d+(st|nd|rd|th)? map\b",
]

SUBSTITUTION_REASON_KEYWORDS = [
    "visa",
    "illness",
    "medical",
    "sick",
    "emergency",
    "technical",
    "power outage",
]


def normalize_name(name: str) -> str:
    """Normalizes a team name to a consistent uppercase format for matching."""
    if not name: return ""
    return name.strip().upper()

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
    """Normalizes format strings like 'bo3' or 'best_of_3' or 'def'."""
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

def detect_is_lan(match_info: List[str]) -> bool:
    """Detects whether a match was played on LAN from the match_info blurbs.
    
    HLTV formats this as 'Best of N (LAN)' or 'Best of N (Online)' in the
    first element of the match_info list.
    """
    if not match_info:
        return False
    for line in match_info:
        if re.search(r'\(lan\)', line, re.IGNORECASE):
            return True
        if re.search(r'\(online\)', line, re.IGNORECASE):
            return False
    return False

def get_showmatch_exclusion_reason(match: dict) -> str | None:
    """Returns the keyword that marks a match as non-standard/showmatch data."""
    haystack = get_match_text_haystack(match)
    for keyword in SHOWMATCH_KEYWORDS:
        if keyword in haystack:
            return keyword

    return None

def get_match_text_haystack(match: dict) -> str:
    """Builds a lower-case searchable text blob from scraped match metadata."""
    text_parts = [
        match.get("event", ""),
        match.get("format", ""),
        " ".join(str(line) for line in match.get("match_info", [])),
        " ".join(str(line) for line in match.get("hltv_vetoes", [])),
    ]

    for map_data in match.get("hltv_maps", []):
        text_parts.extend([
            map_data.get("map_name", ""),
            map_data.get("picker", ""),
        ])

    return " ".join(part for part in text_parts if part).lower()

def get_match_info_lines(match: dict) -> list[str]:
    """Returns searchable match-info lines as strings."""
    return [str(line).strip() for line in match.get("match_info", []) if str(line).strip()]

def get_roster_status_flags(match: dict) -> dict:
    """Extracts roster/anomaly flags from HLTV match blurbs.

    These are metadata-only flags. They do not currently exclude matches from
    training or alter model features.
    """
    lines = get_match_info_lines(match)
    substitution_lines = []
    team_replacement_lines = []
    reason_terms = set()

    has_substitution_note = False
    has_coach_standin = False
    has_mid_map_sub = False
    has_map_specific_sub = False
    has_team_replacement = False

    for line in lines:
        low = line.lower()

        if any(keyword in low for keyword in SUBSTITUTION_KEYWORDS):
            has_substitution_note = True
            substitution_lines.append(line)

            if "(coach)" in low or "(coach" in low or " coach)" in low:
                has_coach_standin = True

            if any(re.search(pattern, low) for pattern in MID_MAP_SUB_PATTERNS):
                has_mid_map_sub = True

            if any(re.search(pattern, low) for pattern in MAP_SCOPE_PATTERNS):
                has_map_specific_sub = True

            for keyword in SUBSTITUTION_REASON_KEYWORDS:
                if keyword in low:
                    reason_terms.add(keyword)

        if any(keyword in low for keyword in TEAM_REPLACEMENT_KEYWORDS):
            has_team_replacement = True
            team_replacement_lines.append(line)

    roster_status = "standard"
    if has_team_replacement:
        roster_status = "team_replacement"
    elif has_mid_map_sub:
        roster_status = "mid_map_substitution"
    elif has_coach_standin:
        roster_status = "coach_standin"
    elif has_substitution_note:
        roster_status = "substitution"

    return {
        "has_substitution_note": has_substitution_note,
        "has_coach_standin": has_coach_standin,
        "has_mid_map_sub": has_mid_map_sub,
        "has_map_specific_sub": has_map_specific_sub,
        "has_team_replacement": has_team_replacement,
        "substitution_reason_terms": ",".join(sorted(reason_terms)),
        "substitution_notes": " | ".join(substitution_lines),
        "team_replacement_notes": " | ".join(team_replacement_lines),
        "roster_status": roster_status,
    }

def get_invalid_veto_exclusion_reason(match: dict) -> str | None:
    """Returns a reason when the match description says veto data is invalid."""
    haystack = get_match_text_haystack(match)
    if "randomized" in haystack:
        return "randomized"

    if "failed to show up" in haystack and "veto process" in haystack:
        return "failed to show up for the veto process"

    return None

def normalize_player_name(player_name: str) -> str:
    """Normalizes a player name for roster-size checks."""
    return str(player_name or "").strip().lower()

def get_nonstandard_roster_exclusion_reason(match: dict) -> str | None:
    """Returns a reason when map stats indicate more than five players per team."""
    series_players = set()

    for map_data in match.get("hltv_maps", []):
        player_stats = map_data.get("player_stats", [])
        if not player_stats:
            continue

        map_players = {
            normalize_player_name(player.get("player", ""))
            for player in player_stats
            if normalize_player_name(player.get("player", ""))
        }

        player_count = len(map_players) if map_players else len(player_stats)
        map_name = map_data.get("map_name", "unknown map")
        if player_count > MAX_STANDARD_PLAYERS:
            return f"{player_count} players recorded on {map_name}"

        series_players.update(map_players)

    if len(series_players) > MAX_STANDARD_PLAYERS:
        return f"{len(series_players)} unique players recorded across the series"

    return None

def process_hltv_map_data(m_data, team_a_name, team_b_name, team_a_id, team_b_id, match_id, match_date, ranks=None, match_format="unknown", is_lan=False, match_flags=None):
    """Common logic to extract a map row from HLTV map object."""
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

    # Normalize names for comparison
    h_t1 = normalize_name(m_data.get("team1", ""))
    h_t2 = normalize_name(m_data.get("team2", ""))
    
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
            n = normalize_name(name)
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
        n_picker = normalize_name(picker_name)
        if n_picker == team_a_id:
            is_team_a_picker = True
        elif n_picker == team_b_id:
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
        "winner_id": winner_id,
        "team_a_picked": is_team_a_picker,
        "team_b_picked": is_team_b_picker,
        "team_a_world_rank": r_a_world,
        "team_a_vrs_rank": r_a_vrs,
        "team_b_world_rank": r_b_world,
        "team_b_vrs_rank": r_b_vrs,
        "match_format": match_format,
        "is_forfeit": map_name.lower() in ["default", "forfeit"],
        "is_lan": is_lan
    }

    if match_flags:
        row.update(match_flags)

    return row

def load_raw_maps() -> pd.DataFrame:
    """Loads raw HLTV match JSON and explodes them into map rows."""
    all_maps = []
    excluded_showmatches = 0
    excluded_invalid_vetoes = 0
    excluded_nonstandard_rosters = 0
    excluded_roster_anomalies = 0
    
    # Pure HLTV files (Canonical Source)
    hltv_pure_path = HLTV_MATCHES_FILE
    if hltv_pure_path.exists():
        with open(hltv_pure_path, "r", encoding="utf-8") as f:
            hltv_data = json.load(f)
            
        for match in hltv_data:
            exclusion_reason = get_showmatch_exclusion_reason(match)
            if exclusion_reason:
                excluded_showmatches += 1
                logger.debug(
                    "Excluded non-standard/showmatch match due to keyword '%s': %s (%s)",
                    exclusion_reason,
                    match.get("event", "Unknown Event"),
                    match.get("url", "no url"),
                )
                continue

            invalid_veto_reason = get_invalid_veto_exclusion_reason(match)
            if invalid_veto_reason:
                excluded_invalid_vetoes += 1
                logger.debug(
                    "Excluded match due to invalid/randomized veto data (%s): %s (%s)",
                    invalid_veto_reason,
                    match.get("event", "Unknown Event"),
                    match.get("url", "no url"),
                )
                continue

            match_flags = get_roster_status_flags(match)
            if match_flags["roster_status"] != "standard":
                excluded_roster_anomalies += 1
                logger.debug(
                    "Excluded match due to roster anomaly (%s): %s (%s)",
                    match_flags["roster_status"],
                    match.get("event", "Unknown Event"),
                    match.get("url", "no url"),
                )
                continue

            roster_exclusion_reason = get_nonstandard_roster_exclusion_reason(match)
            if roster_exclusion_reason:
                excluded_nonstandard_rosters += 1
                logger.debug(
                    "Excluded match due to non-standard roster (%s): %s (%s)",
                    roster_exclusion_reason,
                    match.get("event", "Unknown Event"),
                    match.get("url", "no url"),
                )
                continue

            t1 = match.get("team1")
            t2 = match.get("team2")
            m_date = pd.to_datetime(match.get("date"), utc=True)
            m_id = match.get("url", "hltv_" + str(hash(t1 + t2 + str(m_date))))
            t1_id = normalize_name(t1)
            t2_id = normalize_name(t2)
            ranks = match.get("team_ranks")

            # HLTV format is already in the match object
            m_format = normalize_format(match.get("format", "unknown"))

            # Detect LAN vs Online from match_info blurbs
            is_lan = detect_is_lan(match.get("match_info", []))

            for m_data in match.get("hltv_maps", []):
                row = process_hltv_map_data(
                    m_data,
                    t1,
                    t2,
                    t1_id,
                    t2_id,
                    m_id,
                    m_date,
                    ranks,
                    match_format=m_format,
                    is_lan=is_lan,
                    match_flags=match_flags,
                )
                if row: all_maps.append(row)

        if excluded_showmatches > 0:
            logger.info(f"Excluded {excluded_showmatches} non-standard/showmatch matches before map cleaning.")
        if excluded_invalid_vetoes > 0:
            logger.info(f"Excluded {excluded_invalid_vetoes} matches with invalid/randomized veto data before map cleaning.")
        if excluded_nonstandard_rosters > 0:
            logger.info(f"Excluded {excluded_nonstandard_rosters} matches with non-standard rosters before map cleaning.")
        if excluded_roster_anomalies > 0:
            logger.info(f"Excluded {excluded_roster_anomalies} matches with roster anomalies before map cleaning.")
                
    df = pd.DataFrame(all_maps)
    if df.empty: return df
    return df.sort_values("date").reset_index(drop=True)


def build_clean_maps(raw_maps: pd.DataFrame = None) -> pd.DataFrame:
    """Loads raw maps and applies the same map-level exclusions used for training."""
    df = load_raw_maps() if raw_maps is None else raw_maps.copy()
    if not df.empty:
        # 1. Flag matches that contain a forfeited map
        forfeit_match_ids = df[df["is_forfeit"] == True]["match_id"].unique()
        df["match_has_forfeit"] = df["match_id"].isin(forfeit_match_ids)
        
        # 2. Exclude the "Default" map rows themselves as they contain no comparative gameplay data
        df = df[df["is_forfeit"] == False]
        
        # 3. Exclude matches where the ENTIRE series was marked as 'def'
        df = df[df["match_format"] != "def"]

        df = df.reset_index(drop=True)

    return df


def clean_data():
    logger.info("Starting map-level data cleaning with rank features...")
    initial_df = load_raw_maps()
    df = build_clean_maps(initial_df)
    if not df.empty:
        removed = len(initial_df) - len(df)
        if removed > 0:
            logger.info(f"Excluded {removed} maps from default/forfeit wins.")

        out_path = PROCESSED_DIR / "clean_maps.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Cleaned map data saved to {out_path} with {len(df)} rows.")
    else:
        logger.warning("No valid map data found in RAW_DIR.")

if __name__ == "__main__":
    clean_data()
