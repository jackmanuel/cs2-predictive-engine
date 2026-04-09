import os
import sys
import json
import logging
import difflib
import argparse
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.hltv_client import HLTVClient

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MAPPING_FILE = "data/team_mappings.json"
PANDASCORE_DATA = "data/raw/matches_20260409.json"

def load_mappings():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_mappings(mappings):
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Interactive team name mapping utility.")
    parser.add_argument("--min-count", type=int, default=2, help="Only map teams that appear at least X times in the raw data.")
    parser.add_argument("--limit", type=int, default=100, help="Limit the mapping session to the top X unmapped teams.")
    args = parser.parse_args()

    if not os.path.exists(PANDASCORE_DATA):
        print(f"Error: Could not find {PANDASCORE_DATA}")
        return

    mappings = load_mappings()
    with open(PANDASCORE_DATA, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    # Collect frequency of all unique team names from PandaScore that aren't mapped
    team_counts = {}
    for m in matches:
        for opp in m.get('opponents', []):
            name = opp['opponent']['name']
            if name not in mappings:
                team_counts[name] = team_counts.get(name, 0) + 1

    # Filter by min-count
    filtered_teams = {name: count for name, count in team_counts.items() if count >= args.min_count}
    
    if not filtered_teams:
        print(f"No unmapped teams found with at least {args.min_count} matches.")
        return

    # Sort by frequency (most frequent first)
    sorted_teams = sorted(filtered_teams.items(), key=lambda x: x[1], reverse=True)
    
    # Apply limit
    to_map = sorted_teams[:args.limit]

    print(f"Prioritizing {len(to_map)} unmapped teams (appearing {args.min_count}+ times).")
    print("Fetching current HLTV results to find candidates...")
    
    hltv = HLTVClient()
    try:
        # Fetch 5 pages to have a wide corpus of HLTV names
        hltv_matches = hltv.fetch_recent_results(pages=5)
        hltv_names = set()
        for hm in hltv_matches:
            hltv_names.add(hm['team1'])
            hltv_names.add(hm['team2'])
        
        hltv_names_list = sorted(list(hltv_names))
        print(f"Fetched {len(hltv_names_list)} unique team names from HLTV results.\n")

        # Start interactive loop
        new_mappings = 0
        for team, count in to_map:
            print("-" * 50)
            print(f"PANDASCORE TEAM: {team} ({count} occurrences)")
            
            # Find fuzzy matches
            choices = difflib.get_close_matches(team, hltv_names_list, n=5, cutoff=0.5)
            
            if not choices:
                print("No close matches found on HLTV.")
                print("Options: [s]kip, [m]anual entry, [q]uit")
            else:
                print("HLTV CANDIDATES:")
                for i, choice in enumerate(choices):
                    print(f"  [{i+1}] {choice}")
                print(f"Options: [1-{len(choices)}] to select, [s]kip, [m]anual entry, [q]uit")

            cmd = input("> ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 's' or cmd == '':
                continue
            elif cmd == 'm':
                manual = input("Enter exact HLTV team name: ").strip()
                if manual:
                    mappings[team] = manual
                    new_mappings += 1
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(choices):
                    mappings[team] = choices[idx]
                    new_mappings += 1
                    print(f"Mapped: {team} -> {choices[idx]}")
                else:
                    print("Invalid choice.")
            
            # Save incrementally
            if new_mappings % 5 == 0 and new_mappings > 0:
                save_mappings(mappings)

        save_mappings(mappings)
        print(f"\nDone! Added {new_mappings} new mappings to {MAPPING_FILE}")

    finally:
        hltv.stop()

if __name__ == "__main__":
    main()
