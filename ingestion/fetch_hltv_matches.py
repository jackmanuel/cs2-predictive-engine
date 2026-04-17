import os
import sys
import json
import logging
import random
import time
import argparse
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, HLTV_MATCHES_FILE
from ingestion.hltv_client import HLTVClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Scrape HLTV matches to serve as the canonical dataset.")
    parser.add_argument("--pages", type=int, default=None, help="How many HLTV results pages to scan.")
    parser.add_argument("--matches", "--count", type=int, default=None, dest="count", help="Stop after scraping this many new matches.")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    output_data_file = HLTV_MATCHES_FILE
    
    # Load existing scraped data to skip repeats
    scraped_data = []
    processed_urls = set()
    if os.path.exists(output_data_file):
        try:
            with open(output_data_file, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
                processed_urls = {m.get("url") for m in scraped_data if m.get("url")}
            logger.info(f"Loaded {len(processed_urls)} already scraped matches.")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")

    hltv = HLTVClient()
    
    try:
        pages = args.pages
        if pages is None and args.count is None:
            pages = 3
            
        to_process = []
        
        if args.count is not None and pages is None:
            page_idx = 0
            while len(to_process) < args.count:
                logger.info(f"Fetching HLTV results page {page_idx + 1}...")
                page_results = hltv.fetch_recent_results(pages=1, start_page=page_idx)
                if not page_results:
                    logger.warning("No more results found on HLTV. Stopping pagination.")
                    break
                
                new_matches = [m for m in page_results if m.get("url") and m.get("url") not in processed_urls]
                to_process.extend(new_matches)
                logger.info(f"Found {len(new_matches)} new matches on page {page_idx + 1}. Total queued: {len(to_process)}/{args.count}")
                page_idx += 1
            
            logger.info(f"Found {len(to_process)} total new matches across {page_idx} pages.")
            to_process = to_process[:args.count]
        else:
            pages_to_fetch = pages if pages is not None else 1
            logger.info(f"Fetching {pages_to_fetch} pages of recent HLTV results...")
            hltv_results = hltv.fetch_recent_results(pages=pages_to_fetch)
            logger.info(f"Found {len(hltv_results)} matches on the results pages.")
            
            # Filter down to matches we haven't scraped details for yet
            to_process = [m for m in hltv_results if m.get("url") and m.get("url") not in processed_urls]
            
            if args.count is not None:
                 logger.info(f"Limiting scrape to {args.count} matches as requested.")
                 to_process = to_process[:args.count]

        if not to_process:
            logger.info("No new matches found to scrape.")
            return

        logger.info(f"Preparing to scrape details for {len(to_process)} matches...")
        
        consecutive_errors = 0
        
        for i, match in enumerate(to_process):
            logger.info(f"[{i+1}/{len(to_process)}] Scraping match: {match['team1']} vs {match['team2']} ({match.get('date')})")
            
            try:
                details = hltv.fetch_match_details(match['url'])
                 
                for map_obj in details['maps']:
                    s_url = map_obj.get('stats_url')
                    if s_url:
                        map_stats = hltv.fetch_map_stats(s_url)
                        map_obj['round_history'] = map_stats.get('round_history', "")
                        map_obj['player_stats'] = map_stats.get('player_stats', [])
                    else:
                        map_obj['round_history'] = ""
                        map_obj['player_stats'] = []
                 
                match['match_info'] = details.get('match_info', [])
                match['player_stats'] = details.get('player_stats', [])
                match['team_ranks'] = details.get('team_ranks', {})
                match['hltv_vetoes'] = details.get('vetoes', [])
                match['hltv_maps'] = details.get('maps', [])
                
                played_maps_count = sum(1 for m in details.get('maps', []) if m.get('team1_score') and m.get('team1_score') != '-')
                logger.info(f"  -> Extracted {played_maps_count} maps and {len(details.get('player_stats', []))} players stats.")
                
                consecutive_errors = 0
                
            except Exception as e:
                logger.error(f"  -> Failed to scrape details for {match['url']}: {e}")
                match['error'] = str(e)
                
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    logger.warning("Hit 3 consecutive errors! Restarting Selenium browser to clear memory and recover...")
                    try:
                        hltv.stop()
                    except Exception as stop_err:
                        logger.error(f"Error while trying to stop Selenium during recovery: {stop_err}")
                    
                    time.sleep(5)
                    hltv.start()
                    consecutive_errors = 0
            
            # Add to the running dataset
            scraped_data.append(match)
            processed_urls.add(match['url'])
            
            # Incremental save
            with open(output_data_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=2)
            
            # --- Anti-Scraping Logic ---
            # Every 5 matches, take a long break (human-like)
            if (i + 1) % 5 == 0 and (i + 1) < len(to_process):
                break_time = random.uniform(60.0, 120.0)
                logger.info(f"Taking a strict anti-bot breather... pausing for {break_time:.1f} seconds.")
                time.sleep(break_time)
             
        logger.info(f"Batch complete. Scraped dataset contains {len(scraped_data)} total matches.")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        hltv.stop()

if __name__ == "__main__":
    main()
