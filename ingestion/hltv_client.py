import time
import random
import re
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

logger = logging.getLogger(__name__)

class HLTVClient:
    """
    Client for interacting with HLTV via non-headless Selenium.
    Bypasses Cloudflare protections.
    """
    def __init__(self):
        self.driver = None

    def start(self):
        """Starts the Chrome instance."""
        if not self.driver:
            logger.info("Initializing Selenium Browser (Non-headless)...")
            options = uc.ChromeOptions()
            options.add_argument('--window-size=1024,768')
            # Using version_main=146 to avoid version mismatch issues
            self.driver = uc.Chrome(headless=False, version_main=146, options=options)
            
    def stop(self):
        """Stops the Chrome instance."""
        if self.driver:
            logger.info("Closing Selenium Browser...")
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Error closing driver: {e}")
            self.driver = None

    def _wait_for_cloudflare(self):
        """Simple sleep to ensure CF Javascript challenge completes."""
        # 5 to 8 seconds delay ensures a human-like pause and gives CF time
        sleep_time = random.uniform(5.5, 8.5)
        logger.debug(f"Waiting {sleep_time:.1f}s for page load/Cloudflare...")
        time.sleep(sleep_time)

    def fetch_recent_results(self, pages: int = 1) -> List[Dict]:
        """
        Fetches the recent match results from HLTV's /results page.
        Returns a list of match dictionaries containing URL, team names, and date.
        """
        if not self.driver:
            self.start()
            
        results = []
        base_url = "https://www.hltv.org/results"
        
        for p in range(pages):
            offset = p * 100
            url = f"{base_url}?offset={offset}" if offset > 0 else base_url
            
            logger.info(f"Navigating to {url}")
            self.driver.get(url)
            self._wait_for_cloudflare()
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # The results are grouped by date sublists
            all_sublists = soup.find_all('div', class_='results-sublist')
            for sublist in all_sublists:
                # The date is in a standard-box subtitle
                # Format: "Matches for February 5th 2026"
                date_el = sublist.find('div', class_='standard-box subtitle')
                if not date_el:
                    continue
                
                raw_date = date_el.text.replace("Matches for ", "").strip()
                # Remove ordinals (st, nd, rd, th) for easier parsing
                clean_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', raw_date)
                
                # Convert to YYYY-MM-DD for standard comparison
                try:
                    dt_obj = time.strptime(clean_date, "%B %d %Y")
                    match_date_str = time.strftime("%Y-%m-%d", dt_obj)
                except Exception as e:
                    logger.warning(f"Failed to parse HLTV date '{raw_date}': {e}")
                    match_date_str = None
                
                # Extract matches for this date
                result_links = sublist.find_all('a', class_='a-reset')
                for a in result_links:
                    href = a.get('href', '')
                    if '/matches/' in href:
                        match_url = "https://www.hltv.org" + href
                        
                        t1_div = a.find('div', class_='team1')
                        t2_div = a.find('div', class_='team2')
                        
                        t1_name = t1_div.text.strip() if t1_div else "Unknown"
                        t2_name = t2_div.text.strip() if t2_div else "Unknown"
                        
                        results.append({
                            'team1': t1_name,
                            'team2': t2_name,
                            'url': match_url,
                            'date': match_date_str
                        })
                    
        return results

    def fetch_match_details(self, match_url: str) -> Dict:
        """
        Navigates to a match URL and extracts the veto box text and map results.
        Returns a dictionary with 'vetoes' (list) and 'maps' (list of dicts).
        """
        if not self.driver:
             self.start()
             
        logger.info(f"Fetching details for: {match_url}")
        self.driver.get(match_url)
        self._wait_for_cloudflare()
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # 1. Extract Vetoes
        # 1. Extract Vetoes
        veto_boxes = soup.find_all('div', class_='veto-box')
        vetoes = []
        for box in veto_boxes:
            potential_lines = []
            div_lines = box.find_all('div')
            if div_lines:
                for dl in div_lines:
                    potential_lines.extend(dl.text.split('\n'))
            else:
                potential_lines = box.text.split('\n')
                
            for line in potential_lines:
                line = line.strip()
                if line and any(kw in line.lower() for kw in ["removed", "picked", "left over"]):
                    # Deduplicate: don't add if it's already there OR if it's just a subset/superset
                    if line not in vetoes:
                        vetoes.append(line)
        
        if not vetoes:
             logger.warning(f"No veto records extracted from {match_url}.")
        else:
             logger.info(f"  -> Successfully extracted {len(vetoes)} veto records.")
             
        # 2. Extract Map Scores
        maps_data = []
        mapholders = soup.find_all('div', class_='mapholder')
        for m in mapholders:
            map_name_el = m.find('div', class_='mapname')
            if not map_name_el:
                continue
            map_name = map_name_el.text.strip()
            
            containers = m.find_all('div', class_='results-teamname-container')
            if len(containers) >= 2:
                left_team_el = containers[0]
                right_team_el = containers[1]
                
                left_name_div = left_team_el.find('div', class_='results-teamname')
                left_score_div = left_team_el.find('div', class_='results-team-score')
                right_name_div = right_team_el.find('div', class_='results-teamname')
                right_score_div = right_team_el.find('div', class_='results-team-score')
                
                left_team = left_name_div.text.strip() if left_name_div else "Team1"
                left_score = left_score_div.text.strip() if left_score_div else ""
                right_team = right_name_div.text.strip() if right_name_div else "Team2"
                right_score = right_score_div.text.strip() if right_score_div else ""
                
                half_score_el = m.find('div', class_='results-center-half-score')
                halves = []
                if half_score_el:
                    spans = half_score_el.find_all('span')
                    # Keep track of side classes. Typical order: ['t'/'ct', 'ct'/'t', 'ct'/'t', 't'/'ct']
                    # We just extract the numbers and their assigned side.
                    score_items = []
                    for span in spans:
                        c = span.get('class', [])
                        side = ''
                        if 't' in c: side = 'T'
                        if 'ct' in c: side = 'CT'
                        if side and span.text.isdigit():
                            score_items.append({"score": int(span.text), "side": side})
                            
                    # Combine into half blocks if we have 4 scores (2 halves) or 6 (OT included)
                    for i in range(0, len(score_items), 2):
                        if i+1 < len(score_items):
                            halves.append({
                                "team1_score": score_items[i]["score"],
                                "team1_side": score_items[i]["side"],
                                "team2_score": score_items[i+1]["score"],
                                "team2_side": score_items[i+1]["side"]
                            })
                            
                # Get Stats URL for this map
                stats_link_el = m.find('a', class_='results-stats')
                stats_url = "https://www.hltv.org" + stats_link_el.get('href') if stats_link_el else None
                
                # Determine who picked this map from vetoes
                picker = None
                for v in vetoes:
                    if "picked" in v.lower() and map_name.lower() in v.lower():
                        # Example: "1. Wildcard picked Mirage"
                        raw_picker = v.split("picked")[0].strip()
                        # Clean leading number "1. " and strip any extra characters
                        picker = re.sub(r'^\d+\.\s*', '', raw_picker).strip()
                
                maps_data.append({
                    "map_name": map_name,
                    "team1": left_team,
                    "team1_score": left_score,
                    "team2": right_team,
                    "team2_score": right_score,
                    "halves": halves,
                    "picker": picker,
                    "stats_url": stats_url
                })

        # Add an extra random delay to prevent rate-limiting between successive match queries
        time.sleep(random.uniform(2.0, 4.0))
        return {
            "vetoes": vetoes,
            "maps": maps_data
        }

    def fetch_round_history(self, stats_url: str) -> List[Dict]:
        """
        Navigates to a map stats page and extracts the round-by-round history.
        """
        if not self.driver or not stats_url:
             return []
             
        logger.info(f"Fetching round history from: {stats_url}")
        self.driver.get(stats_url)
        self._wait_for_cloudflare()
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        rounds = []
        rh_con = soup.find('div', class_='round-history-con')
        if rh_con:
            team_rows = rh_con.find_all('div', class_='round-history-team-row')
            if len(team_rows) >= 2:
                # Get all outcomes for both teams
                t1_outcomes = team_rows[0].find_all('img', class_='round-history-outcome')
                t2_outcomes = team_rows[1].find_all('img', class_='round-history-outcome')
                
                num_rounds = max(len(t1_outcomes), len(t2_outcomes))
                for i in range(num_rounds):
                    winner_team = None
                    winner_side = None
                    outcome_title = ""
                    
                    # Check if Team 1 won this round
                    if i < len(t1_outcomes):
                        src = t1_outcomes[i].get('src', '').lower()
                        # If the src isn't 'emptyhistory', they won
                        if 'emptyhistory' not in src:
                            winner_team = "Team1"
                            outcome_title = t1_outcomes[i].get('title', '')
                            # Side is often in the title or can be inferred
                            if 'counter-terrorist' in outcome_title.lower(): winner_side = 'CT'
                            elif 'terrorist' in outcome_title.lower(): winner_side = 'T'
                    
                    # Check if Team 2 won this round
                    if not winner_team and i < len(t2_outcomes):
                        src = t2_outcomes[i].get('src', '').lower()
                        if 'emptyhistory' not in src:
                            winner_team = "Team2"
                            outcome_title = t2_outcomes[i].get('title', '')
                            if 'counter-terrorist' in outcome_title.lower(): winner_side = 'CT'
                            elif 'terrorist' in outcome_title.lower(): winner_side = 'T'
                    
                    rounds.append({
                        "round_num": i + 1,
                        "winner": winner_team,
                        "winner_side": winner_side,
                        "outcome": outcome_title
                    })
        
        time.sleep(random.uniform(2.0, 4.0))
        return rounds

if __name__ == '__main__':
    # Very basic execution test
    logging.basicConfig(level=logging.INFO)
    client = HLTVClient()
    try:
        res = client.fetch_recent_results(pages=1)
        if res:
             print(f"Found {len(res)} matches.")
             print(f"Checking vetos for {res[0]['team1']} vs {res[0]['team2']}...")
             vetoes = client.fetch_match_vetoes(res[0]['url'])
             print("VETOES:")
             for v in vetoes:
                 print(v)
    finally:
        client.stop()
