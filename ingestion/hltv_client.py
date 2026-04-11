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
        Returns a list of match dictionaries containing URL, team names, format, event and date.
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
                # The date is in a standard-headline
                date_el = sublist.find('div', class_='standard-headline')
                if not date_el:
                    continue
                
                raw_date = date_el.text.replace("Results for ", "").replace("Matches for ", "").strip()
                clean_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', raw_date)
                
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
                        event_div = a.find('span', class_='event-name')
                        format_div = a.find('div', class_='map-text')
                        
                        t1_name = t1_div.text.strip() if t1_div else "Unknown"
                        t2_name = t2_div.text.strip() if t2_div else "Unknown"
                        event_name = event_div.text.strip() if event_div else "Unknown Event"
                        match_format = format_div.text.strip() if format_div else "Unknown Format"
                        
                        results.append({
                            'team1': t1_name,
                            'team2': t2_name,
                            'event': event_name,
                            'format': match_format,
                            'url': match_url,
                            'date': match_date_str
                        })
                    
        return results

    def fetch_match_details(self, match_url: str) -> Dict:
        """
        Navigates to a match URL and extracts the veto box text, map results, player stats, and match blurbs.
        """
        if not self.driver:
             self.start()
             
        logger.info(f"Fetching details for: {match_url}")
        self.driver.get(match_url)
        self._wait_for_cloudflare()
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # 1. Extract Match Blurbs / Info
        match_info = []
        try:
            # Often found in preformatted-text like "Best of 3 (LAN)\nSemi-final"
            blurbs = soup.find_all('div', class_='preformatted-text')
            for b in blurbs:
                text = b.text.strip()
                if text and text not in match_info:
                    # split by newlines if it has multiple lines
                    for line in text.split('\n'):
                        if line.strip() and line.strip() not in match_info:
                            match_info.append(line.strip())
            
            event_el = soup.find('div', class_='event text-ellipsis')
            if event_el and event_el.text.strip() not in match_info:
                match_info.append("Event: " + event_el.text.strip())
        except Exception as e:
            logger.warning(f"Error scraping match blurbs: {e}")

        # 2. Extract Player Stats (Rating, KDA)
        player_stats = []
        try:
            stats_tables = soup.find_all('table', class_='table totalstats')
            for table in stats_tables:
                # Find headers dynamically
                header_row = table.find('tr', class_='header-row')
                headers = []
                if header_row:
                    headers = [th.text.strip().lower() for th in header_row.find_all(['th', 'td'])]
                
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')
                for row in rows:
                    if 'header-row' in row.get('class', []): continue
                    cells = row.find_all('td')
                    
                    if not headers:
                        headers = ['player', 'k-d', 'ek-ed', 'swing', 'adr', 'eadr', 'kast', 'ekast', 'rating']

                    if len(cells) >= len(headers):
                        player_data = {}
                        for i, h in enumerate(headers):
                            if h: # some headers might be empty
                                player_data[h] = cells[i].text.strip()
                        
                        # Calculate raw +/- if K-D exists
                        kd = player_data.get('k-d', '')
                        if '-' in kd:
                            try:
                                kills, deaths = kd.split('-')
                                player_data['plus_minus'] = str(int(kills.strip()) - int(deaths.strip()))
                            except:
                                player_data['plus_minus'] = "0"
                        elif '+/-' in headers:
                             # Legacy fallback if HLTV reverts or different format
                             player_data['plus_minus'] = cells[headers.index('+/-')].text.strip()
                             
                        player_stats.append(player_data)
        except Exception as e:
            logger.warning(f"Error scraping player stats: {e}")

        # 3. Extract Vetoes
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
                    if line not in vetoes:
                        vetoes.append(line)
        
        if not vetoes:
             logger.warning(f"No veto records extracted from {match_url}.")
        else:
             logger.info(f"  -> Successfully extracted {len(vetoes)} veto records.")
             
        # 4. Extract Map Scores
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
                    score_items = []
                    for span in spans:
                        c = span.get('class', [])
                        side = ''
                        if 't' in c: side = 'T'
                        if 'ct' in c: side = 'CT'
                        if side and span.text.isdigit():
                            score_items.append({"score": int(span.text), "side": side})
                            
                    for i in range(0, len(score_items), 2):
                        if i+1 < len(score_items):
                            halves.append({
                                "team1_score": score_items[i]["score"],
                                "team1_side": score_items[i]["side"],
                                "team2_score": score_items[i+1]["score"],
                                "team2_side": score_items[i+1]["side"]
                            })
                            
                stats_link_el = m.find('a', class_='results-stats')
                stats_url = "https://www.hltv.org" + stats_link_el.get('href') if stats_link_el else None
                
                picker = None
                for v in vetoes:
                    if "picked" in v.lower() and map_name.lower() in v.lower():
                        raw_picker = v.split("picked")[0].strip()
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

        # 5. Extract Team Ranks from Lineups
        team_ranks = {}
        try:
            lineup_boxes = soup.find_all('div', class_='lineup standard-box')
            for box in lineup_boxes:
                team_name_el = box.find('a', class_='text-ellipsis')
                rank_el = box.find('div', class_='teamRanking')
                
                if team_name_el and rank_el:
                    team_name = team_name_el.text.strip()
                    rank_text = rank_el.text.replace("World rank:", "").strip()
                    team_ranks[team_name] = {"world_rank": rank_text}
        except Exception as e:
            logger.warning(f"Error scraping team ranks: {e}")

        # 6. Extract VRS Data (Before & Result/Forecast)
        try:
            vrs_container = soup.find('div', class_='vrs-forecast-container')
            if vrs_container:
                # Get team names in order
                team_names = []
                team_rows = vrs_container.find_all('div', class_='vrs-forecast-team-name')
                for tr in team_rows:
                    team_names.append(tr.text.strip())
                
                # Helper to extract points/rank from a column
                def extract_vrs_column(col_div):
                    if not col_div: return None, []
                    header_el = col_div.find('div', class_='vrs-forecast-header')
                    header = header_el.text.strip().lower() if header_el else "unknown"
                    
                    data = []
                    wrappers = col_div.find_all('div', class_='vrs-forecast-numbers-wrapper')
                    for wrapper in wrappers:
                        pts = wrapper.find('div', class_='vrs-forecast-points')
                        rnk = wrapper.find('div', class_='vrs-forecast-ranking')
                        data.append({
                            "points": pts.text.strip() if pts else None,
                            "rank": rnk.text.strip() if rnk else None
                        })
                    return header, data

                # Process left column (Before)
                left_col = vrs_container.find('div', class_='vrs-forecast-left') or vrs_container.find('div', class_='vrs-forecast-left-numbers')
                header_l, data_l = extract_vrs_column(left_col)
                if data_l:
                    for i, d in enumerate(data_l):
                        if i < len(team_names):
                            t_name = team_names[i]
                            if t_name not in team_ranks: team_ranks[t_name] = {}
                            team_ranks[t_name]["vrs_before_rank"] = d["rank"]
                            team_ranks[t_name]["vrs_before_points"] = d["points"]

                # Process middle column (Result/Forecast)
                mid_col = vrs_container.find('div', class_='vrs-forecast-middle')
                header_m, data_m = extract_vrs_column(mid_col)
                if data_m:
                    for i, d in enumerate(data_m):
                        if i < len(team_names):
                            t_name = team_names[i]
                            if t_name not in team_ranks: team_ranks[t_name] = {}
                            
                            # For finished matches, HLTV calls this 'result'. 
                            # We map it to 'after' for consistency.
                            if header_m == "result":
                                team_ranks[t_name]["vrs_after_rank"] = d["rank"]
                                team_ranks[t_name]["vrs_after_points"] = d["points"]
                            else:
                                # Fallback for unexpected headers
                                suffix = header_m.replace(" ", "_")
                                team_ranks[t_name][f"vrs_{suffix}_rank"] = d["rank"]
                                team_ranks[t_name][f"vrs_{suffix}_points"] = d["points"]

                # Process right column (if exists in future/upcoming)
                right_col = vrs_container.find('div', class_='vrs-forecast-right')
                header_r, data_r = extract_vrs_column(right_col)
                if data_r:
                    for i, d in enumerate(data_r):
                        if i < len(team_names):
                            t_name = team_names[i]
                            if t_name not in team_ranks: team_ranks[t_name] = {}
                            suffix = header_r.replace(" ", "_")
                            team_ranks[t_name][f"vrs_{suffix}_rank"] = d["rank"]
                            team_ranks[t_name][f"vrs_{suffix}_points"] = d["points"]

        except Exception as e:
            logger.warning(f"Error scraping VRS data: {e}")

        time.sleep(random.uniform(2.0, 4.0))
        return {
            "match_info": match_info,
            "player_stats": player_stats,
            "team_ranks": team_ranks,
            "vetoes": vetoes,
            "maps": maps_data
        }

    def fetch_map_stats(self, stats_url: str) -> Dict:
        """
        Navigates to a map stats page and extracts the round-by-round history
        and map-specific player stats.
        """
        if not self.driver or not stats_url:
             return {"rounds": [], "player_stats": []}
             
        logger.info(f"Fetching map stats from: {stats_url}")
        self.driver.get(stats_url)
        self._wait_for_cloudflare()
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # 1. Round history
        rounds = []
        rh_con = soup.find('div', class_='round-history-con')
        if rh_con:
            team_rows = rh_con.find_all('div', class_='round-history-team-row')
            if len(team_rows) >= 2:
                t1_outcomes = team_rows[0].find_all('img', class_='round-history-outcome')
                t2_outcomes = team_rows[1].find_all('img', class_='round-history-outcome')
                
                num_rounds = max(len(t1_outcomes), len(t2_outcomes))
                for i in range(num_rounds):
                    winner_team = None
                    winner_side = None
                    outcome_title = ""
                    
                    if i < len(t1_outcomes):
                        src = t1_outcomes[i].get('src', '').lower()
                        if 'emptyhistory' not in src:
                            winner_team = "Team1"
                            outcome_title = t1_outcomes[i].get('title', '')
                            if 'counter-terrorist' in outcome_title.lower(): winner_side = 'CT'
                            elif 'terrorist' in outcome_title.lower(): winner_side = 'T'
                    
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
        
        # 2. Player Stats (Map specific)
        player_stats = []
        try:
            stats_tables = soup.find_all('table', class_='totalstats')
            for table in stats_tables:
                header_row = table.find('thead')
                headers = []
                if header_row:
                    headers = [th.text.strip().lower() for th in header_row.find_all('th')]
                
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')
                for row in rows:
                    if 'header-row' in row.get('class', []): continue
                    cells = row.find_all('td')
                    
                    if len(headers) > 0 and len(cells) >= len(headers):
                        player_data = {}
                        for i, h in enumerate(headers):
                            if h and i < len(cells):
                                if i == 0:
                                    player_data["player"] = cells[i].text.strip()
                                else:
                                    player_data[h] = cells[i].text.strip()
                                    
                        # Calculate raw +/- 
                        kills = 0
                        deaths = 0
                        has_kd = False
                        
                        if 'k (hs)' in player_data and 'd (t)' in player_data:
                            try:
                                k_str = player_data['k (hs)'].split('(')[0]
                                kills = int(k_str.strip())
                                d_str = player_data['d (t)'].split('(')[0]
                                deaths = int(d_str.strip())
                                has_kd = True
                            except: pass
                        elif 'k-d' in player_data:
                            try:
                                k_str, d_str = player_data['k-d'].split('-')
                                kills = int(k_str.strip())
                                deaths = int(d_str.strip())
                                has_kd = True
                            except: pass
                            
                        if has_kd:
                            player_data['plus_minus'] = str(kills - deaths)
                        elif '+/-' in headers:
                            player_data['plus_minus'] = cells[headers.index('+/-')].text.strip()
                            
                        player_stats.append(player_data)
        except Exception as e:
            logger.warning(f"Error scraping map player stats: {e}")

        time.sleep(random.uniform(2.0, 4.0))
        return {
            "rounds": rounds,
            "player_stats": player_stats
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    client = HLTVClient()
    try:
        res = client.fetch_recent_results(pages=1)
        if res:
             print(f"Found {len(res)} matches.")
             print(f"Checking details for {res[0]['team1']} vs {res[0]['team2']}...")
             details = client.fetch_match_details(res[0]['url'])
             print(details)
    finally:
        client.stop()
