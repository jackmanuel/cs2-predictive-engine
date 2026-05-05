import time
import random
import re
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

logger = logging.getLogger(__name__)

HLTV_BASE_URL = "https://www.hltv.org"

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

    def _absolute_url(self, href: str) -> str:
        """Returns an absolute HLTV URL for relative links."""
        if not href:
            return ""
        if href.startswith("http"):
            return href
        return HLTV_BASE_URL + href

    def _analytics_url_from_match_url(self, match_url: str) -> Optional[str]:
        """Converts a match page URL to its betting analytics URL."""
        if not match_url:
            return None
        match = re.search(r"/matches/(\d+)/(.+)$", match_url)
        if not match:
            return None
        match_id, slug = match.groups()
        return f"{HLTV_BASE_URL}/betting/analytics/{match_id}/{slug}"

    def _extract_analytics_summary(self, soup: BeautifulSoup) -> Dict:
        """Extracts the human-readable analytics summary block when present."""
        heading = soup.find(
            lambda tag: tag.name in ["h1", "h2", "h3"]
            and "analytics summary" in tag.get_text(" ", strip=True).lower()
        )
        if not heading:
            return {}

        lines = []
        for sibling in heading.find_all_next():
            if sibling is heading:
                continue
            if sibling.name in ["h1", "h2", "h3"] and sibling.get_text(" ", strip=True):
                break
            text = sibling.get_text(" ", strip=True)
            if text and text not in lines:
                lines.append(text)

        return {"text": "\n".join(lines[:80])}

    def _decimal_odds_from_text(self, text: str) -> List[float]:
        """Finds plausible decimal odds values in a short text fragment."""
        values = []
        for raw in re.findall(r"(?<!\d)([1-9]\d?\.\d{2})(?!\d)", text):
            try:
                val = float(raw)
            except ValueError:
                continue
            if 1.01 <= val <= 100.0:
                values.append(val)
        return values

    def _infer_bookmaker_name(self, container, team_names: List[str]) -> str:
        """Best-effort bookmaker name extraction from a likely odds row."""
        ignored = {"", "image", "logo", "hltv.org", "counter-strike", "cs2"}
        team_lows = {t.lower() for t in team_names if t}

        for img in container.find_all("img"):
            for attr in ["alt", "title"]:
                name = (img.get(attr) or "").strip()
                name = re.sub(r"^logo for\s+", "", name, flags=re.I)
                name = re.sub(r"\s+", " ", name)
                if name and name.lower() not in ignored and name.lower() not in team_lows:
                    return name

        text = container.get_text(" ", strip=True)
        for team in team_names:
            if team:
                text = re.sub(re.escape(team), " ", text, flags=re.I)
        text = re.sub(r"(?<!\d)[1-9]\d?\.\d{2}(?!\d)", " ", text)
        text = re.sub(r"\b(bet now|claim|bonus|terms|promocode|odds|pick a winner)\b", " ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" -|:")
        return text[:80] if text else "Unknown"

    def _normalize_provider_name(self, provider: str) -> str:
        """Converts HLTV provider ids into readable bookmaker names."""
        if not provider:
            return "Unknown"

        known = {
            "ggbet": "GG.BET",
            "thunderpick": "Thunderpick",
            "1xbet": "1xBet",
            "vulkan": "Vulkan Bet",
            "bet20": "20Bet",
        }
        key = provider.strip().lower()
        return known.get(key, provider.strip())

    def _parse_bookmaker_odds_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Parses HLTV's bookmaker comparison table using data-provider/data-numeric-odds."""
        odds_rows = []
        seen = set()

        for row in soup.find_all("tr"):
            odds_cells = row.find_all(
                lambda tag: tag.name in ["td", "div"]
                and "odds" in tag.get("class", [])
                and tag.get("data-numeric-odds")
            )
            if len(odds_cells) < 2:
                continue

            by_side = {}
            provider = None
            for cell in odds_cells:
                classes = cell.get("class", [])
                if "team1" in classes:
                    side = "a"
                elif "team2" in classes:
                    side = "b"
                else:
                    continue

                try:
                    value = float(cell.get("data-numeric-odds"))
                except (TypeError, ValueError):
                    continue

                if not provider:
                    provider = cell.get("data-provider")
                by_side[side] = value

            if "a" not in by_side or "b" not in by_side:
                continue

            bookmaker = self._normalize_provider_name(provider)
            key = (bookmaker.lower(), round(by_side["a"], 3), round(by_side["b"], 3))
            if key in seen:
                continue
            seen.add(key)

            odds_a = by_side["a"]
            odds_b = by_side["b"]
            odds_rows.append({
                "bookmaker": bookmaker,
                "provider_id": provider,
                "market": "moneyline",
                "odds_a": odds_a,
                "odds_b": odds_b,
                "implied_prob_a": 1.0 / odds_a,
                "implied_prob_b": 1.0 / odds_b,
                "overround": (1.0 / odds_a) + (1.0 / odds_b),
            })

        return odds_rows

    def parse_betting_analytics(self, html: str, team1: str = None, team2: str = None) -> Dict:
        """
        Parses an HLTV betting analytics page.

        Returns all discovered two-sided moneyline odds rows plus the analytics summary text.
        The DOM has changed a few times, so the odds parser intentionally uses conservative
        structure and class-name heuristics instead of one brittle selector.
        """
        soup = BeautifulSoup(html, 'html.parser')
        team_names = [t for t in [team1, team2] if t]
        summary = self._extract_analytics_summary(soup)

        odds_rows = self._parse_bookmaker_odds_table(soup)
        seen = {
            (row.get("bookmaker", "").lower(), round(row["odds_a"], 3), round(row["odds_b"], 3))
            for row in odds_rows
            if row.get("odds_a") and row.get("odds_b")
        }

        if odds_rows:
            return {
                "analytics_summary": summary,
                "bookmaker_odds": odds_rows,
                "odds_summary": self.summarize_bookmaker_odds(odds_rows),
            }

        class_hint = re.compile(r"(book|odds|betting|provider|market)", re.I)

        candidates = soup.find_all(
            lambda tag: tag.name in ["div", "tr", "li", "a"]
            and class_hint.search(" ".join(tag.get("class", [])) + " " + (tag.get("id") or ""))
        )

        for container in candidates:
            text = container.get_text(" ", strip=True)
            if not text or len(text) > 700:
                continue

            values = self._decimal_odds_from_text(text)
            if len(values) < 2:
                continue

            odds_a, odds_b = values[0], values[1]
            bookmaker = self._infer_bookmaker_name(container, team_names)
            key = (bookmaker.lower(), round(odds_a, 3), round(odds_b, 3))
            if key in seen:
                continue
            seen.add(key)

            odds_rows.append({
                "bookmaker": bookmaker,
                "market": "moneyline",
                "odds_a": odds_a,
                "odds_b": odds_b,
                "implied_prob_a": 1.0 / odds_a,
                "implied_prob_b": 1.0 / odds_b,
                "overround": (1.0 / odds_a) + (1.0 / odds_b),
            })

        return {
            "analytics_summary": summary,
            "bookmaker_odds": odds_rows,
            "odds_summary": self.summarize_bookmaker_odds(odds_rows),
        }

    def summarize_bookmaker_odds(self, odds_rows: List[Dict]) -> Dict:
        """Summarises raw bookmaker odds into neutral market-level fields."""
        clean_rows = [
            row for row in odds_rows
            if row.get("odds_a") and row.get("odds_b")
        ]
        if not clean_rows:
            return {}

        odds_a = [row["odds_a"] for row in clean_rows]
        odds_b = [row["odds_b"] for row in clean_rows]
        return {
            "source": "hltv_analytics_median",
            "book_count": len(clean_rows),
            "odds_a_median": statistics.median(odds_a),
            "odds_b_median": statistics.median(odds_b),
            "odds_a_mean": statistics.mean(odds_a),
            "odds_b_mean": statistics.mean(odds_b),
            "odds_a_min": min(odds_a),
            "odds_b_min": min(odds_b),
            "odds_a_max": max(odds_a),
            "odds_b_max": max(odds_b),
        }

    def fetch_match_betting_analytics(self, match: Dict) -> Dict:
        """Fetches and parses the per-match HLTV betting analytics page."""
        if not self.driver:
            self.start()

        url = match.get("analytics_url") or self._analytics_url_from_match_url(match.get("url", ""))
        if not url:
            return {"bookmaker_odds": [], "odds_summary": {}, "analytics_summary": {}}

        logger.info(f"Fetching betting analytics for: {match.get('team1')} vs {match.get('team2')}")
        self.driver.get(url)
        self._wait_for_cloudflare()

        html = self.driver.page_source
        parsed = self.parse_betting_analytics(html, match.get("team1"), match.get("team2"))
        parsed["source_url"] = url
        return parsed

    def fetch_recent_results(self, pages: int = 1, start_page: int = 0) -> List[Dict]:
        """
        Fetches the recent match results from HLTV's /results page.
        Returns a list of match dictionaries containing URL, team names, format, event and date.
        """
        if not self.driver:
            self.start()
            
        results = []
        base_url = "https://www.hltv.org/results"
        
        for p in range(start_page, start_page + pages):
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
                        match_url = self._absolute_url(href)
                        
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
        
        # 1. Extract Match Metadata (Teams, Date, Event, Result)
        match_metadata = {
            "team1": "Unknown",
            "team2": "Unknown",
            "date": None,
            "time": None,
            "event": "Unknown",
            "format": "Unknown",
            "is_finished": False,
            "winner": None,
            "score": None
        }
        try:
            # Date/Time
            time_el = soup.find('div', class_='date')
            if time_el and time_el.get('data-unix'):
                unix_ms = int(time_el['data-unix'])
                dt = datetime.fromtimestamp(unix_ms / 1000.0)
                match_metadata['date'] = dt.strftime('%Y-%m-%d')
                match_metadata['time'] = dt.strftime('%H:%M')
            
            # Teams and Event
            # Using the standard HLTV classes for match headers
            t1_name_el = soup.find('div', class_='team1-gradient').find('div', class_='teamName') if soup.find('div', class_='team1-gradient') else None
            t2_name_el = soup.find('div', class_='team2-gradient').find('div', class_='teamName') if soup.find('div', class_='team2-gradient') else None
            
            if t1_name_el: match_metadata['team1'] = t1_name_el.text.strip()
            if t2_name_el: match_metadata['team2'] = t2_name_el.text.strip()
            
            event_el = soup.find('div', class_='event text-ellipsis')
            if event_el: match_metadata['event'] = event_el.text.strip()

            # Format (Best of X)
            format_el = soup.find('div', class_='preformatted-text')
            if format_el:
                f_text = format_el.text.strip()
                if "Best of" in f_text:
                    match_metadata['format'] = f_text

            # Result (if finished)
            t1_score_el = soup.find('div', class_='team1-gradient').find('div', class_='won') or soup.find('div', class_='team1-gradient').find('div', class_='lost') or soup.find('div', class_='team1-gradient').find('div', class_='tie')
            t2_score_el = soup.find('div', class_='team2-gradient').find('div', class_='won') or soup.find('div', class_='team2-gradient').find('div', class_='lost') or soup.find('div', class_='team2-gradient').find('div', class_='tie')
            
            if t1_score_el and t2_score_el:
                s1 = t1_score_el.text.strip()
                s2 = t2_score_el.text.strip()
                if s1.isdigit() and s2.isdigit():
                    s1_int, s2_int = int(s1), int(s2)
                    match_metadata['score'] = f"{s1_int}-{s2_int}"
                    
                    # Determine winning threshold
                    winning_threshold = 2 # Default for BO3
                    if "Best of 1" in match_metadata['format']:
                        winning_threshold = 1
                    elif "Best of 5" in match_metadata['format']:
                        winning_threshold = 3
                    
                    if s1_int >= winning_threshold or s2_int >= winning_threshold:
                        match_metadata['is_finished'] = True
                        if s1_int > s2_int:
                            match_metadata['winner'] = match_metadata['team1']
                        elif s2_int > s1_int:
                            match_metadata['winner'] = match_metadata['team2']
            
            # Fallback for blurbs
            blurbs = soup.find_all('div', class_='preformatted-text')
            match_info = []
            for b in blurbs:
                text = b.text.strip()
                if text:
                    for line in text.split('\n'):
                        if line.strip(): match_info.append(line.strip())
        except Exception as e:
            logger.warning(f"Error scraping match metadata: {e}")
            match_info = []

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
                        elif 'ct' in c: side = 'CT'
                        
                        text = span.text.strip()
                        if text.isdigit():
                            score_items.append({"score": int(text), "side": side})
                            
                    for i in range(0, len(score_items), 2):
                        if i+1 < len(score_items):
                            halves.append({
                                "team1_score": score_items[i]["score"],
                                "team1_side": score_items[i]["side"],
                                "team2_score": score_items[i+1]["score"],
                                "team2_side": score_items[i+1]["side"]
                            })
                            
                stats_link_el = m.find('a', class_='results-stats')
                stats_url = self._absolute_url(stats_link_el.get('href')) if stats_link_el else None
                
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
            "metadata": match_metadata,
            "match_info": match_info,
            "player_stats": player_stats,
            "team_ranks": team_ranks,
            "vetoes": vetoes,
            "maps": maps_data
        }

    def fetch_upcoming_matches(self, event_id: Optional[int] = None) -> List[Dict]:
        """
        Fetches the upcoming matches from HLTV's /matches page or a specific event matches page.
        Returns a list of match dictionaries containing URL, teams, format, and betting odds if available.
        """
        if not self.driver:
            self.start()
            
        if event_id:
            url = f"https://www.hltv.org/events/{event_id}/matches"
        else:
            url = "https://www.hltv.org/matches"
            
        logger.info(f"Navigating to {url}")
        self.driver.get(url)
        self._wait_for_cloudflare()
        
        self.last_html = self.driver.page_source
        return self.parse_upcoming_matches(self.last_html)

    def parse_upcoming_matches(self, html: str) -> List[Dict]:
        """
        Parses the upcoming matches from an HLTV matches page HTML.
        Specifically looks for the match-wrapper entries and extracts team names and odds.
        """
        soup = BeautifulSoup(html, 'html.parser')
        matches = []
        seen_ids = set()
        
        # Each match is in a match-wrapper
        match_wrappers = soup.find_all('div', class_='match-wrapper')
        for wrapper in match_wrappers:
            match_id = wrapper.get('data-match-id')
            if not match_id:
                # Some placeholders or live matches might lack IDs; 
                # use URL as fallback if possible or skip
                anchor = wrapper.find('a', href=True)
                match_id = anchor['href'] if anchor else None
            
            if match_id in seen_ids:
                continue
            seen_ids.add(match_id)

            match_data = {}
            
            # Match URL and IDs
            match_el = wrapper.find('div', class_='match')
            if not match_el:
                continue
            
            anchor = match_el.find('a', class_='match-top', href=True) or match_el.find('a', href=True)
            if not anchor:
                continue
            
            href = anchor['href']
            match_data['url'] = self._absolute_url(href)
            match_data['id'] = wrapper.get('data-match-id')

            analytics_anchor = wrapper.find('a', href=lambda h: h and '/betting/analytics/' in h)
            if analytics_anchor:
                match_data['analytics_url'] = self._absolute_url(analytics_anchor.get('href'))
            else:
                match_data['analytics_url'] = self._analytics_url_from_match_url(match_data['url'])
            
            # Match start date/time (from unix timestamp)
            time_el = match_el.find('div', class_='match-time') or match_el.find('div', {'data-unix': True})
            if time_el and time_el.get('data-unix'):
                try:
                    unix_ms = int(time_el['data-unix'])
                    dt = datetime.fromtimestamp(unix_ms / 1000.0)
                    match_data['date'] = dt.strftime('%Y-%m-%d')
                    match_data['time'] = dt.strftime('%H:%M')
                except (ValueError, OSError):
                    match_data['date'] = None
                    match_data['time'] = None
            else:
                match_data['date'] = None
                match_data['time'] = None
            
            # Team Names and Logos
            t1_div = wrapper.find('div', class_='team1')
            t2_div = wrapper.find('div', class_='team2')
            
            if t1_div:
                t1_name_el = t1_div.find('div', class_='match-teamname')
                match_data['team1'] = t1_name_el.text.strip() if t1_name_el else t1_div.text.strip()
                t1_img = t1_div.find('img', class_='match-team-logo')
                match_data['team1_logo'] = t1_img.get('src') if t1_img else None
            else:
                match_data['team1'] = "Unknown"
                match_data['team1_logo'] = None
                
            if t2_div:
                t2_name_el = t2_div.find('div', class_='match-teamname')
                match_data['team2'] = t2_name_el.text.strip() if t2_name_el else t2_div.text.strip()
                t2_img = t2_div.find('img', class_='match-team-logo')
                match_data['team2_logo'] = t2_img.get('src') if t2_img else None
            else:
                match_data['team2'] = "Unknown"
                match_data['team2_logo'] = None
            
            # Format (bo1, bo3, bo5)
            meta_el = wrapper.find('div', class_='match-meta')
            match_data['format'] = meta_el.text.strip().lower() if meta_el else "bo3"
            
            # Event Name
            event_el = wrapper.find('div', class_='match-event-name') or wrapper.find('div', class_='match-event')
            match_data['event'] = event_el.text.strip() if event_el else "Unknown Event"

            # Odds - Found in match-fixtures
            odds_wrapper = wrapper.find('div', class_='odds-wrapper')
            if odds_wrapper:
                odds_els = odds_wrapper.find_all('div', class_='match-fixture-number')
                if len(odds_els) >= 2:
                    try:
                        match_data['odds1'] = float(odds_els[0].text.strip())
                        match_data['odds2'] = float(odds_els[1].text.strip())
                    except ValueError:
                        pass
            
            # Skip matches with "TBD" or "Unknown" teams as they cannot be predicted
            t1_low = match_data['team1'].lower()
            t2_low = match_data['team2'].lower()
            if "tbd" in t1_low or "tbd" in t2_low or "unknown" in t1_low or "unknown" in t2_low:
                continue
                
            matches.append(match_data)
            
        logger.info(f"Parsed {len(matches)} upcoming matches from HTML.")
        return matches

    def fetch_map_stats(self, stats_url: str) -> Dict:
        """
        Navigates to a map stats page and extracts the round-by-round history
        and map-specific player stats.
        """
        if not self.driver or not stats_url:
             return {"round_history": "", "player_stats": []}
             
        logger.info(f"Fetching map stats from: {stats_url}")
        self.driver.get(stats_url)
        self._wait_for_cloudflare()
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # 1. Round history (Can be multiple containers if there was OT)
        round_history = ""
        rh_cons = soup.find_all('div', class_='round-history-con')
        for rh_con in rh_cons:
            team_rows = rh_con.find_all('div', class_='round-history-team-row')
            if len(team_rows) >= 2:
                t1_outcomes = team_rows[0].find_all('img', class_='round-history-outcome')
                t2_outcomes = team_rows[1].find_all('img', class_='round-history-outcome')
                
                num_rounds_in_con = max(len(t1_outcomes), len(t2_outcomes))
                for i in range(num_rounds_in_con):
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
                    
                    if winner_team:
                        round_history += "1" if winner_team == "Team1" else "2"
        
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
            "round_history": round_history,
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
