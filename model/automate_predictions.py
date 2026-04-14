import argparse
import sys
import os
import json
import logging
import re
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import calculate_expected_series_win
from ingestion.hltv_client import HLTVClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Template Loading ---
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def load_template(name):
    path = os.path.join(TEMPLATE_DIR, f"{name}.html")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def simple_format(template, **kwargs):
    """Replaces {key} or {key:spec} with value without clashing with CSS/JS braces."""
    def replacer(match):
        key = match.group(1)
        spec = match.group(2) or ""
        if key in kwargs:
            try:
                return ("{0" + spec + "}").format(kwargs[key])
            except (ValueError, TypeError):
                return match.group(0)
        return match.group(0)
    
    return re.sub(r'\{(\w+)(:[^}]*)?\}', replacer, template)

# Load templates at startup
try:
    HTML_TEMPLATE = load_template("report_layout")
    MATCH_CARD_TEMPLATE = load_template("match_card")
    ODDS_SECTION_TEMPLATE = load_template("odds_section")
    NO_ODDS_SECTION_TEMPLATE = load_template("no_odds_section")
    VETO_ITEM_TEMPLATE = load_template("veto_item")
except FileNotFoundError as e:
    logger.error(f"Failed to load templates: {e}")
    sys.exit(1)

MAP_FILENAME_MAP = {
    "ancient": "de_ancient.png",
    "anubis": "de_anubis.png",
    "dust2": "de_dust2.png",
    "inferno": "de_inferno.png",
    "mirage": "de_mirage.png",
    "nuke": "de_nuke.png",
    "overpass": "de_overpass.png",
    "train": "de_train.png",
    "vertigo": "de_vertigo.png"
}

def convert_odds_to_prob(o1, o2):
    """Converts decimal odds to implied probability percentages."""
    if not o1 or not o2:
        return None, None
    
    p1_raw = 1.0 / o1
    p2_raw = 1.0 / o2
    total = p1_raw + p2_raw
    
    p1 = (p1_raw / total) * 100
    p2 = (p2_raw / total) * 100
    return p1, p2

def archive_hltv_html(html):
    """Saves the raw HLTV matches HTML to data/raw/hltv_archive/ with a timestamp."""
    if not html:
        return
    
    # Path relative to project root (system.path already includes it)
    archive_dir = os.path.join("data", "raw", "hltv_archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"hltv_matches_{timestamp}.html"
    filepath = os.path.join(archive_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Archived raw HLTV matches HTML to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Automated HLTV Match Prediction Pipeline")
    parser.add_argument("--event-id", type=int, help="HLTV Event ID to scrape matches from")
    parser.add_argument("--output", default="predictions_report.html", help="Output file path (default: predictions_report.html)")
    parser.add_argument("--html-file", help="Path to a local HTML file to parse (skips scraping)")
    parser.add_argument("--iters", type=int, default=10000, help="MC iterations for veto simulation")
    parser.add_argument("--threshold", type=float, default=0.90, help="Probability truncation threshold")
    
    args = parser.parse_args()

    client = HLTVClient()
    matches = []

    try:
        if args.html_file:
            logger.info(f"Parsing local HTML file: {args.html_file}")
            with open(args.html_file, 'r', encoding='utf-8') as f:
                html = f.read()
            matches = client.parse_upcoming_matches(html)
        else:
            logger.info("Fetching upcoming matches from HLTV...")
            matches = client.fetch_upcoming_matches(event_id=args.event_id)
            # Capture the raw HTML retrieved by the client
            if hasattr(client, 'last_html'):
                 archive_hltv_html(client.last_html)
    except Exception as e:
        logger.error(f"Failed to fetch/parse matches: {e}")
        sys.exit(1)
    finally:
        if not args.html_file:
            client.stop()

    if not matches:
        logger.warning("No matches found to predict.")
        return

    logger.info(f"Found {len(matches)} matches. Starting predictions...")

    cards_html_list = []

    for i, match in enumerate(matches):
        team_a = match['team1']
        team_b = match['team2']
        
        # Filter Predetermined matches
        t1_low, t2_low = team_a.lower(), team_b.lower()
        if any(x in t1_low or x in t2_low for x in ["unknown", "tbd"]):
            logger.info(f"[{i+1}/{len(matches)}] Skipping: {team_a} vs {team_b} (Undetermined participants)")
            continue

        fmt = match['format']
        match_url = match['url']
        
        if 'bo1' in fmt: fmt = 'bo1'
        elif 'bo5' in fmt: fmt = 'bo5'
        else: fmt = 'bo3'

        logger.info(f"[{i+1}/{len(matches)}] Predicting: {team_a} vs {team_b} ({fmt})")

        try:
            results = calculate_expected_series_win(
                team_a, 
                team_b, 
                series_format=fmt, 
                threshold=args.threshold, 
                iters=args.iters
            )
            
            prob = results["expected_win_prob"]
            t_a_id = results["team_a_id"]
            t_b_id = results["team_b_id"]
            ctx = results["predictor_ctx"]
            seq_counts = results["sequence_counts"]
            
            # Odds Calculation
            o1 = match.get('odds1')
            o2 = match.get('odds2')
            implied_p1, implied_p2 = convert_odds_to_prob(o1, o2)
            
            odds_section = ""
            if implied_p1 and implied_p2:
                odds_section = simple_format(ODDS_SECTION_TEMPLATE,
                    imp1_str=f"{implied_p1:.1f}%", 
                    imp2_str=f"{implied_p2:.1f}%", 
                    o1_str=f"{o1:.2f}", 
                    o2_str=f"{o2:.2f}"
                )
            else:
                odds_section = NO_ODDS_SECTION_TEMPLATE

            # Top Vetoes
            veto_items = ""
            sorted_seqs = sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for seq, count in sorted_seqs:
                s_prob = (count / args.iters) * 100
                
                # Create map thumbnails and formatted name string
                map_names = [m.strip() for m in seq.split(",")]
                formatted_names = " → ".join(map_names)
                
                map_thumbs = ""
                for mname in map_names:
                    m_key = mname.lower().replace(" ", "")
                    fname = MAP_FILENAME_MAP.get(m_key, "placeholder.png")
                    path = f"static/maps/{fname}"
                    map_thumbs += f'<img src="{path}" class="map-thumb" alt="{mname}" title="{mname}">'
                
                veto_items += simple_format(VETO_ITEM_TEMPLATE,
                    s_prob_str=f"{s_prob:5.1f}%",
                    formatted_names=formatted_names,
                    map_thumbs=map_thumbs
                )

            # Build Match Entry
            card = simple_format(MATCH_CARD_TEMPLATE,
                url=match_url,
                event=match.get('id', 'Match'), # Could use actual event name if we scrape it
                format=fmt.upper(),
                team1=team_a,
                team2=team_b,
                t1_logo=match.get('team1_logo', ''),
                t2_logo=match.get('team2_logo', ''),
                prob1_str=f"{prob*100:.1f}%",
                prob2_str=f"{(1-prob)*100:.1f}%",
                prob1_style=f'style="width: {prob*100:.1f}%"',
                team1_short=t_a_id[:12],
                team2_short=t_b_id[:12],
                odds_section=odds_section,
                veto_items=veto_items,
                t1_maps=len(ctx.gen_histories.get(t_a_id, [])),
                t2_maps=len(ctx.gen_histories.get(t_b_id, []))
            )
            cards_html_list.append(card)

        except Exception as e:
            logger.error(f"Error predicting {team_a} vs {team_b}: {e}")

    # Final HTML assembly
    final_html = simple_format(HTML_TEMPLATE,
        gen_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        iters=args.iters,
        threshold_pct=args.threshold * 100,
        cards_html="\n".join(cards_html_list)
    )

    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    logger.info(f"Automation complete. HTML report saved to {args.output}")

if __name__ == "__main__":
    main()
