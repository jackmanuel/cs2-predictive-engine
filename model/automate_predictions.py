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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_report_name = f"predictions_report_{timestamp}.html"
    
    parser = argparse.ArgumentParser(description="Automated HLTV Match Prediction Pipeline")
    parser.add_argument("--event-id", type=int, help="HLTV Event ID to scrape matches from")
    parser.add_argument("--output", default=os.path.join("reports", default_report_name), help=f"Output file path (default: reports/{default_report_name})")
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

    from model.predict import PredictorContext
    try:
        shared_ctx = PredictorContext()
    except Exception as e:
        logger.error(f"Failed to initialize predictor context: {e}")
        sys.exit(1)

    match_results = []

    for i, match in enumerate(matches):
        team_a = match['team1']
        team_b = match['team2']
        
        # Filter Predetermined matches
        t1_low, t2_low = team_a.lower(), team_b.lower()
        is_undetermined = any(x in t1_low or x in t2_low for x in ["unknown", "tbd"])
        is_placeholder = any(re.search(r'.+/.+ (winner|loser)', t) for t in [t1_low, t2_low])

        if is_undetermined or is_placeholder:
            logger.info(f"[{i+1}/{len(matches)}] Skipping: {team_a} vs {team_b} (Undetermined/Placeholder participants)")
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
                iters=args.iters,
                ctx=shared_ctx
            )
            
            prob1 = results["expected_win_prob"]
            prob2 = 1.0 - prob1
            t_a_id = results["team_a_id"]
            t_b_id = results["team_b_id"]
            seq_counts = results["sequence_counts"]
            
            # Odds Calculation
            o1 = match.get('odds1')
            o2 = match.get('odds2')
            
            # Normalised (Implied)
            imp1, imp2 = convert_odds_to_prob(o1, o2)
            
            # Unnormalised (Raw)
            unnorm1, unnorm2 = None, None
            if o1 and o2:
                unnorm1 = (1.0 / o1) * 100
                unnorm2 = (1.0 / o2) * 100

            # Calculate Edge (Model Prob - Unnormalised Prob)
            edge1 = (prob1 * 100) - unnorm1 if unnorm1 is not None else -100
            edge2 = (prob2 * 100) - unnorm2 if unnorm2 is not None else -100
            
            max_edge = max(edge1, edge2)
            is_value_t1 = edge1 > 2.0 # 2% threshold for "value"
            is_value_t2 = edge2 > 2.0
            
            match_results.append({
                "match": match,
                "team_a": team_a,
                "team_b": team_b,
                "t1_logo": match.get('team1_logo', ''),
                "t2_logo": match.get('team2_logo', ''),
                "t_a_id": t_a_id,
                "t_b_id": t_b_id,
                "fmt": fmt,
                "prob1": prob1,
                "prob2": prob2,
                "o1": o1,
                "o2": o2,
                "imp1": imp1,
                "imp2": imp2,
                "unnorm1": unnorm1,
                "unnorm2": unnorm2,
                "edge1": edge1,
                "edge2": edge2,
                "max_edge": max_edge,
                "is_value_t1": is_value_t1,
                "is_value_t2": is_value_t2,
                "seq_counts": seq_counts
            })

        except Exception as e:
            logger.error(f"Error predicting {team_a} vs {team_b}: {e}")

    # Sort by model predicted edge
    match_results.sort(key=lambda x: x['max_edge'], reverse=True)

    cards_html_list = []
    for item in match_results:
        # Odds Section
        odds_section = ""
        if item['imp1'] and item['imp2']:
            odds_section = simple_format(ODDS_SECTION_TEMPLATE,
                imp1_str=f"{item['imp1']:.1f}%", 
                imp2_str=f"{item['imp2']:.1f}%", 
                unnorm1_str=f"{item['unnorm1']:.1f}%",
                unnorm2_str=f"{item['unnorm2']:.1f}%",
                o1_str=f"{item['o1']:.2f}", 
                o2_str=f"{item['o2']:.2f}"
            )
        else:
            odds_section = NO_ODDS_SECTION_TEMPLATE

        # Top Vetoes
        veto_items = ""
        sorted_seqs = sorted(item['seq_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
        for seq, count in sorted_seqs:
            s_prob = (count / args.iters) * 100
            map_names = [m.strip() for m in seq.split(",")]
            formatted_names = " → ".join(map_names)
            
            map_thumbs = ""
            for mname in map_names:
                m_key = mname.lower().replace(" ", "")
                fname = MAP_FILENAME_MAP.get(m_key, "placeholder.png")
                path = f"../static/maps/{fname}"
                map_thumbs += f'<img src="{path}" class="map-thumb" alt="{mname}" title="{mname}">'
            
            veto_items += simple_format(VETO_ITEM_TEMPLATE,
                s_prob_str=f"{s_prob:5.1f}%",
                formatted_names=formatted_names,
                map_thumbs=map_thumbs
            )

        # Value Badges
        value_badge1 = '<span class="value-badge">BEST VALUE</span>' if item['is_value_t1'] else ""
        value_badge2 = '<span class="value-badge">BEST VALUE</span>' if item['is_value_t2'] else ""

        # Edge Labels
        edge1_class = "edge-pos" if item['edge1'] > 0 else "edge-neg"
        edge2_class = "edge-pos" if item['edge2'] > 0 else "edge-neg"
        edge1_str = f"{item['edge1']:+.1f}%" if item['unnorm1'] is not None else "N/A"
        edge2_str = f"{item['edge2']:+.1f}%" if item['unnorm2'] is not None else "N/A"

        # Build Match Entry
        card = simple_format(MATCH_CARD_TEMPLATE,
            url=item['match']['url'],
            event=item['match'].get('id', 'Match'),
            format=item['fmt'].upper(),
            team1=item['team_a'],
            team2=item['team_b'],
            t1_logo=item['t1_logo'],
            t2_logo=item['t2_logo'],
            prob1_str=f"{item['prob1']*100:.1f}%",
            prob2_str=f"{item['prob2']*100:.1f}%",
            edge1_str=edge1_str,
            edge2_str=edge2_str,
            edge1_class=edge1_class,
            edge2_class=edge2_class,
            prob1_style=f'style="width: {item['prob1']*100:.1f}%"',
            team1_short=item['t_a_id'][:12],
            team2_short=item['t_b_id'][:12],
            odds_section=odds_section,
            veto_items=veto_items,
            t1_maps=len(shared_ctx.gen_histories.get(item['t_a_id'], [])),
            t2_maps=len(shared_ctx.gen_histories.get(item['t_b_id'], [])),
            value_badge1=value_badge1,
            value_badge2=value_badge2
        )
        cards_html_list.append(card)

    # Final HTML assembly
    final_html = simple_format(HTML_TEMPLATE,
        gen_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        iters=args.iters,
        threshold_pct=args.threshold * 100,
        cards_html="\n".join(cards_html_list)
    )

    # Save to file
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    logger.info(f"Automation complete. HTML report saved to {args.output}")

if __name__ == "__main__":
    main()
