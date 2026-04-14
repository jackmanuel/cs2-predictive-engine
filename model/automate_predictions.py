import argparse
import sys
import os
import json
import logging
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

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CS2 Series Predictions</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --accent-primary: #38bdf8;
            --accent-secondary: #818cf8;
            --success: #22c55e;
            --danger: #ef4444;
            --gold: #fbbf24;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Inter', sans-serif; 
            background-color: var(--bg-color); 
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{
            margin-bottom: 3rem;
            border-bottom: 2px solid #334155;
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        
        h1 {{ font-weight: 800; font-size: 2.5rem; background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .meta-info {{ text-align: right; color: var(--text-dim); font-size: 0.9rem; }}
        
        .match-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 2rem;
        }}
        
        .match-card {{
            background: var(--card-bg);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid #334155;
            transition: transform 0.2s, border-color 0.2s;
            position: relative;
            overflow: hidden;
        }}
        
        .match-card:hover {{
            transform: translateY(-4px);
            border-color: var(--accent-primary);
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        
        .format-badge {{
            background: #334155;
            padding: 0.2rem 0.6rem;
            border-radius: 0.4rem;
            color: var(--text-main);
        }}
        
        .teams-area {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2rem;
        }}
        
        .team {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            text-align: center;
        }}
        
        .vs {{
            font-weight: 800;
            color: #475569;
            font-size: 1.2rem;
            padding: 0 1rem;
        }}
        
        .logo-box {{
            width: 80px;
            height: 80px;
            background: #0f172a;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.5rem;
            border: 2px solid #334155;
        }}
        
        .logo-box img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        
        .team-name {{ font-weight: 700; font-size: 1.1rem; }}
        
        .prob-bar-container {{
            margin-bottom: 2rem;
        }}
        
        .bar-labels {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-weight: 700;
            font-size: 1.4rem;
        }}
        
        .bar-label-sub {{
            font-size: 0.8rem;
            color: var(--text-dim);
            font-weight: 400;
            margin-top: -0.2rem;
        }}
        
        .progress-track {{
            height: 12px;
            background: rgba(239, 68, 68, 0.2);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            border-radius: 6px;
            transition: width 1s ease-out;
        }}
        
        .odds-comparison {{
            background: #0f172a;
            border-radius: 0.8rem;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }}
        
        .odds-grid {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 1rem;
            align-items: center;
            font-size: 0.9rem;
        }}
        
        .odds-type {{ color: var(--text-dim); text-align: center; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
        
        .prediction-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        
        .detail-box {{
            background: rgba(15, 23, 42, 0.5);
            padding: 0.75rem;
            border-radius: 0.6rem;
        }}
        
        .detail-title {{
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--text-dim);
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}
        
        .veto-list {{ list-style: none; font-size: 0.8rem; }}
        .veto-item {{ 
            margin-bottom: 0.8rem; 
            background: #0f172a; 
            padding: 0.5rem; 
            border-radius: 0.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}
        .veto-meta {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; }}
        .veto-prob {{ color: var(--accent-primary); font-weight: 700; font-size: 0.9rem; }}
        .veto-names {{ 
            font-size: 0.75rem; 
            color: var(--text-dim); 
            background: #1e293b; 
            padding: 0.1rem 0.4rem; 
            border-radius: 0.3rem;
            font-family: monospace;
        }}
        .map-strip {{ display: flex; gap: 0.5rem; }}
        .map-thumb {{ 
            width: 50px; 
            height: 50px; 
            border-radius: 8px; 
            object-fit: contain;
            background: #0f172a;
            border: 1px solid #334155;
            transition: transform 0.2s;
        }}
        .map-thumb:hover {{
            transform: scale(1.1);
            z-index: 5;
            border-color: var(--accent-primary);
        }}
        
        .stat-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.2rem; }}
        .stat-val {{ font-weight: 600; }}
        
        .match-link {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 1;
        }}
        
        @media (max-width: 600px) {{
            .match-grid {{ grid-template-columns: 1fr; }}
            .teams-area {{ gap: 0.5rem; }}
            .logo-box {{ width: 60px; height: 60px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>CS2 Predictor Pro</h1>
                <p style="color: var(--text-dim)">Advanced Veto & Win Probability Analysis</p>
            </div>
            <div class="meta-info">
                <p>Generated: {gen_time}</p>
                <p>MC Sim: {iters:,} iters | {threshold_pct:.0f}% Threshold</p>
            </div>
        </header>
        
        <div class="match-grid">
            {cards_html}
        </div>
    </div>
</body>
</html>
"""

MATCH_CARD_TEMPLATE = """
            <div class="match-card">
                <a href="{url}" target="_blank" class="match-link"></a>
                <div class="card-header">
                    <span>{event}</span>
                    <span class="format-badge">{format}</span>
                </div>
                
                <div class="teams-area">
                    <div class="team">
                        <div class="logo-box">
                            <img src="{t1_logo}" alt="{team1}" onerror="this.src='https://www.hltv.org/img/static/team/placeholder.svg'">
                        </div>
                        <span class="team-name">{team1}</span>
                    </div>
                    <div class="vs">VS</div>
                    <div class="team">
                        <div class="logo-box">
                            <img src="{t2_logo}" alt="{team2}" onerror="this.src='https://www.hltv.org/img/static/team/placeholder.svg'">
                        </div>
                        <span class="team-name">{team2}</span>
                    </div>
                </div>
                
                <div class="prob-bar-container">
                    <div class="bar-labels">
                        <div>
                            <span>{prob1:.1f}%</span>
                            <div class="bar-label-sub">{team1_short}</div>
                        </div>
                        <div style="text-align: right">
                            <span>{prob2:.1f}%</span>
                            <div class="bar-label-sub">{team2_short}</div>
                        </div>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {prob1:.1f}%"></div>
                    </div>
                </div>
                
                {odds_section}
                
                <div class="prediction-details">
                    <div class="detail-box">
                        <div class="detail-title">Top Veto Sequences</div>
                        <ul class="veto-list">
                            {veto_items}
                        </ul>
                    </div>
                    <div class="detail-box">
                        <div class="detail-title">Dataset Stats</div>
                        <div class="stat-row">
                            <span>{team1_short} Maps</span>
                            <span class="stat-val">{t1_maps}</span>
                        </div>
                        <div class="stat-row">
                            <span>{team2_short} Maps</span>
                            <span class="stat-val">{t2_maps}</span>
                        </div>
                    </div>
                </div>
            </div>
"""

ODDS_SECTION_TEMPLATE = """
                <div class="odds-comparison">
                    <div class="odds-type">Market Implied Probabilities</div>
                    <div class="odds-grid">
                        <div style="font-weight: 700">{imp1:.1f}%</div>
                        <div style="color: #475569">← Odds →</div>
                        <div style="font-weight: 700; text-align: right">{imp2:.1f}%</div>
                    </div>
                    <div class="odds-grid" style="font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;">
                         <div>{o1:.2f}</div>
                         <div></div>
                         <div style="text-align: right">{o2:.2f}</div>
                    </div>
                </div>
"""

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
                odds_section = ODDS_SECTION_TEMPLATE.format(
                    imp1=implied_p1, imp2=implied_p2, o1=o1, o2=o2
                )

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
                
                veto_items += f"""
                    <li class="veto-item">
                        <div class="veto-meta">
                            <span class="veto-prob">{s_prob:5.1f}%</span>
                            <span class="veto-names">{formatted_names}</span>
                        </div>
                        <div class="map-strip">
                            {map_thumbs}
                        </div>
                    </li>"""

            # Build Match Entry
            card = MATCH_CARD_TEMPLATE.format(
                url=match_url,
                event=match.get('id', 'Match'), # Could use actual event name if we scrape it
                format=fmt.upper(),
                team1=team_a,
                team2=team_b,
                t1_logo=match.get('team1_logo', ''),
                t2_logo=match.get('team2_logo', ''),
                prob1=prob*100,
                prob2=(1-prob)*100,
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
    final_html = HTML_TEMPLATE.format(
        gen_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        iters=args.iters,
        threshold_pct=args.threshold * 100,
        cards_html="\n".join(cards_html_list)
    )

    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    logger.info(f"Automation complete. Nice HTML report saved to {args.output}")

if __name__ == "__main__":
    main()
