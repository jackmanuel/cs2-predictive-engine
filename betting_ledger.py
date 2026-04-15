import argparse
import os
import pandas as pd
import logging
from ingestion.hltv_client import HLTVClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LEDGER_FILE = r"data\betting_ledger.csv"
COLUMNS = ["date", "time", "game", "link", "bookmaker", "odds", "amount", "bet", "result", "payout", "model_prob", "edge"]

def load_ledger():
    if os.path.exists(LEDGER_FILE):
        df = pd.read_csv(LEDGER_FILE)
        # Ensure all columns exist (migration)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)

def save_ledger(df):
    df.to_csv(LEDGER_FILE, index=False)

def add_bet(args):
    client = HLTVClient()
    try:
        logger.info(f"Fetching match details for {args.url}...")
        details = client.fetch_match_details(args.url)
        meta = details['metadata']
        
        game_text = f"{meta['team1']} vs {meta['team2']}"
        date_text = meta['date'] or "Unknown"
        time_text = meta['time'] or "Unknown"
        
        result = "Pending"
        payout = 0.0
        
        if meta['is_finished']:
            winner = meta['winner']
            bet_target = str(args.bet).strip()
            if winner and bet_target.lower() in winner.lower():
                result = 'Win'
                payout = round(args.odds * args.amount, 2)
            else:
                result = 'Loss'
                payout = 0.0
            logger.info(f"Match is already finished. Result: {result}")

        # Calculate edge: model_prob - implied_prob
        implied_prob = (1.0 / args.odds) if args.odds > 0 else 0
        edge = (args.model_prob - implied_prob) * 100 if args.model_prob else None
        
        new_row = {
            "date": date_text,
            "time": time_text,
            "game": game_text,
            "link": args.url,
            "bookmaker": args.bookmaker,
            "odds": args.odds,
            "amount": args.amount,
            "bet": args.bet,
            "result": result,
            "payout": payout,
            "model_prob": args.model_prob,
            "edge": round(edge, 2) if edge is not None else None
        }
        
        df = load_ledger()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_ledger(df)
        logger.info(f"Added bet: {game_text} on {args.bet} @ {args.odds} ({args.bookmaker})")
        if edge is not None:
            logger.info(f"  Model: {args.model_prob*100:.1f}% | Edge: {edge:+.1f}%")
        
    finally:
        client.stop()

def refresh_ledger():
    df = load_ledger()
    if df.empty:
        logger.info("Ledger is empty.")
        return

    pending_mask = df['result'] == 'Pending'
    if not pending_mask.any():
        logger.info("No pending bets to refresh.")
        return

    client = HLTVClient()
    try:
        updated_count = 0
        for idx, row in df[pending_mask].iterrows():
            logger.info(f"Checking result for: {row['game']}...")
            try:
                details = client.fetch_match_details(row['link'])
                meta = details['metadata']
                
                if meta['is_finished']:
                    winner = meta['winner']
                    bet_target = str(row['bet']).strip()
                    
                    # Determine result
                    if winner and bet_target.lower() in winner.lower():
                        df.at[idx, 'result'] = 'Win'
                        df.at[idx, 'payout'] = round(row['odds'] * row['amount'], 2)
                    else:
                        df.at[idx, 'result'] = 'Loss'
                        df.at[idx, 'payout'] = 0.0
                    
                    updated_count += 1
                    logger.info(f"  -> Match finished. Result: {df.at[idx, 'result']} (Winner: {winner})")
                else:
                    logger.info("  -> Match still pending.")
            except Exception as e:
                logger.error(f"  -> Error refreshing row {idx}: {e}")
                
        if updated_count > 0:
            save_ledger(df)
            logger.info(f"Refreshed {updated_count} bets.")
        else:
            logger.info("No updates found.")
            
    finally:
        client.stop()

def list_ledger():
    df = load_ledger()
    if df.empty:
        print("No bets found.")
        return
    
    # Simple formatted print
    display_cols = ["date", "game", "bet", "odds", "model_prob", "edge", "result", "amount", "payout"]
    display_cols = [c for c in display_cols if c in df.columns]
    print("\n--- Betting Ledger ---")
    print(df[display_cols].to_string(index=False))
    
    total_spent = df['amount'].sum()
    total_payout = df['payout'].sum()
    profit = total_payout - total_spent
    print(f"\nSummary: Spent: {total_spent:.2f} | Payout: {total_payout:.2f} | Profit: {profit:+.2f}")

def main():
    parser = argparse.ArgumentParser(description="CS2 Betting Ledger Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new bet")
    add_parser.add_argument("--url", required=True, help="HLTV match URL")
    add_parser.add_argument("--bookmaker", required=True, help="Bookmaker name")
    add_parser.add_argument("--odds", type=float, required=True, help="Decimal odds")
    add_parser.add_argument("--bet", required=True, help="Team name or outcome you bet on")
    add_parser.add_argument("--amount", type=float, default=1.0, help="Stake amount (default: 1.0)")
    add_parser.add_argument("--model-prob", type=float, help="Model's predicted win probability for your bet (0.0-1.0)")

    # Refresh command
    subparsers.add_parser("refresh", help="Update pending match results")

    # List command
    subparsers.add_parser("list", help="Shows the current ledger")

    args = parser.parse_args()

    if args.command == "add":
        add_bet(args)
    elif args.command == "refresh":
        refresh_ledger()
    elif args.command == "list":
        list_ledger()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
