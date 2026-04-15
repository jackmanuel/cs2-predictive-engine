"""
Shadow Ledger — SQLite-backed model calibration and odds tracking system.

Three tables:
  - model_versions: Registry of every training run (features, hyperparams, weights)
  - matches:        One row per unique match (metadata + result)
  - snapshots:      One row per prediction run × match (odds, model_prob, edge)

Usage:
    python -m evaluation.shadow_ledger refresh   # Resolve pending bets via HLTV
    python -m evaluation.shadow_ledger report    # Show calibration analysis
    python -m evaluation.shadow_ledger list      # Show all shadow bets
    python -m evaluation.shadow_ledger versions  # Show model version history
"""

import os
import sys
import sqlite3
import hashlib
import shutil
import json
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.hltv_client import HLTVClient

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "shadow_ledger.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    version_id      TEXT PRIMARY KEY,
    trained_at       TEXT NOT NULL,
    best_val_loss    REAL,
    epochs_run       INTEGER,
    num_features     INTEGER,
    features_json    TEXT,
    hyperparams_json TEXT,
    data_stats_json  TEXT,
    architecture_hash TEXT,
    weights_path     TEXT,
    scaler_path      TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_url   TEXT PRIMARY KEY,
    team_a      TEXT NOT NULL,
    team_b      TEXT NOT NULL,
    format      TEXT,
    match_date  TEXT,
    match_time  TEXT,
    first_seen  TEXT NOT NULL,
    result      TEXT DEFAULT 'Pending'
);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_url       TEXT NOT NULL REFERENCES matches(match_url),
    version_id      TEXT REFERENCES model_versions(version_id),
    timestamp       TEXT NOT NULL,
    model_prob_a    REAL,
    odds_a          REAL,
    odds_b          REAL,
    implied_prob_a  REAL,
    implied_prob_b  REAL,
    edge_a          REAL,
    edge_b          REAL,
    best_bet        TEXT,
    best_edge       REAL
);
"""


def get_db():
    """Returns a connection to the shadow ledger database, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Model Version Registry
# ---------------------------------------------------------------------------

def compute_architecture_hash():
    """Computes a SHA256 hash of the model architecture source file."""
    net_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "net.py")
    try:
        with open(net_path, "r", encoding="utf-8") as f:
            return hashlib.sha256(f.read().encode()).hexdigest()[:16]
    except FileNotFoundError:
        return "unknown"


def register_model_version(
    trained_at: str,
    best_val_loss: float,
    epochs_run: int,
    features: list,
    hyperparams: dict,
    data_stats: dict,
    weights_src: str,
    scaler_src: str,
):
    """
    Registers a new model version in the database and archives the weights/scaler.
    Called by train.py after each training run.
    Returns the version_id.
    """
    # Generate version ID from training timestamp
    version_id = "v_" + trained_at.replace("-", "").replace(" ", "_").replace(":", "")

    # Archive the weights and scaler
    archive_dir = os.path.join("data", "checkpoints", "archive", version_id)
    os.makedirs(archive_dir, exist_ok=True)

    weights_dst = os.path.join(archive_dir, "best_mvp_model.pt")
    scaler_dst = os.path.join(archive_dir, "scaler.pkl")

    if os.path.exists(weights_src):
        shutil.copy2(weights_src, weights_dst)
    if os.path.exists(scaler_src):
        shutil.copy2(scaler_src, scaler_dst)

    arch_hash = compute_architecture_hash()

    conn = get_db()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO model_versions
               (version_id, trained_at, best_val_loss, epochs_run,
                num_features, features_json, hyperparams_json,
                data_stats_json, architecture_hash, weights_path, scaler_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                trained_at,
                best_val_loss,
                epochs_run,
                len(features),
                json.dumps(features),
                json.dumps(hyperparams),
                json.dumps(data_stats),
                arch_hash,
                weights_dst,
                scaler_dst,
            ),
        )
        conn.commit()
        logger.info(f"Registered model version {version_id} (val_loss={best_val_loss:.4f}, arch={arch_hash})")
    finally:
        conn.close()

    return version_id


def get_latest_version_id():
    """Returns the version_id of the most recently trained model."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT version_id FROM model_versions ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
        return row["version_id"] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prediction Recording
# ---------------------------------------------------------------------------

def record_predictions(match_results: list, version_id: str = None):
    """
    Called by automate_predictions.py after generating predictions.
    Upserts matches and appends snapshot rows for every prediction.
    """
    if version_id is None:
        version_id = get_latest_version_id()

    conn = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    try:
        for item in match_results:
            url = item["match"].get("url", "")
            if not url:
                continue

            # Upsert match (insert-or-ignore keeps first_seen and existing result)
            conn.execute(
                """INSERT OR IGNORE INTO matches
                   (match_url, team_a, team_b, format, match_date, match_time, first_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    url,
                    item["team_a"],
                    item["team_b"],
                    item["fmt"],
                    item["match"].get("date"),
                    item["match"].get("time"),
                    now,
                ),
            )

            prob_a = item["prob1"]
            o_a = item.get("o1")
            o_b = item.get("o2")
            edge_a = item.get("edge1")
            edge_b = item.get("edge2")

            if edge_a is not None and edge_b is not None:
                best_bet = "team_a" if edge_a >= edge_b else "team_b"
                best_edge = max(edge_a, edge_b)
            else:
                best_bet = "team_a" if prob_a >= 0.5 else "team_b"
                best_edge = None

            conn.execute(
                """INSERT INTO snapshots
                   (match_url, version_id, timestamp, model_prob_a,
                    odds_a, odds_b, implied_prob_a, implied_prob_b,
                    edge_a, edge_b, best_bet, best_edge)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    url,
                    version_id,
                    now,
                    round(prob_a, 4),
                    o_a,
                    o_b,
                    round(1.0 / o_a, 4) if o_a else None,
                    round(1.0 / o_b, 4) if o_b else None,
                    round(edge_a, 2) if edge_a is not None else None,
                    round(edge_b, 2) if edge_b is not None else None,
                    best_bet,
                    round(best_edge, 2) if best_edge is not None else None,
                ),
            )
            added += 1

        conn.commit()
        total_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        total_snaps = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        logger.info(
            f"Shadow ledger: {added} snapshots recorded "
            f"(version={version_id}, {total_matches} matches, {total_snaps} total snapshots)"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Refresh (resolve pending results)
# ---------------------------------------------------------------------------

def refresh_shadow():
    """Resolves pending shadow bets by checking HLTV match results.
    Only checks matches whose scheduled start time has already passed."""
    conn = get_db()
    try:
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M")

        # Only check matches that should have started by now.
        # Matches without a date are always checked (no data to filter on).
        pending = conn.execute(
            """SELECT match_url, team_a, team_b, match_date, match_time
               FROM matches
               WHERE result = 'Pending'
                 AND (match_date IS NULL
                      OR match_date || ' ' || COALESCE(match_time, '00:00') <= ?)""",
            (now_str,),
        ).fetchall()

        all_pending = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE result = 'Pending'"
        ).fetchone()[0]
        skipped = all_pending - len(pending)

        if not pending:
            if skipped > 0:
                logger.info(f"No matches ready to resolve ({skipped} still upcoming).")
            else:
                logger.info("No pending shadow bets to resolve.")
            return

        logger.info(f"Resolving {len(pending)} past matches ({skipped} upcoming, skipped)...")
        client = HLTVClient()
        updated = 0

        try:
            for row in pending:
                try:
                    details = client.fetch_match_details(row["match_url"])
                    meta = details["metadata"]

                    if meta["is_finished"]:
                        winner = meta.get("winner", "")
                        t_a = str(row["team_a"]).strip()
                        t_b = str(row["team_b"]).strip()

                        if winner and t_a.lower() in winner.lower():
                            result = "team_a"
                        elif winner and t_b.lower() in winner.lower():
                            result = "team_b"
                        else:
                            result = f"unknown:{winner}"

                        conn.execute(
                            "UPDATE matches SET result = ? WHERE match_url = ?",
                            (result, row["match_url"]),
                        )
                        updated += 1
                        logger.info(f"  Resolved: {t_a} vs {t_b} -> {result}")
                except Exception as e:
                    logger.error(f"  Error resolving {row['match_url']}: {e}")

            if updated > 0:
                conn.commit()
            logger.info(f"Resolved {updated}/{len(pending)} past matches.")
        finally:
            client.stop()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def show_report():
    """Shows calibration analysis of resolved shadow bets."""
    conn = get_db()
    try:
        # Use each match's LATEST snapshot for analysis
        df = pd.read_sql_query(
            """
            SELECT m.match_url, m.team_a, m.team_b, m.result,
                   s.model_prob_a, s.best_bet, s.best_edge, s.version_id
            FROM matches m
            JOIN (
                SELECT match_url, MAX(id) as max_id
                FROM snapshots
                GROUP BY match_url
            ) latest ON m.match_url = latest.match_url
            JOIN snapshots s ON s.id = latest.max_id
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("Shadow ledger is empty.")
        return

    settled = df[df["result"].isin(["team_a", "team_b"])].copy()
    pending = df[df["result"] == "Pending"]

    print(f"\n{'='*60}")
    print(f" SHADOW LEDGER CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f" Total: {len(df)} | Settled: {len(settled)} | Pending: {len(pending)}")
    print(f"{'='*60}")

    if settled.empty:
        print("\n No settled bets to analyse yet.")
        return

    # Model favourite accuracy
    settled["model_fav"] = settled["model_prob_a"].apply(
        lambda p: "team_a" if p >= 0.5 else "team_b"
    )
    settled["fav_correct"] = settled["model_fav"] == settled["result"]
    overall_acc = settled["fav_correct"].mean() * 100
    print(
        f"\n Model Favourite Accuracy: {settled['fav_correct'].sum()}/{len(settled)} ({overall_acc:.1f}%)"
    )

    # Edge-bucket analysis
    has_edge = settled[settled["best_edge"].notna()].copy()
    if not has_edge.empty:
        has_edge["edge_bet_won"] = has_edge["best_bet"] == has_edge["result"]

        print(f"\n{'─'*60}")
        print(f" Edge Bucket Analysis (best edge side)")
        print(f" {'Bucket':>10} | {'W':>4} | {'L':>4} | {'Win%':>6} | {'Avg Edge':>9}")
        print(f" {'-'*10}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*9}")

        buckets = [(0, 2, "  <2%"), (2, 5, " 2-5%"), (5, 10, "5-10%"), (10, 999, " 10%+")]
        for lo, hi, label in buckets:
            bucket = has_edge[
                (has_edge["best_edge"] >= lo) & (has_edge["best_edge"] < hi)
            ]
            if not bucket.empty:
                w = bucket["edge_bet_won"].sum()
                l = len(bucket) - w
                wr = w / len(bucket) * 100
                avg_e = bucket["best_edge"].mean()
                print(f" {label:>10} | {w:>4} | {l:>4} | {wr:>5.1f}% | {avg_e:>+8.1f}%")

    # Favourite vs underdog
    if has_edge is not None and not has_edge.empty:
        has_edge["bet_is_fav"] = has_edge.apply(
            lambda r: (r["best_bet"] == "team_a" and r["model_prob_a"] >= 0.5)
            or (r["best_bet"] == "team_b" and r["model_prob_a"] < 0.5),
            axis=1,
        )

        fav_bets = has_edge[has_edge["bet_is_fav"]]
        dog_bets = has_edge[~has_edge["bet_is_fav"]]

        print(f"\n{'─'*60}")
        print(f" Favourite vs Underdog (edge-side bets only)")

        if not fav_bets.empty:
            fw = fav_bets["edge_bet_won"].sum()
            print(f" Backing favourites:  {fw}/{len(fav_bets)} ({fw/len(fav_bets)*100:.1f}%)")
        if not dog_bets.empty:
            dw = dog_bets["edge_bet_won"].sum()
            print(f" Backing underdogs:   {dw}/{len(dog_bets)} ({dw/len(dog_bets)*100:.1f}%)")

    # Version breakdown
    versions = settled["version_id"].value_counts()
    if len(versions) > 1:
        print(f"\n{'─'*60}")
        print(f" Performance by Model Version")
        for vid, count in versions.items():
            v_df = settled[settled["version_id"] == vid]
            v_acc = v_df["fav_correct"].mean() * 100
            print(f" {vid}: {v_df['fav_correct'].sum()}/{count} ({v_acc:.1f}%)")

    print(f"\n{'='*60}\n")


def show_list():
    """Shows all matches with their latest snapshot."""
    conn = get_db()
    try:
        df = pd.read_sql_query(
            """
            SELECT m.match_date, m.team_a, m.team_b, m.format,
                   s.model_prob_a, s.best_bet, s.best_edge, s.odds_a, s.odds_b,
                   m.result, s.version_id,
                   (SELECT COUNT(*) FROM snapshots WHERE match_url = m.match_url) as num_snapshots
            FROM matches m
            JOIN (
                SELECT match_url, MAX(id) as max_id
                FROM snapshots
                GROUP BY match_url
            ) latest ON m.match_url = latest.match_url
            JOIN snapshots s ON s.id = latest.max_id
            ORDER BY m.match_date, m.match_time
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("Shadow ledger is empty.")
        return

    print("\n--- Shadow Ledger (Latest Snapshots) ---")
    print(df.to_string(index=False))
    total = len(df)
    pending = (df["result"] == "Pending").sum()
    print(f"\nTotal: {total} | Pending: {pending}")


def show_versions():
    """Shows all registered model versions."""
    conn = get_db()
    try:
        df = pd.read_sql_query(
            """
            SELECT v.version_id, v.trained_at, v.best_val_loss, v.epochs_run,
                   v.num_features, v.architecture_hash,
                   (SELECT COUNT(DISTINCT s.match_url) FROM snapshots s WHERE s.version_id = v.version_id) as matches_predicted,
                   json_extract(v.data_stats_json, '$.total_maps') as training_maps
            FROM model_versions v
            ORDER BY v.trained_at DESC
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("No model versions registered yet.")
        return

    print("\n--- Model Version History ---")
    print(df.to_string(index=False))


def show_odds_history(match_url: str):
    """Shows the full odds/prediction history for a specific match."""
    conn = get_db()
    try:
        match = conn.execute(
            "SELECT * FROM matches WHERE match_url = ?", (match_url,)
        ).fetchone()

        if not match:
            print(f"Match not found: {match_url}")
            return

        print(f"\n--- Odds History: {match['team_a']} vs {match['team_b']} ({match['format']}) ---")
        print(f"Result: {match['result']}\n")

        df = pd.read_sql_query(
            """
            SELECT timestamp, version_id, model_prob_a, odds_a, odds_b,
                   implied_prob_a, implied_prob_b, edge_a, edge_b, best_bet, best_edge
            FROM snapshots
            WHERE match_url = ?
            ORDER BY timestamp
            """,
            conn,
            params=(match_url,),
        )
        print(df.to_string(index=False))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shadow Ledger — Model calibration and odds tracking"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("refresh", help="Resolve pending shadow bets via HLTV")
    subparsers.add_parser("report", help="Show calibration analysis")
    subparsers.add_parser("list", help="Show all shadow bets (latest snapshot per match)")
    subparsers.add_parser("versions", help="Show model version history")

    odds_parser = subparsers.add_parser("odds", help="Show odds history for a match")
    odds_parser.add_argument("url", help="HLTV match URL")

    args = parser.parse_args()

    if args.command == "refresh":
        refresh_shadow()
    elif args.command == "report":
        show_report()
    elif args.command == "list":
        show_list()
    elif args.command == "versions":
        show_versions()
    elif args.command == "odds":
        show_odds_history(args.url)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
