import argparse
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HLTV_MATCHES_FILE
from ingestion.hltv_client import HLTVClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_team(name):
    return " ".join(name.split()).casefold()


def load_team_whitelist(path):
    """Map normalized names to the first readable spelling in the CSV."""
    teams = {}
    try:
        with open(path, encoding="utf-8-sig", newline="") as source:
            rows = csv.reader(source, strict=True)
            header = next(rows, [])
            if header.count("team") != 1:
                raise ValueError("expected exactly one 'team' column")
            team_column = header.index("team")
            for row in rows:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f"wrong number of columns at line {rows.line_num}")
                name = " ".join(row[team_column].split())
                if name:
                    teams.setdefault(normalize_team(name), name)
    except (OSError, ValueError, csv.Error) as exc:
        raise ValueError(f"Cannot read team whitelist {path}: {exc}") from exc
    if not teams:
        raise ValueError(f"Team whitelist {path} has no usable teams")
    return teams


def matching_teams(match, whitelist):
    return {normalize_team(match["team1"]), normalize_team(match["team2"])} & whitelist.keys()


def matches_whitelist(match, whitelist, mode):
    if mode == "both":
        return all(normalize_team(match[side]) in whitelist for side in ("team1", "team2"))
    return bool(matching_teams(match, whitelist))


def parse_date(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("expected YYYY-MM-DD")
    return parsed


def date_argument(value):
    try:
        return parse_date(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a valid YYYY-MM-DD date, got {value!r}") from None


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Scrape HLTV matches to serve as the canonical dataset.")
    parser.add_argument("--pages", type=positive_int, help="Maximum HLTV results pages to scan.")
    parser.add_argument("--matches", "--count", type=positive_int, dest="count", help="Maximum eligible matches to scrape.")
    parser.add_argument("--start-date", type=date_argument, help="Inclusive earliest match date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=date_argument, help="Inclusive latest match date (YYYY-MM-DD).")
    parser.add_argument("--team-whitelist", type=Path, help="UTF-8 CSV with a team column.")
    parser.add_argument("--team-match", choices=("any", "both"), help="Require any or both teams to be listed (default: any).")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates using results listings only.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry stored matches with an error field.")
    args = parser.parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    if args.team_match and not args.team_whitelist:
        parser.error("--team-match requires --team-whitelist")
    args.team_match = args.team_match or "any"
    if not (args.pages or args.count or args.start_date or args.end_date):
        args.pages = 3
    try:
        whitelist = load_team_whitelist(args.team_whitelist) if args.team_whitelist else {}
    except ValueError as exc:
        parser.error(str(exc))
    return args, whitelist


def load_matches(path):
    try:
        with path.open(encoding="utf-8") as source:
            records = json.load(source)
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError("expected a JSON array of match objects")
    except FileNotFoundError:
        if not path.is_symlink():
            return [], {}
        raise ValueError(f"Canonical data link {path} has no target") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read canonical data at {path}. Repair or restore the file before retrying. {exc}") from exc

    stored = {}
    for index, record in enumerate(records):
        url = record.get("url")
        if isinstance(url, str) and url:
            # Prefer a successful record if the corpus already contains this URL more than once.
            if url not in stored or "error" in records[stored[url]]:
                stored[url] = index
    return records, stored


def save_matches(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(records, destination, indent=2)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def listing_date(match):
    try:
        return parse_date(match.get("date"))
    except (ValueError, TypeError):
        return None


def discover_matches(client, args, whitelist, records, stored):
    seen = set()
    eligible = []
    counts = Counter()
    pages = 0
    dated = args.start_date is not None or args.end_date is not None
    dated_order = True
    previous_date = None

    while True:
        logger.info("Fetching HLTV results page %d", pages + 1)
        try:
            results = client.fetch_recent_results(pages=1, start_page=pages)
        except Exception as exc:
            reason = f"abnormal listing error on page {pages + 1} ({exc})"
            break
        pages += 1
        if not results:
            reason = "exhaustion"
            break

        previous_total = len(seen)
        full_page_dated = True
        for match in results:
            match_date = listing_date(match) if dated else None
            if dated:
                if match_date is None:
                    full_page_dated = False
                else:
                    if previous_date and match_date > previous_date:
                        dated_order = False
                    previous_date = match_date

            url = match.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            if dated and match_date is None:
                counts["unknown dates"] += 1
            elif (args.start_date and match_date < args.start_date) or (args.end_date and match_date > args.end_date):
                counts["outside date bounds"] += 1
            elif whitelist and not matches_whitelist(match, whitelist, args.team_match):
                counts["team excluded"] += 1
            elif url in stored:
                if "error" not in records[stored[url]]:
                    counts["stored successes skipped"] += 1
                elif args.retry_errors:
                    counts["eligible retries"] += 1
                    eligible.append(match)
                else:
                    counts["stored failures skipped"] += 1
            else:
                counts["eligible new"] += 1
                eligible.append(match)

        if len(seen) == previous_total:
            reason = "abnormal pagination (page contained no new match URLs)"
            break
        # The whole page must support the newest-first ordering before we trust its last date.
        if args.start_date and full_page_dated and dated_order and previous_date < args.start_date:
            reason = "date boundary"
            break
        if args.count and len(eligible) >= args.count:
            reason = "match cap"
            break
        if args.pages and pages >= args.pages:
            reason = "page cap"
            break

    return eligible[:args.count], counts, pages, len(seen), reason


def report_discovery(args, whitelist, selected, counts, pages, unique, reason):
    logger.info(
        "Discovery settings | dates %s through %s (inclusive) | whitelist %s | team match %s",
        args.start_date or "unbounded", args.end_date or "unbounded",
        args.team_whitelist or "none", args.team_match if whitelist else "disabled",
    )
    logger.info("Scanned %d listing pages and discovered %d unique match URLs", pages, unique)
    logger.info("Counts use the first applicable outcome in date, team, then stored-status order.")
    for label in (
        "unknown dates", "outside date bounds", "team excluded", "stored successes skipped",
        "stored failures skipped", "eligible new", "eligible retries",
    ):
        logger.info("%s = %d", label.capitalize(), counts[label])
    logger.info("Selected %d after limits. Discovery stopped by %s.", len(selected), reason)
    if reason not in ("exhaustion", "date boundary"):
        logger.warning("Discovery may cover only part of the requested range or listings.")
    if counts["unknown dates"]:
        logger.warning("Matches with missing or unparseable dates were excluded. Date coverage may be incomplete.")

    if args.dry_run:
        for match in selected:
            logger.info("%s | %s vs %s | %s", match.get("date") or "Unknown date",
                        match["team1"], match["team2"], match["url"])
    if whitelist:
        team_counts = Counter(team for match in selected for team in matching_teams(match, whitelist))
        for team, name in whitelist.items():
            logger.info("%s | %d selected", name, team_counts[team])
        no_matches = [name for team, name in whitelist.items() if not team_counts[team]]
        logger.info("Listed teams with zero selected matches | %s", ", ".join(no_matches) or "none")


def scrape_matches(client, selected, path, records, stored):
    consecutive_errors = 0
    for number, listing in enumerate(selected, start=1):
        match = dict(listing)
        url = match["url"]
        logger.info("[%d/%d] Scraping %s vs %s (%s) | %s", number, len(selected),
                    match["team1"], match["team2"], match.get("date"), url)
        try:
            details = client.fetch_match_details(url)
            for map_data in details["maps"]:
                stats_url = map_data.get("stats_url")
                map_stats = client.fetch_map_stats(stats_url) if stats_url else {}
                map_data["round_history"] = map_stats.get("round_history", "")
                map_data["player_stats"] = map_stats.get("player_stats", [])

            match["match_info"] = details.get("match_info", [])
            match["player_stats"] = details.get("player_stats", [])
            match["team_ranks"] = details.get("team_ranks", {})
            match["hltv_vetoes"] = details.get("vetoes", [])
            match["hltv_maps"] = details["maps"]
            match.pop("error", None)
            consecutive_errors = 0
        except Exception as exc:
            logger.error("Failed to scrape details for %s: %s", url, exc)
            match["error"] = str(exc)
            consecutive_errors += 1

        if url in stored:
            records[stored[url]] = match
        else:
            records.append(match)
        save_matches(path, records)

        if consecutive_errors >= 3:
            raise RuntimeError("Hit 3 consecutive errors during scraping details")
        if number % 5 == 0 and number < len(selected):
            break_time = random.uniform(60.0, 120.0)
            logger.info("Pausing between batches for %.1f seconds", break_time)
            time.sleep(break_time)

    logger.info("Batch complete. Canonical dataset contains %d records.", len(records))


def main(argv=None):
    args, whitelist = parse_args(argv)
    path = Path(HLTV_MATCHES_FILE)
    client = None
    try:
        records, stored = load_matches(path)
        client = HLTVClient()
        selected, counts, pages, unique, reason = discover_matches(client, args, whitelist, records, stored)
        report_discovery(args, whitelist, selected, counts, pages, unique, reason)
        if reason.startswith("abnormal"):
            return 1
        if not args.dry_run and selected:
            scrape_matches(client, selected, path, records, stored)
        return 0
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        return 1
    finally:
        if client is not None:
            client.stop()


if __name__ == "__main__":
    sys.exit(main())
