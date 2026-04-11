import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import RAW_DIR, PANDASCORE_CSGO_PREFIX
from ingestion.client import PandaScoreClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.warning("DEPRECATION: ingestion.fetch_matches (PandaScore) is deprecated. Use ingestion.fetch_hltv_matches instead.")

def fetch_past_matches(client: PandaScoreClient, days_back: int = 90):
    """
    Fetches raw JSON matches from the past `days_back` days.
    """
    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    logger.info(f"Fetching matches since {start_date_str}")
    
    endpoint = f"{PANDASCORE_CSGO_PREFIX}/matches/past"
    params = {
        "range[begin_at]": f"{start_date_str},{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "sort": "begin_at" # oldest first
    }
    
    matches = client.get_paginated(endpoint, params)
    
    logger.info(f"Fetched {len(matches)} matches.")
    
    # Save to raw
    output_path = RAW_DIR / f"matches_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
        
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    client = PandaScoreClient()
    # Getting the last 90 days of matches.
    fetch_past_matches(client, days_back=90)
