import time
import logging
from typing import Dict, Any, List, Optional
import requests
from requests.exceptions import HTTPError, RequestException

from config import PANDASCORE_API_KEY, PANDASCORE_BASE_URL, REQUEST_DELAY_S, MAX_RETRIES, DEFAULT_PAGE_SIZE

logger = logging.getLogger(__name__)

class PandaScoreClient:
    """
    Client for interacting with the PandaScore API.
    Handles authentication, rate limiting, retries, and pagination.
    """
    def __init__(self):
        if not PANDASCORE_API_KEY or PANDASCORE_API_KEY == "your_api_key_here":
            raise ValueError("PANDASCORE_API_KEY is not set or is the default template. Please set it in .env")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {PANDASCORE_API_KEY}",
            "Accept": "application/json"
        })
        self.base_url = PANDASCORE_BASE_URL
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Ensures we do not exceed the rate limit by sleeping if necessary."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < REQUEST_DELAY_S:
            sleep_time = REQUEST_DELAY_S - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """Internal method to make a request with retries and rate limiting."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            
            try:
                response = self.session.request(method, url, params=params)
                
                # Check for rate limit or server errors
                if response.status_code == 429:
                    logger.warning("429 Too Many Requests. Rate limit exceeded.")
                    if attempt < MAX_RETRIES:
                        sleep_time = REQUEST_DELAY_S * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        response.raise_for_status()
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}.")
                    if attempt < MAX_RETRIES:
                        time.sleep(2)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                return response
                
            except RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == MAX_RETRIES:
                    raise

        raise Exception("Max retries exceeded")

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Makes a simple GET request without pagination."""
        response = self._request("GET", endpoint, params)
        return response.json()

    def get_paginated(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Makes a GET request and exhausts cursor-based pagination."""
        all_results = []
        current_page = 1
        
        if params is None:
            params = {}
            
        params["page[size]"] = DEFAULT_PAGE_SIZE

        while True:
            params["page[number]"] = current_page
            logger.info(f"Fetching {endpoint} - Page {current_page}")
            
            response = self._request("GET", endpoint, params)
            data = response.json()
            
            if not data:
                break
                
            all_results.extend(data)
            
            # Check if there are more pages by looking at the Link header, 
            # or just checking if we got a full page of results
            if len(data) < DEFAULT_PAGE_SIZE:
                break
                
            current_page += 1
            
        return all_results
