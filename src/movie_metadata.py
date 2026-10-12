"""Optional TMDb integration for posters and extra metadata.

The core recommender never depends on this module. Without an API key, or on any
network / API problem, functions simply return ``None`` and the UI shows a local
placeholder. The key is read from the TMDB_API_KEY environment variable, a local
``.env`` file, or Streamlit secrets - it is never hard-coded and never logged.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
IMAGE_URL = "https://image.tmdb.org/t/p/w342{path}"
TIMEOUT_SECONDS = 4

_lock = threading.Lock()
_cache: dict[int, dict | None] = {}
_blocked_until = 0.0          # circuit breaker (rate limit / repeated failures)
_failures = 0
_key_rejected = False


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except OSError:
        pass


def get_api_key() -> str | None:
    """API key from environment / .env / Streamlit secrets, or None."""
    _load_dotenv()
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        try:                                    # Streamlit secrets are optional
            import streamlit as st
            secret = st.secrets.get("TMDB_API_KEY", "")
            key = secret.strip() if isinstance(secret, str) else ""
        except Exception:                       # no streamlit / no secrets file
            key = ""
    return key or None


def is_enabled() -> bool:
    return get_api_key() is not None and not _key_rejected


def _trip_breaker(seconds: float) -> None:
    global _blocked_until
    _blocked_until = time.time() + seconds


def fetch_movie_metadata(tmdb_id: int) -> dict | None:
    """Return {'poster_url', 'overview', 'vote_average', 'release_date'} or None.

    Never raises. Results (including misses) are cached for the process lifetime.
    """
    global _failures, _key_rejected
    tmdb_id = int(tmdb_id)
    with _lock:
        if tmdb_id in _cache:
            return _cache[tmdb_id]
        if _key_rejected or time.time() < _blocked_until:
            return None
    key = get_api_key()
    if not key:
        return None

    headers, params = {}, {}
    if key.startswith("eyJ"):                   # v4 read-access token (JWT)
        headers["Authorization"] = f"Bearer {key}"
    else:                                       # v3 API key
        params["api_key"] = key
    result: dict | None = None
    try:
        response = requests.get(API_URL.format(tmdb_id=tmdb_id), params=params, headers=headers,
                                timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            data = response.json()
            poster = data.get("poster_path")
            result = {
                "poster_url": IMAGE_URL.format(path=poster) if poster else None,
                "overview": data.get("overview") or None,
                "vote_average": data.get("vote_average"),
                "release_date": data.get("release_date") or None,
            }
            _failures = 0
        elif response.status_code == 404:
            result = None                       # unknown id: cache the miss
        elif response.status_code in (401, 403):
            _key_rejected = True
            logger.warning("TMDb rejected the API key (HTTP %s); posters disabled.", response.status_code)
            return None
        elif response.status_code == 429:
            retry = response.headers.get("Retry-After", "30")
            _trip_breaker(min(float(retry) if retry.isdigit() else 30.0, 120.0))
            logger.warning("TMDb rate limit reached; pausing requests.")
            return None
        else:
            logger.warning("TMDb returned HTTP %s", response.status_code)
            return None
    except (requests.RequestException, ValueError) as exc:
        # log only the exception class: request errors can echo the URL (and key)
        _failures += 1
        if _failures >= 3:
            _trip_breaker(60.0)
        logger.warning("TMDb request failed (%s)", type(exc).__name__)
        return None
    with _lock:
        _cache[tmdb_id] = result
    return result


def fetch_poster_urls(tmdb_ids: list[int]) -> dict[int, str | None]:
    """Poster URL per TMDb id, fetched concurrently; missing posters map to None."""
    ids = [int(i) for i in dict.fromkeys(tmdb_ids)]
    if not ids or not is_enabled():
        return {i: None for i in ids}
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata = list(pool.map(fetch_movie_metadata, ids))
    return {i: (m or {}).get("poster_url") for i, m in zip(ids, metadata)}
