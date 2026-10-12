"""Central configuration: paths, constants and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

ARTIFACTS_FILENAME = "artifacts.joblib"
METRICS_FILENAME = "metrics.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20                # fraction of each user's ratings held out for testing
RELEVANCE_THRESHOLD = 4.0       # a test rating >= 4.0 means "the user liked it"
DEFAULT_K = 10                  # K for Precision@K / Recall@K / F1@K
MAX_EVAL_USERS = 300            # users sampled for the (slower) ranking evaluation
MIN_RATINGS_FOR_RECS = 5        # hybrid ignores movies with fewer training ratings
RATING_MIN, RATING_MAX = 0.5, 5.0

ALGORITHMS = [
    "Hybrid Recommendation",
    "Content-Based",
    "Collaborative Filtering",
    "SVD",
    "Popularity-Based",
]


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging once (safe to call repeatedly)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
