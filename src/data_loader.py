"""Loading, validating and cleaning the MovieLens dataset.

Expected layout (MovieLens "latest-small")::

    data/ml-latest-small/movies.csv   (movieId, title, genres)
    data/ml-latest-small/ratings.csv  (userId, movieId, rating, timestamp)
    data/ml-latest-small/links.csv    (optional: movieId, imdbId, tmdbId)
    data/ml-latest-small/tags.csv     (optional: userId, movieId, tag, timestamp)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, RANDOM_SEED, RATING_MAX, RATING_MIN, TEST_SIZE
from .exceptions import DatasetError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "movies.csv": ["movieId", "title", "genres"],
    "ratings.csv": ["userId", "movieId", "rating"],
}

DOWNLOAD_HINT = (
    "Download 'ml-latest-small.zip' from https://grouplens.org/datasets/movielens/latest/ "
    "and extract it so that 'movies.csv' and 'ratings.csv' are inside "
    "'data/ml-latest-small/'. See data/README.md for details."
)


@dataclass
class Dataset:
    """Cleaned MovieLens tables plus a report of what cleaning did."""

    movies: pd.DataFrame
    ratings: pd.DataFrame
    links: pd.DataFrame | None = None
    tags: pd.DataFrame | None = None
    source_dir: Path | None = None
    cleaning_report: dict = field(default_factory=dict)


def find_dataset_dir(data_dir: str | Path = DATA_DIR) -> Path:
    """Locate the folder that contains movies.csv and ratings.csv."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        if data_dir == DATA_DIR and (data_dir.parent / "movies.csv").is_file() and (data_dir.parent / "ratings.csv").is_file():
            return data_dir.parent
        raise DatasetError(f"Data folder '{data_dir}' does not exist. {DOWNLOAD_HINT}")

    candidates = [data_dir] + sorted(p for p in data_dir.iterdir() if p.is_dir())
    for candidate in candidates:
        if (candidate / "movies.csv").is_file() and (candidate / "ratings.csv").is_file():
            return candidate

    # Fallback to project root if default DATA_DIR doesn't have the files but root does
    if data_dir == DATA_DIR and (data_dir.parent / "movies.csv").is_file() and (data_dir.parent / "ratings.csv").is_file():
        return data_dir.parent

    raise DatasetError(
        f"Could not find 'movies.csv' and 'ratings.csv' inside '{data_dir}'. {DOWNLOAD_HINT}"
    )


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    """Read a CSV file and verify that the required columns exist."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"'{path.name}' is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"'{path.name}' is not a valid CSV file.") from exc
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise DatasetError(
            f"'{path.name}' is missing required column(s): {', '.join(missing)}. "
            f"Found columns: {', '.join(map(str, frame.columns))}."
        )
    return frame


def clean_movies(movies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop invalid / duplicate movies and fill missing genres."""
    report = {"movies_raw": int(len(movies))}
    movies = movies.dropna(subset=["movieId", "title"]).copy()
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype("int64")
    movies["title"] = movies["title"].astype(str).str.strip()
    movies = movies[movies["title"] != ""]
    before = len(movies)
    movies = movies.drop_duplicates(subset="movieId", keep="first")
    report["duplicate_movies_removed"] = int(before - len(movies))
    missing_genres = int(movies["genres"].isna().sum())
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str)
    report["missing_genres_filled"] = missing_genres
    movies = movies.sort_values("movieId").reset_index(drop=True)
    report["movies_clean"] = int(len(movies))
    return movies, report


def clean_ratings(ratings: pd.DataFrame, valid_movie_ids: set[int]) -> tuple[pd.DataFrame, dict]:
    """Coerce types, drop missing/invalid values and duplicate (user, movie) pairs."""
    report = {"ratings_raw": int(len(ratings))}
    ratings = ratings.copy()
    for col in ("userId", "movieId", "rating"):
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")
    before = len(ratings)
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    report["rows_with_missing_values_removed"] = int(before - len(ratings))

    before = len(ratings)
    ratings = ratings[(ratings["rating"] >= RATING_MIN) & (ratings["rating"] <= RATING_MAX)]
    report["rows_with_out_of_range_rating_removed"] = int(before - len(ratings))

    before = len(ratings)
    ratings = ratings[ratings["movieId"].isin(valid_movie_ids)]
    report["rows_with_unknown_movie_removed"] = int(before - len(ratings))

    ratings["userId"] = ratings["userId"].astype("int64")
    ratings["movieId"] = ratings["movieId"].astype("int64")
    ratings["rating"] = ratings["rating"].astype("float64")

    if "timestamp" in ratings.columns:
        ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")
        ratings = ratings.sort_values("timestamp", kind="stable")
    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    report["duplicate_ratings_removed"] = int(before - len(ratings))
    ratings = ratings.sort_values(["userId", "movieId"]).reset_index(drop=True)
    report["ratings_clean"] = int(len(ratings))
    if ratings.empty:
        raise DatasetError("No valid ratings remain after cleaning; check ratings.csv.")
    return ratings, report


def load_dataset(data_dir: str | Path = DATA_DIR) -> Dataset:
    """Load, validate and clean the MovieLens files.

    Raises:
        DatasetError: if files are missing, unreadable or malformed.
    """
    source = find_dataset_dir(data_dir)
    logger.info("Loading MovieLens data from %s", source)

    movies, movie_report = clean_movies(_read_csv(source / "movies.csv", REQUIRED_COLUMNS["movies.csv"]))
    ratings, rating_report = clean_ratings(
        _read_csv(source / "ratings.csv", REQUIRED_COLUMNS["ratings.csv"]),
        set(movies["movieId"].tolist()),
    )

    links = None
    if (source / "links.csv").is_file():
        try:
            links = _read_csv(source / "links.csv", ["movieId", "tmdbId"])[["movieId", "tmdbId"]]
            links = links.dropna().drop_duplicates("movieId")
            links = links.astype({"movieId": "int64", "tmdbId": "int64"})
        except (DatasetError, ValueError) as exc:
            logger.warning("Ignoring links.csv: %s", exc)
            links = None

    tags = None
    if (source / "tags.csv").is_file():
        try:
            tags = _read_csv(source / "tags.csv", ["movieId", "tag"])[["movieId", "tag"]].dropna()
        except DatasetError as exc:
            logger.warning("Ignoring tags.csv: %s", exc)
            tags = None

    report = {**movie_report, **rating_report}
    return Dataset(movies=movies, ratings=ratings, links=links, tags=tags,
                   source_dir=source, cleaning_report=report)


def train_test_split_ratings(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    min_ratings_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user random hold-out split.

    For every user with at least ``min_ratings_per_user`` ratings, ``test_size`` of
    their ratings (rounded down, at least 1) go to the test set. Users with fewer
    ratings stay entirely in the training set. Every test user therefore also has
    training history, which collaborative models require.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    rng = np.random.RandomState(seed)
    shuffled = ratings.iloc[rng.permutation(len(ratings))].reset_index(drop=True)
    group = shuffled.groupby("userId")["movieId"]
    position = group.cumcount()
    size = group.transform("size")
    n_test = np.maximum(1, np.floor(size * test_size).astype(int))
    is_test = (size >= min_ratings_per_user) & (position < n_test)
    train = shuffled[~is_test].sort_values(["userId", "movieId"]).reset_index(drop=True)
    test = shuffled[is_test].sort_values(["userId", "movieId"]).reset_index(drop=True)
    return train, test
