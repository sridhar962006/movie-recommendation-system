"""Tiny synthetic MovieLens-style dataset used ONLY by the unit tests.

It exists so the tests run in seconds without downloading data. It is never used for
training the real application models or for any reported metric.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror"]


def make_frames(seed: int = 0, n_users: int = 80, n_movies: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    movies = []
    for movie_id in range(1, n_movies + 1):
        picks = rng.choice(GENRES, size=rng.randint(1, 3), replace=False)
        title = f"Film {movie_id:03d} ({1980 + movie_id % 40})"
        if movie_id % 10 == 0:
            title = f"Film {movie_id:03d}, The ({1980 + movie_id % 40})"
        movies.append((movie_id, title, "|".join(picks)))
    movies[-1] = (n_movies, f"Mystery Film ({1990})", "(no genres listed)")
    movies_df = pd.DataFrame(movies, columns=["movieId", "title", "genres"])

    ratings = []
    for user_id in range(1, n_users + 1):
        fav = set(rng.choice(GENRES, size=2, replace=False))
        for movie_id in rng.choice(np.arange(1, n_movies + 1), size=rng.randint(20, 36), replace=False):
            genres = set(movies_df.loc[movies_df.movieId == movie_id, "genres"].iloc[0].split("|"))
            base = 4.0 if genres & fav else 2.5
            rating = float(np.clip(np.round((base + rng.normal(0, 0.7)) * 2) / 2, 0.5, 5.0))
            ratings.append((user_id, int(movie_id), rating, 1_000_000 + len(ratings)))
    ratings_df = pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"])
    links_df = pd.DataFrame({"movieId": movies_df.movieId, "imdbId": 1, "tmdbId": movies_df.movieId + 1000})
    return movies_df, ratings_df, links_df


def write_dataset(folder: Path, seed: int = 0) -> Path:
    """Write movies.csv / ratings.csv / links.csv in the MovieLens layout."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    movies, ratings, links = make_frames(seed)
    movies.to_csv(folder / "movies.csv", index=False)
    ratings.to_csv(folder / "ratings.csv", index=False)
    links.to_csv(folder / "links.csv", index=False)
    return folder


@lru_cache(maxsize=1)
def get_artifacts():
    """Build (once per test run) small trained artifacts from the synthetic data."""
    import tempfile
    from src.pipeline import build_artifacts

    folder = write_dataset(Path(tempfile.mkdtemp()) / "ml-latest-small")
    return build_artifacts(folder.parent, seed=7, n_factors=8, n_epochs=6, evaluate=True,
                           k=5, max_eval_users=30, progress=lambda message: None)
