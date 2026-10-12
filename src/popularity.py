"""Popularity-based recommendation using the IMDb-style weighted rating.

    WR = v / (v + m) * R  +  m / (v + m) * C

v = number of ratings of the movie, R = its mean rating,
m = minimum votes required (a quantile of all rating counts), C = global mean rating.
Movies with few votes are pulled toward the global mean, so a film with a single
5-star rating cannot outrank a film with thousands of 4.5-star ratings.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .preprocessing import compute_movie_stats


def weighted_rating(votes: np.ndarray, mean_rating: np.ndarray, min_votes: float, global_mean: float) -> np.ndarray:
    """Vectorised weighted rating; safe when votes + min_votes == 0."""
    votes = np.asarray(votes, dtype="float64")
    mean_rating = np.asarray(mean_rating, dtype="float64")
    denominator = votes + min_votes
    safe = np.where(denominator > 0, denominator, 1.0)
    value = votes / safe * np.nan_to_num(mean_rating, nan=global_mean) + min_votes / safe * global_mean
    return np.where(denominator > 0, value, global_mean)


class PopularityRecommender:
    """Ranks movies by weighted rating; also offers 'most rated' and 'top rated' lists."""

    def __init__(self, min_votes_quantile: float = 0.80) -> None:
        if not 0.0 <= min_votes_quantile < 1.0:
            raise ValueError("min_votes_quantile must be in [0, 1).")
        self.min_votes_quantile = min_votes_quantile
        self.stats: pd.DataFrame | None = None
        self.global_mean: float = 0.0
        self.min_votes: float = 0.0

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "PopularityRecommender":
        stats = compute_movie_stats(ratings)
        self.global_mean = float(ratings["rating"].mean())
        self.min_votes = float(stats["rating_count"].quantile(self.min_votes_quantile))
        table = movies[["movieId", "genre_list"]].merge(
            stats.reset_index(), on="movieId", how="left"
        )
        table["rating_count"] = table["rating_count"].fillna(0).astype("int64")
        table["weighted_rating"] = weighted_rating(
            table["rating_count"].to_numpy(), table["avg_rating"].to_numpy(),
            self.min_votes, self.global_mean,
        )
        self.stats = table.set_index("movieId")
        return self

    def _require_fit(self) -> pd.DataFrame:
        if self.stats is None:
            raise RuntimeError("PopularityRecommender must be fitted first.")
        return self.stats

    def _format(self, frame: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
        out = frame.head(n).reset_index()[["movieId", "rating_count", "avg_rating", "weighted_rating"]]
        out.insert(1, "score", out[score_col].to_numpy())
        return out.reset_index(drop=True)

    def recommend(
        self,
        n: int = 10,
        genres: Iterable[str] | None = None,
        exclude: Iterable[int] | None = None,
        min_ratings: int = 1,
    ) -> pd.DataFrame:
        """Top-n by weighted rating, optionally restricted to genres (any match)."""
        stats = self._require_fit()
        frame = stats[stats["rating_count"] >= max(1, min_ratings)]
        if genres:
            wanted = set(genres)
            frame = frame[frame["genre_list"].map(lambda g: bool(wanted.intersection(g)))]
        if exclude is not None:
            frame = frame.drop(index=[m for m in set(exclude) if m in frame.index])
        frame = frame.sort_values(["weighted_rating", "rating_count"], ascending=False, kind="stable")
        return self._format(frame, "weighted_rating", n)

    def top_rated(self, n: int = 10, min_ratings: int = 50) -> pd.DataFrame:
        """Highest plain average rating among movies with at least ``min_ratings`` ratings."""
        stats = self._require_fit()
        frame = stats[stats["rating_count"] >= min_ratings]
        frame = frame.sort_values(["avg_rating", "rating_count"], ascending=False, kind="stable")
        return self._format(frame, "avg_rating", n)

    def most_popular(self, n: int = 10) -> pd.DataFrame:
        """Most frequently rated movies."""
        stats = self._require_fit()
        frame = stats.sort_values(["rating_count", "avg_rating"], ascending=False, kind="stable")
        return self._format(frame, "rating_count", n)
