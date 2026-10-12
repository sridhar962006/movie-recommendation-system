"""Hybrid recommender: content similarity + collaborative signal + movie quality.

Every signal is min-max normalised to [0, 1] over the candidate movies and then
combined with configurable weights:

    score = w_content * content + w_collab * collaborative + w_quality * quality

(weights are rescaled to sum to 1).

* content        cosine similarity of TF-IDF genre/metadata vectors
                 (to the chosen movie, or to the user's taste profile)
* collaborative  item-item adjusted-cosine similarity (movie mode) or the SVD
                 predicted rating (user mode)
* quality        weighted rating (IMDb-style) so well-supported, well-rated movies rank higher

Only training data is used, so evaluation on the held-out test split stays leak-free.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .collaborative import ItemBasedCF
from .config import MIN_RATINGS_FOR_RECS
from .content_based import ContentBasedRecommender
from .exceptions import UnknownMovieError, UnknownUserError
from .popularity import PopularityRecommender
from .preprocessing import minmax
from .svd_model import SVDModel


@dataclass(frozen=True)
class HybridWeights:
    """Relative importance of the three signals (rescaled to sum to 1)."""

    content: float = 0.4
    collaborative: float = 0.4
    quality: float = 0.2

    def normalised(self) -> "HybridWeights":
        values = (self.content, self.collaborative, self.quality)
        if any(v < 0 for v in values) or sum(values) <= 0:
            raise ValueError("Hybrid weights must be non-negative and not all zero.")
        total = float(sum(values))
        return HybridWeights(self.content / total, self.collaborative / total, self.quality / total)


class HybridRecommender:
    """Combines the fitted component models into one ranked list."""

    def __init__(self, content: ContentBasedRecommender, item_cf: ItemBasedCF, svd: SVDModel,
                 popularity: PopularityRecommender, movies: pd.DataFrame,
                 min_ratings: int = MIN_RATINGS_FOR_RECS) -> None:
        self.content, self.item_cf, self.svd, self.popularity = content, item_cf, svd, popularity
        self.min_ratings = min_ratings
        self.movie_ids = content.movie_ids                      # global catalogue order
        index = pd.Index(self.movie_ids)
        self._cf_pos = index.get_indexer(item_cf.item_ids)
        self._svd_pos = index.get_indexer(svd.item_ids)
        stats = popularity.stats.reindex(self.movie_ids)
        self._counts = stats["rating_count"].fillna(0).to_numpy()
        self._quality_raw = stats["weighted_rating"].fillna(popularity.global_mean).to_numpy()
        self._index = index

    # --------------------------------------------------------------- helpers
    def _to_global(self, vector: np.ndarray, positions: np.ndarray) -> np.ndarray:
        out = np.full(len(self.movie_ids), np.nan)
        keep = positions >= 0
        out[positions[keep]] = vector[keep]
        return out

    def _position(self, movie_id: int) -> int:
        position = self._index.get_indexer([int(movie_id)])[0]
        if position < 0:
            raise UnknownMovieError(f"Movie id {movie_id!r} is not in the catalogue.")
        return int(position)

    def _rank(self, signals: dict[str, np.ndarray], weights: HybridWeights, blocked: np.ndarray,
              n: int, subject: str, predicted: np.ndarray | None = None) -> pd.DataFrame:
        """Normalise signals over candidates, combine with weights, return the top-n."""
        w = weights.normalised()
        candidates = (~blocked) & (self._counts >= self.min_ratings)
        if not candidates.any():                                # tiny catalogues: relax the filter
            candidates = ~blocked
        idx = np.where(candidates)[0]
        parts = {}
        for name in ("content", "collaborative", "quality"):
            parts[name] = minmax(signals[name][idx])
        total = w.content * parts["content"] + w.collaborative * parts["collaborative"] + w.quality * parts["quality"]
        order = np.lexsort((self.movie_ids[idx], -total))[:n]

        rows = []
        for o in order:
            contributions = {"content": w.content * parts["content"][o],
                             "collaborative": w.collaborative * parts["collaborative"][o],
                             "quality": w.quality * parts["quality"][o]}
            rows.append({
                "movieId": int(self.movie_ids[idx[o]]),
                "score": float(total[o]),
                "content_score": float(parts["content"][o]),
                "collaborative_score": float(parts["collaborative"][o]),
                "quality_score": float(parts["quality"][o]),
                "predicted_rating": float(predicted[idx[o]]) if predicted is not None and np.isfinite(predicted[idx[o]]) else np.nan,
                "reason": self._explain(contributions, parts, int(o), subject),
            })
        return pd.DataFrame(rows, columns=["movieId", "score", "content_score", "collaborative_score",
                                           "quality_score", "predicted_rating", "reason"])

    @staticmethod
    def _explain(contributions: dict[str, float], parts: dict[str, np.ndarray], o: int, subject: str) -> str:
        phrases = {
            "content": f"similar genres/content to {subject}",
            "collaborative": "liked by users with similar taste" if subject != "your taste" else "predicted to match your rating pattern",
            "quality": "well rated by many viewers",
        }
        ranked = [k for k, v in sorted(contributions.items(), key=lambda kv: -kv[1]) if v > 0 and parts[k][o] > 0]
        if not ranked:
            return "Balanced match across all signals"
        text = "; ".join(phrases[k] for k in ranked[:2])
        return text[0].upper() + text[1:]

    def _quality_signal(self) -> np.ndarray:
        return self._quality_raw

    # ----------------------------------------------------------- movie mode
    def recommend_for_movie(self, movie_id: int, n: int = 10, weights: HybridWeights | None = None) -> pd.DataFrame:
        """Movies to watch after ``movie_id`` (excludes the movie itself)."""
        weights = weights or HybridWeights()
        position = self._position(movie_id)
        content = self.content.similarity_vector(movie_id)
        if self.item_cf.has_item(movie_id):
            collaborative = self._to_global(self.item_cf.item_similarity_vector(movie_id), self._cf_pos)
        else:                                                   # movie unseen in training ratings
            collaborative = np.full(len(self.movie_ids), np.nan)
        blocked = np.zeros(len(self.movie_ids), dtype=bool)
        blocked[position] = True
        title_hint = "this movie"
        return self._rank({"content": content, "collaborative": collaborative, "quality": self._quality_signal()},
                          weights, blocked, n, title_hint)

    # ------------------------------------------------------------ user mode
    def recommend_for_user(self, user_id: int, n: int = 10, weights: HybridWeights | None = None,
                           exclude: Iterable[int] | None = None) -> pd.DataFrame:
        """Personalised list of unseen movies for a known user."""
        weights = weights or HybridWeights()
        if not self.item_cf.has_user(user_id) or not self.svd.has_user(user_id):
            raise UnknownUserError(f"User {user_id!r} has no training ratings (cold start).")
        history = self.item_cf.user_history(user_id)
        liked = history[history["rating"] >= 4.0]
        if liked.empty:                                         # fall back to the user's best-rated films
            liked = history.sort_values("rating", ascending=False).head(5)
        content = self.content.profile_scores(liked["movieId"].tolist(), liked["rating"].tolist())
        predicted = self._to_global(self.svd.score_user(user_id), self._svd_pos)

        blocked = np.zeros(len(self.movie_ids), dtype=bool)
        seen = list(history["movieId"]) + list(exclude or [])
        for position in self._index.get_indexer(seen):
            if position >= 0:
                blocked[position] = True
        return self._rank({"content": content, "collaborative": predicted, "quality": self._quality_signal()},
                          weights, blocked, n, "your taste", predicted=predicted)

    # ------------------------------------------------------------ cold start
    def recommend_for_new_user(self, favourite_movie_ids: list[int] | None = None,
                               favourite_genres: list[str] | None = None, n: int = 10,
                               weights: HybridWeights | None = None) -> pd.DataFrame:
        """Cold-start recommendations from the movies/genres the new user selected.

        Nothing is invented: with no selection at all the caller gets popular movies.
        """
        weights = weights or HybridWeights()
        favourite_movie_ids = [int(m) for m in (favourite_movie_ids or []) if self.content.has_movie(m)]
        favourite_genres = list(favourite_genres or [])
        if not favourite_movie_ids and not favourite_genres:
            raise ValueError("Select at least one favourite movie or genre.")

        content_parts = []
        if favourite_movie_ids:
            content_parts.append(self.content.profile_scores(favourite_movie_ids))
        if favourite_genres:
            content_parts.append(self.content.genre_scores(favourite_genres))
        content = np.mean(content_parts, axis=0)

        cf_parts = [self._to_global(self.item_cf.item_similarity_vector(m), self._cf_pos)
                    for m in favourite_movie_ids if self.item_cf.has_item(m)]
        if cf_parts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN columns are expected
                collaborative = np.nanmean(cf_parts, axis=0)
        else:
            collaborative = np.full(len(self.movie_ids), np.nan)

        blocked = np.zeros(len(self.movie_ids), dtype=bool)
        for position in self._index.get_indexer(favourite_movie_ids):
            if position >= 0:
                blocked[position] = True
        return self._rank({"content": content, "collaborative": collaborative, "quality": self._quality_signal()},
                          weights, blocked, n, "your favourites")
