"""Content-based filtering: TF-IDF features + cosine similarity.

Each movie becomes a TF-IDF vector built from genres (and optional decade / user tags).
Only the sparse feature matrix is stored; similarities for one movie are computed on
demand, so no huge N x N matrix is ever built or rebuilt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .exceptions import UnknownMovieError
from .preprocessing import genre_token


class ContentBasedRecommender:
    """Recommend movies whose genre/metadata vectors are close to a movie or a profile."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+", sublinear_tf=True)
        self.matrix: sparse.csr_matrix | None = None
        self.movie_ids: np.ndarray = np.array([], dtype="int64")
        self._index: dict[int, int] = {}
        self._genres: dict[int, list[str]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """``movies`` must be the output of preprocessing.prepare_movies."""
        self.movie_ids = movies["movieId"].to_numpy(dtype="int64")
        self._index = {int(m): i for i, m in enumerate(self.movie_ids)}
        self._genres = {int(m): list(g) for m, g in zip(movies["movieId"], movies["genre_list"])}
        texts = movies["content_text"].fillna("").astype(str)
        if not texts.str.strip().any():
            raise ValueError("No genre or metadata information available to build features.")
        self.matrix = self.vectorizer.fit_transform(texts).tocsr()
        return self

    def _require_fit(self) -> sparse.csr_matrix:
        if self.matrix is None:
            raise RuntimeError("ContentBasedRecommender must be fitted first.")
        return self.matrix

    def has_movie(self, movie_id: int) -> bool:
        return int(movie_id) in self._index

    def _position(self, movie_id: int) -> int:
        try:
            return self._index[int(movie_id)]
        except (KeyError, TypeError, ValueError):
            raise UnknownMovieError(f"Movie id {movie_id!r} is not in the catalogue.") from None

    # ------------------------------------------------------- similarity API
    def similarity_vector(self, movie_id: int) -> np.ndarray:
        """Cosine similarity between ``movie_id`` and every movie (catalogue order)."""
        matrix = self._require_fit()
        row = matrix[self._position(movie_id)]
        return cosine_similarity(row, matrix, dense_output=True).ravel()

    def profile_scores(self, movie_ids: list[int], weights: list[float] | None = None) -> np.ndarray:
        """Cosine similarity of every movie to a weighted taste profile.

        The profile is the weighted mean of the TF-IDF vectors of ``movie_ids``;
        unknown ids are ignored. Returns zeros if the profile is empty.
        """
        matrix = self._require_fit()
        positions, w = [], []
        for i, movie_id in enumerate(movie_ids):
            if int(movie_id) in self._index:
                positions.append(self._index[int(movie_id)])
                w.append(1.0 if weights is None else float(weights[i]))
        if not positions or sum(w) <= 0:
            return np.zeros(matrix.shape[0])
        profile = sparse.csr_matrix(np.asarray(w) @ matrix[positions]) / sum(w)
        return cosine_similarity(profile, matrix, dense_output=True).ravel()

    def genre_scores(self, genres: list[str]) -> np.ndarray:
        """Cosine similarity of every movie to a list of favourite genres (cold start)."""
        matrix = self._require_fit()
        tokens = " ".join(genre_token(g) for g in genres if genre_token(g))
        if not tokens:
            return np.zeros(matrix.shape[0])
        vector = self.vectorizer.transform([tokens])
        if vector.nnz == 0:
            return np.zeros(matrix.shape[0])
        return cosine_similarity(vector, matrix, dense_output=True).ravel()

    def shared_genres(self, movie_a: int, movie_b: int) -> list[str]:
        a, b = self._genres.get(int(movie_a), []), set(self._genres.get(int(movie_b), []))
        return [g for g in a if g in b]

    # --------------------------------------------------------- recommending
    def similar_to(self, movie_id: int, n: int = 10, min_similarity: float = 0.0) -> pd.DataFrame:
        """Top-n movies most similar to ``movie_id`` (never including itself)."""
        position = self._position(movie_id)
        sims = self.similarity_vector(movie_id)
        sims[position] = -1.0
        order = np.lexsort((self.movie_ids, -sims))      # similarity desc, id asc for ties
        order = [i for i in order if sims[i] > min_similarity][:n]
        rows = []
        for i in order:
            shared = self.shared_genres(movie_id, int(self.movie_ids[i]))
            reason = f"Shares genres: {', '.join(shared)}" if shared else "Similar overall content profile"
            rows.append((int(self.movie_ids[i]), float(sims[i]), reason))
        return pd.DataFrame(rows, columns=["movieId", "score", "reason"])
