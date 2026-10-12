"""Item-based collaborative filtering with adjusted cosine similarity.

Idea: two movies are similar if the same users rate them above/below their own
average in the same way. Ratings are first centred on each user's mean rating
(*adjusted cosine*), so a missing rating becomes 0 = "no information" rather than
a fake low rating. Similarities are shrunk toward 0 when two movies share few
raters, and only the top-K neighbours per movie are stored (memory friendly).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .config import RATING_MAX, RATING_MIN
from .exceptions import UnknownMovieError, UnknownUserError

logger = logging.getLogger(__name__)


class ItemBasedCF:
    """Item-item neighbourhood model trained on a (train) rating table."""

    def __init__(self, n_neighbors: int = 50, shrinkage: float = 10.0, damping: float = 1.0,
                 chunk_size: int = 500) -> None:
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage      # sim *= n_common / (n_common + shrinkage)
        self.damping = damping          # prediction = mean + num / (den + damping)
        self.chunk_size = chunk_size
        self.global_mean = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "ItemBasedCF":
        self.user_ids = np.sort(train["userId"].unique()).astype("int64")
        self.item_ids = np.sort(train["movieId"].unique()).astype("int64")
        self._user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self._item_index = {int(m): i for i, m in enumerate(self.item_ids)}
        u = train["userId"].map(self._user_index).to_numpy()
        i = train["movieId"].map(self._item_index).to_numpy()
        r = train["rating"].to_numpy(dtype="float64")
        shape = (len(self.user_ids), len(self.item_ids))

        self.global_mean = float(r.mean())
        counts = np.bincount(u, minlength=shape[0])
        sums = np.bincount(u, weights=r, minlength=shape[0])
        self.user_means = np.divide(sums, counts, out=np.full(shape[0], self.global_mean), where=counts > 0)

        # raw (user x item) matrix: missing entries are simply absent (sparse), never 0 stars
        self.ratings_csr = sparse.csr_matrix((r, (u, i)), shape=shape)
        centred = sparse.csr_matrix((r - self.user_means[u], (u, i)), shape=shape)
        self._centred_items = centred.T.tocsr()                       # items x users
        self._rated_items = sparse.csr_matrix((np.ones_like(r), (i, u)), shape=(shape[1], shape[0]))
        self._build_neighbours()
        logger.info("ItemBasedCF fitted: %d users, %d items, %d ratings", shape[0], shape[1], len(r))
        return self

    def _similarity_block(self, rows: np.ndarray) -> np.ndarray:
        """Shrunk adjusted-cosine similarity of the given items vs all items (dense)."""
        sims = cosine_similarity(self._centred_items[rows], self._centred_items, dense_output=True)
        common = (self._rated_items[rows] @ self._rated_items.T).toarray()
        return sims * (common / (common + self.shrinkage))

    def _build_neighbours(self) -> None:
        n_items = len(self.item_ids)
        k = min(self.n_neighbors, max(n_items - 1, 1))
        self.neighbour_idx = np.zeros((n_items, k), dtype="int32")
        self.neighbour_sim = np.zeros((n_items, k), dtype="float32")
        for start in range(0, n_items, self.chunk_size):
            rows = np.arange(start, min(start + self.chunk_size, n_items))
            block = self._similarity_block(rows)
            block[np.arange(len(rows)), rows] = 0.0                    # never a neighbour of itself
            block[block < 0] = 0.0                                     # keep positively related items
            top = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
            self.neighbour_idx[rows] = top
            self.neighbour_sim[rows] = np.take_along_axis(block, top, axis=1)

    # ----------------------------------------------------------- lookups
    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_index

    def has_item(self, movie_id: int) -> bool:
        return int(movie_id) in self._item_index

    def user_history(self, user_id: int) -> pd.DataFrame:
        """Training ratings of a user as (movieId, rating)."""
        if not self.has_user(user_id):
            raise UnknownUserError(f"User {user_id!r} has no training ratings (cold start).")
        row = self.ratings_csr[self._user_index[int(user_id)]]
        return pd.DataFrame({"movieId": self.item_ids[row.indices], "rating": row.data})

    # --------------------------------------------------------- predictions
    def score_user(self, user_id: int) -> np.ndarray:
        """Predicted rating for every item (``item_ids`` order); NaN = no neighbour evidence."""
        if not self.has_user(user_id):
            raise UnknownUserError(f"User {user_id!r} has no training ratings (cold start).")
        u = self._user_index[int(user_id)]
        start, end = self.ratings_csr.indptr[u], self.ratings_csr.indptr[u + 1]
        rated = self.ratings_csr.indices[start:end]
        centred = self.ratings_csr.data[start:end] - self.user_means[u]
        n_items = len(self.item_ids)
        neighbours, sims = self.neighbour_idx[rated], self.neighbour_sim[rated]
        num = np.bincount(neighbours.ravel(), weights=(sims * centred[:, None]).ravel(), minlength=n_items)
        den = np.bincount(neighbours.ravel(), weights=np.abs(sims).ravel(), minlength=n_items)
        prediction = self.user_means[u] + num / (den + self.damping)
        prediction[den <= 0] = np.nan
        return np.clip(prediction, RATING_MIN, RATING_MAX)

    def predict_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predict ratings for (userId, movieId) rows, with sensible fallbacks.

        Fallback order: neighbour prediction -> user's mean -> global mean.
        """
        out = np.full(len(pairs), self.global_mean, dtype="float64")
        users = pairs["userId"].to_numpy()
        items = pairs["movieId"].to_numpy()
        for user_id in np.unique(users):
            rows = np.where(users == user_id)[0]
            if not self.has_user(user_id):
                continue
            vector = self.score_user(user_id)
            out[rows] = self.user_means[self._user_index[int(user_id)]]
            for row in rows:
                j = self._item_index.get(int(items[row]))
                if j is not None and not np.isnan(vector[j]):
                    out[row] = vector[j]
        return out

    def recommend_for_user(self, user_id: int, n: int = 10, exclude: set[int] | None = None) -> pd.DataFrame:
        """Top-n unseen movies by predicted rating (``score``).

        Movies the user rated in training - and any ids in ``exclude`` - are removed.
        """
        vector = self.score_user(user_id)
        u = self._user_index[int(user_id)]
        seen = self.ratings_csr.indices[self.ratings_csr.indptr[u]:self.ratings_csr.indptr[u + 1]]
        vector[seen] = np.nan
        if exclude:
            idx = [self._item_index[m] for m in exclude if m in self._item_index]
            vector[idx] = np.nan
        valid = np.where(~np.isnan(vector))[0]
        order = valid[np.lexsort((self.item_ids[valid], -vector[valid]))][:n]
        return pd.DataFrame({
            "movieId": self.item_ids[order],
            "score": vector[order],
            "reason": "Liked by users with similar rating patterns",
        })

    # ------------------------------------------------------ item similarity
    def item_similarity_vector(self, movie_id: int) -> np.ndarray:
        """Shrunk adjusted-cosine similarity of one movie to every item (``item_ids`` order)."""
        if not self.has_item(movie_id):
            raise UnknownMovieError(f"Movie {movie_id!r} has no ratings in the training data.")
        position = self._item_index[int(movie_id)]
        return self._similarity_block(np.array([position]))[0]

    def similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """'Users who rated this movie highly also liked ...'."""
        sims = self.item_similarity_vector(movie_id)
        sims[self._item_index[int(movie_id)]] = -1.0
        order = np.lexsort((self.item_ids, -sims))
        order = [i for i in order if sims[i] > 0][:n]
        return pd.DataFrame({
            "movieId": self.item_ids[order],
            "score": sims[order],
            "reason": "Rated similarly by the same users",
        })
