"""Matrix factorisation (FunkSVD / "SVD" as used by the Surprise library) in NumPy.

Model:  r_hat(u, i) = mu + b_u + b_i + p_u . q_i
trained with stochastic gradient descent on the *training* ratings only, minimising
squared error with L2 regularisation. This is the same algorithm as Surprise's
``SVD`` class (it is not a literal linear-algebra SVD of a dense matrix, which is
undefined when most ratings are missing), implemented here with NumPy only so the
project installs everywhere without compiling scikit-surprise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, RATING_MAX, RATING_MIN
from .exceptions import UnknownMovieError, UnknownUserError

logger = logging.getLogger(__name__)


class SVDModel:
    """Biased matrix factorisation trained with SGD."""

    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr: float = 0.005,
                 reg: float = 0.02, seed: int = RANDOM_SEED) -> None:
        self.n_factors, self.n_epochs, self.lr, self.reg, self.seed = n_factors, n_epochs, lr, reg, seed
        self.history: list[dict] = []

    def fit(self, train: pd.DataFrame) -> "SVDModel":
        rng = np.random.RandomState(self.seed)
        self.user_ids = np.sort(train["userId"].unique()).astype("int64")
        self.item_ids = np.sort(train["movieId"].unique()).astype("int64")
        self._user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self._item_index = {int(m): i for i, m in enumerate(self.item_ids)}
        users = train["userId"].map(self._user_index).to_numpy()
        items = train["movieId"].map(self._item_index).to_numpy()
        ratings = train["rating"].to_numpy(dtype="float64")

        self.mu = float(ratings.mean())
        self.bu = np.zeros(len(self.user_ids))
        self.bi = np.zeros(len(self.item_ids))
        self.P = rng.normal(0, 0.1, (len(self.user_ids), self.n_factors))
        self.Q = rng.normal(0, 0.1, (len(self.item_ids), self.n_factors))
        self._seen = {}
        for u, i in zip(users, items):
            self._seen.setdefault(int(u), []).append(int(i))

        lr, reg = self.lr, self.reg
        self.history = []
        for epoch in range(1, self.n_epochs + 1):
            for idx in rng.permutation(len(ratings)):
                u, i = users[idx], items[idx]
                pu, qi = self.P[u].copy(), self.Q[i].copy()
                err = ratings[idx] - (self.mu + self.bu[u] + self.bi[i] + pu @ qi)
                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                self.P[u] += lr * (err * qi - reg * pu)
                self.Q[i] += lr * (err * pu - reg * qi)
            pred = self.mu + self.bu[users] + self.bi[items] + np.einsum("ij,ij->i", self.P[users], self.Q[items])
            rmse = float(np.sqrt(np.mean((ratings - np.clip(pred, RATING_MIN, RATING_MAX)) ** 2)))
            self.history.append({"epoch": epoch, "train_rmse": rmse})
            logger.info("SVD epoch %d/%d  train RMSE %.4f", epoch, self.n_epochs, rmse)
        return self

    # ----------------------------------------------------------- lookups
    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_index

    def has_item(self, movie_id: int) -> bool:
        return int(movie_id) in self._item_index

    # -------------------------------------------------------- predictions
    def predict_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predicted ratings for (userId, movieId) rows; unknown users/items use biases only."""
        u = pairs["userId"].map(lambda x: self._user_index.get(int(x), -1)).to_numpy()
        i = pairs["movieId"].map(lambda x: self._item_index.get(int(x), -1)).to_numpy()
        bu = np.where(u >= 0, self.bu[np.maximum(u, 0)], 0.0)
        bi = np.where(i >= 0, self.bi[np.maximum(i, 0)], 0.0)
        dot = np.einsum("ij,ij->i", self.P[np.maximum(u, 0)], self.Q[np.maximum(i, 0)])
        dot = np.where((u >= 0) & (i >= 0), dot, 0.0)
        return np.clip(self.mu + bu + bi + dot, RATING_MIN, RATING_MAX)

    def score_user(self, user_id: int) -> np.ndarray:
        """Predicted rating of every item (``item_ids`` order) for a known user."""
        if not self.has_user(user_id):
            raise UnknownUserError(f"User {user_id!r} has no training ratings (cold start).")
        u = self._user_index[int(user_id)]
        return np.clip(self.mu + self.bu[u] + self.bi + self.Q @ self.P[u], RATING_MIN, RATING_MAX)

    def recommend_for_user(self, user_id: int, n: int = 10, exclude: set[int] | None = None) -> pd.DataFrame:
        """Top-n unseen movies by predicted rating (``score``)."""
        scores = self.score_user(user_id)
        scores[self._seen.get(self._user_index[int(user_id)], [])] = -np.inf
        if exclude:
            scores[[self._item_index[m] for m in exclude if m in self._item_index]] = -np.inf
        valid = np.where(np.isfinite(scores))[0]
        order = valid[np.lexsort((self.item_ids[valid], -scores[valid]))][:n]
        return pd.DataFrame({
            "movieId": self.item_ids[order],
            "score": scores[order],
            "reason": "High predicted rating from your taste profile (latent factors)",
        })

    def similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Movies with the closest latent-factor vectors (cosine similarity)."""
        if not self.has_item(movie_id):
            raise UnknownMovieError(f"Movie {movie_id!r} has no ratings in the training data.")
        position = self._item_index[int(movie_id)]
        norms = np.linalg.norm(self.Q, axis=1)
        denominator = norms * norms[position]
        sims = np.divide(self.Q @ self.Q[position], denominator, out=np.zeros(len(norms)), where=denominator > 0)
        sims[position] = -np.inf
        order = np.lexsort((self.item_ids, -sims))[:n]
        return pd.DataFrame({
            "movieId": self.item_ids[order],
            "score": sims[order],
            "reason": "Close to this movie in the learned latent-factor space",
        })
