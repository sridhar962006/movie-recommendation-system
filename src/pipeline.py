"""Training / evaluation orchestration and the serving API used by the Streamlit UI.

``build_artifacts`` trains every model on the *training split only*, optionally evaluates
on the held-out test split, and returns an :class:`Artifacts` bundle that can be saved to
``models/artifacts.joblib``. ``Artifacts`` exposes simple methods (``recommend_for_movie``,
``recommend_for_user`` ...) so UI code never touches ML internals.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import evaluation
from .collaborative import ItemBasedCF
from .config import (ALGORITHMS, ARTIFACTS_FILENAME, DATA_DIR, DEFAULT_K, MAX_EVAL_USERS, METRICS_FILENAME,
                     MODELS_DIR, MIN_RATINGS_FOR_RECS, RANDOM_SEED, RELEVANCE_THRESHOLD, TEST_SIZE)
from .content_based import ContentBasedRecommender
from .data_loader import load_dataset, train_test_split_ratings
from .exceptions import ModelNotTrainedError, UnknownMovieError, UnknownUserError
from .hybrid import HybridRecommender, HybridWeights
from .popularity import PopularityRecommender
from .preprocessing import prepare_movies
from .svd_model import SVDModel

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
POPULARITY, CONTENT, COLLABORATIVE, SVD, HYBRID = (
    "Popularity-Based", "Content-Based", "Collaborative Filtering", "SVD", "Hybrid Recommendation")
DISPLAY_COLUMNS = ["movieId", "title", "year", "genres_display", "avg_rating", "rating_count", "tmdbId"]


@dataclass
class Artifacts:
    """Everything the app needs at run time."""

    movies: pd.DataFrame                  # prepared movie table (all movies)
    catalog: pd.DataFrame                 # movies + display stats (from ALL ratings) + tmdbId
    all_ratings: pd.DataFrame             # train + test (for history display only)
    train: pd.DataFrame
    test: pd.DataFrame
    popularity: PopularityRecommender     # fitted on train (used by evaluation)
    popularity_all: PopularityRecommender  # fitted on all ratings (used for display lists)
    content: ContentBasedRecommender
    item_cf: ItemBasedCF
    svd: SVDModel
    hybrid: HybridRecommender
    meta: dict = field(default_factory=dict)
    metrics: dict | None = None
    version: int = ARTIFACT_VERSION

    # ---------------------------------------------------------------- helpers
    def enrich(self, recs: pd.DataFrame) -> pd.DataFrame:
        """Attach title/year/genres/average rating/rating count to a recommendation frame."""
        if recs.empty:
            return recs.assign(**{c: pd.Series(dtype="object") for c in DISPLAY_COLUMNS if c != "movieId"})
        info = self.catalog[DISPLAY_COLUMNS]
        merged = recs.merge(info, on="movieId", how="left", suffixes=("_rec", ""))
        merged = merged[[c for c in merged.columns if not c.endswith("_rec")]]
        return merged.reset_index(drop=True)

    def movie_row(self, movie_id: int) -> pd.Series:
        hit = self.catalog[self.catalog["movieId"] == int(movie_id)]
        if hit.empty:
            raise UnknownMovieError(f"Movie id {movie_id!r} is not in the catalogue.")
        return hit.iloc[0]

    def user_ids(self) -> list[int]:
        return sorted(int(u) for u in self.all_ratings["userId"].unique())

    def has_history(self, user_id: int) -> bool:
        return self.item_cf.has_user(user_id)

    def user_history(self, user_id: int) -> pd.DataFrame:
        """All ratings of a user (train + test) with movie info, newest information first."""
        rows = self.all_ratings[self.all_ratings["userId"] == int(user_id)]
        if rows.empty:
            raise UnknownUserError(f"User {user_id!r} has no ratings in the dataset.")
        out = rows.merge(self.catalog[["movieId", "title", "year", "genres_display", "tmdbId"]], on="movieId", how="left")
        return out.sort_values(["rating", "title"], ascending=[False, True]).reset_index(drop=True)

    def _holdout_ids(self, user_id: int) -> set[int]:
        """Movies a user rated that live in the test split (still 'seen' when serving)."""
        rows = self.test[self.test["userId"] == int(user_id)]
        return set(map(int, rows["movieId"]))

    # ------------------------------------------------------------- movie mode
    def recommend_for_movie(self, algorithm: str, movie_id: int, n: int = 10,
                            weights: HybridWeights | None = None) -> pd.DataFrame:
        """Recommendations related to a chosen movie. May be empty (e.g. movie has no ratings)."""
        seed = self.movie_row(movie_id)
        movie_id = int(movie_id)
        if algorithm == POPULARITY:
            genres = self.movies.loc[self.movies["movieId"] == movie_id, "genre_list"].iloc[0]
            recs = self.popularity_all.recommend(n, genres=genres or None, exclude={movie_id}, min_ratings=5)
            recs["reason"] = recs["weighted_rating"].map(lambda w: f"Popular and highly rated (weighted rating {w:.2f})")
            recs = recs[["movieId", "score", "reason"]]
        elif algorithm == CONTENT:
            recs = self.content.similar_to(movie_id, n)
        elif algorithm == COLLABORATIVE:
            recs = self.item_cf.similar_items(movie_id, n) if self.item_cf.has_item(movie_id) else _empty()
        elif algorithm == SVD:
            recs = self.svd.similar_items(movie_id, n) if self.svd.has_item(movie_id) else _empty()
        elif algorithm == HYBRID:
            recs = self.hybrid.recommend_for_movie(movie_id, n, weights)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")
        recs = recs[recs["movieId"] != int(seed["movieId"])].drop_duplicates("movieId").head(n)
        return self.enrich(recs)

    # -------------------------------------------------------------- user mode
    def recommend_for_user(self, algorithm: str, user_id: int, n: int = 10,
                           weights: HybridWeights | None = None) -> pd.DataFrame:
        """Unseen-movie recommendations for a known user (raises UnknownUserError otherwise)."""
        if not self.has_history(user_id):
            raise UnknownUserError(f"User {user_id!r} has no training ratings (cold start).")
        seen = set(map(int, self.user_history(user_id)["movieId"]))
        if algorithm == POPULARITY:
            recs = self.popularity_all.recommend(n, exclude=seen, min_ratings=5)
            recs["reason"] = recs["weighted_rating"].map(lambda w: f"Popular and highly rated (weighted rating {w:.2f})")
            recs = recs[["movieId", "score", "reason"]]
        elif algorithm == CONTENT:
            recs = self.hybrid.recommend_for_user(user_id, n, HybridWeights(1, 0, 0), exclude=seen)
        elif algorithm == COLLABORATIVE:
            recs = self.item_cf.recommend_for_user(user_id, n, exclude=seen)
            recs["predicted_rating"] = recs["score"]
        elif algorithm == SVD:
            recs = self.svd.recommend_for_user(user_id, n, exclude=seen)
            recs["predicted_rating"] = recs["score"]
        elif algorithm == HYBRID:
            recs = self.hybrid.recommend_for_user(user_id, n, weights, exclude=seen)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")
        recs = recs[~recs["movieId"].isin(seen)].drop_duplicates("movieId").head(n)
        return self.enrich(recs)

    def recommend_new_user(self, genres: list[str] | None, movie_ids: list[int] | None, n: int = 10,
                           weights: HybridWeights | None = None) -> pd.DataFrame:
        """Cold-start recommendations from explicitly selected genres / favourite movies."""
        return self.enrich(self.hybrid.recommend_for_new_user(movie_ids, genres, n, weights))

    def all_genres(self) -> list[str]:
        return sorted({g for genres in self.movies["genre_list"] for g in genres})


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"movieId": pd.Series(dtype="int64"), "score": pd.Series(dtype="float64"),
                         "reason": pd.Series(dtype="object")})


# ============================================================ training pipeline
def _evaluate(art: Artifacts, k: int, max_users: int, seed: int, progress: Callable[[str], None]) -> dict:
    """Run the rating-prediction and ranking evaluations on the held-out test split."""
    train, test = art.train, art.test
    progress("Evaluating rating prediction (RMSE / MAE)")
    actual = test["rating"].to_numpy()
    item_mean = train.groupby("movieId")["rating"].mean()
    global_mean = float(train["rating"].mean())
    rating_rows = [
        evaluation.rating_metrics("Global Mean (baseline)", actual, np.full(len(test), global_mean)),
        evaluation.rating_metrics("Item Mean (baseline)", actual,
                                  test["movieId"].map(item_mean).fillna(global_mean).to_numpy()),
        evaluation.rating_metrics(COLLABORATIVE, actual, art.item_cf.predict_pairs(test)),
        evaluation.rating_metrics(SVD, actual, art.svd.predict_pairs(test)),
    ]

    progress("Evaluating top-K ranking (Precision / Recall / F1)")
    users, relevant = evaluation.select_eval_users(train, test, RELEVANCE_THRESHOLD, max_users, seed)
    seen_train = {int(u): set(map(int, g["movieId"])) for u, g in train.groupby("userId")}
    catalogue = art.movies["movieId"].to_numpy()
    rng = np.random.RandomState(seed)
    size = len(catalogue)

    def popular(u: int, n: int) -> list[int]:
        return art.popularity.recommend(n, exclude=seen_train[u])["movieId"].tolist()

    def content(u: int, n: int) -> list[int]:
        return art.hybrid.recommend_for_user(u, n, HybridWeights(1, 0, 0))["movieId"].tolist()

    def random_baseline(u: int, n: int) -> list[int]:
        pool = [m for m in rng.choice(catalogue, size=min(size, n + len(seen_train[u]) + 50), replace=False)
                if int(m) not in seen_train[u]]
        return [int(m) for m in pool[:n]]

    fns = {
        "Random (baseline)": random_baseline,
        POPULARITY: popular,
        CONTENT: content,
        COLLABORATIVE: lambda u, n: art.item_cf.recommend_for_user(u, n)["movieId"].tolist(),
        SVD: lambda u, n: art.svd.recommend_for_user(u, n)["movieId"].tolist(),
        HYBRID: lambda u, n: art.hybrid.recommend_for_user(u, n)["movieId"].tolist(),
    }
    ranking_rows = [evaluation.evaluate_ranking(name, fn, users, relevant, k, size) for name, fn in fns.items()]
    return {
        "rating_prediction": rating_rows,
        "ranking": ranking_rows,
        "ranking_setup": {
            "k": k, "relevance_threshold": RELEVANCE_THRESHOLD, "users_evaluated": len(users),
            "max_users": max_users,
            "note": ("Content-Based and Hybrid do not output calibrated star ratings, so they have no RMSE/MAE. "
                     "Candidates for ranking are all movies not rated by the user in the training split."),
        },
    }


def build_artifacts(data_dir: str | Path = DATA_DIR, seed: int = RANDOM_SEED, test_size: float = TEST_SIZE,
                    n_factors: int = 50, n_epochs: int = 20, evaluate: bool = True, k: int = DEFAULT_K,
                    max_eval_users: int = MAX_EVAL_USERS,
                    progress: Callable[[str], None] | None = None) -> Artifacts:
    """Load data, split, train all models on the training split and (optionally) evaluate them."""
    progress = progress or (lambda message: logger.info(message))
    started = time.time()
    progress("Loading and validating dataset")
    data = load_dataset(data_dir)
    movies = prepare_movies(data.movies, data.tags)

    progress("Splitting ratings into train / test")
    train, test = train_test_split_ratings(data.ratings, test_size, seed)

    progress("Fitting popularity model")
    popularity = PopularityRecommender().fit(train, movies)
    popularity_all = PopularityRecommender().fit(data.ratings, movies)
    progress("Fitting content-based model (TF-IDF)")
    content = ContentBasedRecommender().fit(movies)
    progress("Fitting item-based collaborative filtering")
    item_cf = ItemBasedCF().fit(train)
    progress(f"Training SVD ({n_epochs} epochs, {n_factors} factors) - this can take a minute")
    svd = SVDModel(n_factors=n_factors, n_epochs=n_epochs, seed=seed).fit(train)
    hybrid = HybridRecommender(content, item_cf, svd, popularity, movies, MIN_RATINGS_FOR_RECS)

    catalog = movies.merge(popularity_all.stats[["rating_count", "avg_rating", "weighted_rating"]].reset_index(),
                           on="movieId", how="left")
    catalog["tmdbId"] = np.nan
    if data.links is not None:
        catalog = catalog.drop(columns="tmdbId").merge(data.links, on="movieId", how="left")
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype("int64")

    meta = {
        "dataset_dir": str(data.source_dir.name if data.source_dir else ""),
        "n_movies": int(len(movies)), "n_users": int(data.ratings["userId"].nunique()),
        "n_ratings": int(len(data.ratings)), "n_train": int(len(train)), "n_test": int(len(test)),
        "test_fraction": test_size, "seed": seed,
        "svd": {"n_factors": n_factors, "n_epochs": n_epochs, "lr": svd.lr, "reg": svd.reg,
                "history": svd.history},
        "hybrid_default_weights": vars(HybridWeights().normalised()),
        "cleaning_report": data.cleaning_report,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    art = Artifacts(movies=movies, catalog=catalog, all_ratings=data.ratings, train=train, test=test,
                    popularity=popularity, popularity_all=popularity_all, content=content, item_cf=item_cf,
                    svd=svd, hybrid=hybrid, meta=meta)
    if evaluate:
        art.metrics = _evaluate(art, k, max_eval_users, seed, progress)
        art.metrics["meta"] = {key: meta[key] for key in ("n_movies", "n_users", "n_ratings", "n_train", "n_test",
                                                          "test_fraction", "seed", "trained_at")}
        art.metrics["meta"]["svd"] = {key: meta["svd"][key] for key in ("n_factors", "n_epochs", "lr", "reg")}
        art.metrics["svd_training_history"] = svd.history
    meta["training_seconds"] = round(time.time() - started, 1)
    progress(f"Done in {meta['training_seconds']} s")
    return art


# ================================================================== persistence
def save_artifacts(art: Artifacts, models_dir: str | Path = MODELS_DIR) -> Path:
    """Write artifacts.joblib (and metrics.json when evaluated). Returns the artifact path."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / ARTIFACTS_FILENAME
    joblib.dump(art, path, compress=3)
    metrics_path = models_dir / METRICS_FILENAME
    if art.metrics is not None:
        metrics_path.write_text(json.dumps(art.metrics, indent=2), encoding="utf-8")
    elif metrics_path.exists():
        metrics_path.unlink()          # never leave metrics from an older run next to new models
    return path


def load_artifacts(models_dir: str | Path = MODELS_DIR) -> Artifacts:
    """Load previously trained artifacts (only load files you created yourself - joblib uses pickle)."""
    path = Path(models_dir) / ARTIFACTS_FILENAME
    if not path.is_file():
        raise ModelNotTrainedError(
            "Trained models were not found. Run:  python train.py   (after placing the MovieLens data in data/).")
    try:
        art = joblib.load(path)
    except Exception as exc:
        logger.error("Could not load artifacts: %s", type(exc).__name__)
        raise ModelNotTrainedError(
            "The saved model file could not be loaded (it may come from a different library version). "
            "Please re-run:  python train.py") from exc
    if getattr(art, "version", None) != ARTIFACT_VERSION:
        raise ModelNotTrainedError("The saved models are outdated. Please re-run:  python train.py")
    return art
