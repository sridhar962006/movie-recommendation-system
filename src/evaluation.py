"""Evaluation metrics for rating prediction and top-K recommendation quality.

Rating prediction (on the held-out test ratings)
    RMSE = sqrt(mean((actual - predicted)^2))     MAE = mean(|actual - predicted|)

Top-K recommendation quality (per user, then averaged over evaluated users)
    * A test movie is *relevant* if the user rated it >= ``threshold`` (default 4.0).
    * Candidates are all movies the user did NOT rate in the training split, so held-out
      test movies compete with everything else the user has not seen.
    * Precision@K = hits / K,  Recall@K = hits / #relevant,  F1@K = harmonic mean.
    Only users with at least one relevant test movie are evaluated (a random sample of
    ``max_users`` users, fixed by the seed). Unrated movies are not necessarily disliked,
    so absolute precision/recall values are low by construction - compare algorithms
    against each other and against the random baseline rather than against 1.0.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_K, MAX_EVAL_USERS, RANDOM_SEED, RELEVANCE_THRESHOLD

logger = logging.getLogger(__name__)

RecommendFn = Callable[[int, int], list[int]]      # (user_id, k) -> ranked movie ids


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.size == 0:
        raise ValueError("Cannot compute RMSE on an empty test set.")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.size == 0:
        raise ValueError("Cannot compute MAE on an empty test set.")
    return float(np.mean(np.abs(actual - predicted)))


def rating_metrics(name: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"algorithm": name, "rmse": rmse(actual, predicted), "mae": mae(actual, predicted), "n_test": int(len(actual))}


def precision_recall_f1_at_k(recommended: list[int], relevant: set[int], k: int) -> tuple[float, float, float]:
    """Metrics for a single user. Safe for empty lists / empty relevant sets."""
    if k <= 0:
        raise ValueError("k must be positive.")
    hits = len(set(recommended[:k]) & relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def select_eval_users(train: pd.DataFrame, test: pd.DataFrame, threshold: float = RELEVANCE_THRESHOLD,
                      max_users: int = MAX_EVAL_USERS, seed: int = RANDOM_SEED) -> tuple[list[int], dict[int, set[int]]]:
    """Users with training history and >= 1 relevant test item (random sample, reproducible)."""
    relevant_rows = test[test["rating"] >= threshold]
    relevant = {int(u): set(map(int, g["movieId"])) for u, g in relevant_rows.groupby("userId")}
    eligible = sorted(set(relevant) & set(map(int, train["userId"].unique())))
    if len(eligible) > max_users:
        rng = np.random.RandomState(seed)
        eligible = sorted(rng.choice(eligible, size=max_users, replace=False).tolist())
    return eligible, {u: relevant[u] for u in eligible}


def evaluate_ranking(name: str, recommend_fn: RecommendFn, users: list[int], relevant: dict[int, set[int]],
                     k: int = DEFAULT_K, catalogue_size: int | None = None) -> dict:
    """Average Precision@K / Recall@K / F1@K (plus hit rate and catalogue coverage)."""
    precisions, recalls, f1s, hits, seen_items = [], [], [], 0, set()
    for user_id in users:
        try:
            recommended = [int(m) for m in recommend_fn(user_id, k)]
        except Exception as exc:                      # a failing user must not hide the others
            logger.warning("%s failed for user %s: %s", name, user_id, type(exc).__name__)
            recommended = []
        p, r, f = precision_recall_f1_at_k(recommended, relevant[user_id], k)
        precisions.append(p), recalls.append(r), f1s.append(f)
        hits += int(bool(set(recommended[:k]) & relevant[user_id]))
        seen_items.update(recommended[:k])
    n_users = len(users)
    return {
        "algorithm": name,
        "precision": float(np.mean(precisions)) if n_users else 0.0,
        "recall": float(np.mean(recalls)) if n_users else 0.0,
        "f1": float(np.mean(f1s)) if n_users else 0.0,
        "hit_rate": hits / n_users if n_users else 0.0,
        "coverage": len(seen_items) / catalogue_size if catalogue_size else 0.0,
        "k": k,
        "n_users": n_users,
    }
