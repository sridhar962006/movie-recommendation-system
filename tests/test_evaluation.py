"""Tests for the evaluation metrics."""

import unittest

import numpy as np

from src.evaluation import (evaluate_ranking, mae, precision_recall_f1_at_k, rmse, select_eval_users)
from tests.helpers import get_artifacts


class MetricTests(unittest.TestCase):
    def test_rmse_mae_known_values(self):
        actual, predicted = np.array([4.0, 3.0, 5.0]), np.array([3.0, 3.0, 3.0])
        self.assertAlmostEqual(mae(actual, predicted), 1.0)
        self.assertAlmostEqual(rmse(actual, predicted), np.sqrt((1 + 0 + 4) / 3))

    def test_empty_inputs_raise(self):
        with self.assertRaises(ValueError):
            rmse(np.array([]), np.array([]))

    def test_precision_recall_f1_by_hand(self):
        p, r, f = precision_recall_f1_at_k([1, 2, 3, 4], {2, 4, 9}, k=4)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 2 / 3)
        self.assertAlmostEqual(f, 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))

    def test_edge_cases_do_not_divide_by_zero(self):
        self.assertEqual(precision_recall_f1_at_k([], {1}, 5), (0.0, 0.0, 0.0))
        self.assertEqual(precision_recall_f1_at_k([1], set(), 5), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            precision_recall_f1_at_k([1], {1}, 0)

    def test_evaluate_ranking_perfect_and_failing_recommender(self):
        relevant = {1: {10, 11}, 2: {20}}
        perfect = evaluate_ranking("perfect", lambda u, k: sorted(relevant[u]), [1, 2], relevant, k=2, catalogue_size=100)
        self.assertAlmostEqual(perfect["recall"], 1.0)
        self.assertAlmostEqual(perfect["hit_rate"], 1.0)

        def broken(u, k):
            raise RuntimeError("boom")
        failing = evaluate_ranking("broken", broken, [1, 2], relevant, k=2)
        self.assertEqual(failing["precision"], 0.0)


class SplitLeakageTests(unittest.TestCase):
    def test_test_pairs_never_in_training_data(self):
        art = get_artifacts()
        keys = lambda df: set(zip(df.userId, df.movieId))
        self.assertTrue(keys(art.train).isdisjoint(keys(art.test)))

    def test_models_only_know_training_pairs(self):
        art = get_artifacts()
        user = int(art.test["userId"].iloc[0])
        trained_items = set(art.item_cf.user_history(user)["movieId"])
        test_items = set(art.test.loc[art.test.userId == user, "movieId"])
        self.assertTrue(trained_items.isdisjoint(test_items))

    def test_eval_users_are_reproducible_and_relevant(self):
        art = get_artifacts()
        a = select_eval_users(art.train, art.test, 4.0, 10, seed=5)
        b = select_eval_users(art.train, art.test, 4.0, 10, seed=5)
        self.assertEqual(a[0], b[0])
        for user, items in a[1].items():
            self.assertTrue(items)

    def test_metrics_were_computed_for_real(self):
        metrics = get_artifacts().metrics
        names = [row["algorithm"] for row in metrics["rating_prediction"]]
        self.assertIn("SVD", names)
        for row in metrics["rating_prediction"]:
            self.assertTrue(np.isfinite(row["rmse"]) and row["rmse"] > 0)
        for row in metrics["ranking"]:
            self.assertTrue(0.0 <= row["precision"] <= 1.0)
            self.assertTrue(0.0 <= row["recall"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
