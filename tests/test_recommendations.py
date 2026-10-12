"""Tests for every recommendation algorithm and the serving API."""

import unittest

import numpy as np
import pandas as pd

from src.collaborative import ItemBasedCF
from src.config import ALGORITHMS
from src.content_based import ContentBasedRecommender
from src.exceptions import UnknownMovieError, UnknownUserError
from src.hybrid import HybridWeights
from src.pipeline import (COLLABORATIVE, CONTENT, HYBRID, POPULARITY, SVD, load_artifacts, save_artifacts)
from src.popularity import PopularityRecommender, weighted_rating
from src.svd_model import SVDModel
from tests.helpers import get_artifacts


def assert_valid(testcase, recs: pd.DataFrame, valid_ids: set, n: int | None = None):
    """Recommendations must be real movies, unique, and not exceed n."""
    testcase.assertTrue(set(recs["movieId"]).issubset(valid_ids))
    testcase.assertFalse(recs["movieId"].duplicated().any())
    if n is not None:
        testcase.assertLessEqual(len(recs), n)


class PopularityTests(unittest.TestCase):
    def test_weighted_rating_penalises_few_votes(self):
        wr = weighted_rating(np.array([1, 1000]), np.array([5.0, 4.5]), min_votes=50, global_mean=3.5)
        self.assertLess(wr[0], wr[1])
        self.assertTrue(np.all(np.isfinite(weighted_rating(np.array([0]), np.array([np.nan]), 0, 3.5))))

    def test_ranking_sorted_and_valid(self):
        art = get_artifacts()
        recs = art.popularity.recommend(10)
        self.assertEqual(recs["weighted_rating"].tolist(), sorted(recs["weighted_rating"], reverse=True))
        assert_valid(self, recs, set(art.movies.movieId), 10)

    def test_exclude_and_genre_filter_and_top_rated(self):
        art = get_artifacts()
        first = int(art.popularity.recommend(1)["movieId"].iloc[0])
        self.assertNotIn(first, art.popularity.recommend(10, exclude={first})["movieId"].tolist())
        drama = art.popularity.recommend(10, genres=["Drama"])
        for movie_id in drama["movieId"]:
            self.assertIn("Drama", art.movies.set_index("movieId").loc[movie_id, "genre_list"])
        top = art.popularity.top_rated(5, min_ratings=10)
        self.assertTrue((top["rating_count"] >= 10).all())

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            PopularityRecommender().recommend()


class ContentBasedTests(unittest.TestCase):
    def test_excludes_itself_and_is_valid(self):
        art = get_artifacts()
        recs = art.content.similar_to(5, 10)
        self.assertNotIn(5, recs["movieId"].tolist())
        assert_valid(self, recs, set(art.movies.movieId), 10)
        self.assertEqual(recs["score"].tolist(), sorted(recs["score"], reverse=True))

    def test_unknown_movie_raises(self):
        with self.assertRaises(UnknownMovieError):
            get_artifacts().content.similar_to(999_999)

    def test_movie_without_genres_is_safe(self):
        art = get_artifacts()
        mystery = int(art.movies.loc[art.movies.genres_display == "Unknown", "movieId"].iloc[0])
        recs = art.content.similar_to(mystery, 5)         # decade token still gives some signal; must not crash
        self.assertNotIn(mystery, recs["movieId"].tolist())

    def test_profile_and_genre_scores(self):
        art = get_artifacts()
        self.assertTrue(np.all(art.content.profile_scores([]) == 0))
        self.assertTrue(np.all(art.content.genre_scores(["NotAGenre"]) == 0))
        self.assertGreater(art.content.genre_scores(["Drama"]).max(), 0)


class CollaborativeTests(unittest.TestCase):
    def setUp(self):
        self.art = get_artifacts()
        self.user = int(self.art.train["userId"].iloc[0])

    def test_excludes_already_rated_movies(self):
        recs = self.art.item_cf.recommend_for_user(self.user, 15)
        rated = set(self.art.item_cf.user_history(self.user)["movieId"])
        self.assertTrue(rated.isdisjoint(recs["movieId"]))
        assert_valid(self, recs, set(self.art.movies.movieId), 15)
        self.assertTrue(recs["score"].between(0.5, 5.0).all())

    def test_invalid_user_raises(self):
        with self.assertRaises(UnknownUserError):
            self.art.item_cf.recommend_for_user(-1)
        with self.assertRaises(UnknownMovieError):
            self.art.item_cf.similar_items(999_999)

    def test_predict_pairs_handles_unknown_ids(self):
        pairs = pd.DataFrame({"userId": [self.user, -5], "movieId": [999_999, 1]})
        preds = self.art.item_cf.predict_pairs(pairs)
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all((preds >= 0.5) & (preds <= 5.0)))

    def test_similar_items_exclude_self(self):
        recs = self.art.item_cf.similar_items(3, 8)
        self.assertNotIn(3, recs["movieId"].tolist())

    def test_sparse_user_matrix_keeps_missing_as_absent(self):
        # a user's centred row must only have entries where the user actually rated
        row = self.art.item_cf.ratings_csr[0]
        self.assertEqual(row.nnz, len(self.art.item_cf.user_history(int(self.art.item_cf.user_ids[0]))))


class SVDTests(unittest.TestCase):
    def setUp(self):
        self.art = get_artifacts()
        self.user = int(self.art.train["userId"].iloc[0])

    def test_training_reduces_error_and_is_logged(self):
        history = self.art.svd.history
        self.assertEqual(len(history), 6)
        self.assertLess(history[-1]["train_rmse"], history[0]["train_rmse"])

    def test_recommend_excludes_seen_and_predictions_in_range(self):
        recs = self.art.svd.recommend_for_user(self.user, 12)
        seen = set(self.art.train.loc[self.art.train.userId == self.user, "movieId"])
        self.assertTrue(seen.isdisjoint(recs["movieId"]))
        self.assertTrue(recs["score"].between(0.5, 5.0).all())
        assert_valid(self, recs, set(self.art.movies.movieId), 12)

    def test_reproducible_with_seed(self):
        small = self.art.train.head(400)
        a = SVDModel(n_factors=4, n_epochs=2, seed=1).fit(small)
        b = SVDModel(n_factors=4, n_epochs=2, seed=1).fit(small)
        np.testing.assert_allclose(a.P, b.P)

    def test_unknown_user_and_movie(self):
        with self.assertRaises(UnknownUserError):
            self.art.svd.recommend_for_user(-1)
        with self.assertRaises(UnknownMovieError):
            self.art.svd.similar_items(999_999)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.art = get_artifacts()
        self.valid = set(self.art.movies.movieId)
        self.user = int(self.art.train["userId"].iloc[0])

    def test_weights_normalise_and_validate(self):
        w = HybridWeights(2, 1, 1).normalised()
        self.assertAlmostEqual(w.content + w.collaborative + w.quality, 1.0)
        with self.assertRaises(ValueError):
            HybridWeights(0, 0, 0).normalised()
        with self.assertRaises(ValueError):
            HybridWeights(-1, 1, 1).normalised()

    def test_movie_mode_output(self):
        recs = self.art.hybrid.recommend_for_movie(4, 10)
        assert_valid(self, recs, self.valid, 10)
        self.assertNotIn(4, recs["movieId"].tolist())
        self.assertTrue(recs["score"].between(0, 1).all())
        self.assertTrue(recs["reason"].str.len().gt(0).all())

    def test_user_mode_excludes_seen_and_has_predicted_rating(self):
        recs = self.art.hybrid.recommend_for_user(self.user, 10)
        seen = set(self.art.item_cf.user_history(self.user)["movieId"])
        self.assertTrue(seen.isdisjoint(recs["movieId"]))
        assert_valid(self, recs, self.valid, 10)
        self.assertTrue(recs["predicted_rating"].dropna().between(0.5, 5.0).all())

    def test_weights_change_the_ranking(self):
        a = self.art.hybrid.recommend_for_user(self.user, 10, HybridWeights(1, 0, 0))["movieId"].tolist()
        b = self.art.hybrid.recommend_for_user(self.user, 10, HybridWeights(0, 0, 1))["movieId"].tolist()
        self.assertNotEqual(a, b)

    def test_invalid_inputs(self):
        with self.assertRaises(UnknownUserError):
            self.art.hybrid.recommend_for_user(-1)
        with self.assertRaises(UnknownMovieError):
            self.art.hybrid.recommend_for_movie(999_999)
        with self.assertRaises(ValueError):
            self.art.hybrid.recommend_for_new_user([], [], 5)

    def test_cold_start_from_genres_and_movies(self):
        by_genre = self.art.hybrid.recommend_for_new_user(None, ["Drama"], 8)
        assert_valid(self, by_genre, self.valid, 8)
        self.assertGreater(len(by_genre), 0)
        by_movie = self.art.hybrid.recommend_for_new_user([2, 3, 999_999], None, 8)
        self.assertTrue({2, 3}.isdisjoint(by_movie["movieId"]))


class ServingApiTests(unittest.TestCase):
    """The same API the Streamlit pages call."""

    def setUp(self):
        self.art = get_artifacts()
        self.valid = set(self.art.movies.movieId)
        self.user = int(self.art.train["userId"].iloc[0])

    def test_all_algorithms_in_movie_mode(self):
        for algorithm in ALGORITHMS:
            recs = self.art.recommend_for_movie(algorithm, 6, 10)
            assert_valid(self, recs, self.valid, 10)
            self.assertNotIn(6, recs["movieId"].tolist(), algorithm)
            for column in ("title", "genres_display", "avg_rating", "rating_count"):
                self.assertIn(column, recs.columns)

    def test_all_algorithms_in_user_mode_exclude_every_rated_movie(self):
        rated = set(self.art.all_ratings.loc[self.art.all_ratings.userId == self.user, "movieId"])
        for algorithm in ALGORITHMS:
            recs = self.art.recommend_for_user(algorithm, self.user, 10)
            self.assertTrue(rated.isdisjoint(recs["movieId"]), algorithm)
            assert_valid(self, recs, self.valid, 10)
            self.assertGreater(len(recs), 0, algorithm)

    def test_unknown_inputs(self):
        with self.assertRaises(UnknownMovieError):
            self.art.recommend_for_movie(HYBRID, 999_999, 5)
        with self.assertRaises(UnknownUserError):
            self.art.recommend_for_user(HYBRID, -3, 5)
        with self.assertRaises(ValueError):
            self.art.recommend_for_movie("Nonsense", 1, 5)

    def test_movie_without_training_ratings_returns_empty_not_error(self):
        unrated = self.art.movies.loc[~self.art.movies.movieId.isin(self.art.train.movieId), "movieId"]
        if len(unrated):
            self.assertTrue(self.art.recommend_for_movie(COLLABORATIVE, int(unrated.iloc[0]), 5).empty)

    def test_save_and_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            save_artifacts(self.art, tmp)
            loaded = load_artifacts(tmp)
            a = self.art.recommend_for_user(SVD, self.user, 5)["movieId"].tolist()
            b = loaded.recommend_for_user(SVD, self.user, 5)["movieId"].tolist()
            self.assertEqual(a, b)

    def test_load_missing_models_gives_helpful_error(self):
        import tempfile
        from src.exceptions import ModelNotTrainedError
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ModelNotTrainedError) as ctx:
            load_artifacts(tmp)
        self.assertIn("train.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
