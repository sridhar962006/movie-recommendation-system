"""Tests for dataset loading, cleaning, splitting and text preprocessing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import find_dataset_dir, load_dataset, train_test_split_ratings
from src.exceptions import DatasetError
from src.preprocessing import (clean_title, extract_year, find_movies, lookup_movie_id, minmax,
                               prepare_movies, split_genres)
from tests.helpers import make_frames, write_dataset


class TextPreprocessingTests(unittest.TestCase):
    def test_extract_year(self):
        self.assertEqual(extract_year("Toy Story (1995)"), 1995)
        self.assertIsNone(extract_year("No Year Here"))
        self.assertIsNone(extract_year(None))

    def test_clean_title_moves_article_and_drops_year(self):
        self.assertEqual(clean_title("Matrix, The (1999)"), "The Matrix")
        self.assertEqual(clean_title("Toy Story (1995)"), "Toy Story")
        self.assertEqual(clean_title(None), "")

    def test_split_genres_handles_missing(self):
        self.assertEqual(split_genres("Action|Sci-Fi"), ["Action", "Sci-Fi"])
        self.assertEqual(split_genres("(no genres listed)"), [])
        self.assertEqual(split_genres(None), [])

    def test_minmax_constant_and_nan(self):
        self.assertTrue(np.all(minmax(np.array([2.0, 2.0])) == 0))
        out = minmax(np.array([1.0, np.nan, 3.0]))
        self.assertEqual(out.tolist(), [0.0, 0.0, 1.0])

    def test_prepare_movies_and_lookup(self):
        movies, _, _ = make_frames()
        prepared = prepare_movies(movies)
        self.assertIn("content_text", prepared.columns)
        self.assertEqual(prepared.loc[prepared.movieId == 60, "genres_display"].iloc[0], "Unknown")
        self.assertEqual(lookup_movie_id(prepared, "film 010, the (1990)"), 10)
        self.assertEqual(lookup_movie_id(prepared, "The Film 010"), 10)
        self.assertIsNone(lookup_movie_id(prepared, "Definitely Not A Movie"))
        self.assertIsNone(lookup_movie_id(prepared, ""))
        hits = find_movies(prepared, "film 01")
        self.assertGreater(len(hits), 0)
        self.assertTrue(find_movies(prepared, "").empty)


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "ml-latest-small"

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_nested_folder(self):
        write_dataset(self.folder)
        data = load_dataset(Path(self.tmp.name))
        self.assertEqual(find_dataset_dir(self.tmp.name), self.folder)
        self.assertGreater(len(data.ratings), 0)
        self.assertIsNotNone(data.links)

    def test_cleaning_removes_duplicates_missing_and_invalid(self):
        write_dataset(self.folder)
        ratings = pd.read_csv(self.folder / "ratings.csv")
        extra = pd.DataFrame([
            (1, 1, 5.0, 9_999_999),        # duplicate (user, movie) pair -> later one wins
            (1, 2, np.nan, 1),             # missing rating
            (1, 3, 9.0, 1),                # out of range
            (1, 99999, 4.0, 1),            # unknown movie
        ], columns=ratings.columns)
        pd.concat([ratings, extra]).to_csv(self.folder / "ratings.csv", index=False)
        data = load_dataset(self.folder)
        report = data.cleaning_report
        self.assertEqual(report["rows_with_missing_values_removed"], 1)
        self.assertEqual(report["rows_with_out_of_range_rating_removed"], 1)
        self.assertEqual(report["rows_with_unknown_movie_removed"], 1)
        self.assertGreaterEqual(report["duplicate_ratings_removed"], 0)
        self.assertFalse(data.ratings.duplicated(["userId", "movieId"]).any())
        self.assertTrue(data.ratings["rating"].between(0.5, 5.0).all())

    def test_missing_files_give_clear_error(self):
        with self.assertRaises(DatasetError) as ctx:
            load_dataset(Path(self.tmp.name) / "does-not-exist")
        self.assertIn("Download", str(ctx.exception))
        self.folder.mkdir(parents=True)
        with self.assertRaises(DatasetError):
            load_dataset(self.folder)

    def test_missing_column_and_empty_file(self):
        write_dataset(self.folder)
        pd.DataFrame({"userId": [1], "movieId": [1]}).to_csv(self.folder / "ratings.csv", index=False)
        with self.assertRaises(DatasetError) as ctx:
            load_dataset(self.folder)
        self.assertIn("rating", str(ctx.exception))
        (self.folder / "ratings.csv").write_text("")
        with self.assertRaises(DatasetError):
            load_dataset(self.folder)


class SplitTests(unittest.TestCase):
    def setUp(self):
        _, self.ratings, _ = make_frames()

    def test_split_is_disjoint_and_complete(self):
        train, test = train_test_split_ratings(self.ratings, 0.2, seed=1)
        self.assertEqual(len(train) + len(test), len(self.ratings))
        keys = lambda df: set(zip(df.userId, df.movieId))
        self.assertTrue(keys(train).isdisjoint(keys(test)))

    def test_every_test_user_has_training_history(self):
        train, test = train_test_split_ratings(self.ratings, 0.2, seed=1)
        self.assertTrue(set(test.userId).issubset(set(train.userId)))

    def test_split_is_reproducible(self):
        a = train_test_split_ratings(self.ratings, 0.2, seed=3)
        b = train_test_split_ratings(self.ratings, 0.2, seed=3)
        pd.testing.assert_frame_equal(a[1], b[1])

    def test_invalid_test_size(self):
        with self.assertRaises(ValueError):
            train_test_split_ratings(self.ratings, 1.5)


if __name__ == "__main__":
    unittest.main()
