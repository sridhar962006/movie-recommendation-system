"""Tests for HTML card safety and the optional TMDb integration."""

import logging
import unittest
from unittest import mock

import requests

from src import cards, movie_metadata


class CardTests(unittest.TestCase):
    def test_titles_are_escaped_and_poster_urls_restricted(self):
        html = cards.movie_card_html("<script>alert(1)</script>", 1999, "Action", 4.0, 10,
                                     "javascript:alert(1)", "<b>why</b>")
        self.assertNotIn("<script>", html)
        self.assertNotIn("<b>why</b>", html)
        self.assertNotIn("javascript:", html)
        self.assertIn("data:image/svg+xml;base64,", html)        # local placeholder used
        self.assertNotIn("\n", html)                             # markdown-safe single line

    def test_tmdb_poster_url_is_allowed(self):
        url = "https://image.tmdb.org/t/p/w342/abc.jpg"
        self.assertIn(url, cards.movie_card_html("T", poster_url=url))

    def test_missing_values_do_not_crash(self):
        html = cards.movie_card_html(None, float("nan"), None, float("nan"), None)
        self.assertIn("Unknown title", html)
        self.assertIn("No ratings yet", html)


def _reset_module_state():
    movie_metadata._cache.clear()
    movie_metadata._failures = 0
    movie_metadata._key_rejected = False
    movie_metadata._blocked_until = 0.0


def _response(status, payload=None, headers=None):
    response = mock.Mock(status_code=status, headers=headers or {})
    response.json.return_value = payload or {}
    return response


class TMDbTests(unittest.TestCase):
    def setUp(self):
        _reset_module_state()
        self.addCleanup(_reset_module_state)

    def test_no_key_means_no_network_and_no_posters(self):
        with mock.patch.object(movie_metadata, "get_api_key", return_value=None), \
                mock.patch("src.movie_metadata.requests.get") as get:
            self.assertIsNone(movie_metadata.fetch_movie_metadata(1))
            self.assertEqual(movie_metadata.fetch_poster_urls([1, 2]), {1: None, 2: None})
            get.assert_not_called()

    def test_success_and_cache(self):
        payload = {"poster_path": "/p.jpg", "overview": "Plot", "vote_average": 7.5, "release_date": "1999-01-01"}
        with mock.patch.object(movie_metadata, "get_api_key", return_value="k" * 32), \
                mock.patch("src.movie_metadata.requests.get", return_value=_response(200, payload)) as get:
            first = movie_metadata.fetch_movie_metadata(5)
            second = movie_metadata.fetch_movie_metadata(5)
        self.assertEqual(first["poster_url"], "https://image.tmdb.org/t/p/w342/p.jpg")
        self.assertEqual(first, second)
        self.assertEqual(get.call_count, 1)                      # second call served from cache

    def test_missing_poster_is_none(self):
        with mock.patch.object(movie_metadata, "get_api_key", return_value="k"), \
                mock.patch("src.movie_metadata.requests.get", return_value=_response(200, {"poster_path": None})):
            self.assertIsNone(movie_metadata.fetch_movie_metadata(6)["poster_url"])

    def test_rejected_key_disables_further_calls(self):
        with mock.patch.object(movie_metadata, "get_api_key", return_value="bad"), \
                mock.patch("src.movie_metadata.requests.get", return_value=_response(401)) as get:
            self.assertIsNone(movie_metadata.fetch_movie_metadata(1))
            self.assertIsNone(movie_metadata.fetch_movie_metadata(2))
            self.assertEqual(get.call_count, 1)
        self.assertFalse(movie_metadata.is_enabled())

    def test_rate_limit_pauses_requests(self):
        with mock.patch.object(movie_metadata, "get_api_key", return_value="k"), \
                mock.patch("src.movie_metadata.requests.get",
                           return_value=_response(429, headers={"Retry-After": "10"})) as get:
            self.assertIsNone(movie_metadata.fetch_movie_metadata(1))
            self.assertIsNone(movie_metadata.fetch_movie_metadata(2))
            self.assertEqual(get.call_count, 1)

    def test_timeout_is_handled_and_secret_not_logged(self):
        secret = "SUPERSECRETKEY123"
        error = requests.Timeout(f"https://api.themoviedb.org/3/movie/1?api_key={secret}")
        with mock.patch.object(movie_metadata, "get_api_key", return_value=secret), \
                mock.patch("src.movie_metadata.requests.get", side_effect=error), \
                self.assertLogs("src.movie_metadata", level=logging.WARNING) as logs:
            self.assertIsNone(movie_metadata.fetch_movie_metadata(1))
        self.assertNotIn(secret, "\n".join(logs.output))

    def test_dotenv_does_not_override_environment(self):
        import os, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"TMDB_API_KEY": "from-env"}):
            env = Path(tmp) / ".env"
            env.write_text("TMDB_API_KEY=from-file\n# comment\n")
            movie_metadata._load_dotenv(env)
            self.assertEqual(os.environ["TMDB_API_KEY"], "from-env")


if __name__ == "__main__":
    unittest.main()
