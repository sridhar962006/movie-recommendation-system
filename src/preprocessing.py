"""Title/genre cleaning and movie-level statistics."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NO_GENRES = "(no genres listed)"
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_ARTICLE_RE = re.compile(r"^(?P<main>.+?),\s*(?P<article>The|A|An)(?P<rest>\s*\(.*\))?$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def extract_year(title: str | None) -> int | None:
    """Return the release year from 'Toy Story (1995)', or None."""
    if not isinstance(title, str):
        return None
    match = _YEAR_RE.search(title.strip())
    return int(match.group(1)) if match else None


def clean_title(title: str | None) -> str:
    """'Matrix, The (1999)' -> 'The Matrix' (year removed, article moved to front)."""
    if not isinstance(title, str):
        return ""
    text = _YEAR_RE.sub("", title.strip()).strip()
    match = _ARTICLE_RE.match(text)
    if match:
        text = f"{match.group('article')} {match.group('main')}{match.group('rest') or ''}"
    return text.strip()


def split_genres(genres: str | None) -> list[str]:
    """'Action|Sci-Fi' -> ['Action', 'Sci-Fi']; missing genres -> []."""
    if not isinstance(genres, str) or not genres.strip() or genres.strip() == NO_GENRES:
        return []
    return [g.strip() for g in genres.split("|") if g.strip()]


def genre_token(genre: str) -> str:
    """Make a genre safe as a vectorizer token: 'Sci-Fi' -> 'sci_fi'."""
    return _NON_ALNUM.sub("_", genre.lower()).strip("_")


def _tag_tokens(tags: pd.DataFrame | None, max_tags: int = 10) -> dict[int, list[str]]:
    """Most frequent user tags per movie, as single tokens."""
    if tags is None or tags.empty:
        return {}
    tidy = tags.copy()
    tidy["tag"] = tidy["tag"].astype(str).str.lower().str.strip()
    tidy = tidy[tidy["tag"] != ""]
    counts = tidy.groupby(["movieId", "tag"]).size().reset_index(name="n")
    counts = counts.sort_values(["movieId", "n", "tag"], ascending=[True, False, True])
    top = counts.groupby("movieId").head(max_tags)
    result: dict[int, list[str]] = {}
    for movie_id, tag in zip(top["movieId"], top["tag"]):
        token = _NON_ALNUM.sub("_", tag).strip("_")
        if token:
            result.setdefault(int(movie_id), []).append("tag_" + token)
    return result


def prepare_movies(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add display and feature columns to the raw movies table.

    New columns: clean_title, year, genre_list, genres_display, search_key, content_text.
    ``content_text`` is the document the content-based vectorizer sees: genre tokens
    (counted twice so they dominate), an optional decade token and optional user tags.
    """
    prepared = movies.copy()
    prepared["movieId"] = prepared["movieId"].astype("int64")
    prepared["title"] = prepared["title"].fillna("").astype(str)
    prepared["clean_title"] = prepared["title"].map(clean_title)
    prepared["year"] = prepared["title"].map(extract_year).astype("Int64")
    prepared["genre_list"] = prepared["genres"].map(split_genres)
    prepared["genres_display"] = prepared["genre_list"].map(lambda g: ", ".join(g) if g else "Unknown")
    prepared["search_key"] = prepared["clean_title"].str.lower()

    tag_map = _tag_tokens(tags)
    texts = []
    for movie_id, genre_list, year in zip(prepared["movieId"], prepared["genre_list"], prepared["year"]):
        tokens = [genre_token(g) for g in genre_list] * 2
        if pd.notna(year):
            tokens.append(f"decade_{int(year) // 10 * 10}")
        tokens.extend(tag_map.get(int(movie_id), []))
        texts.append(" ".join(t for t in tokens if t))
    prepared["content_text"] = texts
    return prepared.reset_index(drop=True)


def compute_movie_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-movie rating count and mean rating (index = movieId)."""
    stats = ratings.groupby("movieId")["rating"].agg(rating_count="count", avg_rating="mean")
    stats["rating_count"] = stats["rating_count"].astype("int64")
    return stats


def find_movies(movies: pd.DataFrame, query: str, limit: int = 50) -> pd.DataFrame:
    """Case-insensitive title search; prefix matches are listed first."""
    query = (query or "").strip().lower()
    if not query:
        return movies.iloc[0:0]
    key = movies["search_key"]
    raw = movies["title"].str.lower()
    contains = key.str.contains(query, regex=False) | raw.str.contains(query, regex=False)
    hits = movies[contains].copy()
    hits["_prefix"] = ~hits["search_key"].str.startswith(query)
    hits = hits.sort_values(["_prefix", "clean_title"], kind="stable").drop(columns="_prefix")
    return hits.head(limit)


def lookup_movie_id(movies: pd.DataFrame, title: str) -> int | None:
    """Exact (case-insensitive) title lookup; accepts raw or cleaned titles."""
    if not isinstance(title, str) or not title.strip():
        return None
    needle = title.strip().lower()
    mask = (movies["title"].str.lower() == needle) | (movies["search_key"] == needle)
    hit = movies.loc[mask, "movieId"]
    return int(hit.iloc[0]) if len(hit) else None


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant (or empty / all-NaN) vector becomes all zeros."""
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    low, high = values[finite].min(), values[finite].max()
    span = high - low
    if span <= 0:
        return np.zeros_like(values)
    scaled = (values - low) / span
    scaled[~finite] = 0.0
    return scaled
