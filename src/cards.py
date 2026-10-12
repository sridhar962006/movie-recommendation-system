"""Pure HTML/CSS helpers for the dark cinematic theme (no Streamlit dependency).

All text coming from the dataset or an API is HTML-escaped, and poster URLs are only
accepted from the TMDb image host, so a malicious title cannot inject markup.
"""

from __future__ import annotations

import base64
import html
import math
from functools import lru_cache
from typing import Any

from .config import ASSETS_DIR

ALLOWED_POSTER_PREFIX = "https://image.tmdb.org/"

CSS = """
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1a1040 0%, #0b0b14 55%) fixed; }
h1, h2, h3 { letter-spacing: .3px; }
.hero { padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(229,9,20,.25), rgba(139,92,246,.22));
        border: 1px solid rgba(255,255,255,.08); }
.hero h1 { margin: 0 0 6px 0; font-size: 2.2rem; }
.hero p { margin: 0; color: #d1d5db; font-size: 1.02rem; }
.section-title { margin: 26px 0 10px 0; padding-left: 12px; border-left: 4px solid #e50914;
                 font-size: 1.35rem; font-weight: 700; }
.movie-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 16px; }
.movie-card { background: #16162a; border: 1px solid rgba(255,255,255,.07); border-radius: 14px;
              overflow: hidden; display: flex; flex-direction: column;
              transition: transform .15s ease, border-color .15s ease; }
.movie-card:hover { transform: translateY(-4px); border-color: #8b5cf6; }
.movie-card img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; background: #0b0b14; display: block; }
.movie-body { padding: 10px 12px 12px 12px; display: flex; flex-direction: column; gap: 4px; }
.movie-title { font-weight: 700; font-size: .95rem; line-height: 1.25; color: #fff; }
.movie-meta { color: #9ca3af; font-size: .78rem; }
.movie-rating { color: #fbbf24; font-size: .82rem; font-weight: 600; }
.movie-badge { display: inline-block; background: rgba(139,92,246,.22); color: #c4b5fd; border-radius: 999px;
               padding: 2px 9px; font-size: .74rem; font-weight: 600; align-self: flex-start; }
.movie-reason { color: #d1d5db; font-size: .76rem; font-style: italic; border-top: 1px dashed rgba(255,255,255,.12);
                padding-top: 6px; margin-top: 4px; }
.algo-box { background: rgba(139,92,246,.10); border-left: 4px solid #8b5cf6; padding: 10px 14px;
            border-radius: 8px; color: #e5e7eb; margin: 8px 0 14px 0; }
[data-testid="stMetric"] { background: #16162a; border: 1px solid rgba(255,255,255,.07);
                           border-radius: 14px; padding: 14px 16px; }
"""


@lru_cache(maxsize=1)
def placeholder_data_uri() -> str:
    """The local placeholder poster as a data: URI (works offline)."""
    try:
        data = (ASSETS_DIR / "placeholder.svg").read_bytes()
    except OSError:
        data = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450"><rect width="300" height="450" fill="#16162a"/></svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")


def safe_poster_url(url: Any) -> str:
    """Return the poster URL if it is a TMDb image URL, otherwise the placeholder."""
    if isinstance(url, str) and url.startswith(ALLOWED_POSTER_PREFIX):
        return url
    return placeholder_data_uri()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def movie_card_html(title: Any, year: Any = None, genres: Any = None, avg_rating: Any = None,
                    rating_count: Any = None, poster_url: Any = None, reason: Any = None,
                    badge: str | None = None) -> str:
    """One movie card as a single-line HTML string (no blank lines, markdown-safe)."""
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [f'<div class="movie-card"><img src="{esc(safe_poster_url(poster_url))}" alt="Poster" loading="lazy"/>',
             '<div class="movie-body">']
    if badge:
        parts.append(f'<span class="movie-badge">{esc(badge)}</span>')
    heading = esc(title if title else "Unknown title")
    parts.append(f'<div class="movie-title">{heading}</div>')
    meta = []
    if _is_number(year) or (year is not None and str(year).isdigit()):
        meta.append(str(int(float(year))))
    if genres:
        meta.append(str(genres))
    if meta:
        separator = " &middot; "
        parts.append(f'<div class="movie-meta">{separator.join(esc(m) for m in meta)}</div>')
    if _is_number(avg_rating):
        count = f" ({int(rating_count):,} ratings)" if _is_number(rating_count) else ""
        parts.append(f'<div class="movie-rating">&#9733; {float(avg_rating):.2f}{esc(count)}</div>')
    else:
        parts.append('<div class="movie-meta">No ratings yet</div>')
    if reason:
        parts.append(f'<div class="movie-reason">{esc(reason)}</div>')
    parts.append("</div></div>")
    return "".join(parts)


def grid_html(card_list: list[str]) -> str:
    return '<div class="movie-grid">' + "".join(card_list) + "</div>"


def section_title_html(text: str) -> str:
    return f'<div class="section-title">{html.escape(text)}</div>'


def hero_html(title: str, subtitle: str) -> str:
    return f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'
