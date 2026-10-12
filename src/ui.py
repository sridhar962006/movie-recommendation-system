"""Streamlit glue shared by all pages: page setup, cached model loading, card grids.

This is the only module (besides app.py and pages/) that imports Streamlit; all ML logic
lives in the other modules.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from . import cards, movie_metadata
from .config import DATA_DIR, configure_logging
from .exceptions import DatasetError, ModelNotTrainedError
from .pipeline import Artifacts, build_artifacts, load_artifacts, save_artifacts

logger = logging.getLogger(__name__)

ALGORITHM_HELP = {
    "Hybrid Recommendation": "Blends content similarity, collaborative evidence and movie quality. "
                             "Each signal is scaled to 0-1 and combined with the weights you choose.",
    "Content-Based": "Finds movies with similar genres/metadata using TF-IDF vectors and cosine similarity.",
    "Collaborative Filtering": "Item-based: movies that users with similar rating behaviour rated alike "
                               "(adjusted cosine similarity on the user-movie matrix).",
    "SVD": "Matrix factorisation: learns hidden taste factors for users and movies and ranks by predicted rating.",
    "Popularity-Based": "Ranks well-rated movies by a weighted rating that discounts movies with few votes. "
                        "Not personalised.",
}


def setup_page(title: str, icon: str = "🎬") -> None:
    """Must be the first Streamlit call of every page."""
    st.set_page_config(page_title=f"{title} | Movie Recommender", page_icon=icon, layout="wide")
    configure_logging()
    st.markdown(f"<style>{cards.CSS}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained models...")
def _load_cached() -> Artifacts:
    return load_artifacts()


def get_artifacts() -> Artifacts:
    """Return trained artifacts, or show setup help (with a train button) and stop the page."""
    try:
        return _load_cached()
    except ModelNotTrainedError as exc:
        st.warning(str(exc))
        st.markdown("**Option 1 - from a terminal (recommended):**")
        st.code("python train.py", language="bash")
        st.markdown("**Option 2 - train from this page** (needs the MovieLens files in `data/`; takes a few minutes):")
        if st.button("Train models now"):
            try:
                with st.status("Training models...", expanded=True) as status:
                    art = build_artifacts(DATA_DIR, progress=lambda message: status.write(message))
                    save_artifacts(art)
                    status.update(label="Training finished", state="complete")
                _load_cached.clear()
                st.rerun()
            except DatasetError as problem:
                st.error(str(problem))
            except Exception:                                   # keep details in the log, not the UI
                logger.exception("Training from the UI failed")
                st.error("Training failed. Check the terminal for details, then try `python train.py`.")
        st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_posters(ids: tuple[int, ...]) -> dict[int, str | None]:
    return movie_metadata.fetch_poster_urls(list(ids))


def attach_posters(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a poster_url column (None everywhere when TMDb is not configured)."""
    frame = frame.copy()
    frame["poster_url"] = None
    if "tmdbId" in frame.columns and movie_metadata.is_enabled():
        ids = tuple(int(i) for i in frame["tmdbId"].dropna().unique())
        if ids:
            urls = _cached_posters(ids)
            frame["poster_url"] = frame["tmdbId"].map(lambda i: urls.get(int(i)) if pd.notna(i) else None)
    return frame


def render_movie_grid(frame: pd.DataFrame, show_reason: bool = True, predicted_col: str | None = None,
                      badge_col: str | None = None, badge_format: str = "{:.1f}") -> None:
    """Render enriched recommendation rows as a responsive grid of poster cards."""
    if frame.empty:
        st.info("Nothing to show here yet.")
        return
    frame = attach_posters(frame)
    html_cards = []
    for values in frame.to_dict("records"):
        badge = None
        if predicted_col and pd.notna(values.get(predicted_col)):
            badge = f"Predicted: {values[predicted_col]:.1f} ★"
        elif badge_col and pd.notna(values.get(badge_col)):
            badge = badge_format.format(values[badge_col])
        html_cards.append(cards.movie_card_html(
            values.get("title"), values.get("year"), values.get("genres_display"), values.get("avg_rating"),
            values.get("rating_count"), values.get("poster_url"),
            values.get("reason") if show_reason else None, badge))
    st.markdown(cards.grid_html(html_cards), unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(cards.section_title_html(title), unsafe_allow_html=True)


def movie_label(row: pd.Series) -> str:
    year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
    return f"{row['clean_title']}{year}"


def tmdb_status_note() -> None:
    """Small caption explaining why posters are missing (only when relevant)."""
    if not movie_metadata.is_enabled():
        st.caption("Posters are optional: add a TMDB_API_KEY (see README) to show real posters. "
                   "Everything else works without it.")
