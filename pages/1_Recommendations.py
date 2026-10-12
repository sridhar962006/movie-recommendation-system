"""Movie-based recommendations with a choice of algorithm."""

import pandas as pd
import streamlit as st

from src.config import ALGORITHMS
from src.exceptions import RecommenderError
from src.hybrid import HybridWeights
from src.preprocessing import find_movies
from src.ui import (ALGORITHM_HELP, get_artifacts, logger, movie_label, render_movie_grid, section, setup_page,
                    tmdb_status_note)

setup_page("Recommendations", "🎯")
art = get_artifacts()
st.title("Movie Recommendations")
st.write("Pick a movie you like, choose an algorithm, and get similar movies to watch next.")

# ---- 1. choose a movie ---------------------------------------------------------------------
query = st.text_input("Search for a movie", placeholder="Type part of a title...", key="rec_query")
if query.strip():
    pool = find_movies(art.movies, query, limit=100)
else:
    top_ids = art.popularity_all.most_popular(200)["movieId"]
    pool = art.movies.set_index("movieId").loc[top_ids].reset_index()      # most-rated first

if pool.empty:
    st.warning(f"No movies match '{query}'. Try another title.")
    st.stop()

labels = {int(row.movieId): movie_label(row) for _, row in pool.iterrows()}
movie_id = st.selectbox("Select a movie", options=list(labels), format_func=lambda m: labels[m],
                        help="Shows the 200 most-rated movies until you type a search.")

# ---- 2. options ----------------------------------------------------------------------------
col_a, col_b = st.columns([2, 1])
algorithm = col_a.selectbox("Algorithm", ALGORITHMS, index=0)
n = col_b.slider("Number of recommendations", 5, 20, 10)
st.markdown(f'<div class="algo-box">{ALGORITHM_HELP[algorithm]}</div>', unsafe_allow_html=True)

weights = HybridWeights()
if algorithm == "Hybrid Recommendation":
    with st.expander("Hybrid weights (optional)"):
        w_content = st.slider("Content similarity", 0.0, 1.0, 0.4, 0.05)
        w_collab = st.slider("Collaborative signal", 0.0, 1.0, 0.4, 0.05)
        w_quality = st.slider("Movie quality (weighted rating)", 0.0, 1.0, 0.2, 0.05)
        if w_content + w_collab + w_quality == 0:
            st.error("At least one weight must be above zero.")
            st.stop()
        weights = HybridWeights(w_content, w_collab, w_quality)
        shown = weights.normalised()
        st.caption(f"Normalised weights: content {shown.content:.2f}, collaborative {shown.collaborative:.2f}, "
                   f"quality {shown.quality:.2f}")

# ---- 3. selected movie -----------------------------------------------------------------------
selected = art.movie_row(movie_id)
section("Selected movie")
render_movie_grid(art.enrich(pd.DataFrame({"movieId": [int(selected["movieId"])]})), show_reason=False)

# ---- 4. recommend ----------------------------------------------------------------------------
if st.button("Recommend", type="primary"):
    try:
        with st.spinner("Finding recommendations..."):
            result = art.recommend_for_movie(algorithm, movie_id, n, weights)
        st.session_state["rec_result"] = {"frame": result, "algorithm": algorithm, "movie": labels[movie_id]}
    except RecommenderError as exc:
        st.session_state.pop("rec_result", None)
        st.error(str(exc))
    except Exception:
        logger.exception("Recommendation failed")
        st.session_state.pop("rec_result", None)
        st.error("Something went wrong while generating recommendations. Please try another movie.")

state = st.session_state.get("rec_result")
if state:
    section(f"{state['algorithm']} - because you picked {state['movie']}")
    if state["frame"].empty:
        st.info("No recommendations are available for this movie with this algorithm "
                "(for example, it has no ratings in the training data). Try Content-Based or Hybrid.")
    else:
        render_movie_grid(state["frame"])
        with st.expander("Show scores"):
            columns = [c for c in ["title", "score", "content_score", "collaborative_score", "quality_score"]
                       if c in state["frame"].columns]
            st.dataframe(state["frame"][columns].round(3), hide_index=True)
tmdb_status_note()
