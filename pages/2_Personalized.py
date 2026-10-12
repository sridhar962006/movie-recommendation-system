"""Personalised recommendations for existing users, plus a cold-start flow for new users."""

import streamlit as st

from src.config import ALGORITHMS
from src.exceptions import RecommenderError
from src.hybrid import HybridWeights
from src.ui import (ALGORITHM_HELP, get_artifacts, logger, movie_label, render_movie_grid, section, setup_page,
                    tmdb_status_note)

setup_page("Personalized", "👤")
art = get_artifacts()
st.title("Personalized Recommendations")

mode = st.radio("Who are you?", ["Existing user (from the dataset)", "New user (cold start)"], horizontal=True)

if mode.startswith("Existing"):
    user_id = st.selectbox("User ID", art.user_ids())
    history = art.user_history(user_id)

    m1, m2, m3 = st.columns(3)
    m1.metric("Movies rated", len(history))
    m2.metric("Average rating given", f"{history['rating'].mean():.2f}")
    m3.metric("Favourite genre", (history.loc[history["rating"] >= 4, "genres_display"].str.split(", ").explode()
                                  .value_counts().index[:1].tolist() or ["-"])[0])

    section("Highest-rated movies in this user's history")
    best = history.head(10)
    render_movie_grid(art.enrich(best[["movieId", "rating"]]), show_reason=False, badge_col="rating",
                      badge_format="Your rating: {:.1f}")
    with st.expander(f"All {len(history)} rated movies"):
        st.dataframe(history[["title", "year", "genres_display", "rating"]].rename(
            columns={"title": "Title", "year": "Year", "genres_display": "Genres", "rating": "Rating"}),
            hide_index=True)

    section("Recommended unseen movies")
    col_a, col_b = st.columns([2, 1])
    algorithm = col_a.selectbox("Algorithm", ALGORITHMS, index=0, key="user_algo")
    n = col_b.slider("How many?", 5, 20, 10, key="user_n")
    st.markdown(f'<div class="algo-box">{ALGORITHM_HELP[algorithm]}</div>', unsafe_allow_html=True)
    weights = HybridWeights()
    if algorithm == "Hybrid Recommendation":
        with st.expander("Hybrid weights (optional)"):
            parts = [st.slider("Content similarity", 0.0, 1.0, 0.4, 0.05, key="uw1"),
                     st.slider("Collaborative (SVD predicted rating)", 0.0, 1.0, 0.4, 0.05, key="uw2"),
                     st.slider("Movie quality", 0.0, 1.0, 0.2, 0.05, key="uw3")]
            if sum(parts) == 0:
                st.error("At least one weight must be above zero.")
                st.stop()
            weights = HybridWeights(*parts)

    if st.button("Recommend for this user", type="primary"):
        try:
            with st.spinner("Computing recommendations..."):
                result = art.recommend_for_user(algorithm, user_id, n, weights)
            if result.empty:
                st.info("No unseen movies could be scored with this algorithm for this user.")
            else:
                render_movie_grid(result, predicted_col="predicted_rating")
                st.caption("'Predicted' is the model's estimate of the star rating this user would give. "
                           "It is an estimate, not a real rating. Popularity and Content-Based do not predict ratings.")
        except RecommenderError as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("Personalised recommendation failed")
            st.error("Something went wrong while generating recommendations.")
else:
    st.info("New users have no rating history, so nothing is guessed about you. "
            "Tell us what you like - your choices are used only for this session and are not stored.")
    genres = st.multiselect("Favourite genres", art.all_genres())
    top_ids = art.popularity_all.most_popular(500)["movieId"].tolist()
    options = art.movies[art.movies["movieId"].isin(top_ids)].set_index("movieId").loc[top_ids].reset_index()
    labels = {int(r.movieId): movie_label(r) for _, r in options.iterrows()}
    favourites = st.multiselect("Movies you like (pick a few)", list(labels), format_func=lambda m: labels[m])
    n = st.slider("How many?", 5, 20, 10, key="cold_n")
    if st.button("Recommend", type="primary", key="cold_go"):
        if not genres and not favourites:
            st.warning("Pick at least one genre or movie first.")
        else:
            try:
                render_movie_grid(art.recommend_new_user(genres, favourites, n))
            except RecommenderError as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Cold-start recommendation failed")
                st.error("Something went wrong while generating recommendations.")
tmdb_status_note()
