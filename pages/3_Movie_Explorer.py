"""Browse and filter the whole catalogue, then inspect a movie and its neighbours."""

import pandas as pd
import streamlit as st

from src.exceptions import RecommenderError
from src.ui import get_artifacts, logger, render_movie_grid, section, setup_page, tmdb_status_note

setup_page("Movie Explorer", "🔎")
art = get_artifacts()
st.title("Movie Explorer")

catalog = art.catalog
max_count = int(catalog["rating_count"].max())

c1, c2 = st.columns([2, 2])
query = c1.text_input("Search by title", placeholder="Type part of a title...")
genres = c2.multiselect("Genres", art.all_genres())
c3, c4, c5, c6 = st.columns(4)
match_all = c3.checkbox("Must match ALL selected genres", value=False)
rating_range = c4.slider("Average rating", 0.5, 5.0, (0.5, 5.0), 0.5)
min_count = c5.slider("Minimum number of ratings", 0, max(max_count, 1), min(5, max_count))
sort_by = c6.selectbox("Sort by", ["Average rating", "Popularity (number of ratings)", "Title (A-Z)"])
ascending = st.checkbox("Ascending order", value=sort_by == "Title (A-Z)")

frame = catalog.copy()
if query.strip():
    needle = query.strip().lower()
    frame = frame[frame["search_key"].str.contains(needle, regex=False)
                  | frame["title"].str.lower().str.contains(needle, regex=False)]
if genres:
    chooser = all if match_all else any
    frame = frame[frame["genre_list"].map(lambda g: chooser(x in g for x in genres))]
frame = frame[frame["rating_count"] >= min_count]
frame = frame[frame["avg_rating"].between(*rating_range)]
sort_column = {"Average rating": "avg_rating", "Popularity (number of ratings)": "rating_count",
               "Title (A-Z)": "search_key"}[sort_by]
frame = frame.sort_values([sort_column, "movieId"], ascending=[ascending, True], kind="stable")

st.caption(f"{len(frame):,} movies match (showing up to 500).")
if frame.empty:
    st.info("No movies match these filters. Try lowering the minimum rating count or widening the rating range.")
    st.stop()

table = frame.head(500)[["title", "year", "genres_display", "avg_rating", "rating_count"]].rename(columns={
    "title": "Title", "year": "Year", "genres_display": "Genres", "avg_rating": "Avg rating",
    "rating_count": "Ratings"})
table["Avg rating"] = table["Avg rating"].round(2)
st.dataframe(table, hide_index=True)

section("Movie details")
shortlist = frame.head(200)
options = {int(r.movieId): f"{r.title}" for r in shortlist.itertuples()}
movie_id = st.selectbox("Inspect a movie from the list above", list(options), format_func=lambda m: options[m])
try:
    render_movie_grid(art.enrich(pd.DataFrame({"movieId": [movie_id]})), show_reason=False)
    left, right = st.columns(2)
    with left:
        st.markdown("**Similar by content (genres/metadata)**")
        render_movie_grid(art.recommend_for_movie("Content-Based", movie_id, 5))
    with right:
        st.markdown("**Liked by similar users (collaborative)**")
        similar = art.recommend_for_movie("Collaborative Filtering", movie_id, 5)
        if similar.empty:
            st.info("This movie has too few ratings for collaborative suggestions.")
        else:
            render_movie_grid(similar)
except RecommenderError as exc:
    st.error(str(exc))
except Exception:
    logger.exception("Movie detail view failed")
    st.error("Could not load the details for this movie.")
tmdb_status_note()
