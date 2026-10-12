"""Home dashboard - run with:  streamlit run app.py"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from src import cards
from src.preprocessing import find_movies
from src.ui import get_artifacts, render_movie_grid, section, setup_page, tmdb_status_note

setup_page("Home")
art = get_artifacts()

st.markdown(cards.hero_html(
    "Movie Recommendation System Using Machine Learning",
    "Discover movies with popularity, content-based, collaborative, SVD and hybrid recommenders "
    "trained on the MovieLens dataset."), unsafe_allow_html=True)

# ---- dataset statistics -------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies", f"{len(art.movies):,}")
c2.metric("Users", f"{art.all_ratings['userId'].nunique():,}")
c3.metric("Ratings", f"{len(art.all_ratings):,}")
c4.metric("Average rating", f"{art.all_ratings['rating'].mean():.2f}")

# ---- navigation ---------------------------------------------------------------------------
n1, n2, n3, n4, n5 = st.columns(5)
n1.page_link("pages/1_Recommendations.py", label="Recommendations", icon="🎯")
n2.page_link("pages/2_Personalized.py", label="Personalized", icon="👤")
n3.page_link("pages/3_Movie_Explorer.py", label="Movie Explorer", icon="🔎")
n4.page_link("pages/4_Model_Performance.py", label="Model Performance", icon="📊")
n5.page_link("pages/5_About.py", label="About", icon="ℹ️")

# ---- search -------------------------------------------------------------------------------
section("Search movies")
query = st.text_input("Search by title", placeholder="e.g. toy story, matrix, godfather", key="home_search")
if query.strip():
    hits = find_movies(art.movies, query, limit=12)
    if hits.empty:
        st.info(f"No movies found for '{query}'. Try a shorter or different title.")
    else:
        render_movie_grid(art.enrich(hits[["movieId"]]), show_reason=False)
        st.caption("Open **Recommendations** to get movies similar to one of these.")

# ---- popular / top rated ------------------------------------------------------------------
section("Popular movies")
st.caption("Ranked by weighted rating, so movies with very few ratings are not ranked unfairly.")
render_movie_grid(art.enrich(art.popularity_all.recommend(10, min_ratings=5)), show_reason=False)

section("Top-rated movies")
enough = 50 if (art.catalog["rating_count"] >= 50).sum() >= 10 else 5
render_movie_grid(art.enrich(art.popularity_all.top_rated(10, min_ratings=enough)), show_reason=False)
st.caption(f"Highest average rating among movies with at least {enough} ratings.")
tmdb_status_note()

# ---- charts -------------------------------------------------------------------------------
section("Dataset overview")
left, right = st.columns(2)
with left:
    genre_counts = pd.Series([g for genres in art.movies["genre_list"] for g in genres]).value_counts()
    fig = px.bar(genre_counts.sort_values(), orientation="h", labels={"value": "Movies", "index": "Genre"},
                 title="Genre distribution", template="plotly_dark", color_discrete_sequence=["#e50914"])
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)
with right:
    counts = art.all_ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor("#0b0b14")
    ax.set_facecolor("#16162a")
    ax.bar(counts.index.astype(str), counts.values, color="#8b5cf6")
    ax.set_title("Rating distribution", color="white")
    ax.set_xlabel("Rating (stars)", color="#d1d5db")
    ax.set_ylabel("Number of ratings", color="#d1d5db")
    ax.tick_params(colors="#d1d5db")
    for spine in ax.spines.values():
        spine.set_color("#3a2a5c")
    st.pyplot(fig)
    plt.close(fig)
