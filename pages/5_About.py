"""Static project description (works even before the models are trained)."""

import streamlit as st

from src.ui import setup_page

setup_page("About", "ℹ️")
st.title("About this Project")
st.subheader("Movie Recommendation System Using Machine Learning")

st.header("Problem statement")
st.write("Streaming catalogues contain thousands of movies, so users struggle to find something they will enjoy. "
         "This project builds a recommender that suggests relevant movies from ratings and movie genres.")

st.header("Objectives")
st.markdown("- Implement popularity, content-based, collaborative, SVD and hybrid recommenders.\n"
            "- Evaluate them fairly on a held-out test split (RMSE, MAE, Precision@K, Recall@K, F1@K).\n"
            "- Present everything in an interactive Streamlit application.\n"
            "- Handle cold-start users, unknown movies and missing data gracefully.")

st.header("Dataset")
st.write("MovieLens (GroupLens Research), the 'latest-small' version: about 100,000 ratings from 600 users on "
         "9,000 movies, with genres, optional tags and TMDb links. Each user's ratings are split into 80% "
         "training and 20% testing.")

st.header("Machine-learning algorithms")
st.markdown(
    "| Algorithm | Idea |\n|---|---|\n"
    "| Popularity | Weighted rating `WR = v/(v+m)*R + m/(v+m)*C` ranks movies without trusting tiny sample sizes |\n"
    "| Content-based | TF-IDF vectors of genres/tags + cosine similarity |\n"
    "| Collaborative (item-based) | Adjusted cosine similarity between movies from the user-movie matrix |\n"
    "| SVD | Biased matrix factorisation `mu + b_u + b_i + p_u.q_i` trained with SGD (NumPy implementation) |\n"
    "| Hybrid | Min-max normalised content, collaborative and quality signals with configurable weights |")

st.header("System architecture")
st.code(
    "MovieLens CSV files\n"
    "      |  data_loader (validate, clean, split)\n"
    "      v\n"
    "preprocessing -> popularity | content_based | collaborative | svd_model -> hybrid\n"
    "      |                                                              |\n"
    "      +--- train.py: fit on train split, evaluate on test split -----+\n"
    "      v\n"
    "models/artifacts.joblib + metrics.json\n"
    "      v\n"
    "Streamlit UI (app.py + pages/)  <-- optional TMDb posters (movie_metadata)",
    language="text")

st.header("Limitations")
st.markdown("- Only explicit star ratings and genres are used; plots, cast and reviews are not.\n"
            "- Models are trained on the 80% training split, so the newest 20% of each user's ratings are not learned.\n"
            "- Offline metrics treat unrated movies as unknown, which understates real-world precision.\n"
            "- Popular movies are over-represented in the data (popularity bias).\n"
            "- New users and brand-new movies lack ratings (cold start); the app falls back to genres and favourites.\n"
            "- Evaluation uses one split and a sample of users, so small differences between algorithms are not conclusive.")

st.header("Future enhancements")
st.markdown("- Neural collaborative filtering or implicit-feedback models (ALS, BPR).\n"
            "- Text features from plot summaries with sentence embeddings.\n"
            "- Time-aware models and k-fold cross-validation with significance tests.\n"
            "- Diversity / novelty metrics and explanations with more detail.\n"
            "- User accounts, feedback buttons and online A/B testing.")
