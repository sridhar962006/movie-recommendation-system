"""Shows the real evaluation results saved by train.py (nothing here is hard-coded)."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import get_artifacts, section, setup_page

setup_page("Model Performance", "📊")
art = get_artifacts()
st.title("Model Performance")

metrics = art.metrics
if not metrics:
    st.warning("These models were trained without evaluation, so there are no results to show. "
               "Run `python train.py` (without --skip-eval) to evaluate them on the held-out test split.")
    st.stop()

meta, setup = metrics["meta"], metrics["ranking_setup"]
st.markdown(
    f"Models were trained on **{meta['n_train']:,}** ratings and evaluated on **{meta['n_test']:,}** held-out ratings "
    f"(per-user split, test fraction {meta['test_fraction']:.0%}, seed {meta['seed']}). "
    f"Trained at {meta['trained_at']} UTC.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Training ratings", f"{meta['n_train']:,}")
m2.metric("Test ratings", f"{meta['n_test']:,}")
m3.metric("Users evaluated for ranking", f"{setup['users_evaluated']:,}")
m4.metric("K", setup["k"])

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

# ---- rating prediction -----------------------------------------------------------------------
section("Rating prediction: RMSE and MAE (lower is better)")
rating = pd.DataFrame(metrics["rating_prediction"])
best = rating.loc[rating["rmse"].idxmin()]
cards = st.columns(len(rating))
for col, row in zip(cards, rating.itertuples()):
    col.metric(row.algorithm, f"RMSE {row.rmse:.3f}", f"MAE {row.mae:.3f}", delta_color="off")
long = rating.melt(id_vars="algorithm", value_vars=["rmse", "mae"], var_name="metric", value_name="value")
long["metric"] = long["metric"].str.upper()
fig = px.bar(long, x="algorithm", y="value", color="metric", barmode="group", text_auto=".3f",
             color_discrete_map={"RMSE": "#e50914", "MAE": "#8b5cf6"}, **PLOT)
fig.update_layout(xaxis_title="", yaxis_title="Error (stars)")
st.plotly_chart(fig)
st.caption(f"Lowest RMSE in this run: **{best['algorithm']}** ({best['rmse']:.3f}). "
           "Differences between close results may not be meaningful with a single train/test split.")
with st.expander("Predicted rating vs actual rating - what do RMSE and MAE mean?"):
    st.markdown(
        "- **Actual rating**: the stars a user really gave a movie (kept hidden from the model in the test split).\n"
        "- **Predicted rating**: the model's estimate of those stars from training data only.\n"
        "- **MAE** is the average size of the mistake in stars; **RMSE** is similar but punishes large "
        "mistakes more. An RMSE of 0.9 means predictions are typically about 0.9 stars away.\n"
        "- Baselines (global mean, item mean) show how much the learned models actually improve on trivial guesses.")

# ---- ranking ---------------------------------------------------------------------------------
section(f"Top-{setup['k']} recommendation quality")
ranking = pd.DataFrame(metrics["ranking"])
view = ranking[["algorithm", "precision", "recall", "f1", "hit_rate", "coverage"]].rename(columns={
    "algorithm": "Algorithm", "precision": f"Precision@{setup['k']}", "recall": f"Recall@{setup['k']}",
    "f1": f"F1@{setup['k']}", "hit_rate": "Hit rate", "coverage": "Catalogue coverage"})
st.dataframe(view.round(4), hide_index=True)
long = ranking.melt(id_vars="algorithm", value_vars=["precision", "recall", "f1"], var_name="metric",
                    value_name="value")
fig = px.bar(long, x="algorithm", y="value", color="metric", barmode="group", text_auto=".3f",
             color_discrete_map={"precision": "#e50914", "recall": "#8b5cf6", "f1": "#fbbf24"}, **PLOT)
fig.update_layout(xaxis_title="", yaxis_title="Score")
st.plotly_chart(fig)
top = ranking.loc[ranking["f1"].idxmax()]
st.caption(f"Highest F1@{setup['k']} in this run: **{top['algorithm']}** ({top['f1']:.3f}). "
           "Treat small gaps with caution: only a sample of users is evaluated.")

with st.expander("How the ranking metrics are calculated"):
    st.markdown(
        f"- **Relevant movie**: a test-split movie the user rated **>= {setup['relevance_threshold']}** stars.\n"
        "- **Candidates**: every movie the user did not rate in the training split (test movies compete with all of them).\n"
        f"- **Precision@K** = relevant movies in the top {setup['k']} / {setup['k']}; **Recall@K** = relevant movies in the "
        "top K / all relevant movies; **F1@K** is their harmonic mean. Values are averaged over users.\n"
        f"- Only users with at least one relevant test movie are evaluated (a random sample of up to "
        f"{setup['max_users']}).\n"
        "- Unrated does not mean disliked, so absolute values are low by design - compare with the **Random** baseline.\n"
        f"- {setup['note']}")

# ---- training information --------------------------------------------------------------------
section("Training information")
svd = meta["svd"]
st.markdown(f"SVD: {svd['n_factors']} latent factors, {svd['n_epochs']} epochs, learning rate {svd['lr']}, "
            f"regularisation {svd['reg']}. Hybrid default weights: "
            + ", ".join(f"{k} {v:.2f}" for k, v in art.meta.get("hybrid_default_weights", {}).items()) + ".")
history = pd.DataFrame(metrics.get("svd_training_history", []))
if not history.empty:
    fig = px.line(history, x="epoch", y="train_rmse", markers=True, title="SVD training error per epoch",
                  color_discrete_sequence=["#e50914"], **PLOT)
    fig.update_layout(yaxis_title="Training RMSE")
    st.plotly_chart(fig)
