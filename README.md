# Movie Recommendation System Using Machine Learning

A complete, modular movie recommender built with **Python, scikit-learn, NumPy/pandas and Streamlit**, trained on the
**MovieLens** dataset. It implements five recommendation approaches (popularity, content-based, collaborative
filtering, SVD matrix factorisation and a hybrid), evaluates them on a held-out test split, and presents everything in
a dark, cinematic multipage web app.

---

## 1. Problem statement

Streaming services have thousands of titles. Users cannot browse them all, so they need help finding movies they will
enjoy. This project builds a recommender that learns from other people's ratings and from movie genres to suggest
relevant movies, and measures honestly how well each technique works.

## 2. Features

* **Five algorithms** - Popularity, Content-Based, Collaborative Filtering (item-based), SVD, Hybrid (default).
* **Real evaluation** - RMSE and MAE for rating prediction; Precision@K, Recall@K, F1@K, hit rate and coverage for
  ranking, compared against baselines. Numbers on the Performance page come from `models/metrics.json`, produced by
  `train.py`. Nothing is hard-coded.
* **No data leakage** - every model is trained on the training split only; test ratings are never used for fitting.
* **Six-page Streamlit app** - Home dashboard, Recommendations, Personalized (with cold-start flow), Movie Explorer,
  Model Performance, About.
* **Optional TMDb posters** - key read from an environment variable / `.env` / Streamlit secrets; the app works fully
  without it and shows a local placeholder.
* **Robust** - dataset validation with readable errors, duplicate/missing-value cleaning, unknown movie / user
  handling, cold start, cached models, no stack traces shown to users, secrets never logged.
* **Unit tests** for loading, preprocessing, every algorithm, evaluation, cards and the TMDb client.

## 3. Technologies

| Purpose | Library |
|---|---|
| UI | Streamlit |
| Data | pandas, NumPy, SciPy (sparse matrices) |
| ML | scikit-learn (TF-IDF, cosine similarity), NumPy (SVD / matrix factorisation) |
| Charts | Plotly, Matplotlib |
| Model storage | joblib |
| Posters (optional) | TMDb API via `requests` |
| Tests | unittest / pytest |

> **Why NumPy for SVD instead of Surprise?** `scikit-surprise` must be compiled and frequently fails to install
> (especially on Windows and newer Python versions). `src/svd_model.py` implements the *same algorithm* Surprise's
> `SVD` uses (biased matrix factorisation trained by SGD), so there is nothing to compile.

## 4. The algorithms in simple English

1. **Popularity-based** - rank movies by a *weighted rating*, `WR = v/(v+m)·R + m/(v+m)·C`
   (`v` = number of ratings, `R` = movie's mean rating, `m` = minimum votes = 80th percentile of rating counts,
   `C` = global mean). A movie with one 5-star rating is pulled toward the global average, so it cannot beat a movie
   loved by thousands. Not personalised.
2. **Content-based** - describe each movie by its genres (plus decade and user tags if `tags.csv` exists), turn that
   into a TF-IDF vector, and recommend the movies whose vectors point in the most similar direction (**cosine
   similarity**). The chosen movie is excluded from its own list.
3. **Collaborative filtering (item-based)** - "people who rated these movies the same way also liked...". Ratings are
   centred on each user's average (*adjusted cosine*), so a missing rating means "no information", not "disliked".
   Similarities between movies sharing few raters are shrunk toward zero, and only each movie's 50 nearest
   neighbours are stored. A user's unseen movies are scored from their rated movies' neighbours.
4. **SVD (matrix factorisation)** - learns a small vector of hidden "taste factors" for every user and movie such that
   `predicted = mean + user_bias + movie_bias + user_factors · movie_factors`. Trained with stochastic gradient
   descent on training ratings only, then used to predict ratings for movies a user has not seen.
5. **Hybrid** - combines three signals, each min-max scaled to 0-1 over the candidate movies:
   `score = w_content·content + w_collaborative·collaborative + w_quality·quality`
   (weights are configurable in the UI and normalised to sum to 1; defaults 0.4 / 0.4 / 0.2).
   *Movie mode*: content = TF-IDF similarity to the chosen movie, collaborative = item-item similarity,
   quality = weighted rating. *User mode*: content = similarity to the user's liked-movie profile,
   collaborative = SVD predicted rating, quality = weighted rating. Movies with fewer than 5 training ratings are
   not recommended by the hybrid.

**Predicted vs actual rating:** the *actual* rating is what a user really gave (hidden from the model in the test
split); the *predicted* rating is the model's estimate from training data. RMSE/MAE measure the average gap in stars.

## 5. Dataset

Use **MovieLens latest-small** (about 100k ratings, 600 users, 9k movies).

1. Download <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
   (or from <https://grouplens.org/datasets/movielens/latest/>).
2. Extract it into the project's `data/` folder so that `data/ml-latest-small/movies.csv` and
   `data/ml-latest-small/ratings.csv` exist (`links.csv` and `tags.csv` are optional but recommended).

See `data/README.md`. The loader checks the files and tells you exactly what is wrong if something is missing.

## 6. Project structure

```
movie-recommendation-system/
├── app.py                    # Home dashboard (entry point: streamlit run app.py)
├── train.py                  # Trains + evaluates all models, saves to models/
├── requirements.txt
├── README.md
├── .env.example              # copy to .env to enable TMDb posters
├── .gitignore
├── pytest.ini / conftest.py
├── .streamlit/config.toml    # dark theme, hides stack traces
├── data/README.md            # where to put MovieLens
├── models/README.md          # artifacts.joblib + metrics.json are written here
├── assets/placeholder.svg    # local poster placeholder
├── src/
│   ├── config.py             # paths, constants, logging
│   ├── exceptions.py         # friendly custom errors
│   ├── data_loader.py        # validation, cleaning, per-user train/test split
│   ├── preprocessing.py      # titles, years, genres, TF-IDF text, movie stats, search
│   ├── popularity.py         # weighted-rating recommender
│   ├── content_based.py      # TF-IDF + cosine similarity
│   ├── collaborative.py      # item-based CF (adjusted cosine, top-K neighbours)
│   ├── svd_model.py          # NumPy biased matrix factorisation (FunkSVD)
│   ├── hybrid.py             # configurable weighted hybrid + cold start
│   ├── evaluation.py         # RMSE, MAE, Precision/Recall/F1@K
│   ├── pipeline.py           # train/evaluate/save/load + serving API for the UI
│   ├── movie_metadata.py     # optional TMDb client (cache, timeouts, rate-limit handling)
│   ├── cards.py              # HTML/CSS for poster cards (pure Python, escaped)
│   └── ui.py                 # Streamlit glue (cached loading, grids)
├── pages/
│   ├── 1_Recommendations.py
│   ├── 2_Personalized.py
│   ├── 3_Movie_Explorer.py
│   ├── 4_Model_Performance.py
│   └── 5_About.py
└── tests/
    ├── helpers.py                      # tiny synthetic dataset (tests only)
    ├── test_preprocessing.py
    ├── test_recommendations.py
    ├── test_evaluation.py
    └── test_cards_and_metadata.py
```

ML logic lives in `src/` (no Streamlit imports except `ui.py`); the pages only call the serving API in `pipeline.py`.

## 7. Installation

Requires **Python 3.10+** (developed against 3.13).

**Windows (PowerShell)**

```powershell
cd movie-recommendation-system
python -m venv .venv
.venv\Scripts\Activate.ps1          # in cmd.exe use: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Linux / macOS**

```bash
cd movie-recommendation-system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Train the models

After placing the dataset (section 5):

```bash
python train.py
```

This loads and cleans the data, splits each user's ratings 80/20, trains every model on the training part, evaluates on
the test part, prints result tables and writes `models/artifacts.joblib` and `models/metrics.json`.
The SVD step is plain NumPy SGD, so expect a few minutes on latest-small. Useful options:

```bash
python train.py --epochs 30 --factors 64        # more SVD capacity
python train.py --max-eval-users 600 --k 10     # evaluate more users
python train.py --seed 7 --test-size 0.25
python train.py --skip-eval                     # faster; Performance page will have no results
```

(If you skip this step, the app offers a **Train models now** button that runs the same pipeline.)

## 9. Launch the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

## 10. Optional: TMDb posters

1. Create a free account at <https://www.themoviedb.org> and request an API key (Settings -> API).
2. Provide the key in **one** of these ways (never commit it):
   * `.env` file in the project root: copy `.env.example` to `.env` and set `TMDB_API_KEY=...`
   * environment variable - Windows: `$env:TMDB_API_KEY="..."`; Linux/macOS: `export TMDB_API_KEY=...`
   * `.streamlit/secrets.toml` containing `TMDB_API_KEY = "..."`
3. Restart the app. Posters are looked up through `links.csv` (`tmdbId`), cached, fetched with timeouts, and fall back
   to the placeholder on any error. Invalid keys and rate limits disable lookups temporarily instead of breaking the
   app. The key is never printed or logged.

## 11. Running the tests

```bash
python -m unittest discover -s tests -t .      # no extra packages needed
pytest                                          # if pytest is installed
```

The tests use a tiny synthetic dataset generated in `tests/helpers.py` (so they run in seconds). It is used **only**
by the tests, never for training the app's models or for reported metrics. Tests check, among other things, that
recommendations contain valid movie IDs, no duplicates, never include already-rated movies, that unknown movies/users
raise clear errors, that cold-start works, and that test ratings never appear in the training split.

## 12. Model evaluation methodology

* **Split** - per-user random hold-out: for each user with at least 5 ratings, 20% of their ratings (at least one)
  go to the test set; everyone keeps training history. Fixed seed (42) for reproducibility.
* **Rating prediction** - RMSE and MAE over all test ratings for Item-based CF and SVD, next to two baselines
  (global mean, item mean). Content-based and Hybrid output relevance scores, not star ratings, so they have no
  RMSE/MAE.
* **Ranking** - a test movie is **relevant** if the user rated it **>= 4.0**. For each evaluated user, each algorithm
  ranks all movies the user did *not* rate in the training split and the top K (default 10) are compared with the
  relevant set: **Precision@K** = hits/K, **Recall@K** = hits/#relevant, **F1@K** = their harmonic mean. A random
  baseline is included. Only users with at least one relevant test movie are evaluated (a reproducible random sample
  of up to 300).
* **Caveats** - an unrated movie is not necessarily disliked, so absolute precision/recall are low by design; compare
  algorithms with each other and with the random baseline. One split and a user sample means small differences are not
  conclusive. The app reports which algorithm scored best on each metric *in the run you executed* and does not claim
  a universal winner.

## 13. Troubleshooting

| Problem | Solution |
|---|---|
| "Could not find movies.csv and ratings.csv" | Extract MovieLens into `data/ml-latest-small/` (section 5). |
| "Trained models were not found" | Run `python train.py` or click **Train models now** in the app. |
| "saved model file could not be loaded" after upgrading libraries | Re-run `python train.py` to regenerate the models. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root (`streamlit run app.py`, `python train.py`). |
| `ModuleNotFoundError: streamlit/plotly/...` | Activate the virtual environment and `pip install -r requirements.txt`. |
| PowerShell cannot run `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `activate.bat` in cmd. |
| Posters never appear | Check that `TMDB_API_KEY` is set and `links.csv` exists; see the terminal log (the key itself is never logged). |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Training feels slow | SVD is pure NumPy; lower `--epochs` (e.g. 10) or `--factors`. |
| Dataset error mentions a column | Make sure you downloaded the *CSV* version (`ml-latest-small`), not the `ml-100k` `u.data` format. |

## 14. Future enhancements

Implicit-feedback models (ALS, BPR) and neural collaborative filtering; text embeddings of plot summaries; k-fold
cross-validation and significance tests; diversity/novelty metrics; time-aware models; training the final serving
model on all ratings after validation; user accounts with feedback buttons and A/B testing.

## 15. Tips for presenting this as a college project

1. Start on **Home**: dataset size, genre distribution, popular vs top-rated (explain weighted rating).
2. **Recommendations**: pick one movie and compare Content-Based vs Collaborative vs Hybrid; change the hybrid weights
   live and show the "why" captions.
3. **Personalized**: show a user's history, then unseen recommendations with predicted ratings; demonstrate the
   cold-start flow.
4. **Model Performance**: explain the split, RMSE/MAE and Precision@K, point out the baselines and the limitations.
5. Be ready to discuss: cold start, popularity bias, sparsity, why unrated is not disliked, why the SVD is trained
   with SGD.

## License and data attribution

Code: for educational use. Dataset: MovieLens by GroupLens Research (see the license in the dataset zip). Posters and
metadata from TMDb are subject to TMDb's terms; this product uses the TMDb API but is not endorsed or certified by TMDb.
