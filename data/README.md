# Data folder

Place the **MovieLens latest-small** dataset here.

1. Download `ml-latest-small.zip` from <https://grouplens.org/datasets/movielens/latest/>
   (direct link: <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>, about 1 MB).
2. Extract it so the layout is:

```
data/
└── ml-latest-small/
    ├── movies.csv     required  (movieId, title, genres)
    ├── ratings.csv    required  (userId, movieId, rating, timestamp)
    ├── links.csv      optional  (movieId, imdbId, tmdbId)   -> used for TMDb posters
    └── tags.csv       optional  (userId, movieId, tag, timestamp) -> extra content features
```

Files placed directly in `data/` (without the `ml-latest-small` sub-folder) also work.

The loader validates the files, drops duplicate and invalid rows, and prints a clear error if something is missing.
Dataset files are git-ignored. Dataset license: see `README.txt` inside the zip (MovieLens, GroupLens Research; for research/education use).
