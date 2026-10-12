# Models folder

`python train.py` writes its output here:

* `artifacts.joblib` - trained models (popularity, TF-IDF content model, item-based CF, SVD, hybrid) plus the
  train/test split and movie tables used by the app.
* `metrics.json` - the real evaluation results shown on the **Model Performance** page.

Both files are generated, git-ignored and safe to delete (re-run `python train.py` to recreate them).
Only load `artifacts.joblib` files that you created yourself: joblib files are pickles.
