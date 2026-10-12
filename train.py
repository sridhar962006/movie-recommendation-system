"""Train all models, evaluate them on the held-out split and save them to models/.

Usage:
    python train.py
    python train.py --epochs 30 --factors 64 --max-eval-users 500
    python train.py --skip-eval          # faster, but the Performance page will be empty
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import DATA_DIR, DEFAULT_K, MAX_EVAL_USERS, MODELS_DIR, RANDOM_SEED, TEST_SIZE, configure_logging
from src.exceptions import DatasetError
from src.pipeline import build_artifacts, save_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the movie recommendation models.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--factors", type=int, default=50, help="latent factors for SVD")
    parser.add_argument("--epochs", type=int, default=20, help="SGD epochs for SVD")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="K for Precision/Recall/F1@K")
    parser.add_argument("--max-eval-users", type=int, default=MAX_EVAL_USERS)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    configure_logging()
    try:
        art = build_artifacts(args.data_dir, args.seed, args.test_size, args.factors, args.epochs,
                              not args.skip_eval, args.k, args.max_eval_users)
    except DatasetError as exc:
        print(f"\nDataset problem: {exc}", file=sys.stderr)
        return 1
    path = save_artifacts(art, args.models_dir)
    print(f"\nSaved trained models to {path}")

    if art.metrics:
        pd.set_option("display.width", 140)
        print("\nRating prediction on the held-out test split:")
        print(pd.DataFrame(art.metrics["rating_prediction"]).round(4).to_string(index=False))
        print(f"\nTop-{args.k} ranking quality ({art.metrics['ranking_setup']['users_evaluated']} users):")
        print(pd.DataFrame(art.metrics["ranking"]).drop(columns=["k"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
