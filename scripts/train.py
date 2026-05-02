"""Train the natural-gas-price model and save it.

Usage:
    python scripts/train.py [--max-features 30]
                            [--threshold-step 0.10]
                            [--importance-csv path/to/audit.csv]

Reads config from .env, pulls historical data (real EIA + NOAA when
keys are set, synthetic otherwise), builds features, runs walk-forward
feature selection, trains the quantile-GBM ensemble + per-strike
classifier ensemble, and saves the artifact to models/natgas_price.pkl.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_loaders import build_panel
from src.features import build_features
from src.model import save_model, train_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("train")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None,
                        help="Override TARGET_COLUMN env var")
    parser.add_argument("--max-features", type=int, default=30,
                        help="Cap on selected features after walk-forward + "
                             "correlation pruning")
    parser.add_argument("--ensemble-seeds", type=int, default=5,
                        help="Number of seeded GBMs per quantile")
    parser.add_argument("--classifier-seeds", type=int, default=3,
                        help="Number of seeded classifiers per strike")
    parser.add_argument("--threshold-step", type=float, default=0.10,
                        help="$/MMBTU step for the per-strike training grid")
    parser.add_argument("--importance-csv", default=None,
                        help="Optional path to dump feature-importance audit CSV")
    args = parser.parse_args()
    if args.target:
        import os
        os.environ["TARGET_COLUMN"] = args.target

    cfg = load_config()
    log.info("target=%s history_days=%d test_size=%d",
             cfg.target_column,
             cfg.history_days_for_training, cfg.test_size_days)

    panel = build_panel(cfg)
    log.info("panel: %d rows %s..%s",
             len(panel), panel.index.min().date(), panel.index.max().date())

    df, feature_cols = build_features(panel, target=cfg.target_column)
    log.info("features built: %d rows × %d candidate feature cols",
             len(df), len(feature_cols))

    model = train_model(
        df, feature_cols,
        test_size_days=cfg.test_size_days,
        random_state=cfg.random_state,
        ensemble_seeds=args.ensemble_seeds,
        classifier_seeds=args.classifier_seeds,
        max_features=args.max_features,
        threshold_training_step_usd=args.threshold_step,
        importance_csv_path=args.importance_csv,
    )

    save_model(model, cfg.model_path)
    m = model.metrics
    print(f"\nTrained {model.model_name}")
    print(f"  Median forecast: MAE=${m['mae']:.3f}/MMBTU  "
          f"RMSE=${m['rmse']:.3f}/MMBTU  R2={m['r2']:.3f}")
    print(f"  Residual std: ${model.residual_std:.3f}/MMBTU")
    print(f"  Per-strike classifiers: {m['per_strike_count']} strikes  |  "
          f"avg acc={m['per_strike_avg_accuracy']:.3f}  "
          f"F1={m['per_strike_avg_f1']:.3f}  "
          f"AUC={m['per_strike_avg_roc_auc']:.3f}")
    print(f"  Features selected: {m['n_features_selected']}/"
          f"{len(feature_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
