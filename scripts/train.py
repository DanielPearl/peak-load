"""Train the peak-load model and save it.

Usage:
    python scripts/train.py [--region ercot|nyiso|pjm|caiso]
                            [--target daily_peak_load_mw|net_peak_load_mw]

Reads config from .env, pulls historical data (real APIs if keys are
set, synthetic otherwise), builds features, trains baseline + stronger
models, picks the better one, and saves it to models/peak_load.pkl.
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
    parser.add_argument("--region", default=None,
                        help="Override ENERGY_REGION env var")
    parser.add_argument("--target", default=None,
                        help="Override TARGET_COLUMN env var")
    args = parser.parse_args()
    if args.region:
        import os
        os.environ["ENERGY_REGION"] = args.region.lower()
    if args.target:
        import os
        os.environ["TARGET_COLUMN"] = args.target

    cfg = load_config()
    log.info("region=%s target=%s history_days=%d test_size=%d",
             cfg.region, cfg.target_column,
             cfg.history_days_for_training, cfg.test_size_days)

    panel = build_panel(cfg)
    log.info("panel: %d rows %s..%s, columns=%s",
             len(panel), panel.index.min().date(), panel.index.max().date(),
             list(panel.columns))

    df, feature_cols = build_features(panel, target=cfg.target_column)
    log.info("features built: %d rows × %d feature cols", len(df), len(feature_cols))

    model = train_model(
        df, feature_cols,
        test_size_days=cfg.test_size_days,
        random_state=cfg.random_state,
    )

    save_model(model, cfg.model_path)
    log.info("metrics: %s", model.metrics)
    print(f"\nTrained {model.model_name}: "
          f"MAE={model.metrics['mae']:.0f} MW  "
          f"RMSE={model.metrics['rmse']:.0f} MW  "
          f"R²={model.metrics['r2']:.3f}  "
          f"residual_std={model.residual_std:.0f} MW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
