"""Smoke test for the natural-gas-price bot.

Verifies imports + config loading + basic feature pipeline don't
crash. Exit 0 = good.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from src import config, data_loaders, features, kalshi, model
    from src import signals, simulator

    cfg = config.load_config()
    assert cfg.kalshi_series_prefix == "KXNATGASD"
    assert cfg.threshold_grid_usd, "threshold grid empty"
    print(f"[ok] config — series={cfg.kalshi_series_prefix} "
          f"thresholds={len(cfg.threshold_grid_usd)}")

    panel = data_loaders.build_panel(cfg, days=180)
    assert not panel.empty, "panel is empty"
    df, fcols = features.build_features(panel, target=cfg.target_column)
    assert len(fcols) >= 10, f"too few feature cols: {len(fcols)}"
    assert cfg.target_column not in fcols, "target leakage in features"
    print(f"[ok] feature pipeline — panel={len(panel)} rows, "
          f"{len(fcols)} feature cols")

    # Use a real temp file (not :memory:) — sqlite3.connect(":memory:")
    # opens a new empty DB each call, so multi-connection patterns
    # can't see the schema across calls.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "smoke.db"
        sim = simulator.NatGasSimulator(db, cfg)
        assert sim.open_positions() == [], "fresh sim should start empty"
        # Open + close one position end-to-end to exercise the trade path.
        pid = sim.open_position(
            ticker="KXNATGASD-26MAY0117-T3.000", side="YES", ask_cents=42,
            threshold_value=3.000, forecast_value=3.10, signal_edge=0.12)
        assert pid is not None
        assert len(sim.open_positions()) == 1
        sim.close_position(pid, exit_price_cents=100)
        assert sim.open_positions() == []
    print("[ok] simulator (open/close lifecycle)")

    print("[ok] all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
