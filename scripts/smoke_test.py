"""Smoke test for the peak-load bot.

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
    assert cfg.region in config.REGION_PRESETS
    assert cfg.threshold_grid_mw, "threshold grid empty"
    print(f"[ok] config — region={cfg.region} thresholds={len(cfg.threshold_grid_mw)}")

    panel = data_loaders.build_panel(cfg, days=180)
    assert not panel.empty, "panel is empty"
    df, fcols = features.build_features(panel, target=cfg.target_column)
    assert len(fcols) >= 10, f"too few feature cols: {len(fcols)}"
    assert "daily_peak_load_mw" not in fcols, "load leakage in features"
    assert "net_peak_load_mw" not in fcols, "net-load leakage in features"
    print(f"[ok] feature pipeline — panel={len(panel)} rows, "
          f"{len(fcols)} feature cols")

    # Use a real temp file (not :memory:) — sqlite3.connect(":memory:")
    # opens a NEW empty DB each call, so a multi-connection pattern
    # like ours can't see the schema across calls.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "smoke.db"
        sim = simulator.PeakLoadSimulator(db, cfg)
        assert sim.open_positions() == [], "fresh sim should start empty"
        # Open + close one position end-to-end to exercise the trade path.
        pid = sim.open_position(
            ticker="KXTEST-DEMO-100", side="YES", ask_cents=42,
            threshold_mw=70000, forecast_mw=72000, signal_edge=0.12)
        assert pid is not None
        assert len(sim.open_positions()) == 1
        sim.close_position(pid, exit_price_cents=100)
        assert sim.open_positions() == []
    print("[ok] simulator (open/close lifecycle)")

    print("[ok] all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
