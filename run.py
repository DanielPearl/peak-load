#!/usr/bin/env python3
"""Standardized entry point — matches the run.py in every other bot.

Delegates to the existing one-shot daily script. Kept as a thin wrapper
so the systemd unit (which still points at scripts/run_daily.py) keeps
working untouched. New tooling and the runbook reference this top-level
``run.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.run_daily import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
