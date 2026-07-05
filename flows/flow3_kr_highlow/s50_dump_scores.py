"""Stage 50-dump (remote): reuse s49.build_scored to emit (ticker,date,score)
for the 5-min-covered tickers only, so the intraday backtest can run locally."""
from __future__ import annotations
# --- flow bootstrap: root config + sibling flow scripts importable ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _p in (_ROOT, *sorted((_ROOT / "flows").glob("flow*"))):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------
import os, sys, json, pandas as pd
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s49_portfolio_backtest import build_scored, ART

TICK = json.loads(open(os.path.join(ART, "min5_tickers.json")).read())
scored, split = build_scored()
scored = scored[scored.ticker.isin(set(TICK))][["ticker", "date", "o", "gap", "score"]].copy()
out = os.path.join(ART, "kr50_scores_min5univ.parquet")
scored.to_parquet(out)
print("saved", out, "rows", len(scored), "tickers", scored.ticker.nunique(),
      "split", str(split)[:10], flush=True)
