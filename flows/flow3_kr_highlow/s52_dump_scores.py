"""Stage 52 dump (remote) - FULL-universe scored table (all 626 tickers) for the
exact no-stop daily backtest. Reuses s49.build_scored (GBM UP5 prob; GBM fitted
strictly on rows < 2021-11-29, scores everything)."""
from __future__ import annotations
# --- flow bootstrap: root config + sibling flow scripts importable ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _p in (_ROOT, *sorted((_ROOT / "flows").glob("flow*"))):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s49_portfolio_backtest import build_scored

def main():
    scored, split = build_scored()
    sub = scored[scored.date >= pd.Timestamp("2021-06-01")].copy()
    for c in ("o", "h", "l", "c", "gap", "vol", "score"):
        if c in sub.columns:
            sub[c] = sub[c].astype("float32")
    out = os.path.join(os.path.expanduser("~"), "dlfe", "artifacts",
                       "kr52_scores_full.parquet")
    sub.to_parquet(out)
    print("saved", out, "rows", len(sub), "tickers", sub.ticker.nunique(),
          "split", str(split)[:10], "cols", sub.columns.tolist(), flush=True)

if __name__ == "__main__":
    main()
