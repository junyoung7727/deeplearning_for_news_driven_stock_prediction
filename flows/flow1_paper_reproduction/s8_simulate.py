"""
Stage 8 - Market simulation (paper Sec 4.3 "Market Simulation"; Lavrenko 2000).

Per predicted day, trade $10,000 of NVDA at the open:
  LONG  (predict up)   : sell at +2% if reachable intraday (high>=open*1.02),
                         else sell at the close.
  SHORT (predict down) : cover at -1% if reachable intraday (low<=open*0.99),
                         else cover at the close.
Profit is summed over the evaluated days.

Threshold strategy (paper Fig.5): trade only when confident -
  P(up) > beta -> long ; P(up) < 1-beta -> short ; otherwise no trade.

Randomization test (paper): 1000 trials of random long/short to assess
significance of the model's profit.
"""
from __future__ import annotations
# --- flow bootstrap: root config + sibling flow scripts importable ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _p in (_ROOT, *sorted((_ROOT / "flows").glob("flow*"))):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------
import numpy as np
import config as C


def _trade(direction, o, h, l, c):
    if direction == 1:                                   # long
        sell = o * (1 + C.SIM_TAKEPROFIT) if h >= o * (1 + C.SIM_TAKEPROFIT) else c
        return C.SIM_CAPITAL * (sell / o - 1.0)
    else:                                                # short
        cover = o * (1 - C.SIM_COVER) if l <= o * (1 - C.SIM_COVER) else c
        return C.SIM_CAPITAL * (1.0 - cover / o)


def simulate(prob, o, h, l, c, threshold=None):
    prof = np.zeros(len(prob))
    for i in range(len(prob)):
        if threshold is None:
            d = 1 if prob[i] > 0.5 else -1
        elif prob[i] > threshold:
            d = 1
        elif prob[i] < 1 - threshold:
            d = -1
        else:
            continue
        prof[i] = _trade(d, o[i], h[i], l[i], c[i])
    return prof


def randomization_dist(o, h, l, c, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    tot = np.empty(n)
    for t in range(n):
        dirs = rng.integers(0, 2, size=len(o)) * 2 - 1
        tot[t] = sum(_trade(int(dirs[i]), o[i], h[i], l[i], c[i]) for i in range(len(o)))
    return tot


def randomization_test(o, h, l, c, model_profit, n=1000, seed=C.SEED):
    tot = randomization_dist(o, h, l, c, n, seed)
    return float(tot.mean()), float((tot >= model_profit).mean())
