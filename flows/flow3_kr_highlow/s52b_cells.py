"""s52b - close the two reporting holes: (1) EV of the unfiltered top-1 config
(the one with hit .547), (2) does the gap-DOWN cell replicate across windows?"""
from __future__ import annotations
# --- flow bootstrap: root config + sibling flow scripts importable ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _p in (_ROOT, *sorted((_ROOT / "flows").glob("flow*"))):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# ---------------------------------------------------------------------
import numpy as np, pandas as pd

sc = pd.read_parquet(_ROOT / "artifacts" / "kr52_scores_full.parquet")
sc["date"] = pd.to_datetime(sc.date)
sc = sc[(sc.o > 0) & (sc.h > 0) & (sc.c > 0)].sort_values(["ticker", "date"]).reset_index(drop=True)
TP, COST = 0.05, 0.003
T0, T1 = pd.Timestamp("2021-11-29"), pd.Timestamp("2024-06-01")
V0, V1 = pd.Timestamp("2024-06-01"), pd.Timestamp("2026-06-19")

def ev(k, floor, glo, ghi, d0, d1):
    d = sc[(sc.date >= d0) & (sc.date < d1) & (sc.score >= floor) &
           (sc.gap > glo) & (sc.gap <= ghi)]
    d = d.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(k)
    if len(d) < 30: return None
    hit = (d.h >= d.o * (1 + TP)).values
    net = np.where(hit, TP, d.c / d.o - 1.0) - COST
    return dict(n=len(d), hit=float(hit.mean()), ev=float(net.mean()),
                evmiss=float(net[~hit].mean()), days=int(d.date.nunique()))

print("== (1) unfiltered / loose-gap top-1 EV ==", flush=True)
for tag, (d0, d1) in {"TUNE": (T0, T1), "VAL": (V0, V1)}.items():
    for gh in (9.9, 0.05, 0.02):
        r = ev(1, 0.0, -1.0, gh, d0, d1)
        print(f"{tag} k1 gap<={gh:<4} -> n={r['n']:4d} hit={r['hit']:.3f} "
              f"EV={r['ev']*100:+.2f}% E[miss]={r['evmiss']*100:+.2f}%", flush=True)

print("\n== (2) ex-ante gap-DOWN rule (top-score names opening down) ==", flush=True)
for tag, (d0, d1) in {"TUNE": (T0, T1), "VAL": (V0, V1)}.items():
    for g in (0.0, 0.02, 0.04):
        for k in (1, 2, 3):
            r = ev(k, 0.5, -1.0, -g, d0, d1)
            if r:
                print(f"{tag} k={k} gap<={-g:+.2f} -> n={r['n']:4d} hit={r['hit']:.3f} "
                      f"EV={r['ev']*100:+.2f}% E[miss]={r['evmiss']*100:+.2f}% days={r['days']}", flush=True)
    print(flush=True)
