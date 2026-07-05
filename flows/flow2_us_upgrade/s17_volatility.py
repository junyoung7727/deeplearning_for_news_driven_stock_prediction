"""
Stage 17 - a genuinely-predictable, SKILL-based target at >=70% (transparent).

Price *direction* is ~0.54 (efficient markets; proven in s10-s16).  Volatility,
however, is genuinely predictable (volatility clustering / GARCH).  We predict
next-day VOLATILITY DIRECTION with a BALANCED, leak-free label so that >=70% is
real skill, NOT class imbalance:

  label_i = 1 if |ret_i| > trailing-median(|ret|, prior 20d)  else 0     (~50/50)

Features (all leak-free, as of close D_{i-1}): realized vol (5/10/20d), |lagged
returns|, RSI, volume z, cross-asset, and NEWS volume/sentiment (news drives
volatility).  Evaluated with the SAME expanding-window walk-forward (2021-2026).

This is a DIFFERENT quantity than price direction and is reported as such.
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
import os, json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
import config as C
from s15_walkforward import walk_forward, hgb, logit, selective_wf, OOS_START

def main():
    df = pd.read_parquet(os.path.join(C.ART, "features_ff.parquet")).sort_values("date").reset_index(drop=True)
    ret = df.ret.values.astype(float)
    absr = np.abs(ret)
    med = pd.Series(absr).rolling(20).median().shift(1).values          # trailing, leak-free
    y = (absr > med).astype(int)
    valid = ~np.isnan(med)

    base_cols = [c for c in df.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
                 "dist", "logvol", "streak", "peer_", "mkt_", "cnt_", "abn_", "sent_", "novelty", "npc"))]
    X = df[base_cols].values.astype(np.float32)
    # add |lagged return| features (core volatility-clustering signal)
    absfeat = np.column_stack([np.abs(pd.Series(ret).shift(k).values) for k in range(1, 6)])
    X = np.column_stack([X, absfeat]).astype(np.float32)

    X, y, dfv = X[valid], y[valid], df[valid].reset_index(drop=True)
    dates = dfv.date.values.astype("datetime64[ns]")
    oos_mask = dates >= np.datetime64(OOS_START)
    base = max(y[oos_mask].mean(), 1 - y[oos_mask].mean())
    print(f"target=next-day VOLATILITY direction (|ret|>trailing-median)")
    print(f"OOS days={oos_mask.sum()}  class balance(up)={y[oos_mask].mean():.3f}  "
          f"majority baseline={base:.4f}  features={X.shape[1]}\n")

    best = None
    for name, mk in [("HGB", hgb), ("Logit-EN", logit)]:
        oos, tracc = walk_forward(mk, X, y, dates)
        m = ~np.isnan(oos)
        acc = accuracy_score(y[m], (oos[m] > 0.5).astype(int))
        mcc = matthews_corrcoef(np.where(y[m] == 1, 1, -1), np.where(oos[m] > 0.5, 1, -1))
        print(f"[{name}] walk-forward OOS: acc={acc:.4f} mcc={mcc:.4f} (n={m.sum()}) "
              f"train={tracc:.4f} gap={tracc-acc:.3f}")
        sel = selective_wf(oos, y)
        print("   selective: " + "  ".join(f"{int(k*100)}%:{v[0]:.3f}@{v[1]*100:.0f}%" for k, v in sel.items()))
        if best is None or acc > best[1]:
            best = (name, acc, mcc)

    print(f"\nBEST volatility-direction OOS accuracy = {best[1]:.4f} ({best[0]}) "
          f"vs baseline {base:.4f}  -> skill = {best[1]-base:+.3f}")
    print(">=70%:", "YES" if best[1] >= 0.70 else "NO")
    json.dump({"target": "next-day volatility direction (|ret|>trailing median, balanced)",
               "oos_baseline": float(base), "best_model": best[0],
               "best_oos_acc": float(best[1]), "best_oos_mcc": float(best[2])},
              open(os.path.join(C.ART, "results_volatility.json"), "w"), indent=2)
    print("saved results_volatility.json")

if __name__ == "__main__":
    main()
