"""
Stage 18 - honest high-confidence (selective) accuracy curve.

Answers: how high does accuracy go on the model's most-confident days, using an
HONEST pre-committed threshold (walk-forward: tau from PRIOR OOS confidences only,
never test-peeking)?  Reported for full OOS (2021-2026) and recent (2024-2026),
for price-direction and (balanced) volatility-direction, with a calibrated
ensemble (Logit-EN + HGB + ExtraTrees).
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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import config as C
from s15_walkforward import walk_forward, hgb, logit, OOS_START

def et():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=25,
                                random_state=C.SEED, n_jobs=1)

def ensemble_oos(X, y, dates):
    ps = [walk_forward(mk, X, y, dates)[0] for mk in (logit, hgb, et)]
    return np.nanmean(np.vstack(ps), 0)

def wf_curve(prob, y, dates, recent_start, kappas=(0.4, 0.3, 0.2, 0.1, 0.05), warmup=150):
    idx = np.where(~np.isnan(prob))[0]
    conf = np.abs(prob - 0.5)
    rec = dates >= np.datetime64(recent_start)
    out = {}
    for kappa in kappas:
        prior = []
        cf = ct = cr = rt = 0
        for k in idx:
            if len(prior) >= warmup:
                tau = np.quantile(conf[prior], 1 - kappa)
                if conf[k] >= tau:
                    ok = int((prob[k] > 0.5) == (y[k] == 1))
                    ct += 1; cf += ok
                    if rec[k]:
                        rt += 1; cr += ok
            prior.append(k)
        out[kappa] = {"full_acc": cf / ct if ct else float("nan"), "full_n": ct,
                      "recent_acc": cr / rt if rt else float("nan"), "recent_n": rt}
    return out

def feats(df):
    cols = [c for c in df.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
            "dist", "logvol", "streak", "peer_", "mkt_", "cnt_", "abn_", "sent_", "novelty", "npc"))]
    return df[cols].values.astype(np.float32)

def main():
    df = pd.read_parquet(os.path.join(C.ART, "features_ff.parquet")).sort_values("date").reset_index(drop=True)
    dates = df.date.values.astype("datetime64[ns]")
    X = feats(df)
    ydir = ((df.label.values + 1) // 2).astype(int)

    # volatility target (balanced, leak-free) + |lag ret| features
    absr = np.abs(df.ret.values.astype(float))
    med = pd.Series(absr).rolling(20).median().shift(1).values
    yvol = (absr > med).astype(int)
    Xv = np.column_stack([X] + [np.abs(pd.Series(df.ret.values).shift(k).values) for k in range(1, 6)]).astype(np.float32)

    for tag, (Xt, yt) in {"PRICE-direction": (X, ydir), "VOLATILITY-direction": (Xv, yvol)}.items():
        m = ~np.isnan(yt.astype(float))
        oos = ensemble_oos(Xt[m], yt[m], dates[m])
        yy = yt[m]; dd = dates[m]
        full = dd >= np.datetime64(OOS_START)
        base_full = max(yy[full].mean(), 1 - yy[full].mean())
        acc_full = accuracy_score(yy[~np.isnan(oos)], (oos[~np.isnan(oos)] > 0.5).astype(int))
        print(f"\n===== {tag} (ensemble, walk-forward) =====")
        print(f"full-coverage OOS acc={acc_full:.4f}  baseline={base_full:.4f}")
        cur = wf_curve(oos, yy, dd, "2024-01-01")
        print(f"{'coverage':>9s} {'full OOS acc':>13s} (n) {'recent 24-26 acc':>18s} (n)")
        for k, v in cur.items():
            print(f"{int(k*100):8d}% {v['full_acc']:13.4f} ({v['full_n']:4d}) "
                  f"{v['recent_acc']:18.4f} ({v['recent_n']:4d})")
        if tag.startswith("PRICE"):
            json.dump({k: v for k, v in cur.items()},
                      open(os.path.join(C.ART, "results_confidence.json"), "w"),
                      indent=2, default=float)
    print("\nsaved results_confidence.json")

if __name__ == "__main__":
    main()
