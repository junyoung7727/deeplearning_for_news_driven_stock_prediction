"""
Stage 16 - divergent technique sweep under honest walk-forward OOS.
Answers "try many financial-ML methods without overfitting": compares model
families, an ensemble, |return|-weighted training, a momentum rule, price-only vs
+news, all evaluated with the SAME leak-free expanding-window walk-forward.
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
import numpy as np, os, json, pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef
import config as C
from s15_walkforward import walk_forward, hgb, logit, selective_wf, OOS_START, STEP, EMBARGO

def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=30,
                                  random_state=C.SEED, n_jobs=1)
def et():
    return ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=25,
                                random_state=C.SEED, n_jobs=1)
def logit_l2():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.2, max_iter=4000))

def wf_weighted(X, y, w, dates):
    n = len(X); oos = np.full(n, np.nan)
    start = int(np.searchsorted(dates, np.datetime64(OOS_START))); i = start
    while i < n:
        j = min(i + STEP, n); tr = i - EMBARGO
        if tr >= 200:
            m = hgb(); m.fit(X[:tr], y[:tr], sample_weight=w[:tr])
            oos[i:j] = m.predict_proba(X[i:j])[:, 1]
        i = j
    return oos

def acc_mcc(y, oos):
    m = ~np.isnan(oos)
    a = accuracy_score(y[m], (oos[m] > 0.5).astype(int))
    c = matthews_corrcoef(np.where(y[m] == 1, 1, -1), np.where(oos[m] > 0.5, 1, -1))
    return a, c, int(m.sum())

def main():
    df = pd.read_parquet(os.path.join(C.ART, "features_ff.parquet")).sort_values("date").reset_index(drop=True)
    allcols = [c for c in df.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
              "dist", "logvol", "streak", "peer_", "mkt_", "cnt_", "abn_", "sent_", "novelty", "npc"))]
    price_cols = [c for c in allcols if not c.startswith(("cnt_", "abn_", "sent_", "novelty", "npc"))]
    X = df[allcols].values.astype(np.float32); Xp = df[price_cols].values.astype(np.float32)
    y = ((df.label.values + 1) // 2).astype(int)
    dates = df.date.values.astype("datetime64[ns]")
    ret = np.abs(df.ret.values.astype(float))
    oos_mask = dates >= np.datetime64(OOS_START)
    base = max(y[oos_mask].mean(), 1 - y[oos_mask].mean())
    print(f"OOS days={oos_mask.sum()} majority baseline={base:.4f} feats(all)={len(allcols)} price={len(price_cols)}\n")

    rows = []
    # rule baselines
    lag1 = df["lagret_1"].values
    mom_pred = (lag1 > 0).astype(float)
    a = accuracy_score(y[oos_mask], mom_pred[oos_mask].astype(int)); rows.append(("momentum rule", a, 0.0))
    rows.append(("always-up (majority)", y[oos_mask].mean(), 0.0))

    oos_store = {}
    for name, mk in [("Logit-L2", logit_l2), ("Logit-EN", logit), ("HGB", hgb),
                     ("RandomForest", rf), ("ExtraTrees", et)]:
        oos, _ = walk_forward(mk, X, y, dates); oos_store[name] = oos
        a, c, n = acc_mcc(y, oos); rows.append((name + " (all feats)", a, c))
    # price-only HGB (does news help OOS?)
    oosp, _ = walk_forward(hgb, Xp, y, dates); a, c, _ = acc_mcc(y, oosp); rows.append(("HGB price-only", a, c))
    # |return|-weighted HGB (focus informative moves)
    w = 1.0 + 5.0 * ret; ow = wf_weighted(X, y, w, dates); a, c, _ = acc_mcc(y, ow); rows.append(("HGB |ret|-weighted", a, c))
    # ensemble average of Logit-EN + HGB + RF
    ens = np.nanmean(np.vstack([oos_store["Logit-EN"], oos_store["HGB"], oos_store["RandomForest"]]), 0)
    a, c, _ = acc_mcc(y, ens); rows.append(("ENSEMBLE(LE+HGB+RF)", a, c))

    print(f"{'method':28s} {'OOS_acc':>8s} {'OOS_mcc':>8s}  vs base {base:.3f}")
    for n, a, c in rows:
        flag = "  <-- > base" if a > base + 0.01 else ""
        print(f"{n:28s} {a:8.4f} {c:8.4f}{flag}")

    best = max(rows, key=lambda r: r[1])
    print(f"\nbest OOS accuracy: {best[0]} = {best[1]:.4f} (baseline {base:.4f})")
    # selective for ensemble
    sel = selective_wf(ens, y)
    print("ENSEMBLE selective (WF thr): " + "  ".join(f"{int(k*100)}%:{v[0]:.3f}@{v[1]*100:.0f}%" for k, v in sel.items()))
    json.dump({"baseline": float(base), "results": [{"method": n, "oos_acc": float(a), "oos_mcc": float(c)} for n, a, c in rows]},
              open(os.path.join(C.ART, "results_divergent.json"), "w"), indent=2)
    print("saved results_divergent.json")

if __name__ == "__main__":
    main()
