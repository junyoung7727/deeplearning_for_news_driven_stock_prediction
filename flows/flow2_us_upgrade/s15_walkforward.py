"""
Stage 15 - Walk-forward OOS evaluation + meta-labeling/confidence (anti-overfit).

- Expanding-window walk-forward: retrain every `STEP` trading days on ALL prior
  data (embargo 1 day), predict the next block. Every prediction uses only past
  data => leak-free, regime-adaptive, and gives a long OOS span (2021-2026).
- Confidence-selective: threshold set from PRIOR OOS confidences only
  (pre-committable, no test-peeking).
- Meta-labeling: a secondary model trained (walk-forward) to predict whether the
  primary call is correct; act only when P(correct) is high.

Reports full-coverage OOS accuracy and selective accuracy vs coverage, plus the
train-vs-OOS gap (overfit check).
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef
import config as C

OOS_START = "2021-01-01"
STEP = 63          # retrain quarterly
EMBARGO = 1

def hgb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=C.SEED)
def logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.3,
                                            solver="saga", max_iter=4000))

def walk_forward(make_model, X, y, dates):
    n = len(X); oos = np.full(n, np.nan); tr_accs = []
    start = int(np.searchsorted(dates, np.datetime64(OOS_START)))
    i = start
    while i < n:
        j = min(i + STEP, n)
        tr_end = i - EMBARGO
        if tr_end >= 200:
            m = make_model(); m.fit(X[:tr_end], y[:tr_end])
            oos[i:j] = m.predict_proba(X[i:j])[:, 1]
            tr_accs.append(accuracy_score(y[:tr_end], (m.predict_proba(X[:tr_end])[:, 1] > 0.5)))
        i = j
    return oos, float(np.mean(tr_accs))

def selective_wf(prob, y, warmup=120):
    """threshold from PRIOR OOS confidences only; report acc/coverage per kappa."""
    idx = np.where(~np.isnan(prob))[0]
    conf = np.abs(prob - 0.5)
    res = {}
    for kappa in [0.5, 0.3, 0.2, 0.1]:
        correct = total = 0
        prior = []
        for k in idx:
            if len(prior) >= warmup:
                tau = np.quantile(conf[prior], 1 - kappa)
                if conf[k] >= tau:
                    total += 1
                    correct += int((prob[k] > 0.5) == (y[k] == 1))
            prior.append(k)
        res[kappa] = (correct / total if total else float("nan"), total / len(idx))
    return res

def meta_label(prob_primary, X, y, dates):
    """Walk-forward secondary model P(primary correct); act when high."""
    n = len(X); metap = np.full(n, np.nan)
    idx = np.where(~np.isnan(prob_primary))[0]
    correct = ((prob_primary > 0.5).astype(int) == y).astype(int)
    conf = np.abs(prob_primary - 0.5)
    Xm = np.column_stack([X, prob_primary, conf])
    start = idx[0]
    i = start
    while i < n:
        j = min(i + STEP, n)
        prior = idx[(idx < i - EMBARGO)]
        if len(prior) >= 150:
            m = hgb(); m.fit(Xm[prior], correct[prior])
            blk = np.arange(i, j)
            blk = blk[~np.isnan(prob_primary[blk])]
            if len(blk):
                metap[blk] = m.predict_proba(Xm[blk])[:, 1]
        i = j
    return metap

def main():
    df = pd.read_parquet(os.path.join(C.ART, "features_ff.parquet")).sort_values("date").reset_index(drop=True)
    fcols = [c for c in df.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
             "dist", "logvol", "streak", "peer_", "mkt_", "cnt_", "abn_", "sent_", "novelty", "npc"))]
    X = df[fcols].values.astype(np.float32)
    y = ((df.label.values + 1) // 2).astype(int)
    dates = df.date.values.astype("datetime64[ns]")
    oos_mask0 = dates >= np.datetime64(OOS_START)
    base = max(y[oos_mask0].mean(), 1 - y[oos_mask0].mean())
    print(f"features={len(fcols)}  OOS days={oos_mask0.sum()}  OOS majority baseline={base:.4f}\n")

    for name, mk in [("HGB", hgb), ("Logit-EN", logit)]:
        oos, tracc = walk_forward(mk, X, y, dates)
        m = ~np.isnan(oos)
        acc = accuracy_score(y[m], (oos[m] > 0.5).astype(int))
        mcc = matthews_corrcoef(np.where(y[m] == 1, 1, -1), np.where(oos[m] > 0.5, 1, -1))
        print(f"[{name}] walk-forward OOS: full-cov acc={acc:.4f} mcc={mcc:.4f} "
              f"(n={m.sum()}) | train_acc={tracc:.4f} gap={tracc-acc:.3f}")
        sel = selective_wf(oos, y)
        print("   selective (WF threshold): " +
              "  ".join(f"{int(k*100)}%:{v[0]:.3f}@{v[1]*100:.0f}%" for k, v in sel.items()))
        if name == "HGB":
            metap = meta_label(oos, X, y, dates)
            mm = ~np.isnan(metap)
            # act on top meta-confidence via WF threshold
            for kappa in [0.3, 0.2, 0.1]:
                idx = np.where(mm)[0]; prior = []; correct = total = 0
                for k in idx:
                    if len(prior) >= 120:
                        tau = np.quantile(metap[prior], 1 - kappa)
                        if metap[k] >= tau:
                            total += 1; correct += int((oos[k] > 0.5) == (y[k] == 1))
                    prior.append(k)
                print(f"   meta-label act top {int(kappa*100)}%: acc={correct/total if total else float('nan'):.3f} "
                      f"@{total/len(idx)*100:.0f}%")
            best = {"model": "HGB", "oos_full_acc": float(acc), "oos_full_mcc": float(mcc),
                    "selective": {f"{int(k*100)}pct": [v[0], v[1]] for k, v in sel.items()}}
            json.dump(best, open(os.path.join(C.ART, "results_walkforward.json"), "w"), indent=2)
    print("\nsaved results_walkforward.json")

if __name__ == "__main__":
    main()
