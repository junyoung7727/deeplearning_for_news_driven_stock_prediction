"""
Stage 11 - Selective (high-confidence) operating mode.

The model ABSTAINS on low-confidence days and only predicts on the most-confident
ones.  To stay honest / out-of-sample the confidence threshold is chosen on the
DEV set (never on test):

  rule: predict iff |p - 0.5| >= tau,  where tau = the (1 - kappa) quantile of
        DEV confidences  ->  targets ~kappa coverage, threshold fixed before test.

We report the resulting TEST accuracy and TEST coverage at that pre-committed
threshold, plus the full descriptive accuracy-coverage curve for transparency.

Base model: bagged HistGradientBoosting (5 bootstraps) on leak-free price/tech
(+news) features; operating feature-set chosen by DEV MCC.
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
from sklearn.metrics import accuracy_score, matthews_corrcoef
import config as C
from s10_boost import price_features

N_BAG = 5

def bagged_proba(Xtr, ytr, X, seeds=N_BAG):
    rng = np.random.default_rng(C.SEED)
    ps = []
    for s in range(seeds):
        idx = rng.integers(0, len(Xtr), len(Xtr))          # bootstrap
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=400, l2_regularization=1.0,
            min_samples_leaf=30, random_state=C.SEED + s).fit(Xtr[idx], ytr[idx])
        ps.append(m.predict_proba(X)[:, 1])
    return np.mean(ps, 0)

def main():
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    daily = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close", "volume"])
    daily = daily[daily.ticker == C.TARGET].copy()
    daily["date"] = pd.to_datetime(daily.trade_date)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["close"] = daily.close.astype(float); daily["volume"] = daily.volume.astype(float)
    daily["ret"] = daily.close / daily.close.shift(1) - 1
    samp = samp.merge(price_features(daily), on="date", how="left")
    price_cols = [c for c in samp.columns if c.startswith(("lagret", "mom", "vol", "maratio",
                  "rsi", "dist", "logvol", "streak"))]
    TEB = np.load(os.path.join(C.ART, "feats_TEB.npz"))
    news = np.concatenate([TEB["short"], TEB["mid"].mean(1), TEB["long"].mean(1)], axis=1)

    y = ((samp.label.values + 1) // 2).astype(int)
    ypm = samp.label.values
    sp = samp.split.values
    tr, dv, te = sp == "train", sp == "dev", sp == "test"
    Xp = samp[price_cols].values.astype(np.float32)
    feats = {"price": Xp, "price+news": np.concatenate([Xp, news], 1)}

    # pick operating feature-set by DEV MCC (honest, no test peeking)
    probs = {}
    for name, X in feats.items():
        p_dv = bagged_proba(X[tr], y[tr], X[dv])
        p_te = bagged_proba(X[tr], y[tr], X[te])
        probs[name] = (p_dv, p_te)
    def dev_mcc(name):
        p = probs[name][0]; pr = (p > 0.5).astype(int)
        return matthews_corrcoef(y[dv], pr) if len(np.unique(pr)) > 1 else 0.0
    op = max(feats, key=dev_mcc)
    p_dv, p_te = probs[op]
    print(f"operating model (by dev MCC): {op}  "
          f"| full-cov test_acc={accuracy_score(y[te],(p_te>0.5).astype(int)):.4f}")

    conf_dv, conf_te = np.abs(p_dv - 0.5), np.abs(p_te - 0.5)

    # pre-committed dev-quantile thresholds targeting kappa coverage, BOTH models
    print("\n== honest selective (threshold = dev (1-kappa) quantile, fixed pre-test) ==")
    rows = []
    for name in feats:
        pdv, pte = probs[name]
        cdv, cte = np.abs(pdv - 0.5), np.abs(pte - 0.5)
        print(f"  [{name}{' (dev-selected)' if name == op else ''}]")
        for kappa in [1.0, 0.5, 0.3, 0.2, 0.1]:
            tau = 0.0 if kappa >= 1.0 else np.quantile(cdv, 1 - kappa)
            mask = cte >= tau
            if mask.sum() == 0:
                continue
            acc = accuracy_score(y[te][mask], (pte[mask] > 0.5).astype(int))
            rows.append({"model": name, "target_cov": kappa, "test_cov": float(mask.mean()),
                         "test_acc": float(acc), "n": int(mask.sum())})
            print(f"     target {kappa*100:4.0f}%  ->  test cov {mask.mean()*100:5.1f}%  "
                  f"acc {acc:.4f}  (n={mask.sum()})")

    # descriptive test curve (ranked by test confidence) - transparency only
    print("\n== descriptive test accuracy-coverage curve (test-ranked) ==")
    order = np.argsort(-conf_te)
    for kappa in [0.5, 0.3, 0.2, 0.1]:
        n = max(1, int(len(p_te) * kappa)); idx = order[:n]
        print(f"  top {kappa*100:4.0f}%  acc {accuracy_score(y[te][idx],(p_te[idx]>0.5).astype(int)):.4f}")

    # save the operating predictions on test (date, prob, decision, actual)
    te_dates = samp.date.values[te]
    dec = np.where(conf_te >= (np.quantile(conf_dv, 0.8)), np.where(p_te > 0.5, 1, -1), 0)
    pred_df = pd.DataFrame({"date": te_dates, "prob_up": p_te,
                            "decision": dec, "actual": ypm[te]})
    pred_df.to_parquet(os.path.join(C.ART, "selective_test_predictions.parquet"))
    json.dump({"operating_model": op, "dev_quantile_rule": rows},
              open(os.path.join(C.ART, "results_selective.json"), "w"), indent=2)
    print("\nsaved results_selective.json + selective_test_predictions.parquet")

if __name__ == "__main__":
    main()
