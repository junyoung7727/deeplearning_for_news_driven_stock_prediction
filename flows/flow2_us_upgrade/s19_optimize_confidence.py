"""
Stage 19 - optimize the honest high-confidence operating point.

Techniques (all leak-free, walk-forward, pre-committed thresholds):
  - recency-weighted training (weight recent days more -> regime match)
  - probability calibration (isotonic on prior OOS)
  - volatility filter: act on direction only when the day is predicted HIGH-vol
    (big, news-driven moves are more directionally predictable)
Reports robust (n>=40) high-confidence accuracy, full OOS and recent 2024-26.
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
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import config as C

OOS_START = "2021-01-01"; STEP = 42; EMBARGO = 1; HALFLIFE = 252.0

def _weights(dts_train, block_date):
    age = (block_date - dts_train).astype("timedelta64[D]").astype(float)
    return 0.5 ** (age / HALFLIFE)

def wf_weighted(X, y, dates, weighted=True):
    n = len(X); oos = np.full(n, np.nan)
    imp = SimpleImputer(strategy="median")
    start = int(np.searchsorted(dates, np.datetime64(OOS_START))); i = start
    while i < n:
        j = min(i + STEP, n); tr = i - EMBARGO
        if tr >= 250:
            w = _weights(dates[:tr], dates[i]) if weighted else None
            Xi = imp.fit_transform(X[:tr]); Xt = imp.transform(X[i:j])
            m1 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                    min_samples_leaf=40, l2_regularization=1.0, random_state=C.SEED)
            m1.fit(X[:tr], y[:tr], sample_weight=w)
            m2 = ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=25,
                    random_state=C.SEED, n_jobs=1)
            m2.fit(Xi, y[:tr], sample_weight=w)
            p = 0.5 * m1.predict_proba(X[i:j])[:, 1] + 0.5 * m2.predict_proba(Xt)[:, 1]
            oos[i:j] = p
        i = j
    return oos

def calibrate_wf(prob, y, warmup=200):
    """isotonic calibration fit on PRIOR OOS only."""
    idx = np.where(~np.isnan(prob))[0]; cal = prob.copy()
    for pos, k in enumerate(idx):
        if pos >= warmup:
            pri = idx[:pos]
            try:
                ir = IsotonicRegression(out_of_bounds="clip").fit(prob[pri], y[pri])
                cal[k] = ir.predict([prob[k]])[0]
            except Exception:
                pass
    return cal

def curve(prob, y, dates, recent="2024-01-01", kappas=(0.4, 0.3, 0.2, 0.15, 0.1), warmup=150, vol=None):
    idx = np.where(~np.isnan(prob))[0]; conf = np.abs(prob - 0.5)
    rec = dates >= np.datetime64(recent); out = {}
    for kappa in kappas:
        prior = []; cf = ct = cr = rt = 0
        for k in idx:
            if len(prior) >= warmup:
                tau = np.quantile(conf[prior], 1 - kappa)
                act = conf[k] >= tau
                if vol is not None:
                    vth = np.quantile(vol[prior][~np.isnan(vol[prior])], 0.5) if np.any(~np.isnan(vol[prior])) else 0.5
                    act = act and (vol[k] >= vth)
                if act:
                    ok = int((prob[k] > 0.5) == (y[k] == 1)); ct += 1; cf += ok
                    if rec[k]:
                        rt += 1; cr += ok
            prior.append(k)
        out[kappa] = {"full": (cf / ct if ct else float("nan"), ct),
                      "recent": (cr / rt if rt else float("nan"), rt)}
    return out

def main():
    df = pd.read_parquet(os.path.join(C.ART, "features_ff.parquet")).sort_values("date").reset_index(drop=True)
    dates = df.date.values.astype("datetime64[ns]")
    cols = [c for c in df.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
            "dist", "logvol", "streak", "peer_", "mkt_", "cnt_", "abn_", "sent_", "novelty", "npc"))]
    X = df[cols].values.astype(np.float32)
    ydir = ((df.label.values + 1) // 2).astype(int)
    absr = np.abs(df.ret.values.astype(float))
    med = pd.Series(absr).rolling(20).median().shift(1).values
    yvol = (absr > med).astype(int)
    Xv = np.column_stack([X] + [np.abs(pd.Series(df.ret.values).shift(k).values) for k in range(1, 6)]).astype(np.float32)

    print("computing walk-forward (recency-weighted) direction + volatility ...", flush=True)
    dir_oos = wf_weighted(X, ydir, dates, weighted=True)
    dir_cal = calibrate_wf(dir_oos, ydir)
    vol_oos = wf_weighted(Xv, yvol, dates, weighted=True)

    def show(name, prob, vol=None):
        c = curve(prob, ydir, dates, vol=vol)
        print(f"\n== {name} ==")
        print(f"{'cov':>5s} {'full acc(n)':>16s} {'recent24-26 acc(n)':>22s}")
        for k, v in c.items():
            print(f"{int(k*100):4d}% {v['full'][0]:9.4f}({v['full'][1]:4d}) "
                  f"{v['recent'][0]:14.4f}({v['recent'][1]:4d})")
        return c

    show("direction (recency-wtd)", dir_oos)
    show("direction (recency-wtd + isotonic)", dir_cal)
    cvol = show("direction ∩ predicted-high-vol", dir_cal, vol=vol_oos)
    json.dump({"note": "walk-forward, pre-committed thresholds; acc(n) = accuracy(sample count)",
               "dir_highvol_recent": {int(k*100): cvol[k]["recent"] for k in cvol}},
              open(os.path.join(C.ART, "results_confidence_opt.json"), "w"), indent=2, default=float)
    print("\nsaved results_confidence_opt.json")

if __name__ == "__main__":
    main()
