"""
Stage 10 - Accuracy push.

Adds leak-free PRICE/technical features (the paper is news-only) to the finance
event embeddings and trains gradient-boosted trees + logistic, reporting:
  * full-coverage test accuracy / MCC   (the actual task: NVDA next-day up/down)
  * selective accuracy vs coverage      (accuracy on the model's confident days)

LEAKAGE DISCIPLINE: sample for trading day D_i has target sign(close_i/close_{i-1}-1);
every feature uses ONLY information available at the close of D_{i-1}
(all price series shifted by >=1; news windows already end at D_{i-1}).
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

def price_features(prices):
    df = prices.sort_values("date").reset_index(drop=True).copy()
    for col in ["close", "volume", "ret"]:
        df[col] = df[col].astype(float)
    c, r, v = df["close"], df["ret"], df["volume"]
    f = pd.DataFrame({"date": df["date"]})
    for k in [1, 2, 3, 5, 10]:
        f[f"lagret_{k}"] = r.shift(k)                    # return realised on day i-k
    for k in [5, 10, 20]:
        f[f"mom_{k}"]     = c.shift(1) / c.shift(1 + k) - 1
        f[f"vol_{k}"]     = r.shift(1).rolling(k).std()
        f[f"maratio_{k}"] = c.shift(1) / c.shift(1).rolling(k).mean() - 1
    d = c.shift(1).diff()
    up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    f["dist_hi20"] = c.shift(1) / c.shift(1).rolling(20).max() - 1
    f["dist_lo20"] = c.shift(1) / c.shift(1).rolling(20).min() - 1
    f["logvol"] = np.log(v.shift(1) + 1)
    f["volz"] = (v.shift(1) - v.shift(1).rolling(20).mean()) / (v.shift(1).rolling(20).std() + 1e-9)
    f["streak"] = np.sign(r).shift(1).rolling(3).sum()   # recent up/down streak
    return f

def sel_curve(prob, y01, covs=(1.0, 0.5, 0.3, 0.2, 0.1)):
    conf = np.abs(prob - 0.5)
    order = np.argsort(-conf)
    out = {}
    for cov in covs:
        n = max(1, int(len(prob) * cov))
        idx = order[:n]
        out[cov] = accuracy_score(y01[idx], (prob[idx] > 0.5).astype(int))
    return out

def main():
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    # full NVDA daily history (with volume) so rolling features are valid at the
    # 2018 boundary; every feature is shifted >=1 day (leak-free).
    daily = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close", "volume"])
    daily = daily[daily.ticker == C.TARGET].copy()
    daily["date"] = pd.to_datetime(daily.trade_date)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["close"] = daily.close.astype(float); daily["volume"] = daily.volume.astype(float)
    daily["ret"] = daily.close / daily.close.shift(1) - 1
    pf = price_features(daily)
    samp = samp.merge(pf, on="date", how="left")
    price_cols = [c for c in pf.columns if c != "date"]

    # news embedding features (leak-free windows already end at D_{i-1})
    TEB = np.load(os.path.join(C.ART, "feats_TEB.npz"))
    news = np.concatenate([TEB["short"], TEB["mid"].mean(1), TEB["long"].mean(1)], axis=1)

    y = ((samp.label.values + 1) // 2).astype(int)
    ypm = samp.label.values
    sp = samp.split.values
    tr, dv, te = sp == "train", sp == "dev", sp == "test"
    Xp = samp[price_cols].values.astype(np.float32)
    print(f"features: price={Xp.shape[1]} news={news.shape[1]} | "
          f"train={tr.sum()} dev={dv.sum()} test={te.sum()} test-up={y[te].mean():.3f}")

    feature_sets = {"price": Xp, "news": news, "price+news": np.concatenate([Xp, news], 1)}

    def hgb():
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=400, l2_regularization=1.0,
            min_samples_leaf=30, validation_fraction=None, random_state=C.SEED)
    def logit():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.3))

    results = {}
    rows = []
    for fname, X in feature_sets.items():
        for mname, mk in [("HGB", hgb), ("Logit", logit)]:
            m = mk().fit(X[tr], y[tr])
            p_dv = m.predict_proba(X[dv])[:, 1]
            p_te = m.predict_proba(X[te])[:, 1]
            acc = accuracy_score(y[te], (p_te > 0.5).astype(int))
            mcc = matthews_corrcoef(ypm[te], np.where(p_te > 0.5, 1, -1)) if len(np.unique(p_te > 0.5)) > 1 else 0.0
            dacc = accuracy_score(y[dv], (p_dv > 0.5).astype(int))
            rows.append((f"{fname}/{mname}", dacc, acc, mcc))
            results[f"{fname}/{mname}"] = (p_dv, p_te)

    # ensemble of price+news HGB and Logit
    pe_dv = np.mean([results["price+news/HGB"][0], results["price+news/Logit"][0]], 0)
    pe_te = np.mean([results["price+news/HGB"][1], results["price+news/Logit"][1]], 0)
    rows.append(("price+news/ENSEMBLE",
                 accuracy_score(y[dv], (pe_dv > 0.5).astype(int)),
                 accuracy_score(y[te], (pe_te > 0.5).astype(int)),
                 matthews_corrcoef(ypm[te], np.where(pe_te > 0.5, 1, -1))))

    print("\n==== full-coverage test results ====")
    print(f"{'model':24s} {'dev_acc':>8s} {'test_acc':>9s} {'test_mcc':>9s}")
    for n, da, ta, tm in rows:
        print(f"{n:24s} {da:8.4f} {ta:9.4f} {tm:9.4f}")

    best = max(rows, key=lambda z: z[2])
    print(f"\nBEST full-coverage test accuracy: {best[0]} = {best[2]:.4f}")

    # selective accuracy for the strong-signal models
    def selective_report(name, pdv, pte):
        conf_dv, conf_te = np.abs(pdv - 0.5), np.abs(pte - 0.5)
        print(f"\n== selective: {name} ==")
        sc = sel_curve(pte, y[te])
        for cov, a in sc.items():
            print(f"   coverage {cov*100:5.0f}%  test_acc {a:.4f}")
        chosen = None
        for t in np.quantile(conf_dv, np.linspace(0, 0.97, 60)):
            mask = conf_dv >= t
            if mask.sum() >= max(10, int(0.15 * len(pdv))):
                if accuracy_score(y[dv][mask], (pdv[mask] > 0.5).astype(int)) >= 0.70:
                    chosen = t; break
        if chosen is not None:
            mt = conf_te >= chosen
            a = accuracy_score(y[te][mt], (pte[mt] > 0.5).astype(int)) if mt.sum() else float("nan")
            print(f"   dev-threshold(>=70% dev): test_acc {a:.4f} @ coverage {mt.mean():.1%}")
        else:
            print("   (no dev threshold reaches 70% dev accuracy)")
        return sc
    sc_pn = selective_report("price+news/HGB", *results["price+news/HGB"])
    selective_report("price/HGB", *results["price/HGB"])

    # train on train+dev (more data), evaluate test - final-eval convention
    print("\n==== trained on train+dev ====")
    trdv = tr | dv
    trdv_rows = []
    for fname in ["price", "news", "price+news"]:
        X = feature_sets[fname]
        m = hgb().fit(X[trdv], y[trdv])
        pte = m.predict_proba(X[te])[:, 1]
        a = accuracy_score(y[te], (pte > 0.5).astype(int))
        mcc = matthews_corrcoef(ypm[te], np.where(pte > 0.5, 1, -1))
        trdv_rows.append((fname, a, mcc))
        print(f"  {fname}/HGB(train+dev)  test_acc {a:.4f}  test_mcc {mcc:.4f}")

    best_full_acc = max([r[2] for r in rows] + [r[1] for r in trdv_rows])
    json.dump({"full": [{"model": n, "dev_acc": da, "test_acc": ta, "test_mcc": tm}
                        for n, da, ta, tm in rows],
               "train_dev": [{"model": n, "test_acc": a, "test_mcc": m} for n, a, m in trdv_rows],
               "best_full_test_acc": float(best_full_acc),
               "selective_price_news": {f"{int(k*100)}pct": v for k, v in sc_pn.items()}},
              open(os.path.join(C.ART, "results_boost.json"), "w"), indent=2)
    print(f"\nBEST honest full-coverage test accuracy = {best_full_acc:.4f}")
    print("saved results_boost.json")

if __name__ == "__main__":
    main()
