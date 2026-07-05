"""
Stage 12 - add cross-asset / market features (last legitimate lever).

Daily data holds 9 mega-caps; build a market/peer factor from the 8 non-NVDA
names: each peer's PRIOR-trading-day return + an equal-weight market proxy
(prior-day and 5-day). All leak-free (shifted to <= D_{i-1}).

Feature sets compared (bagged HistGBM, honest dev-thresholded selective):
  price            NVDA price/technical (from s10)
  price+xasset     + peer/market factor
  price+xasset+news+ TEB news embeddings
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
from s10_boost import price_features
from s11_selective import bagged_proba

PEERS = ["AAPL", "MSFT", "JPM", "V", "BRK-B", "CAT", "RTX", "GE"]

def xasset_features():
    d = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close"])
    d = d[d.ticker.isin(PEERS + [C.TARGET])].copy()
    d["date"] = pd.to_datetime(d.trade_date); d["close"] = d.close.astype(float)
    d = d.sort_values(["ticker", "date"])
    d["ret"] = d.groupby("ticker").close.pct_change()
    wide = d.pivot_table(index="date", columns="ticker", values="ret").sort_index()
    lag1 = wide.shift(1)                                    # prior trading-day returns
    f = pd.DataFrame(index=wide.index)
    for p in PEERS:
        if p in lag1.columns:
            f[f"peer_{p}"] = lag1[p]
    mkt = wide[PEERS].mean(axis=1)
    f["mkt_lag1"] = mkt.shift(1)
    f["mkt_mom5"] = mkt.rolling(5).mean().shift(1)
    f["mkt_lag2"] = mkt.shift(2)
    return f.reset_index()

def honest_selective(p_dv, p_te, y, dv, te):
    cdv, cte = np.abs(p_dv - 0.5), np.abs(p_te - 0.5)
    out = {}
    for k in [1.0, 0.5, 0.3, 0.2, 0.1]:
        tau = 0.0 if k >= 1.0 else np.quantile(cdv, 1 - k)
        m = cte >= tau
        if m.sum():
            out[k] = (float(accuracy_score(y[te][m], (p_te[m] > 0.5).astype(int))), float(m.mean()))
    return out

def main():
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    daily = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close", "volume"])
    daily = daily[daily.ticker == C.TARGET].copy()
    daily["date"] = pd.to_datetime(daily.trade_date); daily = daily.sort_values("date").reset_index(drop=True)
    daily["close"] = daily.close.astype(float); daily["volume"] = daily.volume.astype(float)
    daily["ret"] = daily.close / daily.close.shift(1) - 1
    samp = samp.merge(price_features(daily), on="date", how="left")
    samp = samp.merge(xasset_features(), on="date", how="left")

    price_cols = [c for c in samp.columns if c.startswith(("lagret", "mom", "vol", "maratio",
                  "rsi", "dist", "logvol", "streak"))]
    xa_cols = [c for c in samp.columns if c.startswith(("peer_", "mkt_"))]
    TEB = np.load(os.path.join(C.ART, "feats_TEB.npz"))
    news = np.concatenate([TEB["short"], TEB["mid"].mean(1), TEB["long"].mean(1)], axis=1)

    y = ((samp.label.values + 1) // 2).astype(int); ypm = samp.label.values
    sp = samp.split.values; tr, dv, te = sp == "train", sp == "dev", sp == "test"
    Xp = samp[price_cols].values.astype(np.float32)
    Xx = samp[xa_cols].values.astype(np.float32)
    sets = {"price": Xp, "price+xasset": np.concatenate([Xp, Xx], 1),
            "price+xasset+news": np.concatenate([Xp, Xx, news], 1)}
    print(f"price={Xp.shape[1]} xasset={Xx.shape[1]} news={news.shape[1]} test-up={y[te].mean():.3f}")

    report = {}
    print(f"\n{'set':22s} {'dev_acc':>8s} {'test_acc':>9s} {'test_mcc':>9s}")
    for name, X in sets.items():
        p_dv = bagged_proba(X[tr], y[tr], X[dv])
        p_te = bagged_proba(X[tr], y[tr], X[te])
        acc = accuracy_score(y[te], (p_te > 0.5).astype(int))
        mcc = matthews_corrcoef(ypm[te], np.where(p_te > 0.5, 1, -1))
        dacc = accuracy_score(y[dv], (p_dv > 0.5).astype(int))
        sel = honest_selective(p_dv, p_te, y, dv, te)
        report[name] = {"dev_acc": dacc, "test_acc": acc, "test_mcc": mcc,
                        "selective": {f"{int(k*100)}pct": v for k, v in sel.items()}}
        print(f"{name:22s} {dacc:8.4f} {acc:9.4f} {mcc:9.4f}")
        print("     honest selective (test_acc @ cov): " +
              "  ".join(f"{int(k*100)}%:{v[0]:.3f}@{v[1]*100:.0f}%" for k, v in sel.items()))

    best_full = max(r["test_acc"] for r in report.values())
    best_sel = max((v[0] for r in report.values() for v in
                    [tuple(x) for x in r["selective"].values()]), default=0)
    print(f"\nBEST honest full-coverage test acc = {best_full:.4f}")
    print(f"BEST honest selective test acc      = {best_sel:.4f}")
    json.dump(report, open(os.path.join(C.ART, "results_xasset.json"), "w"), indent=2)
    print("saved results_xasset.json")

if __name__ == "__main__":
    main()
