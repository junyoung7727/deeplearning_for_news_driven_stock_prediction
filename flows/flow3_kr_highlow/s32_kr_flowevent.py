"""
Stage 32 - KR foreign-flow EVENT study: how high does next-day direction accuracy
go, honestly, when conditioned on flow strength?  Leak-free (flow at t-1 predicts
day t). Two views: (a) pure event rule sign(foreign flow_{t-1}) -> day t;
(b) HGB model, accuracy by flow-magnitude decile + model-confidence.
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
import os, numpy as np, pandas as pd
import config as C
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.join(C.DATA_ROOT, "analysis_outputs", "kr_ff5_foreign_regression_20260619T105218Z")

def main():
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"))
    flow = pd.read_parquet(os.path.join(BASE, "kr_foreign_flow_daily.parquet"))
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"))
    for d in (sec, flow, cap):
        d["date"] = pd.to_datetime(d.trade_date)
    df = sec[["ticker", "date", "ret_1d"]].merge(
        flow[["ticker", "date", "foreign_net_value_krw", "institution_net_value_krw",
              "foreign_buy_value_krw", "foreign_sell_value_krw"]], on=["ticker", "date"], how="left").merge(
        cap[["ticker", "date", "market_cap", "trading_value"]], on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df[df.ret_1d.notna()].reset_index(drop=True)
    mc = df.market_cap.replace(0, np.nan)
    df["fnv_norm"] = df.foreign_net_value_krw / mc
    g = df.groupby("ticker", group_keys=False)
    df["fnv_l1"] = g["fnv_norm"].shift(1)
    df["fnv5_l1"] = g["fnv_norm"].apply(lambda s: s.shift(1).rolling(5).sum())
    df["y"] = (df.ret_1d > 0).astype(int)
    d = df.dropna(subset=["fnv_l1"]).copy()

    print(f"n={len(d)}  overall up-rate={d.y.mean():.4f}")
    # (a) pure event rule: predict up if foreign bought yesterday
    for col, lbl in [("fnv_l1", "1-day foreign flow"), ("fnv5_l1", "5-day foreign flow")]:
        dd = d.dropna(subset=[col])
        acc = (np.sign(dd[col]) == np.where(dd.ret_1d > 0, 1, -1)).mean()
        print(f"  rule sign({lbl}_{{t-1}}) -> day t : acc={acc:.4f} (n={len(dd)})")
        # extreme deciles
        dd = dd.copy(); dd["dec"] = pd.qcut(dd[col].rank(method="first"), 10, labels=False)
        top = dd[dd.dec == 9]; bot = dd[dd.dec == 0]
        print(f"     heavy-BUY decile: up-rate={top.y.mean():.4f} (n={len(top)}) | "
              f"heavy-SELL decile: up-rate={bot.y.mean():.4f} (n={len(bot)})")
        # combined event rule accuracy on extreme deciles only (predict buy->up / sell->down)
        ext = dd[dd.dec.isin([0, 9])]
        eacc = (np.sign(ext[col]) == np.where(ext.ret_1d > 0, 1, -1)).mean()
        print(f"     extreme-decile event rule acc={eacc:.4f} (n={len(ext)})")

    # (b) model, walk-forward, accuracy by |flow| decile
    feats = ["fnv_l1", "fnv5_l1"]
    for k in [1, 2, 5]:
        d[f"lr{k}"] = g["ret_1d"].shift(k)
    d["mom5"] = g["ret_1d"].apply(lambda s: s.shift(1).rolling(5).sum())
    feats += ["lr1", "lr2", "lr5", "mom5"]
    dates = np.sort(d.date.unique()); X = d[feats].values.astype(np.float32)
    y = d.y.values; dt = d.date.values; oos = np.full(len(d), np.nan)
    cut = dates[int(len(dates)*0.4)]
    while cut <= dates[-1]:
        nxt = cut + np.timedelta64(63, "D"); tr = dt < cut; blk = (dt >= cut) & (dt < nxt)
        if tr.sum() > 4000 and blk.sum() > 0:
            m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                    min_samples_leaf=60, l2_regularization=1.0, random_state=C.SEED).fit(X[tr], y[tr])
            oos[blk] = m.predict_proba(X[blk])[:, 1]
        cut = nxt
    mm = ~np.isnan(oos)
    print(f"\nmodel WF full-cov acc={accuracy_score(y[mm],(oos[mm]>0.5).astype(int)):.4f} "
          f"baseline={max(y[mm].mean(),1-y[mm].mean()):.4f}")
    conf = np.abs(oos - 0.5)
    for kap in [0.1, 0.05, 0.02, 0.01]:
        thr = np.nanquantile(conf, 1 - kap); s = mm & (conf >= thr)
        print(f"  model selective top {int(kap*100) if kap>=0.01 else 1}%: "
              f"acc={accuracy_score(y[s],(oos[s]>0.5).astype(int)):.4f} (n={int(s.sum())})")

if __name__ == "__main__":
    main()
