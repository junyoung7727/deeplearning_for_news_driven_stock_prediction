"""
Stage 31 - KR: verify excess target + test WEEKLY (5-day) horizon with foreign flow.
Foreign-flow signals persist over weeks; test if 5-day-ahead direction is more
predictable (non-overlapping samples to avoid label autocorrelation inflation).
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

BASE = r"D:\Github\homeserver\alphamale\data\analysis_outputs\kr_ff5_foreign_regression_20260619T105218Z"

def main():
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"))
    flow = pd.read_parquet(os.path.join(BASE, "kr_foreign_flow_daily.parquet"))
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"))
    for d in (sec, flow, cap):
        d["date"] = pd.to_datetime(d.trade_date)
    df = sec[["ticker", "date", "ret_1d", "excess_ret"]].merge(
        flow[["ticker", "date", "foreign_net_value_krw", "institution_net_value_krw",
              "foreign_buy_value_krw", "foreign_sell_value_krw"]], on=["ticker", "date"], how="left").merge(
        cap[["ticker", "date", "market_cap", "trading_value"]], on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df[df.ret_1d.notna()].reset_index(drop=True)

    # verify excess vs raw
    v = df.dropna(subset=["excess_ret"])
    agree = (np.sign(v.ret_1d) == np.sign(v.excess_ret)).mean()
    print(f"sign agreement ret_1d vs excess_ret = {agree:.3f}  "
          f"corr={np.corrcoef(v.ret_1d, v.excess_ret)[0,1]:.3f}")

    mc = df.market_cap.replace(0, np.nan)
    df["fnv_norm"] = df.foreign_net_value_krw / mc
    df["inst_norm"] = df.institution_net_value_krw / mc
    df["for_imb"] = (df.foreign_buy_value_krw - df.foreign_sell_value_krw) / \
                    (df.foreign_buy_value_krw + df.foreign_sell_value_krw + 1)
    df["turnover"] = df.trading_value / mc
    g = df.groupby("ticker", group_keys=False)
    # cumulative log price to build multi-day forward returns
    df["logp"] = g["ret_1d"].apply(lambda s: np.log1p(s).cumsum())

    def roll(c, w, fn="sum"): return g[c].apply(lambda s: getattr(s.shift(1).rolling(w), fn)())
    df["fnv_l1"] = g["fnv_norm"].shift(1); df["fnv_5"] = roll("fnv_norm", 5)
    df["fnv_20"] = roll("fnv_norm", 20); df["inst_5"] = roll("inst_norm", 5)
    df["for_imb_l1"] = g["for_imb"].shift(1); df["for_imb_5"] = roll("for_imb", 5, "mean")
    df["fbuy_streak"] = g["fnv_norm"].apply(lambda s: (s.shift(1) > 0).rolling(10).sum())
    df["turnover_l1"] = g["turnover"].shift(1)
    for k in [1, 2, 5]:
        df[f"lagret_{k}"] = g["ret_1d"].shift(k)
    df["mom20"] = roll("ret_1d", 20); df["vol20"] = roll("ret_1d", 20, "std")
    df["logcap"] = np.log(mc)
    feats = ["fnv_l1", "fnv_5", "fnv_20", "inst_5", "for_imb_l1", "for_imb_5",
             "fbuy_streak", "turnover_l1", "lagret_1", "lagret_2", "lagret_5",
             "mom20", "vol20", "logcap"]

    for H in [1, 5, 10]:
        df["fwd"] = g["logp"].apply(lambda s: s.shift(-H) - s)   # H-day forward log return
        d = df.dropna(subset=["fwd"] + ["fnv_5"]).copy()
        d["y"] = (d.fwd > 0).astype(int)
        # non-overlapping samples for H>1: keep every H-th trading day per ticker
        if H > 1:
            d["rk"] = d.groupby("ticker").cumcount()
            d = d[d.rk % H == 0]
        dates = np.sort(d.date.unique()); X = d[feats].values.astype(np.float32)
        y = d.y.values; dt = d.date.values
        oos = np.full(len(d), np.nan); cut = dates[int(len(dates)*0.4)]
        while cut <= dates[-1]:
            nxt = cut + np.timedelta64(63, "D")
            tr = dt < cut - np.timedelta64(H, "D"); blk = (dt >= cut) & (dt < nxt)
            if tr.sum() > 4000 and blk.sum() > 0:
                m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_depth=4,
                        min_samples_leaf=50, l2_regularization=1.0, random_state=C.SEED).fit(X[tr], y[tr])
                oos[blk] = m.predict_proba(X[blk])[:, 1]
            cut = nxt
        mm = ~np.isnan(oos); base = max(y[mm].mean(), 1 - y[mm].mean())
        acc = accuracy_score(y[mm], (oos[mm] > 0.5).astype(int))
        print(f"\n=== {H}-day forward direction (non-overlap) === n={mm.sum()} base={base:.4f} full_acc={acc:.4f}")
        conf = np.abs(oos - 0.5)
        for kap in [0.3, 0.2, 0.1, 0.05]:
            thr = np.nanquantile(conf, 1 - kap); s = mm & (conf >= thr)
            print(f"   top {int(kap*100):2d}%: acc={accuracy_score(y[s],(oos[s]>0.5).astype(int)):.4f} (n={int(s.sum())})")

if __name__ == "__main__":
    main()
