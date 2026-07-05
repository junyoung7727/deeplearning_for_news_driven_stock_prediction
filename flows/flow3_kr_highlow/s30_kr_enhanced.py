"""
Stage 30 - KR enhanced: richer foreign/institution FLOW features + price, 626
stocks, walk-forward. Push selective accuracy toward 0.70 honestly (large robust
samples, leak-free, pre-committed thresholds).
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
from sklearn.metrics import accuracy_score, matthews_corrcoef

BASE = os.path.join(C.DATA_ROOT, "analysis_outputs", "kr_ff5_foreign_regression_20260619T105218Z")

def main():
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"))
    flow = pd.read_parquet(os.path.join(BASE, "kr_foreign_flow_daily.parquet"))
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"))
    for d in (sec, flow, cap):
        d["date"] = pd.to_datetime(d.trade_date)
    df = sec[["ticker", "date", "ret_1d", "excess_ret"]].merge(
        flow[["ticker", "date", "foreign_net_value_krw", "foreign_net_qty",
              "institution_net_value_krw", "institution_net_qty",
              "foreign_buy_value_krw", "foreign_sell_value_krw"]], on=["ticker", "date"], how="left").merge(
        cap[["ticker", "date", "market_cap", "volume", "trading_value"]], on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df[df.ret_1d.notna()].reset_index(drop=True)
    mc = df.market_cap.replace(0, np.nan)
    df["fnv_norm"] = df.foreign_net_value_krw / mc
    df["inst_norm"] = df.institution_net_value_krw / mc
    df["for_imb"] = (df.foreign_buy_value_krw - df.foreign_sell_value_krw) / \
                    (df.foreign_buy_value_krw + df.foreign_sell_value_krw + 1)
    df["turnover"] = df.trading_value / mc

    g = df.groupby("ticker", group_keys=False)
    def shift1(c): return g[c].shift(1)
    def roll(c, w, fn="sum"): return g[c].apply(lambda s: getattr(s.shift(1).rolling(w), fn)())
    df["fnv_norm_l1"] = shift1("fnv_norm"); df["inst_norm_l1"] = shift1("inst_norm")
    df["for_imb_l1"] = shift1("for_imb"); df["turnover_l1"] = shift1("turnover")
    df["fnv_norm_5"] = roll("fnv_norm", 5); df["fnv_norm_20"] = roll("fnv_norm", 20)
    df["inst_norm_5"] = roll("inst_norm", 5)
    df["for_imb_5"] = roll("for_imb", 5, "mean")
    df["fbuy_streak"] = g["fnv_norm"].apply(lambda s: (s.shift(1) > 0).rolling(5).sum())
    for k in [1, 2, 3, 5]:
        df[f"lagret_{k}"] = g["ret_1d"].shift(k)
    df["mom5"] = roll("ret_1d", 5); df["mom20"] = roll("ret_1d", 20)
    df["vol5"] = roll("ret_1d", 5, "std"); df["vol20"] = roll("ret_1d", 20, "std")
    df["logcap"] = np.log(mc)
    mkt = df.groupby("date").ret_1d.transform("mean")
    df["mkt_l1"] = g.apply(lambda x: None) if False else mkt.groupby(df.ticker).shift(1)

    feats = ["fnv_norm_l1", "inst_norm_l1", "for_imb_l1", "fnv_norm_5", "fnv_norm_20",
             "inst_norm_5", "for_imb_5", "fbuy_streak", "turnover_l1", "logcap",
             "lagret_1", "lagret_2", "lagret_3", "lagret_5", "mom5", "mom20", "vol5", "vol20", "mkt_l1"]

    def evalt(tcol, label):
        d = df[df[tcol].notna()].copy(); d["y"] = (d[tcol] > 0).astype(int)
        dates = np.sort(d.date.unique())
        X = d[feats].values.astype(np.float32); y = d.y.values; dt = d.date.values
        # walk-forward: retrain quarterly, expanding window
        oos = np.full(len(d), np.nan)
        qs = pd.to_datetime(dates); starts = dates[(pd.Series(qs).dt.to_period("Q").drop_duplicates().index)]
        first_test = dates[int(len(dates)*0.4)]
        cut = first_test
        while cut <= dates[-1]:
            nxt = cut + np.timedelta64(63, "D")
            tr = dt < cut; blk = (dt >= cut) & (dt < nxt)
            if tr.sum() > 5000 and blk.sum() > 0:
                m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_depth=4,
                        min_samples_leaf=50, l2_regularization=1.0, random_state=C.SEED).fit(X[tr], y[tr])
                oos[blk] = m.predict_proba(X[blk])[:, 1]
            cut = nxt
        mm = ~np.isnan(oos)
        base = max(y[mm].mean(), 1 - y[mm].mean())
        acc = accuracy_score(y[mm], (oos[mm] > 0.5).astype(int))
        mcc = matthews_corrcoef(np.where(y[mm]==1,1,-1), np.where(oos[mm]>0.5,1,-1))
        print(f"\n=== {label} === WF OOS n={mm.sum()} baseline={base:.4f}")
        print(f"  full-coverage acc={acc:.4f} mcc={mcc:.4f}")
        conf = np.abs(oos - 0.5)
        for kap in [0.3, 0.2, 0.1, 0.05, 0.02]:
            thr = np.nanquantile(conf, 1 - kap); s = mm & (conf >= thr)
            print(f"  selective top {int(kap*100):2d}%: acc={accuracy_score(y[s],(oos[s]>0.5).astype(int)):.4f} (n={int(s.sum())})")

    evalt("ret_1d", "raw direction")
    evalt("excess_ret", "excess (market-neutral) direction")

if __name__ == "__main__":
    main()
