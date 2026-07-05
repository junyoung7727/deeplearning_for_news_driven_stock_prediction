"""
Stage 29 - KR full universe (626 stocks) next-day direction with FOREIGN/INSTITUTION
FLOW + price. Foreign net-buying is a documented leading signal in Korea.
Leak-free: features as of day t-1; target = sign(return on day t). Walk-forward.
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

BASE = r"D:\Github\homeserver\alphamale\data\analysis_outputs\kr_ff5_foreign_regression_20260619T105218Z"

def main():
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"))
    flow = pd.read_parquet(os.path.join(BASE, "kr_foreign_flow_daily.parquet"))
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"))
    for d in (sec, flow, cap):
        d["date"] = pd.to_datetime(d.trade_date)
    df = sec[["ticker", "date", "ret_1d", "excess_ret", "foreign_net_value_krw",
              "foreign_flow_intensity"]].merge(
        flow[["ticker", "date", "foreign_net_qty", "institution_net_value_krw",
              "institution_net_qty"]], on=["ticker", "date"], how="left").merge(
        cap[["ticker", "date", "market_cap", "volume", "trading_value"]], on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df[df.ret_1d.notna()].copy()
    print(f"KR universe: tickers={df.ticker.nunique()} rows={len(df)} "
          f"dates {df.date.min().date()}..{df.date.max().date()}")

    g = df.groupby("ticker", group_keys=False)
    # leak-free lagged features (known at t-1)
    df["turnover"] = df.trading_value / df.market_cap.replace(0, np.nan)
    for col in ["foreign_net_value_krw", "foreign_flow_intensity", "foreign_net_qty",
                "institution_net_value_krw", "institution_net_qty", "turnover", "volume"]:
        df[col + "_l1"] = g[col].shift(1)
    df["fnv_5"] = g["foreign_net_value_krw"].apply(lambda s: s.shift(1).rolling(5).sum())
    df["inst_5"] = g["institution_net_value_krw"].apply(lambda s: s.shift(1).rolling(5).sum())
    for k in [1, 2, 3, 5]:
        df[f"lagret_{k}"] = g["ret_1d"].shift(k)
    df["mom5"] = g["ret_1d"].apply(lambda s: s.shift(1).rolling(5).sum())
    df["vol5"] = g["ret_1d"].apply(lambda s: s.shift(1).rolling(5).std())
    df["logcap"] = np.log(df.market_cap.replace(0, np.nan))

    feats = ["foreign_net_value_krw_l1", "foreign_flow_intensity_l1", "foreign_net_qty_l1",
             "institution_net_value_krw_l1", "institution_net_qty_l1", "fnv_5", "inst_5",
             "turnover_l1", "volume_l1", "lagret_1", "lagret_2", "lagret_3", "lagret_5",
             "mom5", "vol5", "logcap"]

    for target_name, tcol in [("raw direction", "ret_1d"), ("excess (mkt-neutral) direction", "excess_ret")]:
        d = df[df[tcol].notna()].copy()
        d["y"] = (d[tcol] > 0).astype(int)
        dates = np.sort(d.date.unique())
        split = dates[int(len(dates) * 0.6)]
        tr = (d.date < split).values; te = (d.date >= split).values
        X = d[feats].values.astype(np.float32); y = d.y.values
        base = max(y[te].mean(), 1 - y[te].mean())
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_depth=4,
                                           min_samples_leaf=50, l2_regularization=1.0,
                                           random_state=C.SEED).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        acc = accuracy_score(y[te], (p > 0.5).astype(int))
        mcc = matthews_corrcoef(np.where(y[te] == 1, 1, -1), np.where(p > 0.5, 1, -1))
        print(f"\n=== {target_name} === test n={te.sum()} baseline={base:.4f}")
        print(f"  full-coverage: acc={acc:.4f} mcc={mcc:.4f}")
        conf = np.abs(p - 0.5)
        for kap in [0.3, 0.2, 0.1, 0.05]:
            thr = np.quantile(conf, 1 - kap); mm = conf >= thr
            print(f"  selective top {int(kap*100):2d}%: acc={accuracy_score(y[te][mm],(p[mm]>0.5).astype(int)):.4f} (n={int(mm.sum())})")
        # small-cap subset (bottom market-cap tercile)
        capq = d.loc[te, "market_cap"]
        small = te.copy(); small[te] = (capq <= capq.quantile(0.33)).values
        if small.sum() > 50:
            ps = m.predict_proba(X[small])[:, 1]
            print(f"  small-cap subset: acc={accuracy_score(y[small],(ps>0.5).astype(int)):.4f} (n={int(small.sum())})")

if __name__ == "__main__":
    main()
