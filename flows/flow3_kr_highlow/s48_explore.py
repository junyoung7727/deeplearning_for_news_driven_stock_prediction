"""
Stage 48-E - EXPLORATION before the performance push. CPU-only, ~5-10 min.

Questions that decide the s48 design:
 E1 which features actually carry UP5/DN5 signal (GBM permutation importance)
 E2 conditional base rates: gap x vol, day-of-week, news presence, novelty|vol
 E3 market-day regime structure: per-date rate dispersion, autocorr, x-sec corr
 E4 per-ticker persistence of hit rates (train-half vs test-half)
 E5 data-scaling headroom: all tickers vs small-cap, no-news days, big-cap rates
 E6 error anatomy of s47 ENS: top-decile FP vs TP feature profile
 E7 label co-occurrence UP5&DN5 (volatility vs direction), by vol quintile
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
import os, time, numpy as np, pandas as pd

HOME = os.path.expanduser("~")
ART = os.path.join(HOME, "dlfe", "artifacts")
DATA = os.path.join(HOME, "dlfe", "data")
t0 = time.time()
def log(m): print(f"{m}  ({time.time()-t0:.0f}s)", flush=True)

Z = np.load(os.path.join(ART, "kr46_features_bidirectional.npz"), allow_pickle=True)
VOLF, NOVF, SENTF, PSEQ = (Z[k].astype(np.float32) for k in ("VOLF", "NOVF", "SENTF", "PSEQ"))
HIT, DROP = Z["HIT"].astype(np.float32), Z["DROP"].astype(np.float32)
DT = Z["DT"].astype("datetime64[ns]")
up5, dn5 = HIT[:, 2], DROP[:, 2]
N = len(up5)
dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
log(f"loaded N={N} train={len(trN)} test={len(teN)}")

VN = ["gap_open", "hl20", "std20", "hl_prev", "absr1_prev",
      "nov_mean", "nov_max", "ln_n_over", "ln_n7", "simfrac", "has_over",
      "sent_mean", "sent_min", "sent_max"]
PN = [f"{s}_{w}{a}" for w in (5, 10, 30) for a in ("mu", "sd", "mx", "mn")
      for s in ("r1", "hl", "gap", "volz", "hit2")] + \
     [f"{s}_last" for s in ("r1", "hl", "gap", "volz", "hit2")]

def pseq_aggs(P):
    feats = []
    for w in (5, 10, 30):
        seg = P[:, -w:, :]
        feats += [seg.mean(1), seg.std(1), seg.max(1), seg.min(1)]
    feats.append(P[:, -1, :])
    return np.concatenate(feats, 1).astype(np.float32)

def zs(X, idx):
    mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-9
    return ((X - mu) / sd).astype(np.float32)

V14 = np.concatenate([zs(VOLF, trN), zs(NOVF[:, :5], trN), NOVF[:, 5:6], zs(SENTF, trN)], 1)
Xtab = np.concatenate([V14, pseq_aggs(PSEQ)], 1)
FN = VN + PN
log(f"tabular {Xtab.shape}, names {len(FN)}")

# ---------------------------------------------------------------- E1
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    dvN = trN[DT[trN] >= np.sort(np.unique(DT[trN]))[int(len(np.unique(DT[trN])) * 0.85)]]
    trS = trN[DT[trN] < np.sort(np.unique(DT[trN]))[int(len(np.unique(DT[trN])) * 0.85)]]
    print("\n=== E1 permutation importance (AUC, dev) ===", flush=True)
    for name, y in (("UP5", up5), ("DN5", dn5)):
        pos = y[trS].mean(); w = float(np.clip((1 - pos) / pos, 1, 25))
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
            max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=13)
        gbm.fit(Xtab[trS], y[trS], sample_weight=np.where(y[trS] == 1, w, 1.0))
        r = permutation_importance(gbm, Xtab[dvN], y[dvN], scoring="roc_auc",
                                   n_repeats=3, random_state=13, n_jobs=8)
        order = np.argsort(-r.importances_mean)[:18]
        print(f"[{name}] top features by dAUC:", flush=True)
        for i in order:
            print(f"   {FN[i]:<14s} {r.importances_mean[i]*1e3:6.2f}e-3", flush=True)
except Exception as e:
    print("E1 FAIL", repr(e), flush=True)

# ---------------------------------------------------------------- E2
try:
    print("\n=== E2 conditional base rates (test only) ===", flush=True)
    g = VOLF[teN, 0]; v20 = VOLF[teN, 2]
    u, d = up5[teN], dn5[teN]
    gb = np.digitize(g, [-0.02, 0.0, 0.02])
    vq = np.digitize(v20, np.quantile(v20, [0.2, 0.4, 0.6, 0.8]))
    print("gap bucket (<-2%,-2..0,0..2%,>2%) x vol quintile -> UP5 | DN5 | n", flush=True)
    for bi, bl in enumerate(["gap<-2", "-2..0", "0..+2", "gap>+2"]):
        row = []
        for q in range(5):
            m = (gb == bi) & (vq == q)
            row.append(f"{u[m].mean():.3f}/{d[m].mean():.3f}/{m.sum()}" if m.sum() > 50 else "-")
        print(f"  {bl:<7s} " + " | ".join(row), flush=True)
    dow = pd.DatetimeIndex(DT[teN]).dayofweek
    print("day-of-week UP5:", [f"{u[dow==k].mean():.3f}" for k in range(5)], flush=True)
    has = NOVF[teN, 5] > 0
    print(f"news presence: UP5 with={u[has].mean():.3f} without={u[~has].mean():.3f} "
          f"| DN5 with={d[has].mean():.3f} without={d[~has].mean():.3f} (n_with={has.sum()})", flush=True)
    nm = NOVF[teN, 0]
    hv = vq >= 3
    for lbl, m in (("hiVol", hv & has), ("loVol", ~hv & has)):
        if m.sum() > 200:
            qs = np.quantile(nm[m], [0.25, 0.5, 0.75])
            b = np.digitize(nm[m], qs)
            print(f"novelty quartiles ({lbl}, n={m.sum()}): UP5 " +
                  " ".join(f"{u[m][b==k].mean():.3f}" for k in range(4)) + " | DN5 " +
                  " ".join(f"{d[m][b==k].mean():.3f}" for k in range(4)), flush=True)
except Exception as e:
    print("E2 FAIL", repr(e), flush=True)

# ---------------------------------------------------------------- E3
try:
    print("\n=== E3 market-day regime ===", flush=True)
    df = pd.DataFrame({"d": DT, "u": up5, "dn": dn5, "v": VOLF[:, 2], "g": VOLF[:, 0]})
    day = df.groupby("d").agg(u=("u", "mean"), dn=("dn", "mean"), v=("v", "mean"),
                              g=("g", "mean"), n=("u", "size"))
    day = day[day.n >= 30]
    print(f"per-day UP5 rate: mean={day.u.mean():.3f} std={day.u.std():.3f} "
          f"p10={day.u.quantile(.1):.3f} p90={day.u.quantile(.9):.3f}", flush=True)
    print(f"lag-1 autocorr: UP5 {day.u.autocorr(1):+.3f} DN5 {day.dn.autocorr(1):+.3f}", flush=True)
    print(f"corr(day mean vol, day UP5 rate)={day.v.corr(day.u):+.3f}, "
          f"corr(day mean gap, day UP5)={day.g.corr(day.u):+.3f}, "
          f"corr(day mean gap, day DN5)={day.g.corr(day.dn):+.3f}", flush=True)
    print(f"corr(prev-day UP5 rate, day UP5 rate)={day.u.shift(1).corr(day.u):+.3f}", flush=True)
except Exception as e:
    print("E3 FAIL", repr(e), flush=True)

# ---------------------------------------------------------------- E4  (ticker persistence needs ticker ids - reconstruct from ohlcv order is unsafe; use per-sample joins via dates+features hash? skip precise: use PSEQ fingerprint) 
try:
    print("\n=== E4 ticker persistence (via ohlcv rebuild) ===", flush=True)
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    ohlcv = ohlcv[ohlcv.date >= pd.Timestamp("2015-01-01")].sort_values(["ticker", "date"])
    g = ohlcv.groupby("ticker")
    o, h, l, c = (ohlcv[k].values.astype(float) for k in ("open", "high", "low", "close"))
    up = (h >= o * 1.05) & (o > 0)
    dn = (l <= o * 0.95) & (o > 0)
    ohlcv["u5"], ohlcv["d5"] = up, dn
    mid = ohlcv.date.quantile(0.6)
    a = ohlcv[ohlcv.date < mid].groupby("ticker")[["u5", "d5"]].mean()
    b = ohlcv[ohlcv.date >= mid].groupby("ticker")[["u5", "d5"]].mean()
    j = a.join(b, lsuffix="_a", rsuffix="_b").dropna()
    j = j[(ohlcv.groupby("ticker").size().reindex(j.index) > 400)]
    print(f"tickers={len(j)} corr(u5 early, u5 late)={j.u5_a.corr(j.u5_b):+.3f} "
          f"corr(d5)={j.d5_a.corr(j.d5_b):+.3f}", flush=True)
except Exception as e:
    print("E4 FAIL", repr(e), flush=True)

# ---------------------------------------------------------------- E5
try:
    print("\n=== E5 scaling headroom ===", flush=True)
    capf = pd.read_parquet(os.path.join(DATA, "kr_market_cap_daily.parquet"))
    capf["date"] = pd.to_datetime(capf.trade_date)
    ov = capf.merge(ohlcv[["ticker", "date", "close"]], on=["ticker", "date"])
    ov = ov[(ov.market_cap > 0) & (ov.close > 0)]
    shares = (ov.market_cap / ov.close).groupby(ov.ticker).median()
    ohlcv["pcap"] = ohlcv.close * ohlcv.ticker.map(shares)
    ohlcv["month"] = ohlcv.date.dt.strftime("%Y-%m")
    mcap = ohlcv[ohlcv.pcap.notna()].groupby(["ticker", "month"]).pcap.median().reset_index()
    med = mcap.groupby("month").pcap.median().rename("xmed").reset_index()
    mcap = mcap.merge(med, on="month")
    mcap["small"] = mcap.pcap <= mcap.xmed
    sm = set(map(tuple, mcap[mcap.small][["ticker", "month"]].values))
    ohlcv["is_small"] = [ (t, m) in sm for t, m in zip(ohlcv.ticker, ohlcv.month) ]
    valid = (ohlcv.open > 0) & (ohlcv.volume > 0)
    big = ohlcv[valid & ~ohlcv.is_small]
    small = ohlcv[valid & ohlcv.is_small]
    print(f"ticker-days valid: small={len(small)} big={len(big)} "
          f"(current samples {N} = small ex-warmup/limitup)", flush=True)
    print(f"big-cap rates: UP5={big.u5.mean():.4f} DN5={big.d5.mean():.4f} | "
          f"small: UP5={small.u5.mean():.4f} DN5={small.d5.mean():.4f}", flush=True)
    print(f"tickers: {ohlcv.ticker.nunique()}", flush=True)
except Exception as e:
    print("E5 FAIL", repr(e), flush=True)

# ---------------------------------------------------------------- E6
try:
    print("\n=== E6 s47 ENS error anatomy (test top-decile) ===", flush=True)
    P = np.load(os.path.join(ART, "kr47_probs.npz"), allow_pickle=True)
    e_te, y_te = P["e_te"], P["y_te"]
    feats = {"gap": VOLF[teN, 0], "std20": VOLF[teN, 2], "hl_prev": VOLF[teN, 3],
             "nov_mean": NOVF[teN, 0], "has_over": NOVF[teN, 5],
             "sent_mean": SENTF[teN, 0], "volz_last": pseq_aggs(PSEQ[teN])[:, 60 + 3],
             "hit2_30mu": pseq_aggs(PSEQ[teN])[:, 40 + 4]}
    for j, name in ((2, "UP5"), (5, "DN5")):
        p = e_te[:, j]; y = y_te[:, j]
        order = np.argsort(-p); sel = order[: len(p) // 10]
        tp = sel[y[sel] == 1]; fp = sel[y[sel] == 0]
        print(f"[{name}] top10% n={len(sel)} hit={y[sel].mean():.3f}", flush=True)
        for k, v in feats.items():
            print(f"   {k:<10s} TP={v[tp].mean():+.4f} FP={v[fp].mean():+.4f} rest={v[order[len(p)//10:]].mean():+.4f}", flush=True)
except Exception as e:
    print("E6 FAIL", repr(e), flush=True)

# ---------------------------------------------------------------- E7
try:
    print("\n=== E7 UP5/DN5 co-occurrence ===", flush=True)
    both = (up5 == 1) & (dn5 == 1)
    print(f"P(UP5)={up5.mean():.4f} P(DN5)={dn5.mean():.4f} P(both)={both.mean():.4f} "
          f"P(both|UP5)={both[up5==1].mean():.4f}", flush=True)
    vq = np.digitize(VOLF[:, 2], np.quantile(VOLF[trN, 2], [0.2, 0.4, 0.6, 0.8]))
    print("by vol quintile: " + " | ".join(
        f"q{q}: U={up5[vq==q].mean():.3f} D={dn5[vq==q].mean():.3f} B={both[vq==q].mean():.3f}"
        for q in range(5)), flush=True)
except Exception as e:
    print("E7 FAIL", repr(e), flush=True)

log("exploration done")
