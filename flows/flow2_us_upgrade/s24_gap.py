"""
Stage 24 - honest ceiling of the ONLY leading signal: overnight news -> GAP.

Target: sign of overnight gap = open_t / close_{t-1} - 1 (balanced-ish).
Features (all known before open_t, leak-free for this target):
  overnight news (ET close_{t-1} 16:00 -> open_t 09:30): sentiment, count, emb-PCA
  + prev-day price/technical + cross-asset (from features_ff, as-of close_{t-1}).
Walk-forward OOS. NOTE: gap direction is NOT tradable (news arrives after the
prior close), reported only to bound how much *predictable* signal exists.
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, matthews_corrcoef
import config as C
from s15_walkforward import walk_forward, hgb, logit, OOS_START

def main():
    ff = pd.read_parquet(os.path.join(C.ART, "features_ff.parquet")).sort_values("date").reset_index(drop=True)
    fcols = [c for c in ff.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
             "dist", "logvol", "streak", "peer_", "mkt_"))]  # price/xasset only (prev-day)

    d = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "open", "close"])
    d = d[d.ticker == C.TARGET].copy(); d["date"] = pd.to_datetime(d.trade_date)
    for c in ["open", "close"]:
        d[c] = d[c].astype(float)
    d = d.sort_values("date").reset_index(drop=True)
    d["prev_close"] = d.close.shift(1); d["prev_date"] = d.date.shift(1)
    d["gap"] = d.open / d.prev_close - 1.0

    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    emb = np.load(os.path.join(C.ART, "tf_title_emb.npy")).astype(np.float32)
    et = pd.to_datetime(nv.published_at).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    nv["et"] = et

    m = ff.merge(d[["date", "gap", "prev_date"]], on="date", how="inner")
    m = m[m.date >= "2019-01-01"].reset_index(drop=True)
    trmask = (m.date < pd.Timestamp(C.DEV_START)).values
    pca = PCA(n_components=6, random_state=C.SEED).fit(emb[et.values < np.datetime64(C.DEV_START)][:20000])

    rows = []
    etv = nv.et.values.astype("datetime64[ns]")
    for i in range(len(m)):
        hi = (m.date.iloc[i] + pd.Timedelta(hours=9, minutes=30)).to_datetime64()
        lo = (pd.Timestamp(m.prev_date.iloc[i]) + pd.Timedelta(hours=16)).to_datetime64()
        sel = (etv > lo) & (etv <= hi)
        if sel.sum():
            s = float(nv.sent.values[sel].mean()); c = int(sel.sum())
            pcs = pca.transform(emb[sel].mean(0, keepdims=True))[0]
        else:
            s = 0.0; c = 0; pcs = np.zeros(6, np.float32)
        rows.append([s, c, np.log1p(c)] + list(pcs))
    on = np.array(rows, np.float32)
    X = np.column_stack([m[fcols].values.astype(np.float32), on])
    y = (m.gap.values > 0).astype(int)
    dates = m.date.values.astype("datetime64[ns]")
    oosm = dates >= np.datetime64(OOS_START)
    base = max(y[oosm].mean(), 1 - y[oosm].mean())
    print(f"GAP-direction target | OOS days={oosm.sum()} up-rate={y[oosm].mean():.3f} "
          f"baseline={base:.4f} feats={X.shape[1]}")

    for name, mk in [("HGB", hgb), ("Logit-EN", logit)]:
        oos, tr = walk_forward(mk, X, y, dates)
        mm = ~np.isnan(oos)
        acc = accuracy_score(y[mm], (oos[mm] > 0.5).astype(int))
        mcc = matthews_corrcoef(np.where(y[mm] == 1, 1, -1), np.where(oos[mm] > 0.5, 1, -1))
        print(f"[{name}] gap-dir OOS acc={acc:.4f} mcc={mcc:.4f} train={tr:.4f} gap={tr-acc:.3f}")
    print(">=70%:", "check above")
    print("NOTE: gap direction is NOT tradable (news arrives after prior close);"
          " this only bounds predictable signal.")

if __name__ == "__main__":
    main()
