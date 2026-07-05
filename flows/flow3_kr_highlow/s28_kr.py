"""
Stage 28 - Korean replication: KR large-caps next-day direction from BigKinds
news sentiment + price, leak-free, walk-forward.  Measures (not assumes) whether
KR reaches 70%.

Data: daily_returns_for_factor (market=KR, 17 tickers, 2024-07..2026-06);
BigKinds econ_* news (title/published_at) linked to tickers by name/alias in the
TITLE; pre-computed FinBERT sentiment joined by news_id.
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
import os, glob, re, numpy as np, pandas as pd
import config as C
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef

BK = os.path.join(C.DATA_ROOT, "news", "bigkinds")
PR = os.path.join(C.DATA_ROOT, "price")

def main():
    # ---- KR prices ----
    dr = pd.read_parquet(os.path.join(PR, "daily_returns_for_factor.parquet"))
    kr = dr[dr.market == "KR"].copy()
    kr["date"] = pd.to_datetime(kr.trade_date)
    kr = kr.sort_values(["ticker", "date"]).reset_index(drop=True)
    kr["ret"] = kr["ret"].astype(float)
    kr["label"] = (kr.ret > 0).astype(int)
    tickers = sorted(kr.ticker.unique())
    dmin, dmax = kr.date.min(), kr.date.max()
    print(f"KR prices: {len(tickers)} tickers, {dmin.date()}..{dmax.date()}, rows={len(kr)}")

    # ---- ticker name/alias map ----
    uni = pd.read_parquet(os.path.join(BK, "kr_universe_enriched.parquet"))
    uni = uni[uni.ticker.isin(tickers)]
    names = {}
    for _, r in uni.iterrows():
        al = set()
        if isinstance(r["name"], str): al.add(r["name"])
        try:
            for a in (r["aliases"] or []):
                if isinstance(a, str) and len(a) >= 2: al.add(a)
        except Exception:
            pass
        names[r.ticker] = [a for a in al if len(a) >= 2]
    print("name map sample:", {t: names[t] for t in tickers[:5]})

    # ---- load news (2024+) titles, link by title mention ----
    files = [f for f in glob.glob(os.path.join(BK, "econ_*.parquet"))
             if re.search(r"econ_(\d{4}-\d{2}-\d{2})_", f).group(1) >= "2024-01-01"]
    print("econ_ files 2024+:", len(files))
    frames = []
    for f in files:
        d = pd.read_parquet(f, columns=["news_id", "published_at", "title"])
        frames.append(d)
    news = pd.concat(frames, ignore_index=True)
    news["date"] = pd.to_datetime(news.published_at).dt.normalize()
    print("news rows 2024+:", len(news))

    # link: title contains a ticker name/alias
    rows = []
    for t in tickers:
        pat = "|".join(re.escape(a) for a in names.get(t, []))
        if not pat:
            continue
        hit = news[news.title.str.contains(pat, regex=True, na=False)]
        if len(hit):
            tmp = hit[["news_id", "date"]].copy(); tmp["ticker"] = t
            rows.append(tmp)
    link = pd.concat(rows, ignore_index=True)
    print("linked (ticker,news) rows:", len(link), "| per-ticker:",
          link.ticker.value_counts().to_dict())

    # ---- join FinBERT sentiment by news_id ----
    fb = pd.read_parquet(os.path.join(BK, "bigkinds_finbert_scores.parquet"),
                         columns=["news_id", "finbert_sentiment"])
    link = link.merge(fb, on="news_id", how="left")
    daily = link.groupby(["ticker", "date"]).agg(
        sent=("finbert_sentiment", "mean"), cnt=("finbert_sentiment", "size")).reset_index()

    # ---- build features per (ticker, trading day), leak-free ----
    out = []
    for t, g in kr.groupby("ticker"):
        g = g.sort_values("date").copy()
        ds = daily[daily.ticker == t].set_index("date")
        g["sent"] = g.date.map(ds.sent).astype(float)
        g["cnt"] = g.date.map(ds.cnt).fillna(0).astype(float)
        # leak-free: features as of previous trading day
        g["sent_prev"] = g.sent.shift(1)
        g["cnt_prev"] = g.cnt.shift(1)
        g["sent_3"] = g.sent.shift(1).rolling(3).mean()
        g["cnt_3"] = g.cnt.shift(1).rolling(3).sum()
        for k in [1, 2, 3, 5]:
            g[f"lagret_{k}"] = g.ret.shift(k)
        g["mom5"] = g.ret.shift(1).rolling(5).sum()
        g["vol5"] = g.ret.shift(1).rolling(5).std()
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    feat = ["sent_prev", "cnt_prev", "sent_3", "cnt_3", "lagret_1", "lagret_2",
            "lagret_3", "lagret_5", "mom5", "vol5"]
    df = df.dropna(subset=["ret"]).reset_index(drop=True)

    # ---- walk-forward by date (pooled across tickers) ----
    df = df.sort_values("date").reset_index(drop=True)
    dates = np.sort(df.date.unique())
    split = dates[int(len(dates) * 0.6)]
    tr = df.date < split
    te = df.date >= split
    X = df[feat].values.astype(np.float32); y = df.label.values
    news_cov = (df.cnt_prev > 0).mean()
    print(f"\nsamples={len(df)} train={tr.sum()} test={te.sum()} "
          f"test up-rate={y[te].mean():.3f} | days with prev-news={news_cov:.1%}")
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                                       min_samples_leaf=40, l2_regularization=1.0,
                                       random_state=C.SEED).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    acc = accuracy_score(y[te], (p > 0.5).astype(int))
    mcc = matthews_corrcoef(np.where(y[te] == 1, 1, -1), np.where(p > 0.5, 1, -1))
    base = max(y[te].mean(), 1 - y[te].mean())
    print(f"\nKR pooled next-day direction: test_acc={acc:.4f} mcc={mcc:.4f} baseline={base:.4f}")
    # selective
    conf = np.abs(p - 0.5)
    for kap in [0.3, 0.2, 0.1]:
        thr = np.quantile(conf, 1 - kap); mm = conf >= thr
        print(f"  selective top {int(kap*100)}%: acc={accuracy_score(y[te][mm],(p[mm]>0.5).astype(int)):.4f} (n={mm.sum()})")
    # sentiment-sign coincidence check (same-day)
    dd = df.dropna(subset=["sent"])
    c_same = np.corrcoef(dd.sent, dd.ret)[0, 1]
    c_next = np.corrcoef(dd.sent_prev.fillna(0), dd.ret)[0, 1]
    print(f"\ncorr(sentiment, SAME-day ret)={c_same:+.4f}  corr(sent_prev, ret)={c_next:+.4f}")

if __name__ == "__main__":
    main()
