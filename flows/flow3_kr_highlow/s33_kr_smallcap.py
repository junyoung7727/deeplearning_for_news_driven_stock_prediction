"""
Stage 33 - KR SMALL-CAP news -> next-day direction (same logic as US pipeline).

Small-caps are less efficient => news may be more predictive. Universe = 626
priced KR stocks; take the small-cap half by median market cap. Map BigKinds news
to them by company name/alias in the TITLE; join pre-computed FinBERT sentiment;
aggregate daily; predict next-day direction (leak-free, walk-forward).
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
BASE = os.path.join(C.DATA_ROOT, "analysis_outputs", "kr_ff5_foreign_regression_20260619T105218Z")

def main():
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"),
                          columns=["ticker", "trade_date", "ret_1d"])
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"),
                          columns=["ticker", "trade_date", "market_cap"])
    sec["date"] = pd.to_datetime(sec.trade_date); cap["date"] = pd.to_datetime(cap.trade_date)
    px = sec.merge(cap[["ticker", "date", "market_cap"]], on=["ticker", "date"], how="left")
    px = px[px.ret_1d.notna()].sort_values(["ticker", "date"]).reset_index(drop=True)

    medcap = px.groupby("ticker").market_cap.median()
    small = set(medcap[medcap <= medcap.median()].index)   # smaller half
    px = px[px.ticker.isin(small)].reset_index(drop=True)
    print(f"small-cap tickers: {len(small)} | rows={len(px)} | "
          f"median mktcap(small)={medcap[medcap<=medcap.median()].median()/1e8:.0f}억 "
          f"vs large={medcap[medcap>medcap.median()].median()/1e8:.0f}억")

    # names/aliases for small-caps
    uni = pd.read_parquet(os.path.join(BK, "kr_universe_enriched.parquet"))
    uni = uni[uni.ticker.isin(small)]
    name2t = {}
    for _, r in uni.iterrows():
        cand = set()
        if isinstance(r["name"], str) and len(r["name"]) >= 2: cand.add(r["name"])
        try:
            for a in (r["aliases"] or []):
                if isinstance(a, str) and len(a) >= 3: cand.add(a)   # >=3 to cut noise
        except Exception:
            pass
        for c in cand:
            name2t.setdefault(c, []).append(r.ticker)

    # load 2024+ news titles
    files = [f for f in glob.glob(os.path.join(BK, "econ_*.parquet"))
             if re.search(r"econ_(\d{4}-\d{2}-\d{2})_", f).group(1) >= "2024-01-01"]
    news = pd.concat([pd.read_parquet(f, columns=["news_id", "published_at", "title"]) for f in files],
                     ignore_index=True)
    news["date"] = pd.to_datetime(news.published_at).dt.normalize()
    print("news 2024+ rows:", len(news))

    # prefilter titles containing any small-cap name, then map
    big = re.compile("|".join(sorted((re.escape(k) for k in name2t), key=len, reverse=True)))
    hit = news[news.title.str.contains(big, na=False)].copy()
    print("titles mentioning a small-cap:", len(hit))
    rows = []
    for name, tks in name2t.items():
        sub = hit[hit.title.str.contains(re.escape(name), na=False)]
        if len(sub):
            for t in tks:
                tmp = sub[["news_id", "date"]].copy(); tmp["ticker"] = t; rows.append(tmp)
    link = pd.concat(rows, ignore_index=True).drop_duplicates(["news_id", "ticker"])
    print("linked (ticker,news):", len(link), "| distinct tickers:", link.ticker.nunique())

    fb = pd.read_parquet(os.path.join(BK, "bigkinds_finbert_scores.parquet"),
                         columns=["news_id", "finbert_sentiment"])
    link = link.merge(fb, on="news_id", how="left")
    daily = link.groupby(["ticker", "date"]).agg(
        sent=("finbert_sentiment", "mean"), cnt=("finbert_sentiment", "size")).reset_index()

    # features (same logic), leak-free
    out = []
    for t, g in px.groupby("ticker"):
        g = g.sort_values("date").copy()
        ds = daily[daily.ticker == t].set_index("date")
        g["sent"] = g.date.map(ds.sent).astype(float)
        g["cnt"] = g.date.map(ds.cnt).fillna(0).astype(float)
        g["sent_prev"] = g.sent.shift(1); g["cnt_prev"] = g.cnt.shift(1)
        g["sent_3"] = g.sent.shift(1).rolling(3).mean(); g["cnt_3"] = g.cnt.shift(1).rolling(3).sum()
        for k in [1, 2, 3, 5]:
            g[f"lagret_{k}"] = g.ret_1d.shift(k)
        g["mom5"] = g.ret_1d.shift(1).rolling(5).sum(); g["vol5"] = g.ret_1d.shift(1).rolling(5).std()
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df["y"] = (df.ret_1d > 0).astype(int)
    feat = ["sent_prev", "cnt_prev", "sent_3", "cnt_3", "lagret_1", "lagret_2", "lagret_3", "lagret_5", "mom5", "vol5"]

    # news-day subset coverage
    has_news = (df.cnt_prev > 0)
    print(f"\nsmall-cap ticker-days with prev-day news: {has_news.mean():.1%} ({int(has_news.sum())})")

    # walk-forward pooled
    dates = np.sort(df.date.unique()); X = df[feat].values.astype(np.float32)
    y = df.y.values; dt = df.date.values; oos = np.full(len(df), np.nan)
    cut = dates[int(len(dates)*0.4)]
    while cut <= dates[-1]:
        nxt = cut + np.timedelta64(63, "D"); tr = dt < cut; blk = (dt >= cut) & (dt < nxt)
        if tr.sum() > 3000 and blk.sum() > 0:
            m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                    min_samples_leaf=40, l2_regularization=1.0, random_state=C.SEED).fit(X[tr], y[tr])
            oos[blk] = m.predict_proba(X[blk])[:, 1]
        cut = nxt
    mm = ~np.isnan(oos)
    base = max(y[mm].mean(), 1 - y[mm].mean())
    print(f"\nSMALL-CAP next-day direction (all days) WF: acc={accuracy_score(y[mm],(oos[mm]>0.5).astype(int)):.4f} "
          f"mcc={matthews_corrcoef(np.where(y[mm]==1,1,-1),np.where(oos[mm]>0.5,1,-1)):.4f} baseline={base:.4f} n={mm.sum()}")
    # restrict to news-days (where news exists -> where news signal could matter)
    nd = mm & has_news.values
    if nd.sum() > 200:
        bn = max(y[nd].mean(), 1 - y[nd].mean())
        print(f"SMALL-CAP on NEWS-DAYS only: acc={accuracy_score(y[nd],(oos[nd]>0.5).astype(int)):.4f} "
              f"baseline={bn:.4f} n={nd.sum()}")
    conf = np.abs(oos - 0.5)
    for kap in [0.2, 0.1, 0.05]:
        thr = np.nanquantile(conf, 1 - kap); s = mm & (conf >= thr)
        print(f"  selective top {int(kap*100)}%: acc={accuracy_score(y[s],(oos[s]>0.5).astype(int)):.4f} (n={int(s.sum())})")
    # sentiment coincidence check
    dd = df.dropna(subset=["sent"])
    print(f"\ncorr(sent, SAME-day ret)={np.corrcoef(dd.sent,dd.ret_1d)[0,1]:+.4f} "
          f"corr(sent_prev, ret)={np.corrcoef(dd.sent_prev.fillna(0),dd.ret_1d)[0,1]:+.4f}")

if __name__ == "__main__":
    main()
