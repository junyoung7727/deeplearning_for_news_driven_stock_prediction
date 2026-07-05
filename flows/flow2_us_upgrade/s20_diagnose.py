"""
Stage 20 - CONCRETE failure diagnosis (not "EMH").

Tests specific, fixable pipeline problems:
  1) news timestamp timezone / intraday timing (are we mis-dating news?)
  2) is NVDA news LEADING or LAGGING? corr(daily sentiment, SAME-day return) vs
     corr(daily sentiment, NEXT-day return)  -> reactive news can't predict.
  3) signal dilution: dozens of articles/day averaged into one vector.
  4) target noise floor: how many days are tiny (near-random sign) moves.
  5) is there directional accuracy CONDITIONAL on move size (descriptive)?
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

def main():
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    ts = pd.to_datetime(nv.published_at)
    print("=== 1) news timestamp timing ===")
    print("hour-of-day distribution (published_at as stored):")
    print(ts.dt.hour.value_counts().sort_index().to_dict())
    print("weekday dist (0=Mon):", ts.dt.dayofweek.value_counts().sort_index().to_dict())

    sent = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    nv = nv.assign(sent=sent)
    daily_sent = nv.groupby(nv.published_at.dt.normalize()).sent.mean()
    daily_cnt = nv.groupby(nv.published_at.dt.normalize()).sent.size()
    print(f"\n=== 3) signal dilution ===\narticles/day: median={daily_cnt.median():.0f} "
          f"mean={daily_cnt.mean():.1f} p90={daily_cnt.quantile(.9):.0f} max={daily_cnt.max()}")

    # prices
    d = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close"])
    d = d[d.ticker == C.TARGET].copy()
    d["date"] = pd.to_datetime(d.trade_date); d = d.sort_values("date")
    d["close"] = d.close.astype(float); d["ret"] = d.close.pct_change()
    d = d[(d.date >= C.START_DATE) & (d.date <= C.END_DATE)].reset_index(drop=True)

    print(f"\n=== 4) target noise floor (|daily ret|) ===")
    ar = d.ret.abs()
    for th in [0.005, 0.01, 0.02, 0.03]:
        print(f"  fraction of days with |ret| < {th*100:.0f}%: {(ar < th).mean():.1%}")
    print(f"  median |ret| = {ar.median()*100:.2f}%")

    # align daily sentiment to same-day and next-day trading returns
    ds = daily_sent.reindex(d.date).values          # sentiment on the trading day's date
    df = pd.DataFrame({"date": d.date, "ret": d.ret.values, "sent_today": ds})
    df["sent_prev"] = pd.Series(ds).shift(1).values   # yesterday's news
    df["ret_next"] = df.ret.shift(-1)
    v = df.dropna(subset=["sent_today", "ret"])
    def corr(a, b):
        m = a.notna() & b.notna()
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else float("nan")
    print(f"\n=== 2) LEADING vs LAGGING news (correlations) ===")
    print(f"  corr(sentiment_today , SAME-day ret)   = {corr(df.sent_today, df.ret):+.4f}  (coincident)")
    print(f"  corr(sentiment_today , NEXT-day ret)   = {corr(df.sent_today, df.ret_next):+.4f}  (predictive)")
    print(f"  corr(sentiment_prev  , today's ret)    = {corr(df.sent_prev, df.ret):+.4f}  (what the model uses)")
    # sign-rule accuracy
    def acc(sig, r):
        m = sig.notna() & r.notna() & (r != 0)
        return float(((np.sign(sig[m]) == np.sign(r[m]))).mean()), int(m.sum())
    a_same, n1 = acc(df.sent_today, df.ret)
    a_next, n2 = acc(df.sent_today, df.ret_next)
    print(f"  sentiment-sign accuracy SAME-day: {a_same:.4f} (n={n1}) | NEXT-day: {a_next:.4f} (n={n2})")

    print(f"\n=== 5) directional predictability conditional on move size ===")
    print("  corr(sentiment_prev, next-day ret) within |ret| buckets:")
    df2 = df.dropna(subset=["sent_prev", "ret"]).copy()
    df2["absret"] = df2.ret.abs()
    for lo, hi in [(0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 1)]:
        b = df2[(df2.absret >= lo) & (df2.absret < hi)]
        if len(b) > 30:
            c = np.corrcoef(b.sent_prev, np.sign(b.ret))[0, 1]
            print(f"    |ret| in [{lo*100:.0f},{hi*100:.0f}%): n={len(b):4d}  corr(prev-sent, sign)={c:+.3f}")

if __name__ == "__main__":
    main()
