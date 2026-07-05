"""
Stage 25 - is the news reaction in the OVERNIGHT/extended-hours GAP?

Decompose each day's move into GAP (prev_close->open, = after-hours + pre-market
+ overnight) and INTRADAY (open->close, = regular session).  Test where the news
signal lands, in DIRECTION and in MAGNITUDE.
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
    m5 = pd.read_parquet(C.MIN5_NVDA); m5["dt"] = pd.to_datetime(m5["datetime"])
    hrs = sorted(m5.dt.dt.hour.unique())
    print(f"5-min session hours present: {hrs}  -> {'REGULAR-hours only' if min(hrs)>=9 and max(hrs)<=16 else 'includes extended'}")

    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    et = pd.to_datetime(nv.published_at).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    nv["et"] = et

    d = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "open", "close"])
    d = d[d.ticker == C.TARGET].copy(); d["date"] = pd.to_datetime(d.trade_date)
    for c in ["open", "close"]:
        d[c] = d[c].astype(float)
    d = d.sort_values("date").reset_index(drop=True)
    d["prev_close"] = d.close.shift(1); d["prev_date"] = d.date.shift(1)
    d = d[(d.date >= "2019-01-01")].reset_index(drop=True)
    d["gap"] = d.open / d.prev_close - 1
    d["intraday"] = d.close / d.open - 1

    # overnight news per trading day (prev close 16:00 ET -> open 09:30 ET)
    etv = nv.et.values.astype("datetime64[ns]"); sv = nv.sent.values
    sent_on, cnt_on = [], []
    for i in range(len(d)):
        if pd.isna(d.prev_date.iloc[i]):
            sent_on.append(0.0); cnt_on.append(0); continue
        lo = (pd.Timestamp(d.prev_date.iloc[i]) + pd.Timedelta(hours=16)).to_datetime64()
        hi = (d.date.iloc[i] + pd.Timedelta(hours=9, minutes=30)).to_datetime64()
        sel = (etv > lo) & (etv <= hi)
        sent_on.append(float(sv[sel].mean()) if sel.sum() else 0.0); cnt_on.append(int(sel.sum()))
    d["on_sent"] = sent_on; d["on_cnt"] = cnt_on
    v = d.dropna(subset=["gap", "intraday"]).copy()

    def c(a, b): return float(np.corrcoef(a, b)[0, 1])
    print("\n=== DIRECTION: overnight news sentiment vs ... ===")
    print(f"  gap (overnight/ext-hours) : corr {c(v.on_sent, v.gap):+.4f}")
    print(f"  intraday (regular session): corr {c(v.on_sent, v.intraday):+.4f}")

    print("\n=== MAGNITUDE: overnight news volume vs |move| ===")
    print(f"  corr(news_count, |gap|)      = {c(v.on_cnt, v.gap.abs()):+.4f}")
    print(f"  corr(news_count, |intraday|) = {c(v.on_cnt, v.intraday.abs()):+.4f}")

    print("\n=== where does the daily move happen? ===")
    print(f"  std(gap)={v.gap.std()*100:.2f}%  std(intraday)={v.intraday.std()*100:.2f}%")
    hi_news = v[v.on_cnt >= v.on_cnt.quantile(0.9)]
    lo_news = v[v.on_cnt <= v.on_cnt.quantile(0.5)]
    print(f"  mean |gap| on HIGH-overnight-news days (top10%): {hi_news.gap.abs().mean()*100:.2f}%")
    print(f"  mean |gap| on LOW-overnight-news days  (bot50%): {lo_news.gap.abs().mean()*100:.2f}%")
    print("\n=> news direction shows in the GAP (ext-hours), ~0 in the regular session;")
    print("   heavy-news nights have bigger gaps. The move is repriced before the open.")

if __name__ == "__main__":
    main()
