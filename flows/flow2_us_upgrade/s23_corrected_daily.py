"""
Stage 23 - TZ-corrected diagnosis + the clean LEADING test.

published_at is naive UTC (shown in s22). Convert to ET, then:
  (a) corrected daily corr(sentiment, same-day vs next-day return)
  (b) LEADING test: pre-open news (ET 16:00 prev -> 09:30) -> that day's
      OPEN->CLOSE direction (news strictly before the session; leak-free).
If (b) is ~0, the reactive-news conclusion is complete: no forward edge exists.
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
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    t_utc = pd.to_datetime(nv.published_at).dt.tz_localize("UTC")
    nv["et"] = t_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    nv["et_date"] = nv.et.dt.normalize()
    nv["et_time"] = nv.et.dt.hour + nv.et.dt.minute / 60.0

    d = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "open", "close"])
    d = d[d.ticker == C.TARGET].copy()
    d["date"] = pd.to_datetime(d.trade_date)
    for c in ["open", "close"]:
        d[c] = d[c].astype(float)
    d = d.sort_values("date")
    d = d[(d.date >= "2019-01-01") & (d.date <= C.END_DATE)].reset_index(drop=True)
    d["cc"] = d.close / d.close.shift(1) - 1        # close->close
    d["oc"] = d.close / d.open - 1                    # open->close (intraday)
    d["prev_date"] = d.date.shift(1)

    # (a) daily sentiment by ET date
    ds = nv.groupby("et_date").sent.mean()
    d["sent_today"] = ds.reindex(d.date).values
    d["sent_prev"] = pd.Series(d["sent_today"].values).shift(1).values
    d["cc_next"] = d.cc.shift(-1)
    def corr(a, b):
        m = a.notna() & b.notna()
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 50 else float("nan")
    print("=== (a) TZ-corrected daily correlations ===")
    print(f"  corr(sent_today, SAME-day cc ret) = {corr(d.sent_today, d.cc):+.4f}  (coincident)")
    print(f"  corr(sent_today, NEXT-day cc ret) = {corr(d.sent_today, d.cc_next):+.4f}  (predictive)")
    print(f"  corr(sent_prev , today cc ret)    = {corr(d.sent_prev, d.cc):+.4f}  (model uses)")

    # (b) LEADING test: pre-open news -> open->close
    pre = []
    for i in range(1, len(d)):
        lo = d.prev_date.iloc[i] + pd.Timedelta(hours=16)     # prev close (ET)
        hi = d.date.iloc[i] + pd.Timedelta(hours=9, minutes=30)  # today open (ET)
        w = nv[(nv.et > lo) & (nv.et <= hi)]
        pre.append(w.sent.mean() if len(w) else np.nan)
    d2 = d.iloc[1:].copy(); d2["sent_preopen"] = pre
    m = d2.sent_preopen.notna() & d2.oc.notna() & (d2.oc != 0)
    a, b = d2.sent_preopen[m], d2.oc[m]
    acc = float((np.sign(a) == np.sign(b)).mean())
    print("\n=== (b) LEADING test: pre-open news -> OPEN->CLOSE (leak-free) ===")
    print(f"  n={m.sum()}  corr(preopen sent, open->close ret) = {np.corrcoef(a, b)[0,1]:+.4f}  "
          f"sign-acc = {acc:.4f}")
    # and pre-open -> overnight gap (open vs prev close), the direct reaction
    d2["gap"] = d2.open / d.close.shift(1).iloc[1:].values - 1
    mg = d2.sent_preopen.notna() & d2.gap.notna() & (d2.gap != 0)
    ag = float((np.sign(d2.sent_preopen[mg]) == np.sign(d2.gap[mg])).mean())
    print(f"  n={mg.sum()}  corr(preopen sent, overnight GAP) = "
          f"{np.corrcoef(d2.sent_preopen[mg], d2.gap[mg])[0,1]:+.4f}  sign-acc(gap) = {ag:.4f}")
    print("\n=> if open->close ~0 but GAP>0, even overnight news is priced into the OPEN,"
          " leaving no intraday edge (reactive/instant).")

if __name__ == "__main__":
    main()
