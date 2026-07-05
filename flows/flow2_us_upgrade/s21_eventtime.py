"""
Stage 21 - event-time test: is the signal LEADING at short horizons?

For each NVDA news at time t (sentiment s), measure the FORWARD return strictly
AFTER t using 5-min bars, at horizons 30/60/120 min, to session close, and
next-day close.  If sign(s) predicts short-horizon forward returns >> next-day,
the signal is intraday/leading and the daily next-day task was the wrong horizon.

Leakage guard: entry = first 5-min bar STRICTLY AFTER t (move must be after news).
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
    m5 = pd.read_parquet(C.MIN5_NVDA)
    m5["dt"] = pd.to_datetime(m5["datetime"])
    m5 = m5.sort_values("dt").reset_index(drop=True)
    m5["close"] = m5["close"].astype(float)
    print("5-min bars:", len(m5), "range", m5.dt.min(), "->", m5.dt.max())
    print("5-min hour range (ET market):", sorted(m5.dt.dt.hour.unique())[:3],
          "...", sorted(m5.dt.dt.hour.unique())[-3:])
    bt = m5.dt.values.astype("datetime64[ns]"); bc = m5.close.values

    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    nv["t"] = pd.to_datetime(nv.published_at)
    nv = nv[nv.t >= pd.Timestamp("2019-01-01")].reset_index(drop=True)
    T = nv.t.values.astype("datetime64[ns]"); S = nv.sent.values

    def price_at_or_after(times):
        idx = np.searchsorted(bt, times, side="left")
        ok = idx < len(bt); out = np.full(len(times), np.nan)
        out[ok] = bc[idx[ok]]; return out, idx

    entry, ei = price_at_or_after(T + np.timedelta64(1, "s"))
    horizons = {"30min": 30, "60min": 60, "120min": 120}
    res = {}
    for name, mins in horizons.items():
        exitp, _ = price_at_or_after(T + np.timedelta64(mins, "m"))
        # keep intraday: exit within same calendar day as entry
        same_day = (bt[np.clip(ei, 0, len(bt)-1)].astype("datetime64[D]") ==
                    (T + np.timedelta64(mins, "m")).astype("datetime64[D]"))
        fwd = exitp / entry - 1.0
        valid = ~np.isnan(entry) & ~np.isnan(exitp) & same_day & (np.abs(S) > 1e-6)
        r = fwd[valid]; s = S[valid]
        acc = float((np.sign(s) == np.sign(r)).mean()); cor = float(np.corrcoef(s, r)[0, 1])
        res[name] = (acc, cor, int(valid.sum()))

    # to session close
    day = bt.astype("datetime64[D]")
    last_close = {}
    for dd, cc in zip(day, bc):
        last_close[dd] = cc            # overwritten -> last bar of day
    entry_day = bt[np.clip(ei, 0, len(bt)-1)].astype("datetime64[D]")
    close_px = np.array([last_close.get(d, np.nan) for d in entry_day])
    fwd_c = close_px / entry - 1.0
    valid = ~np.isnan(entry) & ~np.isnan(close_px) & (np.abs(S) > 1e-6)
    res["to_close"] = (float((np.sign(S[valid]) == np.sign(fwd_c[valid])).mean()),
                       float(np.corrcoef(S[valid], fwd_c[valid])[0, 1]), int(valid.sum()))

    print("\nsign(sentiment) predicts FORWARD return (strictly after news):")
    print(f"{'horizon':>10s} {'acc':>7s} {'corr':>8s} {'n':>7s}")
    for k, (a, c, n) in res.items():
        print(f"{k:>10s} {a:7.4f} {c:+8.4f} {n:7d}")
    print("\n(compare: daily next-day close-close accuracy was ~0.52, corr ~0.00)")

if __name__ == "__main__":
    main()
