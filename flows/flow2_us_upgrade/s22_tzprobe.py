"""
Stage 22 - resolve timezone + settle leading-vs-reactive.

For each news at t (sentiment s), using 5-min ET bars, measure the move JUST
BEFORE the news [t-60m -> t] and JUST AFTER [t -> t+60m], under two timezone
assumptions for published_at (treat-as-ET, and UTC->ET).  The correct TZ is the
one with a strong COINCIDENT signal; BEFORE>>AFTER means news is reactive
(reports moves already happened) => not usable to predict forward.
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

def run(bt, bc, T, S, label):
    def px_after(times):
        i = np.searchsorted(bt, times, side="left"); ok = i < len(bt)
        o = np.full(len(times), np.nan); o[ok] = bc[i[ok]]; return o
    def px_before(times):
        i = np.searchsorted(bt, times, side="right") - 1; ok = i >= 0
        o = np.full(len(times), np.nan); o[ok] = bc[i[ok]]; return o
    p0 = px_after(T + np.timedelta64(1, "s"))
    pb = px_before(T - np.timedelta64(60, "m"))
    pf = px_after(T + np.timedelta64(60, "m"))
    et_h = pd.to_datetime(T).hour
    inh = (et_h >= 10) & (et_h <= 14)                  # windows fit in session
    rb = p0 / pb - 1.0                                  # move BEFORE news
    rf = pf / p0 - 1.0                                  # move AFTER news
    v = inh & ~np.isnan(rb) & ~np.isnan(rf) & (np.abs(S) > 1e-6)
    s = S[v]
    cb = np.corrcoef(s, rb[v])[0, 1]; cf = np.corrcoef(s, rf[v])[0, 1]
    print(f"[{label}] n={v.sum()}  corr(sent, BEFORE move)={cb:+.4f}  "
          f"corr(sent, AFTER move)={cf:+.4f}")
    return cb, cf

def main():
    m5 = pd.read_parquet(C.MIN5_NVDA); m5["dt"] = pd.to_datetime(m5["datetime"])
    m5 = m5.sort_values("dt").reset_index(drop=True); m5["close"] = m5["close"].astype(float)
    bt = m5.dt.values.astype("datetime64[ns]"); bc = m5.close.values

    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    t = pd.to_datetime(nv.published_at)
    nv = nv[t >= pd.Timestamp("2019-01-01")].reset_index(drop=True)
    t = pd.to_datetime(nv.published_at); t = t[t >= pd.Timestamp("2019-01-01")].reset_index(drop=True)
    S = nv.sent.values
    print("published_at tz-aware:", t.dt.tz is not None)
    Traw = t.values.astype("datetime64[ns]")
    print("\nWhich TZ shows a strong COINCIDENT signal reveals the true timezone:")
    run(bt, bc, Traw, S, "assume ET (raw)")
    run(bt, bc, Traw - np.timedelta64(4, "h"), S, "assume UTC->EDT (-4h)")
    run(bt, bc, Traw - np.timedelta64(5, "h"), S, "assume UTC->EST (-5h)")
    print("\nRule: BEFORE>>AFTER => news is REACTIVE (already-priced) => no forward edge.")

if __name__ == "__main__":
    main()
