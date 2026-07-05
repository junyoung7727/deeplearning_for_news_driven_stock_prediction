"""
Stage 27 - post-news direction: immediate news-bar reaction + volume-spike news.

Entry at the OPEN of the news bar (captures the intra-bar move from the news),
and condition on the news bar having a VOLUME SPIKE (real, market-moving news).
If even these are ~50%, there is no exploitable intraday post-news direction.
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
import os, re, numpy as np, pandas as pd
import config as C

SPAM = re.compile(r"law firm|law offices|llp|class action|investor alert|investigation|"
                  r"lawsuit|securities fraud|deadline|glancy|schall|pomerantz|rosen|"
                  r"reminds investors|encourages investors|shareholder rights", re.I)

def main():
    m5 = pd.read_parquet(C.MIN5_NVDA); m5["dt"] = pd.to_datetime(m5["datetime"])
    m5 = m5.sort_values("dt").reset_index(drop=True)
    for c in ["open", "close", "volume"]:
        m5[c] = m5[c].astype(float)
    bt = m5.dt.values.astype("datetime64[ns]"); bo = m5.open.values; bc = m5.close.values
    bv = m5.volume.values; bday = bt.astype("datetime64[D]")
    # rolling median volume by bar position (per time-of-day) approx: global rolling median
    volmed = pd.Series(bv).rolling(78*20, min_periods=78).median().values  # ~20-day median

    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    et = pd.to_datetime(nv.published_at).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    nv["et"] = et
    nv = nv[~nv.title_clean.str.contains(SPAM)].reset_index(drop=True)
    et = nv.et
    ins = (et.dt.time >= pd.Timestamp("09:35").time()) & (et.dt.time <= pd.Timestamp("15:00").time())
    nv = nv[ins].reset_index(drop=True)
    T = nv.et.values.astype("datetime64[ns]"); S = nv.sent.values
    bi = np.searchsorted(bt, T, side="right") - 1
    good = (bi >= 5) & (bi < len(bt) - 25) & (bday[np.clip(bi, 0, len(bt)-1)] == T.astype("datetime64[D]"))
    volspike = good & (bv[np.clip(bi, 0, len(bt)-1)] > 3.0 * np.nan_to_num(volmed[np.clip(bi, 0, len(bt)-1)], nan=1e18))

    print("entry at NEWS-BAR OPEN (captures intra-bar move):")
    print(f"{'horizon':>8s} {'n(all)':>7s} {'acc(all)':>9s} {'n(volspike)':>12s} {'acc(volspike)':>14s}")
    for name, hb in {"bar(5m)": 0, "15m": 2, "30m": 5, "60m": 11}.items():
        entry = bo[np.clip(bi, 0, len(bt)-1)]
        xi = bi + hb
        same = good & (xi < len(bt)) & (bday[np.clip(xi, 0, len(bt)-1)] == bday[np.clip(bi, 0, len(bt)-1)])
        fwd = bc[np.clip(xi, 0, len(bt)-1)] / entry - 1.0
        m = same & (np.abs(S) > 1e-6) & (fwd != 0)
        acc = float((np.sign(S[m]) == np.sign(fwd[m])).mean()) if m.sum() else float("nan")
        mv = m & volspike
        accv = float((np.sign(S[mv]) == np.sign(fwd[mv])).mean()) if mv.sum() > 20 else float("nan")
        print(f"{name:>8s} {int(m.sum()):7d} {acc:9.4f} {int(mv.sum()):12d} {accv:14.4f}")

    print("\nvolume-spike news bars found:", int(volspike.sum()),
          "(these are the genuinely market-moving items)")
    print("=> if acc ~0.50 even here, no exploitable intraday post-news direction with 5-min data.")

if __name__ == "__main__":
    main()
