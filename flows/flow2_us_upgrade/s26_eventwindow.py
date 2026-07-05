"""
Stage 26 - RESEARCH: post-news short-horizon direction with 5-min bars.

Question: after an NVDA news item, is the direction over the next N minutes
predictable at ~70%?  5-min bars are REGULAR hours (09:30-15:55) only, so this
covers intraday news whose window fits in the session.

Method (leak-free): convert published_at UTC->ET; for news in-session, entry =
CLOSE of the 5-min bar containing the news (i.e., AFTER the news bar), exit =
H minutes later (same session). Compare sign(sentiment) to sign(forward return),
per horizon.  Also measure the PRE-news move (reactive detection) and check
timestamp granularity (release vs ingestion-batch).
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
    for c in ["open", "close"]:
        m5[c] = m5[c].astype(float)
    bt = m5.dt.values.astype("datetime64[ns]"); bo = m5.open.values; bc = m5.close.values
    bday = bt.astype("datetime64[D]")

    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv["sent"] = np.load(os.path.join(C.ART, "tf_title_sent.npy")).astype(float)
    et = pd.to_datetime(nv.published_at).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    nv["et"] = et
    print("=== timestamp granularity (ET) ===")
    print("  seconds==0:", f"{(et.dt.second==0).mean():.1%}", "| minute%5==0:", f"{(et.dt.minute%5==0).mean():.1%}")
    print("  minute-of-hour top5:", et.dt.minute.value_counts().head(5).to_dict())

    keep = ~nv.title_clean.str.contains(SPAM)
    nv = nv[keep].reset_index(drop=True)
    et = nv.et
    # in-session news (leave room for 120-min window before 16:00)
    ins = (et.dt.time >= pd.Timestamp("09:35").time()) & (et.dt.time <= pd.Timestamp("13:55").time())
    nv = nv[ins].reset_index(drop=True)
    T = nv.et.values.astype("datetime64[ns]"); S = nv.sent.values
    bar_idx = np.searchsorted(bt, T, side="right") - 1
    valid0 = (bar_idx >= 5) & (bar_idx < len(bt) - 25)
    # entry bar must actually contain T (same day & within 5 min)
    contains = valid0 & (bday[np.clip(bar_idx, 0, len(bt)-1)] == T.astype("datetime64[D]"))
    print(f"\nin-session non-spam news with a matching bar: {contains.sum()} / {len(nv)}")

    hbars = {"5m": 1, "10m": 2, "15m": 3, "30m": 6, "60m": 12, "120m": 24}
    print(f"\n{'horizon':>8s} {'n':>6s} {'sent-sign acc':>13s} {'corr':>8s} {'PRE-move acc':>13s}")
    for name, hb in hbars.items():
        ei = bar_idx.copy()
        entry = bc[np.clip(ei, 0, len(bt)-1)]
        xi = ei + hb
        pi = ei - hb
        ok = contains & (xi < len(bt)) & (pi >= 0)
        same = ok & (bday[np.clip(xi, 0, len(bt)-1)] == bday[np.clip(ei, 0, len(bt)-1)])
        exitp = bc[np.clip(xi, 0, len(bt)-1)]
        prep = bc[np.clip(pi, 0, len(bt)-1)]
        fwd = exitp / entry - 1.0
        pre = entry / prep - 1.0
        m = same & (np.abs(S) > 1e-6) & (fwd != 0)
        acc = float((np.sign(S[m]) == np.sign(fwd[m])).mean())
        cor = float(np.corrcoef(S[m], fwd[m])[0, 1])
        pacc = float((np.sign(S[m]) == np.sign(pre[m])).mean())
        print(f"{name:>8s} {int(m.sum()):6d} {acc:13.4f} {cor:+8.4f} {pacc:13.4f}")

    # strong-news subset (|sentiment| high) at 15m/30m
    print("\n=== strong-news subset (|sent| >= 0.6) ===")
    for name, hb in {"15m": 3, "30m": 6, "60m": 12}.items():
        ei = bar_idx; entry = bc[np.clip(ei, 0, len(bt)-1)]
        xi = ei + hb; ok = contains & (xi < len(bt))
        same = ok & (bday[np.clip(xi, 0, len(bt)-1)] == bday[np.clip(ei, 0, len(bt)-1)])
        fwd = bc[np.clip(xi, 0, len(bt)-1)] / entry - 1.0
        m = same & (np.abs(S) >= 0.6) & (fwd != 0)
        if m.sum() > 20:
            acc = float((np.sign(S[m]) == np.sign(fwd[m])).mean())
            print(f"  {name}: n={int(m.sum())} sent-sign acc={acc:.4f}")

if __name__ == "__main__":
    main()
