"""
Stage 39 - fetch KR daily OHLCV via pykrx for the priced universe/date range
(needed for the HIGH-prediction target: high >= open*(1+k), entry at open).

Per-TICKER date-range calls (the by-date all-market endpoint is currently
broken server-side; the by-ticker route is verified working). Resume-safe.
Output: artifacts/kr_ohlcv.parquet  (ticker, date, open, high, low, close, volume)
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
import os, time, pickle, pandas as pd
from pykrx import stock
import config as C

OUT = os.path.join(C.ART, "kr_ohlcv.parquet")
TMP = os.path.join(C.ART, "kr_ohlcv_partial.parquet")

def main():
    px, link, corpus, events_by_nid = pickle.load(open(os.path.join(C.ART, "kr36_corpus.pkl"), "rb"))
    tickers = sorted(px.ticker.unique())
    d0 = pd.Timestamp(px.date.min()).strftime("%Y%m%d")
    d1 = pd.Timestamp(px.date.max()).strftime("%Y%m%d")
    print(f"universe {len(tickers)} tickers, {d0} -> {d1}", flush=True)

    frames, done = [], set()
    if os.path.exists(TMP):
        prev = pd.read_parquet(TMP)
        frames.append(prev); done = set(prev.ticker.unique())
        print(f"resuming: {len(done)} tickers already fetched", flush=True)

    t0 = time.time(); new = []
    for i, t in enumerate(tickers):
        if t in done:
            continue
        df = None
        for attempt in range(3):
            try:
                df = stock.get_market_ohlcv(d0, d1, t)
                break
            except Exception as e:
                print(f"  {t} attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(2.0 * (attempt + 1))
        if df is None:
            print(f"  !! skipping {t}", flush=True); continue
        if len(df) == 0:
            continue
        df = df.reset_index().rename(columns={
            "날짜": "date", "시가": "open", "고가": "high", "저가": "low",
            "종가": "close", "거래량": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["ticker"] = t
        new.append(df)
        if len(new) % 50 == 0:
            pd.concat(frames + new, ignore_index=True).to_parquet(TMP)
            el = time.time() - t0
            print(f"  {len(done) + len(new)}/{len(tickers)} tickers  ({el:.0f}s)", flush=True)
        time.sleep(0.15)

    out = pd.concat(frames + new, ignore_index=True)
    out.to_parquet(OUT)
    print(f"saved {OUT}  rows={len(out)}  tickers={out.ticker.nunique()}", flush=True)

if __name__ == "__main__":
    main()
