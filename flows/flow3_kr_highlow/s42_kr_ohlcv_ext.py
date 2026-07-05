"""
Stage 42 - extended-history KR daily OHLCV + market cap via pykrx (2015->2026)
for the scaling experiment (s43). Universe = the 626 tickers with 2024-26 price
data (SURVIVORSHIP-BIASED for earlier years - documented; acceptable for a
capacity study, not for a live backtest).

Outputs: artifacts/kr_ohlcv_ext.parquet   (ticker,date,open,high,low,close,volume)
         artifacts/kr_cap_ext.parquet     (ticker,date,market_cap)
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

D0, D1 = "20150101", "20260618"
OUT_P = os.path.join(C.ART, "kr_ohlcv_ext.parquet")
OUT_C = os.path.join(C.ART, "kr_cap_ext.parquet")
TMP_P = os.path.join(C.ART, "kr_ohlcv_ext_partial.parquet")
TMP_C = os.path.join(C.ART, "kr_cap_ext_partial.parquet")

def fetch(tickers, getter, rename, tmp, out, cols):
    frames, done = [], set()
    if os.path.exists(tmp):
        prev = pd.read_parquet(tmp)
        frames.append(prev); done = set(prev.ticker.unique())
        print(f"  resume {len(done)}", flush=True)
    new = []
    t0 = time.time()
    for i, t in enumerate(tickers):
        if t in done:
            continue
        df = None
        for a in range(3):
            try:
                df = getter(D0, D1, t); break
            except Exception as e:
                print(f"  {t} try{a+1}: {e}", flush=True); time.sleep(2 * (a + 1))
        if df is None or len(df) == 0:
            continue
        df = df.reset_index().rename(columns=rename)
        df = df[[c for c in cols if c in df.columns]]
        df["ticker"] = t
        new.append(df)
        if len(new) % 50 == 0:
            pd.concat(frames + new, ignore_index=True).to_parquet(tmp)
            print(f"  {len(done)+len(new)}/{len(tickers)} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.12)
    outdf = pd.concat(frames + new, ignore_index=True)
    outdf.to_parquet(out)
    print(f"saved {out} rows={len(outdf)} tickers={outdf.ticker.nunique()} "
          f"range={outdf.date.min()}..{outdf.date.max()}", flush=True)

def main():
    px, *_ = pickle.load(open(os.path.join(C.ART, "kr36_corpus.pkl"), "rb"))
    tickers = sorted(px.ticker.unique())
    print(f"{len(tickers)} tickers, {D0}->{D1}", flush=True)
    fetch(tickers, stock.get_market_ohlcv,
          {"날짜": "date", "시가": "open", "고가": "high", "저가": "low",
           "종가": "close", "거래량": "volume"},
          TMP_P, OUT_P, ["date", "open", "high", "low", "close", "volume"])
    fetch(tickers, stock.get_market_cap,
          {"날짜": "date", "시가총액": "market_cap"},
          TMP_C, OUT_C, ["date", "market_cap"])

if __name__ == "__main__":
    main()
