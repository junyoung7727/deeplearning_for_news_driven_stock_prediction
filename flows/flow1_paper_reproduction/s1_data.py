"""
Stage 1 - Data preparation.

Outputs (in artifacts/):
  prices.parquet : NVDA trading calendar with adjusted OHLC, close-to-close
                   return, binary label, and chronological split tag.
  news.parquet   : cleaned news titles for all CORPUS_TICKERS, one row per
                   (ticker, calendar-date, title), with lowercase token list.

Leak-free labelling convention (documented in report):
  For trading day D_i (i>=1):
     ret_i   = close(D_i)/close(D_{i-1}) - 1
     label_i = +1 if ret_i > 0 else -1            (rows with ret==0 dropped)
  Features for D_i may use ONLY news dated <= D_{i-1} (previous trading day),
  which is fully known before D_i. (See s5_features.py.)
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
import re, numpy as np, pandas as pd
import config as C

# ----------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z][a-z0-9'\-]*")

def clean_title(t: str) -> str:
    if t is None:
        return ""
    return _WS.sub(" ", str(t).replace("\n", " ")).strip()

def tokenize(t: str) -> list[str]:
    return [w for w in _TOK.findall(clean_title(t).lower()) if len(w) >= 2]

# ----------------------------------------------------------------------------
def build_prices() -> pd.DataFrame:
    df = pd.read_parquet(C.DAILY_PARQUET,
                         columns=["ticker", "trade_date", "open", "high",
                                  "low", "close", "volume"])
    df = df[df.ticker == C.TARGET].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["date"] = pd.to_datetime(df.trade_date)
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df.date >= pd.Timestamp(C.START_DATE)) &
            (df.date <= pd.Timestamp(C.END_DATE))].reset_index(drop=True)

    df["prev_close"] = df["close"].shift(1)
    df["prev_date"]  = df["date"].shift(1)
    df["ret"] = df["close"] / df["prev_close"] - 1.0
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    df = df[df.ret != 0.0].reset_index(drop=True)          # drop flat days
    df["label"] = np.where(df.ret > 0, 1, -1).astype(int)

    def split(d):
        if d < pd.Timestamp(C.DEV_START):  return "train"
        if d < pd.Timestamp(C.TEST_START): return "dev"
        return "test"
    df["split"] = df["date"].map(split)
    return df[["date", "prev_date", "open", "high", "low", "close",
               "prev_close", "ret", "label", "split"]]

def build_news() -> pd.DataFrame:
    df = pd.read_parquet(C.NEWS_PARQUET, columns=["ticker", "published_at", "title"])
    df = df[df.ticker.isin(C.CORPUS_TICKERS)].copy()
    # `published_at` is tz-naive UTC (proven in s22): convert to US/Eastern
    # before taking the calendar date, else evening-ET news (>= ~19:00 ET) is
    # dated one day late and every daily alignment downstream inherits the
    # shift. (Bug found in s22, fixed here; s37 realigns cached artifacts.)
    ts_et = (pd.to_datetime(df.published_at).dt.tz_localize("UTC")
               .dt.tz_convert("America/New_York").dt.tz_localize(None))
    df["date"] = ts_et.dt.normalize()
    df = df[(df.date >= C.START_DATE) & (df.date <= C.END_DATE)]
    df["title_clean"] = df.title.map(clean_title)
    df = df[df.title_clean.str.len() > 0]
    # drop exact duplicate (ticker, date, title) - removes repeated wire spam
    df = df.drop_duplicates(subset=["ticker", "date", "title_clean"])
    df["tokens"] = df.title_clean.map(tokenize)
    df = df[df.tokens.map(len) >= 2]
    df["emb_eligible"] = df.date < pd.Timestamp(C.EMB_TRAIN_END)
    return df[["ticker", "date", "published_at", "title_clean",
               "tokens", "emb_eligible"]].sort_values(["date"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    prices = build_prices()
    news   = build_news()
    prices.to_parquet(os.path.join(C.ART, "prices.parquet"))
    news.to_parquet(os.path.join(C.ART, "news.parquet"))

    print("PRICES", prices.shape)
    print(prices.split.value_counts())
    print("label balance by split:")
    print(prices.groupby("split").label.apply(lambda s: (s == 1).mean()).round(3))
    print("date range:", prices.date.min().date(), "->", prices.date.max().date())
    print()
    print("NEWS", news.shape, "| tickers:", news.ticker.nunique())
    print("news per ticker:\n", news.ticker.value_counts())
    nvda = news[news.ticker == C.TARGET]
    print("NVDA news rows:", len(nvda),
          "| emb-eligible:", int(news.emb_eligible.sum()),
          "| total tokens(corpus):", int(news.tokens.map(len).sum()))
    print("sample tokens:", nvda.tokens.iloc[0][:12])
