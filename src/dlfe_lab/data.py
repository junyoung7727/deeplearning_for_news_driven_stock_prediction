"""artifacts/ 실데이터 로더 + 요약 통계."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .paths import ART


def load_news() -> pd.DataFrame:
    return pd.read_parquet(ART / "news.parquet")


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(ART / "prices.parquet")


def load_events() -> pd.DataFrame:
    return pd.read_parquet(ART / "events.parquet")


def load_samples() -> pd.DataFrame:
    return pd.read_parquet(ART / "samples.parquet")


def load_event_ok() -> np.ndarray:
    return np.load(ART / "event_ok.npy")


def news_stats(news: pd.DataFrame) -> dict[str, Any]:
    tok_len = news.tokens.map(len)
    return {
        "rows": int(len(news)),
        "tickers": int(news.ticker.nunique()),
        "date_min": str(pd.Timestamp(news.date.min()).date()),
        "date_max": str(pd.Timestamp(news.date.max()).date()),
        "per_ticker": news.ticker.value_counts(),
        "tokens_mean": float(tok_len.mean()),
        "tokens_p50": float(tok_len.median()),
        "tokens_max": int(tok_len.max()),
        "emb_eligible_ratio": float(news.emb_eligible.mean()),
    }


def price_stats(prices: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(prices))}
    for split, g in prices.groupby("split"):
        out[split] = {
            "days": int(len(g)),
            "start": str(pd.Timestamp(g.date.min()).date()),
            "end": str(pd.Timestamp(g.date.max()).date()),
            "up_rate": float((g.label == 1).mean()),
            "ret_std": float(g.ret.std()),
        }
    return out


def event_stats(events: pd.DataFrame, ok: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(events)),
        "in_vocab": int(ok.sum()),
        "in_vocab_ratio": float(ok.mean()),
        "per_ticker": events.ticker.value_counts(),
        "emb_eligible_ratio": float(events.emb_eligible.mean()),
    }


def triple_str(events: pd.DataFrame, idx: int) -> str:
    row = events.iloc[idx]
    return f"({' '.join(row.o1)} | {' '.join(row.p)} | {' '.join(row.o2)})"
