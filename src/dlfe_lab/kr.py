"""Flow 3 (KR HIGH/LOW) 실데이터/사용자 산출물 로더."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .paths import ART


def load_kr_ohlcv() -> pd.DataFrame:
    df = pd.read_parquet(ART / "kr_ohlcv_ext.parquet")
    df["date"] = pd.to_datetime(df.date)
    return df


def exceedance_rates(df: pd.DataFrame, ks=(0.02, 0.03, 0.05)) -> dict[str, float]:
    d = df[(df.open > 0) & (df.high > 0) & (df.low > 0)]
    out: dict[str, float] = {}
    for k in ks:
        out[f"UP{int(round(k * 100))}%"] = float((d.high >= d.open * (1 + k)).mean())
    for k in ks:
        out[f"DN{int(round(k * 100))}%"] = float((d.low <= d.open * (1 - k)).mean())
    return out


def load_kr_scores() -> pd.DataFrame:
    df = pd.read_parquet(ART / "kr52_scores_full.parquet")
    df["date"] = pd.to_datetime(df.date)
    return df


def daily_topk_hit(scores: pd.DataFrame, k: int = 1, tp: float = 0.05) -> dict[str, Any]:
    d = scores[(scores.o > 0) & (scores.h > 0)].copy()
    d["hit"] = d.h >= d.o * (1 + tp)
    picks = d.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(k)
    return {
        "k": int(k),
        "tp": float(tp),
        "hit_rate": float(picks.hit.mean()),
        "base_rate": float(d.hit.mean()),
        "n_days": int(picks.date.nunique()),
        "n_picks": int(len(picks)),
    }


def load_equity_curves() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    kr50 = pd.read_csv(ART / "kr50_equity_oos.csv", parse_dates=["date"])
    out["kr50"] = kr50[["date", "equity"]]

    kr51 = pd.read_csv(ART / "kr51_equity_champion.csv", parse_dates=["date"])
    out["kr51"] = kr51[["date", "equity"]]

    # s52_daily_nostop.py는 이미 누적된 equity 시리즈(eq)를 저장한다.
    # 컬럼명이 'net'인 것은 cumprod 이전 daily 시리즈 이름이 남은 것일 뿐이므로
    # 다시 복리 누적하면 안 된다.
    kr52 = pd.read_csv(ART / "kr52_equity_val.csv")
    first = kr52.columns[0]
    date = pd.to_datetime(kr52[first], errors="coerce")
    if date.notna().mean() > 0.9:
        kr52["date"] = date
    else:
        kr52["date"] = pd.RangeIndex(len(kr52))
    kr52["equity"] = kr52.net.astype(float)
    out["kr52"] = kr52[["date", "equity"]]
    return out


def _load_json(name: str) -> dict[str, Any]:
    with open(ART / name, encoding="utf-8") as f:
        return json.load(f)


def load_survival() -> dict[str, Any]:
    return _load_json("s53_teb_survival.json")


def load_gap_results() -> dict[str, Any]:
    return _load_json("s54_kr_gap.json")


def load_xai() -> dict[str, Any]:
    return _load_json("s55_kr_teb_xai.json")
