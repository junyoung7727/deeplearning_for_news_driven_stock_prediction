"""논문식 시장 시뮬레이션 (s8_simulate 실코드 사용)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .paths import ART, bootstrap


def _split_rows(samples: pd.DataFrame | None, split: str) -> pd.DataFrame:
    if samples is None:
        from .data import load_samples

        samples = load_samples()
    return samples[samples.split == split].reset_index(drop=True)


def paper_simulate(
    prob,
    samples: pd.DataFrame | None = None,
    split: str = "test",
    threshold: float | None = None,
) -> dict[str, Any]:
    bootstrap()
    from s8_simulate import simulate

    rows = _split_rows(samples, split)
    prob = np.asarray(prob, np.float64)
    if len(prob) != len(rows):
        raise ValueError(f"prob length {len(prob)} != {split} rows {len(rows)}")
    daily = simulate(prob, rows.open.values, rows.high.values, rows.low.values, rows.close.values, threshold=threshold)
    if threshold is None:
        n_trades = int(len(prob))
    else:
        n_trades = int(((prob > threshold) | (prob < 1 - threshold)).sum())
    return {
        "dates": pd.to_datetime(rows.date).values,
        "daily": daily,
        "total": float(daily.sum()),
        "n_trades": n_trades,
    }


def randomization(
    model_profit: float,
    samples: pd.DataFrame | None = None,
    split: str = "test",
    n: int = 1000,
):
    bootstrap()
    from s8_simulate import randomization_dist

    rows = _split_rows(samples, split)
    dist = randomization_dist(rows.open.values, rows.high.values, rows.low.values, rows.close.values, n=n)
    p = float((dist >= model_profit).mean())
    return dist, p


def load_teb_daily() -> pd.DataFrame:
    """s53에서 저장된, 사용자 학습 TEB-CNN 4-seed 앙상블의 test 추론/손익."""
    return pd.read_csv(ART / "s53_teb_daily_profit.csv", parse_dates=["date"])
