"""
Stage 53 - survival check for the NVDA TEB-CNN paper-style trading result.

Question: the news-only TEB-CNN row in artifacts/report.md made +$1,260 on the
TEST market simulation. Does that survive honest statistical validation, or is it
post-hoc selection noise?

Checks:
  1. Reproduce TEB-CNN 4-seed ensemble probabilities/profit.
  2. Higher-resolution random long/short randomization test.
  3. Test-day bootstrap CI and subperiod stability.
  4. DEV-selected threshold test versus TEST-peek threshold curve.
  5. Multiple-testing correction using the existing paper-model matrix results.
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

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as C
from s6_models import DenseModel, fit, predict, metrics
from s8_simulate import simulate, _trade

N_BOOT = 50_000
N_RAND = 100_000
THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.01), 2)


def idx_of(samp: pd.DataFrame, split: str) -> np.ndarray:
    return np.where(samp.split.values == split)[0].astype(np.int64)


def dense_inputs(tensors):
    return lambda ix: (tensors[0][ix], tensors[1][ix], tensors[2][ix])


def simulate_np(prob, o, h, l, c, threshold=None):
    return simulate(prob, o, h, l, c, threshold=threshold)


def random_profit_dist(o, h, l, c, n=N_RAND, seed=C.SEED + 5300):
    rng = np.random.default_rng(seed)
    n_days = len(o)
    dirs = rng.integers(0, 2, size=(n, n_days), dtype=np.int8) * 2 - 1
    long_profit = C.SIM_CAPITAL * (np.where(h >= o * (1 + C.SIM_TAKEPROFIT), o * (1 + C.SIM_TAKEPROFIT), c) / o - 1.0)
    short_cover = np.where(l <= o * (1 - C.SIM_COVER), o * (1 - C.SIM_COVER), c)
    short_profit = C.SIM_CAPITAL * (1.0 - short_cover / o)
    prof = np.where(dirs == 1, long_profit[None, :], short_profit[None, :]).sum(axis=1)
    return prof


def bootstrap_total(prof, n=N_BOOT, seed=C.SEED + 5301):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(prof), size=(n, len(prof)))
    return prof[idx].sum(axis=1)


def summarize_strategy(name, prob, samp, ix, threshold=None):
    o = samp.open.values[ix]
    h = samp.high.values[ix]
    l = samp.low.values[ix]
    c = samp.close.values[ix]
    dts = pd.to_datetime(samp.date.values[ix])
    prof = simulate_np(prob, o, h, l, c, threshold=threshold)
    active = prof != 0
    if threshold is None:
        trades = len(prof)
    else:
        trades = int(active.sum())
    total = float(prof.sum())
    mean = float(prof[active].mean()) if active.any() else 0.0
    std = float(prof[active].std(ddof=1)) if active.sum() > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(252)) if std > 0 else 0.0
    boot = bootstrap_total(prof)
    ci = np.quantile(boot, [0.025, 0.5, 0.975])
    p_nonpos = float((boot <= 0).mean())
    # subperiods
    sub = []
    df = pd.DataFrame({"date": dts, "profit": prof})
    for year, g in df.groupby(df.date.dt.year):
        sub.append({"period": str(int(year)), "n": int(len(g)), "profit": float(g.profit.sum())})
    half = len(df) // 2
    sub_halves = [
        {"period": "first_half", "n": int(half), "profit": float(df.profit.iloc[:half].sum())},
        {"period": "second_half", "n": int(len(df) - half), "profit": float(df.profit.iloc[half:].sum())},
    ]
    return {
        "name": name,
        "threshold": threshold,
        "n_days": int(len(ix)),
        "n_trades": trades,
        "profit_total": total,
        "profit_per_trade": mean,
        "trade_sharpe_like": sharpe,
        "bootstrap_profit_ci95": [float(x) for x in ci],
        "bootstrap_prob_profit_le_0": p_nonpos,
        "yearly_profit": sub,
        "half_profit": sub_halves,
        "daily_profit": prof,
    }


def threshold_sweep(prob, samp, ix):
    o = samp.open.values[ix]
    h = samp.high.values[ix]
    l = samp.low.values[ix]
    c = samp.close.values[ix]
    rows = []
    for th in THRESHOLDS:
        prof = simulate_np(prob, o, h, l, c, threshold=float(th))
        rows.append({
            "threshold": float(th),
            "profit": float(prof.sum()),
            "trades": int((prof != 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    art = Path(C.ART)
    samp = pd.read_parquet(art / "samples.parquet")
    y_pm1 = samp.label.values.astype(int)
    y01 = ((y_pm1 + 1) // 2).astype(int)
    tr, dv, te = idx_of(samp, "train"), idx_of(samp, "dev"), idx_of(samp, "test")

    d = np.load(art / "feats_TEB.npz")
    tensors = (torch.from_numpy(d["short"]), torch.from_numpy(d["mid"]), torch.from_numpy(d["long"]))
    inputs = dense_inputs(tensors)
    dim = int(d["short"].shape[1])

    print(f"samples train={len(tr)} dev={len(dv)} test={len(te)} TEB_dim={dim}", flush=True)
    dps, tps, seed_rows = [], [], []
    for s in range(4):
        torch.manual_seed(C.SEED + s)
        np.random.seed(C.SEED + s)
        model = DenseModel(dim, nn_only=False)
        model, dev_mcc = fit(model, inputs, y01, tr, dv, seed=C.SEED + s)
        dp = predict(model, inputs, dv)
        tp = predict(model, inputs, te)
        dps.append(dp)
        tps.append(tp)
        prof = simulate_np(tp, samp.open.values[te], samp.high.values[te], samp.low.values[te], samp.close.values[te]).sum()
        seed_rows.append({"seed": int(s), "dev_mcc": float(dev_mcc), "test_profit_always": float(prof)})
        print(f"seed{s} dev_mcc={dev_mcc:+.4f} test_profit_always={prof:+.2f}", flush=True)

    dp = np.mean(dps, axis=0)
    tp = np.mean(tps, axis=0)
    dev_acc, dev_mcc = metrics(y_pm1[dv], dp)
    test_acc, test_mcc = metrics(y_pm1[te], tp)
    print(f"ensemble dev_acc={dev_acc:.4f} dev_mcc={dev_mcc:+.4f} test_acc={test_acc:.4f} test_mcc={test_mcc:+.4f}", flush=True)

    always = summarize_strategy("TEB-CNN always", tp, samp, te, None)
    beta = summarize_strategy("TEB-CNN beta=0.70", tp, samp, te, C.SIM_THRESHOLD)

    rand = random_profit_dist(samp.open.values[te], samp.high.values[te], samp.low.values[te], samp.close.values[te])
    rand_p_always = float((rand >= always["profit_total"]).mean())
    rand_p_beta = float((rand >= beta["profit_total"]).mean())
    rand_summary = {"n": int(len(rand)), "mean": float(rand.mean()), "ci95": [float(x) for x in np.quantile(rand, [0.025, 0.975])], "p_ge_always": rand_p_always, "p_ge_beta": rand_p_beta}
    print(f"randomization n={len(rand)} mean={rand.mean():+.2f} p(always)={rand_p_always:.4f} p(beta)={rand_p_beta:.4f}", flush=True)

    dev_curve = threshold_sweep(dp, samp, dv)
    test_curve = threshold_sweep(tp, samp, te)
    dev_best = dev_curve.loc[dev_curve.profit.idxmax()].to_dict()
    test_at_dev_best = test_curve.loc[test_curve.threshold.eq(dev_best["threshold"])].iloc[0].to_dict()
    test_best = test_curve.loc[test_curve.profit.idxmax()].to_dict()
    print(f"dev-selected threshold={dev_best['threshold']:.2f} dev_profit={dev_best['profit']:+.2f}; test_profit_at_dev_best={test_at_dev_best['profit']:+.2f} trades={test_at_dev_best['trades']}", flush=True)
    print(f"test-peek best threshold={test_best['threshold']:.2f} test_profit={test_best['profit']:+.2f} trades={test_best['trades']}", flush=True)

    results_path = art / "results.json"
    existing = json.loads(results_path.read_text())
    sims = existing["simulation"]
    min_p = min(x["p_value"] for x in sims)
    teb_row = next(x for x in sims if x["model"] == "TEB-CNN")
    n_models = len(sims)
    bonf_11 = min(1.0, teb_row["p_value"] * n_models)
    bonf_22 = min(1.0, teb_row["p_value"] * n_models * 2)  # always + beta variants
    dev_best_model = max(existing["metrics"], key=lambda x: x["dev_mcc"])
    test_best_model = max(existing["metrics"], key=lambda x: x["test_mcc"])
    mult = {
        "n_models": n_models,
        "teb_existing_p": float(teb_row["p_value"]),
        "min_existing_p": float(min_p),
        "teb_bonferroni_11_models": bonf_11,
        "teb_bonferroni_22_model_strategy_variants": bonf_22,
        "dev_best_by_mcc": dev_best_model,
        "test_best_by_mcc": test_best_model,
        "dev_best_test_simulation": next(x for x in sims if x["model"] == dev_best_model["model"]),
    }

    out = {
        "seed_rows": seed_rows,
        "ensemble_metrics": {"dev_acc": dev_acc, "dev_mcc": dev_mcc, "test_acc": test_acc, "test_mcc": test_mcc},
        "always": {k: v for k, v in always.items() if k != "daily_profit"},
        "beta_0_70": {k: v for k, v in beta.items() if k != "daily_profit"},
        "randomization": rand_summary,
        "threshold_validation": {"dev_best": dev_best, "test_at_dev_best": test_at_dev_best, "test_peek_best": test_best},
        "multiple_testing": mult,
    }

    (art / "s53_teb_survival.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({
        "date": pd.to_datetime(samp.date.values[te]),
        "prob_up": tp,
        "profit_always": always["daily_profit"],
        "profit_beta_070": beta["daily_profit"],
    }).to_csv(art / "s53_teb_daily_profit.csv", index=False)
    dev_curve.to_csv(art / "s53_teb_dev_threshold_curve.csv", index=False)
    test_curve.to_csv(art / "s53_teb_test_threshold_curve.csv", index=False)
    print("saved artifacts/s53_teb_survival.json and daily/threshold CSVs", flush=True)


if __name__ == "__main__":
    main()
