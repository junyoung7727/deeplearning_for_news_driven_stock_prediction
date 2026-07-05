"""
Stage 52 - EXACT (path-unambiguous) no-stop daily backtest of the UP5 signal on
the FULL 626-ticker universe.

Answers the user's challenge: "predicted-high >= open*1.05 with ~45-55% top-k
hit rate MUST be monetizable with proper entry/exit/sizing."

What earlier stages missed:
  (a) s49 resolved same-bar TP-vs-SL as SL-first (pessimistic); that single
      assumption was the whole -19.8% vs +9.4% spread.
  (b) s50/s51 tested only the 65-ticker 5-min subset where top-k hit ~30%
      (full-universe top-1 = 54.7%).
  (c) A stop converts recoverable dips into realized losses; E[ret|no-TP] under
      NO stop was never measured.

Policy simulated here needs NO intraday path assumption at all:
  buy top-k by score at the open (market), place a limit sell at open*1.05.
  Daily OHLC tells us EXACTLY whether it filled (high >= tp). If unfilled,
  exit at the close (EOD) or next open (NO). No stop-loss.
Sizing: equal weight 1/k of deployed capital per pick; report worst realized
single-day portfolio hit; risk-capped variant deploy=1/3 also reported.

Protocol: GBM scores are model-OOS from 2021-11-29 (fitted strictly before).
Strategy knobs (k, score floor, gap cap, exit) tuned on 2021-11-29..2024-06-01,
FROZEN, validated on 2024-06-01..2026-06-18. Costs 0.30% round trip
(stress 0.60%); strict-tick TP fill (high >= tp*1.002) sensitivity included.
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
import os, time, itertools, numpy as np, pandas as pd

ART = str(_ROOT / "artifacts")
SC = os.path.join(ART, "kr52_scores_full.parquet")
SC65 = os.path.join(ART, "kr50_scores_min5univ.parquet")
COST = 0.003; TP = 0.05
TUNE0, TUNE1 = pd.Timestamp("2021-11-29"), pd.Timestamp("2024-06-01")
VAL0, VAL1 = pd.Timestamp("2024-06-01"), pd.Timestamp("2026-06-19")
t0 = time.time()
def log(m): print(f"{m}  ({time.time()-t0:.0f}s)", flush=True)

def load():
    sc = pd.read_parquet(SC)
    sc["date"] = pd.to_datetime(sc.date)
    sc = sc[(sc.o > 0) & (sc.h > 0) & (sc.c > 0)].copy()
    sc = sc.sort_values(["ticker", "date"]).reset_index(drop=True)
    sc["next_o"] = sc.groupby("ticker").o.shift(-1)
    return sc

def picks_for(sc, k, floor, gcap, d0, d1):
    d = sc[(sc.date >= d0) & (sc.date < d1) & (sc.score >= floor) & (sc.gap <= gcap)]
    d = d.sort_values(["date", "score"], ascending=[True, False])
    return d.groupby("date").head(k).copy()

def outcomes(p, exit_mode, cost=COST, strict=False):
    tp_px = p.o * (1 + TP)
    hit = (p.h >= tp_px * (1.002 if strict else 1.0)).values
    if exit_mode == "NO":
        miss = np.where(p.next_o.notna(), p.next_o / p.o - 1.0, p.c / p.o - 1.0)
    else:
        miss = (p.c / p.o - 1.0).values
    gross = np.where(hit, TP, miss)
    return hit, gross - cost

def portfolio(p, net, k, d0, d1, deploy=1.0):
    """EW 1/k slots, unfilled slots cash. Daily compounding over all universe days."""
    pp = p.copy(); pp["net"] = net
    daily = pp.groupby("date").net.sum() / k * deploy
    days = pd.DatetimeIndex(np.sort(pd.unique(p.date)))  # candidate days
    idx = pd.date_range(d0, d1 - pd.Timedelta(days=1), freq="D")
    r = daily.reindex(idx).fillna(0.0)
    r = r[r.index.dayofweek < 5]
    eq = (1 + r).cumprod()
    yrs = max((d1 - d0).days / 365.25, 0.25)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    live = r[r != 0]
    sharpe = float(live.mean() / (live.std() + 1e-12) * np.sqrt(248)) if len(live) > 20 else -9.0
    mdd = float((eq / eq.cummax() - 1).min())
    return {"cagr": cagr, "sharpe": sharpe, "mdd": mdd, "final": float(eq.iloc[-1]),
            "worst_day": float(r.min()), "n": len(net)}, eq

def stats_line(tag, p, hit, net, res):
    miss_net = net[~hit]
    ev = float(net.mean()); evm = float(miss_net.mean()) if len(miss_net) else 0.0
    return (f"[{tag}] n={res['n']} hit={hit.mean():.3f} EV/trade={ev*100:+.2f}% "
            f"E[net|miss]={evm*100:+.2f}% | CAGR={res['cagr']*100:+.1f}% "
            f"Sharpe={res['sharpe']:+.2f} MDD={res['mdd']*100:.1f}% "
            f"final={res['final']:.3f} worstDay={res['worst_day']*100:+.2f}%")

def run(sc, k, floor, gcap, exit_mode, d0, d1, deploy=1.0, cost=COST, strict=False):
    p = picks_for(sc, k, floor, gcap, d0, d1)
    if len(p) < 50: return None
    hit, net = outcomes(p, exit_mode, cost, strict)
    res, eq = portfolio(p, net, k, d0, d1, deploy)
    return p, hit, net, res, eq

def main():
    sc = load()
    log(f"scored rows={len(sc)} tickers={sc.ticker.nunique()} "
        f"range {sc.date.min().date()}..{sc.date.max().date()}")

    # s48-consistency check: unconditional daily top-1 hit rate on full test
    p1 = picks_for(sc, 1, 0.0, 9.9, TUNE0, VAL1)
    hit1 = (p1.h >= p1.o * (1 + TP)).values
    log(f"consistency: top-1 (no filter) TP-touch rate {hit1.mean():.3f} "
        f"(s48 reported 0.547) n={len(p1)}")

    # ---- tune on 2021-11..2024-06
    print("\n=== TUNE 2021-11-29..2024-06-01 (grid, sorted by Sharpe) ===", flush=True)
    rows = []
    for exit_mode, k, floor, gcap in itertools.product(
            ("EOD", "NO"), (1, 2, 3), (0.0, 0.5, 0.65, 0.8), (0.02, 0.05, 9.9)):
        r = run(sc, k, floor, gcap, exit_mode, TUNE0, TUNE1)
        if r is None: continue
        p, hit, net, res, _ = r
        if res["n"] < 150: continue
        rows.append((res["sharpe"], res["cagr"], exit_mode, k, floor, gcap, res["n"],
                     hit.mean(), float(net.mean())))
    rows.sort(reverse=True)
    for sh, cg, em, k, fl, gc, n, h, ev in rows[:8]:
        print(f"  {em:>3s} k={k} floor={fl:.2f} gap<={gc:.2f} -> Sharpe={sh:+.2f} "
              f"CAGR={cg*100:+.1f}% n={n} hit={h:.3f} EV={ev*100:+.2f}%", flush=True)
    if not rows:
        print("no tune config with enough trades", flush=True); return
    _, _, em, k, fl, gc, *_ = rows[0]
    print(f"\nFROZEN config: exit={em} k={k} floor={fl} gap<={gc}", flush=True)

    # ---- validation
    print("\n=== VALIDATION 2024-06-01..2026-06-18 (frozen) ===", flush=True)
    p, hit, net, res, eq = run(sc, k, fl, gc, em, VAL0, VAL1)
    print(stats_line("VAL frozen", p, hit, net, res), flush=True)
    eq.to_csv(os.path.join(ART, "kr52_equity_val.csv"))
    # robustness on validation
    for tag, kw in [("cost 0.60%", dict(cost=0.006)),
                    ("strict tick fill", dict(strict=True)),
                    ("deploy 1/3 (risk-capped)", dict(deploy=1/3.0))]:
        r = run(sc, k, fl, gc, em, VAL0, VAL1, **kw)
        if r: print(stats_line(f"VAL {tag}", r[0], r[1], r[2], r[3]), flush=True)
    # a-priori default (no tuning): k=3 floor=0.65 gap<=0.02 EOD
    r = run(sc, 3, 0.65, 0.02, "EOD", VAL0, VAL1)
    if r: print(stats_line("VAL a-priori k3/0.65/gap2%/EOD", r[0], r[1], r[2], r[3]), flush=True)
    # neighbours of frozen config
    print("\n--- VAL neighbours (overfit check) ---", flush=True)
    for k2, fl2 in itertools.product((max(1, k - 1), k, k + 1), (fl,)):
        for em2 in ("EOD", "NO"):
            r = run(sc, k2, fl2, gc, em2, VAL0, VAL1)
            if r: print(stats_line(f"VAL k={k2} {em2}", r[0], r[1], r[2], r[3]), flush=True)

    # ---- per-trade bootstrap CI on validation EV
    rng = np.random.default_rng(13)
    bs = [net[rng.integers(0, len(net), len(net))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\nVAL EV/trade 95% CI: [{lo*100:+.2f}%, {hi*100:+.2f}%]  "
          f"(point {net.mean()*100:+.2f}%)", flush=True)
    # day-level Sharpe CI
    pp = p.copy(); pp["net"] = net
    dr = (pp.groupby("date").net.sum() / k).values
    bs2 = []
    for _ in range(2000):
        s = dr[rng.integers(0, len(dr), len(dr))]
        bs2.append(s.mean() / (s.std() + 1e-12) * np.sqrt(248))
    lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
    print(f"VAL Sharpe 95% CI: [{lo2:+.2f}, {hi2:+.2f}]", flush=True)

    # ---- quarterly
    pp["yq"] = pd.PeriodIndex(pp.date, freq="Q")
    qq = pp.groupby("yq").net.agg(["mean", "size"])
    print("VAL quarterly mean net/trade: " + " | ".join(
        f"{i}:{m*100:+.2f}%({int(s)})" for i, (m, s) in qq.iterrows()), flush=True)

    # ---- slices (validation): the user's scenario = gap buckets
    print("\n--- VAL slices ---", flush=True)
    pp["hit"] = hit
    gb = pd.cut(pp.gap, [-1, -0.02, 0.0, 0.02, 0.05, 1.0],
                labels=["gap<-2%", "-2..0%", "0..2%", "2..5%", ">5%"])
    for g, gg in pp.groupby(gb, observed=True):
        if len(gg) < 15: continue
        print(f"  {g:>8s}: n={len(gg):4d} hit={gg.hit.mean():.3f} "
              f"EV={gg.net.mean()*100:+.2f}%", flush=True)
    sb = pd.cut(pp.score, [0, 0.5, 0.65, 0.8, 0.9, 1.01])
    for g, gg in pp.groupby(sb, observed=True):
        if len(gg) < 15: continue
        print(f"  score {str(g):>12s}: n={len(gg):4d} hit={gg.hit.mean():.3f} "
              f"EV={gg.net.mean()*100:+.2f}%", flush=True)
    q = pp[~pp.hit].net
    print(f"  miss-day net quantiles 5/25/50/75/95: "
          + "/".join(f"{v*100:+.1f}%" for v in np.percentile(q, [5, 25, 50, 75, 95])), flush=True)

    # ---- reconcile with the 65-ticker 5-min subset (why s50/s51 looked dead)
    if os.path.exists(SC65):
        t65 = set(pd.read_parquet(SC65, columns=["ticker"]).ticker.unique())
        r = run(sc[sc.ticker.isin(t65)], k, fl, gc, em, VAL0, VAL1)
        if r: print("\n" + stats_line("VAL restricted to 65-ticker 5-min subset",
                                      r[0], r[1], r[2], r[3]), flush=True)

    log("s52 done")

if __name__ == "__main__":
    main()
