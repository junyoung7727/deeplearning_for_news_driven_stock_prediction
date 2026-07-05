"""
Stage 50 - INTRADAY 5-min backtest of the UP5 strategy (local; KR 5-min bars).

Why: the daily-bar test could not order same-day TP vs stop (pessimistic =
catastrophic, optimistic = profitable). 5-min bars RESOLVE the ordering.

Strategy (user spec):
  - decide at 09:00 open using the s48 GBM UP5 score (>= score_min), gap<=gap_max
  - BUY at the session open (first 5-min bar open)
  - TP: sell ALL at open*1.05 the moment a 5-min high >= TP
  - SL: liquidate at open*(1-n) the moment a 5-min low <= SL; n OPTIMIZED
  - if neither by 15:30, exit at the last bar close
  - first-touch decided by walking bars in time; only a single bar straddling
    both barriers is ambiguous -> assume SL first (pessimistic, logged)
  - position sizing: pos_value = equity * 1% / n  (stop-out ~= 1% of equity)
  - full-portfolio equity backtest; optimize n (+k, score_min, gap) on the
    first half of the 5-min period, OOS on the second half.

Scores come from the remote s48 GBM (kr50_scores_min5univ.parquet); 5-min bars,
open, gap, and UP5 labels are derived locally. The GBM was fit on daily data
<=2021-08, so all 5-min dates (2022-11+) are out-of-sample for the signal.
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
import os, glob, json, time, itertools, numpy as np, pandas as pd
import config as C

D5 = os.environ.get("DLFE_MIN5_KR_DIR", os.path.join(C.DATA_ROOT, "prices", "fmp_5min"))
ART = str(_ROOT / "artifacts")
SCORES = os.path.join(ART, "kr50_scores_min5univ.parquet")
COST = 0.0033
RISK_FRAC = 0.01
START_EQ = 1.0
TP = 0.05
t0 = time.time()
def log(m): print(f"{m}  ({time.time()-t0:.0f}s)", flush=True)

def load_5min():
    fs = sorted(glob.glob(os.path.join(D5, "*.parquet")))
    out = {}
    for f in fs:
        df = pd.read_parquet(f)
        if not set(["symbol", "datetime", "open", "high", "low", "close"]).issubset(df.columns):
            continue
        t = str(df.symbol.iloc[0]).replace(".KS", "").replace(".KQ", "")
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df["d"] = df["datetime"].dt.normalize()
        out[t] = df
    log(f"loaded 5min tickers={len(out)}")
    return out

def precompute(bars, scores, n_sl):
    """for each (ticker,date) in scores with intraday bars, compute realized
    return with TP=+5%, SL=-n_sl, first-touch on 5-min path, else close."""
    recs = []
    sc_by_t = {t: g for t, g in scores.groupby("ticker")}
    for t, g in sc_by_t.items():
        b = bars.get(t)
        if b is None: continue
        bd = {pd.Timestamp(d).normalize(): x for d, x in b.groupby("d")}
        gg = g.copy(); gg["nd"] = pd.to_datetime(gg.date).dt.normalize()
        sc_map = dict(zip(gg.nd, gg.score)); gap_map = dict(zip(gg.nd, gg.gap))
        for day in gg.nd.unique():
            day = pd.Timestamp(day)
            x = bd.get(day)
            if x is None or len(x) < 2: continue
            o = float(x.open.iloc[0])
            if o <= 0: continue
            hi = x.high.values.astype(float); lo = x.low.values.astype(float)
            cl = x.close.values.astype(float)
            tp_px = o * (1 + TP); sl_px = o * (1 - n_sl)
            up5 = float(hi.max() >= tp_px)     # label: did high reach +5% intraday
            exit_px = cl[-1]; reason = 2; bar_hit = len(x)
            for bi in range(len(x)):
                lo_b, hi_b = lo[bi], hi[bi]
                s_touch = lo_b <= sl_px; t_touch = hi_b >= tp_px
                if s_touch and t_touch:        # same-bar straddle -> SL first
                    exit_px = sl_px; reason = 3; bar_hit = bi + 1; break
                if s_touch:
                    exit_px = sl_px; reason = 0; bar_hit = bi + 1; break
                if t_touch:
                    exit_px = tp_px; reason = 1; bar_hit = bi + 1; break
            recs.append((t, day, o, float(sc_map.get(day, np.nan)),
                         float(gap_map.get(day, np.nan)),
                         exit_px / o - 1.0, reason, up5, bar_hit))
    r = pd.DataFrame(recs, columns=["ticker", "date", "o", "score", "gap",
                                    "ret_gross", "reason", "up5", "bar_hit"])
    return r.dropna(subset=["score"])

def backtest(cand_all, n_sl, k, score_min, gap_max, t0d, t1d, label="", verbose=True,
             daily_risk_cap=None, pos_cap=0.20):
    """intraday portfolio: all picks open at 09:00, close same day. Sequential
    risk-based sizing under a cash constraint; daily compounding."""
    stop_dist = n_sl
    w = cand_all[(cand_all.date >= t0d) & (cand_all.date < t1d)]
    days = np.sort(w.date.unique())
    by_day = {d: g.sort_values("score", ascending=False) for d, g in w.groupby("date")}
    eq = START_EQ; curve = []; trades = []
    for d in days:
        curve.append((d, eq))
        g = by_day.get(d)
        if g is None: continue
        cand = g[(g.score >= score_min) & (g.gap <= gap_max)].head(k)
        cash_left = eq; risk_used = 0.0; day_pnl = 0.0
        for _, r in cand.iterrows():
            if daily_risk_cap is not None and risk_used + RISK_FRAC > daily_risk_cap + 1e-9: break
            pos_value = min(eq * RISK_FRAC / stop_dist, eq * pos_cap, cash_left)
            if pos_value <= 1e-6: break
            ret = float(r.ret_gross) - COST
            pnl = pos_value * ret
            day_pnl += pnl; cash_left -= pos_value; risk_used += RISK_FRAC
            trades.append({"date": d, "ticker": r.ticker, "ret": ret,
                           "reason": int(r.reason), "loss_vs_eq": pnl / max(eq, 1e-9),
                           "bar_hit": int(r.bar_hit)})
        eq += day_pnl
    eqs = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date")
    tr = pd.DataFrame(trades)
    if len(tr) == 0:
        if verbose: print(f"[{label}] no trades", flush=True)
        return {"sharpe": -9, "cagr": -9, "mdd": 0, "n": 0}, eqs, tr
    rets = eqs.equity.pct_change().dropna()
    yrs = max((eqs.index[-1] - eqs.index[0]).days / 365.25, 0.1)
    cagr = float(eqs.equity.iloc[-1] ** (1 / yrs) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(248))
    peak = eqs.equity.cummax(); mdd = float((eqs.equity / peak - 1).min())
    res = {"sharpe": sharpe, "cagr": cagr, "mdd": mdd, "n": len(tr),
           "final": float(eqs.equity.iloc[-1]), "win": float((tr.ret > 0).mean()),
           "tp_rate": float((tr.reason == 1).mean()), "worst": float(tr.loss_vs_eq.min()),
           "straddle": float((tr.reason == 3).mean())}
    if verbose:
        rc = {0: "sl", 1: "tp", 2: "close", 3: "straddle"}
        rd = {rc[k_]: round(v, 2) for k_, v in tr.reason.value_counts(normalize=True).items()}
        print(f"[{label}] n={res['n']} final={res['final']:.3f} CAGR={cagr*100:+.1f}% "
              f"Sharpe={sharpe:.2f} MDD={mdd*100:.1f}% win={res['win']:.3f} "
              f"tp_rate={res['tp_rate']:.3f} worstHit={res['worst']*100:.2f}%eq "
              f"straddle={res['straddle']:.3f} reasons={rd}", flush=True)
    return res, eqs, tr

def main():
    bars = load_5min()
    scores = pd.read_parquet(SCORES)
    scores["date"] = pd.to_datetime(scores.date)
    log(f"scores rows={len(scores)} tickers={scores.ticker.nunique()} "
        f"range {str(scores.date.min())[:10]}..{str(scores.date.max())[:10]}")
    # precompute per SL level
    grid_sl = [0.02, 0.03, 0.04, 0.05]
    cand = {n: precompute(bars, scores, n) for n in grid_sl}
    any_c = next(iter(cand.values()))
    log(f"candidates/day-rows={len(any_c)} up5_rate={any_c.up5.mean():.3f} "
        f"date {str(any_c.date.min())[:10]}..{str(any_c.date.max())[:10]}")
    dts = np.sort(any_c.date.unique()); mid = dts[len(dts) // 2]
    tr0, tr1 = dts[0], mid; te0, te1 = mid, dts[-1] + np.timedelta64(1, "D")
    log(f"intraday train {str(tr0)[:10]}..{str(mid)[:10]} test {str(mid)[:10]}..end")

    print("\n=== TRAIN optimization (Sharpe; TP=5% fixed; optimize n=SL) ===", flush=True)
    best = None
    for n_sl, k, sm, gm in itertools.product(grid_sl, [1, 2, 3], [0.5, 0.65, 0.8], [0.02, 1.0]):
        r = backtest(cand[n_sl], n_sl, k, sm, gm, tr0, tr1, verbose=False)
        r0 = r[0] if isinstance(r, tuple) else r
        if r0["n"] >= 80 and (best is None or r0["sharpe"] > best[0]["sharpe"]):
            best = (r0, (n_sl, k, sm, gm))
    b, (bn, bk, bsm, bgm) = best
    print(f"BEST train: SL={bn:.0%} k={bk} score>={bsm} gap<={bgm} -> "
          f"Sharpe={b['sharpe']:.2f} CAGR={b['cagr']*100:+.1f}% MDD={b['mdd']*100:.1f}% "
          f"n={b['n']} tp_rate={b['tp_rate']:.3f}", flush=True)

    print("\n=== TEST OOS (optimized) ===", flush=True)
    res, eqs, tr = backtest(cand[bn], bn, bk, bsm, bgm, te0, te1, label="OOS best")
    print("\n--- OOS robustness (SL sweep, best k/score/gap) ---", flush=True)
    for n_sl in grid_sl:
        backtest(cand[n_sl], n_sl, bk, bsm, bgm, te0, te1, label=f"SL={n_sl:.0%}")
    print("\n--- OOS k / score / gap sweep at best SL ---", flush=True)
    for k in (1, 2, 3):
        backtest(cand[bn], bn, k, bsm, bgm, te0, te1, label=f"k={k}")
    for sm in (0.5, 0.65, 0.8):
        backtest(cand[bn], bn, bk, sm, bgm, te0, te1, label=f"score>={sm}")
    for gm in (0.02, 1.0):
        backtest(cand[bn], bn, bk, bsm, gm, te0, te1, label=f"gap<={gm}")
    if isinstance((r := backtest(cand[bn], bn, bk, bsm, bgm, te0, te1, verbose=False)), tuple):
        _, eqs2, tr2 = r
        tr2["yq"] = pd.PeriodIndex(tr2.date, freq="Q")
        qq = tr2.groupby("yq").ret.agg(["mean", "size"])
        print("\nOOS quarterly mean net/trade: " + " | ".join(
            f"{i}:{m*100:+.2f}%({int(s)})" for i, (m, s) in qq.iterrows()), flush=True)
        print(f"OOS worst single-trade equity hit {tr2.loss_vs_eq.min()*100:.2f}% "
              f"| >1% eq losses {(tr2.loss_vs_eq<-0.01).sum()} "
              f"({(tr2.loss_vs_eq<-0.01).mean()*100:.2f}%)", flush=True)
        eqs2.to_csv(os.path.join(ART, "kr50_equity_oos.csv"))
    log("s50 done")

if __name__ == "__main__":
    main()
