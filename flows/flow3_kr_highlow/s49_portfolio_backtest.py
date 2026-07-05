"""
Stage 49 - portfolio backtest + optimization for the UP5 long strategy
(remote box, CPU). Replaces naive buy-open/close-exit.

User spec:
  - close-exit is a poor sell rule -> multi-day hold with TP / trailing-stop /
    max-hold, evaluated on the daily-bar path (pessimistic same-bar ordering).
  - size each position so a stop-out costs ~1% of TOTAL equity
    (risk-based: pos_value = equity * RISK_FRAC / stop_distance).
  - full-portfolio equity backtest across ALL names.
  - OPTIMIZE on train, OOS backtest on test (no test peeking).

Design (verifiable): precompute each candidate trade's EXIT outcome over its
hold window from the ticker's own daily path, then run a simple portfolio
accountant (cash, concurrency, daily-risk cap, risk-based sizing) over the
sorted candidates. Score = s48 primary GBM UP5 prob, rebuilt over full period.
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
import os, sys, time, itertools, numpy as np, pandas as pd

HOME = os.path.expanduser("~")
ART = os.path.join(HOME, "dlfe", "artifacts")
DATA = os.path.join(HOME, "dlfe", "data")
sys.path.insert(0, os.path.join(HOME, "dlfe", "code"))
os.environ.setdefault("KR45_FEAT", os.path.join(ART, "kr46_features_bidirectional.npz"))

COST = 0.0033          # round-trip fees+tax+slippage
RISK_FRAC = 0.01       # target: a stop-out ~= 1% of equity
START_EQ = 1.0
t0 = time.time()
def log(m): print(f"{m}  ({time.time()-t0:.0f}s)", flush=True)

# ---- rebuild master + GBM UP5 score over the full period ------------------
def build_scored():
    from s48_kr_max_push import build_buckets, build_master, pseq_aggs, TABCOLS
    from sklearn.ensemble import HistGradientBoostingClassifier
    ohlcv, TD, over, info = build_buckets()
    df, PSEQ = build_master(ohlcv, TD, over, info)
    Xtab = np.nan_to_num(np.concatenate(
        [df[TABCOLS].values.astype(np.float32), pseq_aggs(PSEQ)], 1),
        nan=0.0, posinf=0.0, neginf=0.0)
    DT = df.date.values.astype("datetime64[ns]")
    y = df["UP5"].values.astype(np.float32)
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    trFit = trN[DT[trN] < dsplit]
    pos = y[trFit].mean(); w = float(np.clip((1 - pos) / max(pos, 1e-4), 1, 25))
    gbm = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08,
        max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
        random_state=13)
    gbm.fit(Xtab[trFit], y[trFit], sample_weight=np.where(y[trFit] == 1, w, 1.0))
    df["score"] = gbm.predict_proba(Xtab)[:, 1].astype(np.float32)
    px = ohlcv[["ticker", "date", "open", "high", "low", "close", "volume"]].rename(
        columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "vol"})
    m = df.merge(px, on=["ticker", "date"], how="left")
    m = m.rename(columns={"gap_open": "gap"})
    log(f"scored rows={len(df)} split={str(split)[:10]} score_mean={df.score.mean():.3f}")
    return m[["ticker", "date", "o", "h", "l", "c", "gap", "vol", "score"]].copy(), split

# ---- precompute per-candidate exit outcomes (both fill assumptions) --------
def precompute_exits(scored, sl, tp, trail, H):
    """simulate hold up to H days on each ticker's own path. Computes BOTH
    fill orderings for same-bar TP+stop touch:
      pess = stop assumed first; opt = TP assumed first.
    Gap-through-stop at a later open always exits at that open (both modes).
    Returns scored with ret_pess/ret_opt/held/reason."""
    s = scored.sort_values(["ticker", "date"]).reset_index(drop=True)
    O, Hh, L, C = (s[x].values.astype(np.float64) for x in ("o", "h", "l", "c"))
    tick = s.ticker.values; n = len(s)
    rp = np.full(n, np.nan); ro = np.full(n, np.nan)
    held = np.zeros(n, np.int32); reason = np.full(n, 2, np.int8)
    uniq, first = np.unique(tick, return_index=True)
    order = np.argsort(first); first = first[order]; ends = np.append(first[1:], n)
    for f, e in zip(first, ends):
        for i in range(f, e):
            entry = O[i]
            if entry <= 0:
                rp[i] = ro[i] = 0.0; continue
            stop = entry * (1 + sl); tpp = entry * (1 + tp); hi = Hh[i]
            done = False
            for j in range(i, min(i + H, e)):
                o, h, l, c = O[j], Hh[j], L[j], C[j]; hd = j - i + 1
                if j > i and o <= stop:                       # gap through stop
                    rp[i] = ro[i] = o / entry - 1.0; held[i] = hd; reason[i] = 0; done = True; break
                s_touch = l <= stop; t_touch = h >= tpp
                if s_touch and t_touch:                        # same-bar both
                    rp[i] = stop / entry - 1.0                 # pess: stop first
                    ro[i] = tpp / entry - 1.0                  # opt : tp first
                    held[i] = hd; reason[i] = 3; done = True; break
                if s_touch:
                    rp[i] = ro[i] = stop / entry - 1.0; held[i] = hd; reason[i] = 0; done = True; break
                if t_touch:
                    rp[i] = ro[i] = tpp / entry - 1.0; held[i] = hd; reason[i] = 1; done = True; break
                if trail is not None:
                    hi = max(hi, h); stop = max(stop, hi * (1 + trail))
                if hd >= H:
                    rp[i] = ro[i] = c / entry - 1.0; held[i] = hd; reason[i] = 2; done = True; break
            if not done:
                j = min(i + H, e) - 1
                rp[i] = ro[i] = C[j] / entry - 1.0; held[i] = j - i + 1; reason[i] = 2
    s["ret_pess"] = rp; s["ret_opt"] = ro; s["held"] = held; s["reason"] = reason
    return s

# ---- portfolio accountant over precomputed trades -------------------------
def backtest(sx, params, t_start, t_end, label="", verbose=True, fill="pess"):
    k = params["k_pick"]; scut = params.get("score_min", 0.0)
    gmax = params.get("gap_max", 1.0); max_conc = params["max_conc"]
    drc = params.get("daily_risk_cap", None); sl = params["sl"]
    pcap = params.get("pos_cap", 0.10)            # <=10% equity/pos (gap-tail guard)
    retcol = "ret_pess" if fill == "pess" else "ret_opt"
    w = sx[(sx.date >= t_start) & (sx.date < t_end)].copy()
    days = np.sort(w.date.unique())
    by_day = {d: g.sort_values("score", ascending=False) for d, g in w.groupby("date")}
    stop_dist = -sl
    equity = START_EQ; cash = START_EQ
    free_on = {}                       # exit day-index -> list of (value, pnl)
    open_val = 0.0
    eq_curve = []; trades = []
    for di, d in enumerate(days):
        for val, pnl in free_on.pop(di, []):
            cash += val + pnl; open_val -= val
        equity = cash + open_val
        eq_curve.append((d, equity))
        g = by_day.get(d)
        if g is None: continue
        cand = g[(g.score >= scut) & (g.gap <= gmax)].head(k)
        risk_used = 0.0; opened = 0
        cur_open_count = sum(len(v) for kk, v in free_on.items() if kk > di)
        for _, r in cand.iterrows():
            if cur_open_count + opened >= max_conc: break
            if drc is not None and risk_used + RISK_FRAC > drc + 1e-9: break
            pos_value = min(equity * RISK_FRAC / stop_dist, equity * pcap, cash)
            if pos_value <= 1e-6: break
            ret = float(r[retcol]) - COST
            pnl = pos_value * ret
            exit_di = min(di + int(r.held), len(days) - 1)
            free_on.setdefault(exit_di, []).append((pos_value, pnl))
            cash -= pos_value; open_val += pos_value
            trades.append({"date": d, "ticker": r.ticker, "ret": ret,
                           "held": int(r.held), "value": pos_value,
                           "reason": int(r.reason), "loss_vs_eq": pnl / max(equity, 1e-9)})
            risk_used += RISK_FRAC; opened += 1
    for di in list(free_on):
        for val, pnl in free_on[di]:
            cash += val + pnl
    eqs = pd.DataFrame(eq_curve, columns=["date", "equity"]).set_index("date")
    tr = pd.DataFrame(trades)
    if len(eqs) < 5 or len(tr) == 0:
        if verbose: print(f"[{label}] no trades", flush=True)
        return {"sharpe": -9, "cagr": -9, "mdd": 0, "n": 0}, eqs, tr
    rets = eqs.equity.pct_change().dropna()
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    cagr = float(eqs.equity.iloc[-1] ** (1 / max(yrs, 0.1)) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(248))
    peak = eqs.equity.cummax(); mdd = float((eqs.equity / peak - 1).min())
    worst = float(tr.loss_vs_eq.min()); over1 = float((tr.loss_vs_eq < -0.01).mean())
    res = {"sharpe": sharpe, "cagr": cagr, "mdd": mdd, "n": len(tr),
           "final": float(eqs.equity.iloc[-1]), "win": float((tr.ret > 0).mean()),
           "worst_hit": worst, "frac_over_1pct": over1, "avg_hold": float(tr.held.mean())}
    if verbose:
        rc = {0: "stop", 1: "tp", 2: "time", 3: "both"}
        rd = {rc[k_]: round(v, 2) for k_, v in tr.reason.value_counts(normalize=True).items()}
        print(f"[{label}] n={res['n']} final={res['final']:.3f} CAGR={cagr*100:+.1f}% "
              f"Sharpe={sharpe:.2f} MDD={mdd*100:.1f}% win={res['win']:.3f} "
              f"avgHold={res['avg_hold']:.1f} worstHit={worst*100:.2f}%eq over1%={over1:.3f} "
              f"reasons={rd}", flush=True)
    return res, eqs, tr

def main():
    scored, split = build_scored()
    scored["date"] = pd.to_datetime(scored.date)
    dates = np.sort(scored.date.unique())
    tr_start, tr_end = dates[0], split
    te_start, te_end = split, dates[-1] + np.timedelta64(1, "D")
    log(f"train {str(tr_start)[:10]}..{str(split)[:10]} test {str(split)[:10]}..end")

    # exit cache keyed by (sl, tp, trail, hold); grid also tunes score_min/k/gap
    grid = {"hold": [3, 5], "sl": [-0.03, -0.05], "tp": [0.03, 0.05],
            "trail": [None, -0.03], "k_pick": [1, 3], "gap_max": [0.02, 1.0],
            "score_min": [0.5, 0.65, 0.8]}
    keys = list(grid)
    print("\n=== TRAIN optimization (Sharpe, PESSIMISTIC fill, min 100 trades) ===", flush=True)
    exit_cache = {}; best = None
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals)); p["max_conc"] = p["k_pick"] * p["hold"]
        ek = (p["sl"], p["tp"], p["trail"], p["hold"])
        if ek not in exit_cache:
            exit_cache[ek] = precompute_exits(scored, *ek)
        r, _, _ = backtest(exit_cache[ek], p, tr_start, tr_end, verbose=False, fill="pess")
        if r["n"] >= 100 and (best is None or r["sharpe"] > best[0]["sharpe"]):
            best = (r, p, ek)
    if best is None:
        print("no train config cleared 100 trades", flush=True); log("s49 done"); return
    br, bp, bek = best
    print(f"BEST train {bp} -> Sharpe={br['sharpe']:.2f} CAGR={br['cagr']*100:+.1f}% "
          f"MDD={br['mdd']*100:.1f}% n={br['n']} worstHit={br['worst_hit']*100:.2f}% "
          f"over1%={br['frac_over_1pct']:.3f}", flush=True)

    print("\n=== TEST OOS - fill-assumption bracket (truth lies between) ===", flush=True)
    sxb = exit_cache[bek]
    _, eqs_p, tr = backtest(sxb, bp, te_start, te_end, label="OOS pess (stop-first)", fill="pess")
    backtest(sxb, bp, te_start, te_end, label="OOS opt  (tp-first) ", fill="opt")

    print("\n--- TEST robustness (pessimistic fill) ---", flush=True)
    for name, over in [("k1", {"k_pick": 1}), ("k3", {"k_pick": 3}),
                       ("score>=0.8", {"score_min": 0.8}), ("gap<=2%", {"gap_max": 0.02}),
                       ("noTrail", {"trail": None}), ("dailyRiskCap2%", {"daily_risk_cap": 0.02})]:
        p = dict(bp); p.update(over); p["max_conc"] = p["k_pick"] * p["hold"]
        ek = (p["sl"], p["tp"], p["trail"], p["hold"])
        if ek not in exit_cache: exit_cache[ek] = precompute_exits(scored, *ek)
        backtest(exit_cache[ek], p, te_start, te_end, label=name, fill="pess")

    if len(tr):
        tr["yq"] = pd.PeriodIndex(tr.date, freq="Q")
        qq = tr.groupby("yq").ret.agg(["mean", "size"])
        print("\nOOS(pess) quarterly mean net/trade: " + " | ".join(
            f"{i}:{m*100:+.2f}%({int(s)})" for i, (m, s) in qq.iterrows()), flush=True)
        print(f"OOS(pess) worst single-trade equity hit: {tr.loss_vs_eq.min()*100:.2f}% "
              f"| trades losing >1% eq: {(tr.loss_vs_eq<-0.01).sum()} "
              f"({(tr.loss_vs_eq<-0.01).mean()*100:.2f}%)  [1%-risk sizing check]", flush=True)
    eqs_p.to_csv(os.path.join(ART, "kr49_equity_test.csv"))
    log("s49 done")

if __name__ == "__main__":
    main()
