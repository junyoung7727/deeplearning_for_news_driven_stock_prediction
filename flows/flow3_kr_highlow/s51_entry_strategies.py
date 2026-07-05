"""
Stage 51 - divergent REALISTIC entry strategies on 5-min bars (local).

s50 showed market-buy-at-open is negative OOS: the open captures no edge. Here
we search entry styles that a real trader would use, keeping the target faithful
to the signal (TP = session_open * 1.05, "predicted +5% high") and a stop n%
BELOW the actual entry (risk-based sizing = equity*1%/n on the true stop
distance). Same universe (65 KR names w/ 5-min), same score>=thr / gap filters,
same portfolio engine (s50.backtest). Optimize per family on the first-half,
report OOS on the second-half. Pick the best that is also LOGICAL.

Entry families (fill rule; look-ahead-free, uses only bars up to entry):
  A open           market buy at 09:00 open              (s50 baseline)
  B dip d%         limit buy at open*(1-d); fills iff a bar low<=that (buy dip)
  C delay HH:MM    market buy at first bar >= HH:MM       (skip open auction)
  D firstclose     buy at close of the first 5-min bar    (skip auction print)
  E orb R          opening-range breakout: stop-buy at max-high of first R min,
                   fills iff a later bar high>=range-high (momentum confirm)
  F dipreclaim d%  wait for dip to open*(1-d), then buy when price reclaims open
Exit (all): TP=open*1.05 (full), SL=entry*(1-n) first-touch on 5-min path
(same-bar straddle=SL first), else close. n optimized.
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
sys.path.insert(0, str(_ROOT / "flows" / "flow3_kr_highlow"))
from s50_intraday_5min_backtest import load_5min, backtest, ART, COST, TP

SCORES = os.path.join(ART, "kr50_scores_min5univ.parquet")
t0 = time.time()
def log(m): print(f"{m}  ({time.time()-t0:.0f}s)", flush=True)

def sim_day(o, h, l, c, mins, entry, param, sl_n):
    """returns (entered, entry_px, ret_gross, reason, bar_hit). reason 0 sl,1 tp,
    2 close,3 straddle. Look-ahead-free."""
    n = len(o); op = o[0]; tp_px = op * (1 + TP)
    e = -1; entry_px = np.nan
    if entry == "open":
        e, entry_px = 0, op
    elif entry == "firstclose":
        e, entry_px = 0, c[0]
    elif entry == "delay":
        idx = np.where(mins >= param)[0]
        if len(idx): e, entry_px = idx[0], o[idx[0]]
    elif entry == "dip":
        tgt = op * (1 - param)
        idx = np.where(l <= tgt)[0]
        if len(idx): e, entry_px = idx[0], tgt
    elif entry == "orb":
        win = mins < (mins[0] + param)
        if win.any() and (~win).any():
            rh = h[win].max()
            later = np.where((~win) & (h >= rh))[0]
            if len(later): e, entry_px = later[0], rh
    elif entry == "dipreclaim":
        tgt = op * (1 - param)
        dip = np.where(l <= tgt)[0]
        if len(dip):
            after = np.where((np.arange(n) >= dip[0]) & (h >= op))[0]
            if len(after): e, entry_px = after[0], op
    if e < 0:
        return (False, np.nan, 0.0, 4, 0)
    sl_px = entry_px * (1 - sl_n)
    exit_px = c[-1]; reason = 2; bar_hit = n - e
    for j in range(e, n):
        s_t = l[j] <= sl_px; t_t = h[j] >= tp_px
        if s_t and t_t:
            exit_px, reason, bar_hit = sl_px, 3, j - e + 1; break
        if s_t:
            exit_px, reason, bar_hit = sl_px, 0, j - e + 1; break
        if t_t:
            exit_px, reason, bar_hit = tp_px, 1, j - e + 1; break
    return (True, entry_px, exit_px / entry_px - 1.0, reason, bar_hit)

def build_cache(bars, scores, score_floor=0.5):
    """prebuild per-(ticker,day) bar arrays + candidate list once."""
    daycache = {}; cands = []
    sc = scores[scores.score >= score_floor]
    for t, g in sc.groupby("ticker"):
        b = bars.get(t)
        if b is None: continue
        bd = {pd.Timestamp(d).normalize(): x for d, x in b.groupby("d")}
        gg = g.copy(); gg["nd"] = pd.to_datetime(gg.date).dt.normalize()
        sc_map = dict(zip(gg.nd, gg.score)); gap_map = dict(zip(gg.nd, gg.gap))
        for day in gg.nd.unique():
            day = pd.Timestamp(day); x = bd.get(day)
            if x is None or len(x) < 3: continue
            arr = (x.open.values.astype(float), x.high.values.astype(float),
                   x.low.values.astype(float), x.close.values.astype(float),
                   (x["datetime"].dt.hour * 60 + x["datetime"].dt.minute).values)
            daycache[(t, day)] = arr
            cands.append((t, day, float(sc_map.get(day, np.nan)), float(gap_map.get(day, np.nan))))
    log(f"cache days={len(daycache)} candidates={len(cands)}")
    return daycache, cands

def precompute(daycache, cands, entry, param, sl_n):
    recs = []
    for t, day, sco, gp in cands:
        arr = daycache.get((t, day))
        if arr is None or np.isnan(sco): continue
        o, h, l, c, mins = arr
        ent, epx, ret, reason, bh = sim_day(o, h, l, c, mins, entry, param, sl_n)
        if not ent: continue
        recs.append((t, day, epx, sco, gp, ret, reason, 0.0, bh))
    return pd.DataFrame(recs, columns=["ticker", "date", "o", "score", "gap",
                                       "ret_gross", "reason", "up5", "bar_hit"])

def main():
    bars = load_5min()
    scores = pd.read_parquet(SCORES); scores["date"] = pd.to_datetime(scores.date)
    scores = scores[scores.date >= pd.Timestamp("2022-11-01")]
    dts = np.sort(scores.date.dt.normalize().unique()); mid = dts[len(dts) // 2]
    tr0, tr1 = dts[0], mid; te0, te1 = mid, dts[-1] + np.timedelta64(1, "D")
    log(f"train {str(tr0)[:10]}..{str(mid)[:10]} test {str(mid)[:10]}..end")
    daycache, cands = build_cache(bars, scores)

    families = {
        "A_open":       [("open", None)],
        "B_dip":        [("dip", d) for d in (0.005, 0.01, 0.02, 0.03)],
        "C_delay":      [("delay", m) for m in (570, 600, 630)],   # 09:30,10:00,10:30
        "D_firstclose": [("firstclose", None)],
        "E_orb":        [("orb", r) for r in (15, 30)],
        "F_dipreclaim": [("dipreclaim", d) for d in (0.01, 0.02)],
    }
    sls = [0.03, 0.04, 0.05]; ks = [1, 2, 3]; sms = [0.65, 0.8]; gm = 0.02
    champions = {}
    for fam, variants in families.items():
        best = None
        for (entry, param), sl_n in itertools.product(variants, sls):
            cand = precompute(daycache, cands, entry, param, sl_n)
            if len(cand) == 0: continue
            for k, sm in itertools.product(ks, sms):
                r = backtest(cand, sl_n, k, sm, gm, tr0, tr1, verbose=False)
                r0 = r[0]
                if r0["n"] >= 60 and (best is None or r0["sharpe"] > best["sh"]):
                    best = {"sh": r0["sharpe"], "cagr": r0["cagr"], "entry": entry,
                            "param": param, "sl": sl_n, "k": k, "sm": sm, "cand": cand}
        if best is None:
            print(f"[{fam}] no train config", flush=True); continue
        champions[fam] = best
        rtr = backtest(best["cand"], best["sl"], best["k"], best["sm"], gm, tr0, tr1, verbose=False)[0]
        print(f"\n[{fam}] TRAIN best: entry={best['entry']}({best['param']}) "
              f"SL={best['sl']:.0%} k={best['k']} score>={best['sm']} -> "
              f"train Sharpe={rtr['sharpe']:.2f} CAGR={rtr['cagr']*100:+.1f}%", flush=True)
        res, eqs, tr = backtest(best["cand"], best["sl"], best["k"], best["sm"], gm,
                                te0, te1, label=f"{fam} OOS")

    # champion of champions by OOS Sharpe
    print("\n=== OOS leaderboard (train-selected configs) ===", flush=True)
    board = []
    for fam, b in champions.items():
        res, eqs, tr = backtest(b["cand"], b["sl"], b["k"], b["sm"], gm, te0, te1, verbose=False)
        board.append((fam, res["sharpe"], res["cagr"], res["mdd"], res["n"],
                      res["tp_rate"], res["win"], res["worst"]))
    board.sort(key=lambda z: -z[1])
    for fam, sh, cg, md, n, tpr, wn, wr in board:
        print(f"  {fam:<14s} OOS Sharpe={sh:+.2f} CAGR={cg*100:+6.1f}% MDD={md*100:6.1f}% "
              f"n={n:4d} tp={tpr:.2f} win={wn:.2f} worstHit={wr*100:.2f}%", flush=True)
    if board:
        champ = board[0][0]; b = champions[champ]
        print(f"\n=== CHAMPION: {champ} -> full OOS detail + quarterly ===", flush=True)
        res, eqs, tr = backtest(b["cand"], b["sl"], b["k"], b["sm"], gm, te0, te1, label=f"{champ} OOS")
        tr["yq"] = pd.PeriodIndex(tr.date, freq="Q")
        qq = tr.groupby("yq").ret.agg(["mean", "size"])
        print("quarterly mean net/trade: " + " | ".join(
            f"{i}:{m*100:+.2f}%({int(s)})" for i, (m, s) in qq.iterrows()), flush=True)
        print(f"worst single-trade equity hit {tr.loss_vs_eq.min()*100:.2f}% | "
              f">1% eq losses {(tr.loss_vs_eq<-0.01).sum()} "
              f"({(tr.loss_vs_eq<-0.01).mean()*100:.2f}%)", flush=True)
        eqs.to_csv(os.path.join(ART, "kr51_equity_champion.csv"))
    log("s51 done")

if __name__ == "__main__":
    main()
