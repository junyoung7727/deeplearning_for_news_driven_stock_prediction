"""
Stage 49 - REAL-TRADE viability check for the UP5 strategy (remote box, CPU).

Strategy under test (user-specified):
  when model predicts high >= open*1.05, BUY at open, place a +5% limit sell
  (TP); if TP not reached, exit at close. Variants with a -3% stop.

Verification axes:
  1. fills: TP filled iff high >= open*1.05; PESSIMISTIC variant: if both TP
     and SL touched intraday, assume SL fills FIRST (OHLC cannot order them).
  2. costs: commission 2x0.015% + sell tax 0.20% + slippage 2x0.05% ~ 0.33%
     round trip (parametrized COST).
  3. liquidity: 20d turnover filter; report pick turnover distribution ->
     realistic position size at 1% participation.
  4. gap conditioning ("open below X"): net PnL by open-gap bucket.
  5. risk anatomy: MAE (open->low), DN5 co-touch rate, quarterly stability,
     top-1/top-3 daily equity curves, Sharpe, max drawdown.

Alignment: rebuilds the s48 master row set (price-only; row inclusion never
depended on news) and asserts exact date/label agreement with kr48_probs.npz.
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
import os, time, numpy as np, pandas as pd

HOME = os.path.expanduser("~")
ART = os.path.join(HOME, "dlfe", "artifacts")
DATA = os.path.join(HOME, "dlfe", "data")
COST = 0.0033          # round-trip: fees 0.0003 + tax 0.0020 + slippage 0.0010
SLIP_IN = 0.0005       # applied inside COST split for entry/exit prices
t0 = time.time()
def log(m): print(f"{m}  ({time.time()-t0:.0f}s)", flush=True)

def build_rows():
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    ohlcv = ohlcv[ohlcv.date >= pd.Timestamp("2015-01-01")].sort_values(
        ["ticker", "date"]).reset_index(drop=True)
    capf = pd.read_parquet(os.path.join(DATA, "kr_market_cap_daily.parquet"))
    capf["date"] = pd.to_datetime(capf.trade_date)
    ov = capf.merge(ohlcv[["ticker", "date", "close"]], on=["ticker", "date"])
    ov = ov[(ov.market_cap > 0) & (ov.close > 0)]
    shares = (ov.market_cap / ov.close).groupby(ov.ticker).median()
    ohlcv["pcap"] = ohlcv.close * ohlcv.ticker.map(shares)
    ohlcv["month"] = ohlcv.date.dt.strftime("%Y-%m")
    mcap = ohlcv[ohlcv.pcap.notna()].groupby(["ticker", "month"]).pcap.median().reset_index()
    med = mcap.groupby("month").pcap.median().rename("xmed").reset_index()
    mcap = mcap.merge(med, on="month")
    small_tm = {(r.ticker, r.month) for r in mcap[mcap.pcap <= mcap.xmed].itertuples()}
    rows = []
    for t, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        o = g.open.values.astype(float); h = g.high.values.astype(float)
        lo = g.low.values.astype(float); c = g.close.values.astype(float)
        v = g.volume.values.astype(float); gd = g.date.values
        n = len(g)
        turn20 = pd.Series(v * c).shift(1).rolling(20, min_periods=10).mean().values
        for i in range(31, n):
            if o[i] <= 0 or v[i] <= 0 or c[i - 1] <= 0: continue
            gp = o[i] / c[i - 1] - 1.0
            if gp >= 0.295: continue
            rows.append((t, gd[i], o[i], h[i], lo[i], c[i], c[i - 1], gp,
                         float(turn20[i]) if np.isfinite(turn20[i]) else 0.0,
                         float((t, str(gd[i])[:7]) in small_tm)))
    df = pd.DataFrame(rows, columns=["ticker", "date", "o", "h", "l", "c",
                                     "cprev", "gap", "turn20", "is_small"])
    log(f"rows {len(df)}")
    return df

def simulate(df, score, label, k_pick, sl=None, min_turn=None, tag=""):
    """daily top-k picks; long at open, TP at +5% limit, else close; optional SL
    with pessimistic both-touch ordering. Returns per-trade frame."""
    d = df.copy()
    d["score"] = score
    if min_turn is not None:
        d = d[d.turn20 >= min_turn]
    d = d.sort_values(["date", "score"], ascending=[True, False])
    picks = d.groupby("date").head(k_pick).copy()
    o, h, lo, c = (picks[x].values for x in ("o", "h", "l", "c"))
    tp_price = o * 1.05
    tp_hit = h >= tp_price
    if sl is None:
        exit_px = np.where(tp_hit, tp_price, c)
        sl_fired = np.zeros(len(picks), bool)
    else:
        sl_price = o * (1 + sl)
        sl_touch = lo <= sl_price
        # pessimistic: if both touched, SL first
        sl_fired = sl_touch
        exit_px = np.where(sl_fired, sl_price, np.where(tp_hit, tp_price, c))
    gross = exit_px / o - 1.0
    net = gross - COST
    picks["gross"], picks["net"] = gross, net
    picks["tp"], picks["slf"] = tp_hit & ~sl_fired, sl_fired
    picks["mae"] = lo / o - 1.0
    picks["dn5_touch"] = lo <= o * 0.95
    r = picks.groupby("date").net.mean()  # equal-weight k picks per day
    eq = (1 + r).cumprod()
    peak = eq.cummax(); mdd = float((eq / peak - 1).min())
    ann = float(r.mean() * 248); sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(248))
    print(f"[{tag}] n={len(picks)} days={r.shape[0]} tp={picks.tp.mean():.3f} "
          f"sl={picks.slf.mean():.3f} | gross/trade={gross.mean()*100:+.3f}% "
          f"net/trade={net.mean()*100:+.3f}% med={np.median(net)*100:+.3f}% | "
          f"win={float((net>0).mean()):.3f} PF={float(net[net>0].sum()/max(-net[net<0].sum(),1e-9)):.2f} | "
          f"ann={ann*100:+.1f}% sharpe={sharpe:.2f} MDD={mdd*100:.1f}% | "
          f"MAE_med={np.median(picks.mae)*100:.2f}% dn5_touch={picks.dn5_touch.mean():.3f}",
          flush=True)
    # quarterly stability
    q = picks.copy(); q["yq"] = pd.PeriodIndex(pd.DatetimeIndex(q.date), freq="Q")
    qq = q.groupby("yq").net.agg(["mean", "size"])
    print("   quarters: " + " | ".join(f"{i}:{m*100:+.2f}%({int(s)})"
          for i, (m, s) in qq.iterrows()), flush=True)
    # gap conditioning
    gb = pd.cut(picks.gap, [-1, -0.02, 0.0, 0.02, 0.05, 1],
                labels=["<-2%", "-2..0", "0..2%", "2..5%", ">5%"])
    gg = picks.groupby(gb, observed=True).net.agg(["mean", "size"])
    print("   by gap: " + " | ".join(f"{i}:{m*100:+.2f}%({int(s)})"
          for i, (m, s) in gg.iterrows()), flush=True)
    return picks

def main():
    df = build_rows()
    Z = np.load(os.path.join(ART, "kr48_probs.npz"), allow_pickle=True)
    y_full, dt_full = Z["y_full"], Z["dt_full"]
    g_te, meta_up5 = Z["g_te"], Z["meta_up5"]
    DT = df.date.values.astype("datetime64[ns]")
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    teN = np.where(DT >= split)[0]
    dte = df.iloc[teN].copy()
    a1 = np.array_equal(DT[teN].astype("datetime64[D]").astype(str), dt_full)
    up5 = ((dte.h.values >= dte.o.values * 1.05)).astype(np.float32)
    a2 = np.array_equal(up5, y_full[:, 2])
    log(f"alignment dates={a1} labels={a2} n={len(dte)}")
    assert a1 and a2, "row alignment failed"
    dte = dte.reset_index(drop=True)

    print(f"\n=== UP5 long: buy open, +5% limit TP, else close (COST {COST*100:.2f}%) ===",
          flush=True)
    for k in (1, 3):
        simulate(dte, meta_up5, "UP5", k, sl=None, tag=f"META top{k} noSL")
    for k in (1, 3):
        simulate(dte, g_te[:, 2], "UP5", k, sl=None, tag=f"GBM  top{k} noSL")
    print("\n--- pessimistic -3% stop (both-touch => SL first) ---", flush=True)
    for k in (1, 3):
        simulate(dte, meta_up5, "UP5", k, sl=-0.03, tag=f"META top{k} SL-3")
    print("\n--- liquidity-filtered (turn20 >= 1e9 KRW) ---", flush=True)
    for k in (1, 3):
        p = simulate(dte, meta_up5, "UP5", k, sl=None, min_turn=1e9,
                     tag=f"META top{k} noSL liq")
        if k == 1:
            tq = p.turn20.quantile([0.1, 0.5, 0.9]).values / 1e8
            print(f"   pick turn20 quantiles (1e8 KRW): p10={tq[0]:.1f} med={tq[1]:.1f} "
                  f"p90={tq[2]:.1f}; 1% participation size med ~ {tq[1]*0.01*100:.1f}M KRW",
                  flush=True)
    print("\n--- entry-gap condition: only enter if open gap <= +2% ---", flush=True)
    m = dte.gap.values <= 0.02
    sc = np.where(m, meta_up5, -1.0)
    for k in (1, 3):
        simulate(dte, sc, "UP5", k, sl=None, tag=f"META top{k} gap<=2%")
    print("\n--- informational: DN5 short side (borrow NOT assumed available) ---",
          flush=True)
    meta_dn5 = Z["meta_dn5"]
    d = dte.copy(); d["score"] = meta_dn5
    picks = d.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(1)
    o, lo, c = picks.o.values, picks.l.values, picks.c.values
    tp = o * 0.95; hit = lo <= tp
    exitp = np.where(hit, tp, c)
    gross = 1.0 - exitp / o
    net = gross - COST - 0.0025  # borrow fee guess
    print(f"[DN5 short top1] n={len(picks)} tp={hit.mean():.3f} "
          f"gross={gross.mean()*100:+.3f}% net~={net.mean()*100:+.3f}%/trade", flush=True)
    log("s49 done")

if __name__ == "__main__":
    main()
