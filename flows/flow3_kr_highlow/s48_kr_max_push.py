"""
Stage 48 - maximum push on UP5/DN5 (remote GPU box).

Exploration (s48e) findings -> design:
  E3 market-day regime (lag1 autocorr +0.41, day-rate p10 3% -> p90 17%)
      -> cross-sectional market context features
  E4 ticker persistence (corr +0.56)          -> rolling per-ticker hit priors
  E5 big-cap base rates similar               -> scale universe 313 -> 626
     tickers, drop news-window requirement for the tabular learner (1.6M rows)
  E6 top-decile TP/FP separable by gap/volz/nov/sent -> meta-labeling stage
  E2 novelty adds signal inside high-vol      -> keep news aggregates + TF

Pipeline:
  1. news buckets (cached corpus/embs) -> overnight + info aggregates
  2. master table over ALL valid ticker-days 2015-2026 (~1.6M): 103 tabular
     features = 65 price-window aggs + 5 vol/gap + 12 news + 12 ticker
     (priors/shape/liquidity) + 9 market-day context
  3. GBM x 6 labels on full train; eval full test + small-cap event-window
     subset (s47-comparable) with date-cluster bootstrap CIs
  4. meta-labeling (UP5/DN5): expanding 3-fold OOF primary -> meta-GBM on
     top-20% candidates -> test re-rank
  5. TF on small-cap event-window tensors with market-enriched V token
     (14 -> 30 dims), 2 seeds; per-label ENS with GBM; final significance
Outputs: ~/dlfe/artifacts/kr48_probs.npz + s48.log
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
import os, sys, math, time, numpy as np, pandas as pd, torch

os.environ.setdefault("KR45_FEAT", os.path.join(os.path.expanduser("~"), "dlfe",
                      "artifacts", "kr46_features_bidirectional.npz"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s43_kr_scale_remote import build_corpus, build_w2v, build_event_emb, START, DATA
from s46_kr_bidirectional_rare import best_thresholds, NAMES
from s47_kr_ensemble_significance import (train_tf, date_codes, boot_mcc_ci,
                                          boot_rate_ci, eval_label, daily_topk)
from s45_kr_feature_ladder import DEV, log

ART = os.path.join(os.path.expanduser("~"), "dlfe", "artifacts")
SCORES = "/home/junyoung/bk_scores/bigkinds_finbert_scores.parquet"
SEED = 13
KS = (0.02, 0.03, 0.05)
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ news buckets
def build_buckets():
    link, corpus, events_by_nid = build_corpus()
    vocab, W = build_w2v(corpus)
    nid2emb = build_event_emb(events_by_nid, vocab, W)
    del corpus, events_by_nid
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    ohlcv = ohlcv[ohlcv.date >= pd.Timestamp(START)].sort_values(["ticker", "date"]).reset_index(drop=True)
    TD = np.sort(ohlcv.date.unique())
    open_ts = TD + np.timedelta64(9, "h")
    close_ts = TD + np.timedelta64(15 * 60 + 30, "m")
    le = link[link.news_id.isin(nid2emb)].copy()
    ts = le.ts.values.astype("datetime64[ns]")
    eff_i = np.searchsorted(open_ts, ts, side="left")
    info_i = np.searchsorted(close_ts, ts, side="left")
    valid = eff_i < len(TD)
    le = le[valid].reset_index(drop=True)
    ei, ii = eff_i[valid], info_i[valid]
    le["eff"] = TD[ei]
    le["info"] = TD[np.minimum(ii, len(TD) - 1)]
    le.loc[ii >= len(TD), "info"] = pd.NaT
    le["fresh"] = ei == ii
    le = le.sort_values(["ticker", "ts"]).reset_index(drop=True)
    E = np.stack([nid2emb[n] for n in le.news_id.values]).astype(np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    nov = np.ones(len(le), np.float32); n7 = np.zeros(len(le), np.float32)
    simfrac = np.zeros(len(le), np.float32)
    tsv = le.ts.values.astype("datetime64[ns]"); week = np.timedelta64(7, "D")
    pos = 0
    for t, g in le.groupby("ticker", sort=False):
        n = len(g); base = pos
        gts = tsv[base:base + n]; gE = En[base:base + n]; lo = 0
        for i in range(n):
            while gts[lo] < gts[i] - week: lo += 1
            if lo < i:
                sims = gE[lo:i] @ gE[i]
                nov[base + i] = 1.0 - float(sims.max())
                n7[base + i] = i - lo
                simfrac[base + i] = float((sims > 0.8).mean())
        pos += n
    le["nov"] = nov; le["n7"] = n7; le["simfrac"] = simfrac
    sc = pd.read_parquet(SCORES, columns=["news_id", "finbert_sentiment"])
    sc = sc[sc.news_id.isin(set(le.news_id.unique()))]
    le["sent"] = le.news_id.map(dict(zip(sc.news_id.values,
        sc.finbert_sentiment.values.astype(np.float32)))).fillna(0.0).astype(np.float32)
    del sc
    le["emb_i"] = np.arange(len(le))
    def bucket(df, key):
        out = {}
        for (t, d), g in df.dropna(subset=[key]).groupby(["ticker", key]):
            idx = g.emb_i.values
            s = g.sent.values
            out[(t, d)] = (E[idx].mean(0), float(s.mean()), float(np.abs(s).max()),
                           float(g.nov.values.mean()), float(g.nov.values.max()),
                           float(len(g)), float(np.log1p(g.n7.values).mean()),
                           float(g.simfrac.values.mean()), float(s.min()), float(s.max()))
        return out
    over = bucket(le[le.fresh], "eff"); info = bucket(le, "info")
    log(f"buckets over={len(over)} info={len(info)}")
    return ohlcv, TD, over, info

# ------------------------------------------------------------------ master table
def build_master(ohlcv, TD, over, info):
    capf = pd.read_parquet(os.path.join(DATA, "kr_market_cap_daily.parquet"))
    capf["date"] = pd.to_datetime(capf.trade_date)
    ov = capf.merge(ohlcv[["ticker", "date", "close"]], on=["ticker", "date"])
    ov = ov[(ov.market_cap > 0) & (ov.close > 0)]
    shares = (ov.market_cap / ov.close).groupby(ov.ticker).median()
    ohlcv = ohlcv.copy()
    ohlcv["pcap"] = ohlcv.close * ohlcv.ticker.map(shares)
    ohlcv["month"] = ohlcv.date.dt.strftime("%Y-%m")
    mcap = ohlcv[ohlcv.pcap.notna()].groupby(["ticker", "month"]).pcap.median().reset_index()
    med = mcap.groupby("month").pcap.median().rename("xmed").reset_index()
    mcap = mcap.merge(med, on="month")
    small_tm = {(r.ticker, r.month) for r in mcap[mcap.pcap <= mcap.xmed].itertuples()}

    td_pos = {d: i for i, d in enumerate(TD)}
    rows, psq = [], []
    for t, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        o = g.open.values.astype(float); h = g.high.values.astype(float)
        lo = g.low.values.astype(float); c = g.close.values.astype(float)
        v = g.volume.values.astype(float); gd = g.date.values
        n = len(g)
        hl = (h - lo) / np.where(c > 0, c, np.nan)
        r1 = np.concatenate([[0.0], c[1:] / c[:-1] - 1.0])
        gap = np.concatenate([[0.0], o[1:] / c[:-1] - 1.0])
        up5 = ((h >= o * 1.05) & (o > 0)).astype(np.float32)
        dn5 = ((lo <= o * 0.95) & (o > 0)).astype(np.float32)
        hit2 = ((h >= o * 1.02) & (o > 0)).astype(np.float32)
        vmed = pd.Series(v).rolling(60, min_periods=10).median().values
        with np.errstate(divide="ignore", invalid="ignore"):
            volz = np.log(np.where((v > 0) & (vmed > 0), v / vmed, 1.0))
        volz = np.clip(np.nan_to_num(volz), -3, 3)
        pday = np.stack([r1, np.nan_to_num(hl), gap, volz, hit2], 1).astype(np.float32)
        # ticker priors (shifted: known before open of day i)
        s_up = pd.Series(up5); s_dn = pd.Series(dn5)
        u120 = s_up.shift(1).rolling(120, min_periods=40).mean().values
        d120 = s_dn.shift(1).rolling(120, min_periods=40).mean().values
        u20 = s_up.shift(1).rolling(20, min_periods=10).mean().values
        d20 = s_dn.shift(1).rolling(20, min_periods=10).mean().values
        turn = pd.Series(v * c).shift(1).rolling(20, min_periods=10).mean().values
        for i in range(31, n):
            if o[i] <= 0 or v[i] <= 0 or c[i - 1] <= 0: continue
            gp = o[i] / c[i - 1] - 1.0
            if gp >= 0.295: continue
            di = gd[i]
            month = str(di)[:7]
            a = gd[i - 1]
            lo_p, hi_p = td_pos[a], td_pos[di]
            ods = [over[(t, TD[kk])] for kk in range(lo_p + 1, hi_p + 1) if (t, TD[kk]) in over]
            if ods:
                novf = (float(np.mean([x[3] for x in ods])), float(np.max([x[4] for x in ods])),
                        float(np.log1p(sum(x[5] for x in ods))), float(np.mean([x[6] for x in ods])),
                        float(np.mean([x[7] for x in ods])), 1.0,
                        float(np.mean([x[1] for x in ods])), float(np.min([x[8] for x in ods])),
                        float(np.max([x[9] for x in ods])))
            else:
                novf = (0.,) * 5 + (0., 0., 0., 0.)
            n7d = n30d = 0.0; s7 = []
            has_any = bool(ods)
            for kk in range(1, 31):
                if hi_p - kk < 0: break
                x = info.get((t, TD[hi_p - kk]))
                if x is not None:
                    has_any = True
                    n30d += x[5]
                    if kk <= 7:
                        n7d += x[5]; s7.append(x[1])
            prev_rng = (h[i - 1] - lo[i - 1])
            open_pos = (o[i] - lo[i - 1]) / prev_rng if prev_rng > 0 else 0.5
            co_prev = (c[i - 1] - o[i - 1]) / o[i - 1] if o[i - 1] > 0 else 0.0
            ush = (h[i - 1] - max(o[i - 1], c[i - 1])) / c[i - 1] if c[i - 1] > 0 else 0.0
            lsh = (min(o[i - 1], c[i - 1]) - lo[i - 1]) / c[i - 1] if c[i - 1] > 0 else 0.0
            rows.append((t, di, month,
                         float(h[i] >= o[i] * 1.02), float(h[i] >= o[i] * 1.03), float(up5[i]),
                         float(lo[i] <= o[i] * 0.98), float(lo[i] <= o[i] * 0.97), float(dn5[i]),
                         gp, float(np.nanmean(hl[max(0, i - 20):i])),
                         float(np.std(r1[max(0, i - 20):i])), 
                         float(hl[i - 1]) if np.isfinite(hl[i - 1]) else 0.0, float(abs(r1[i - 1])),
                         *novf, n7d and float(np.log1p(n7d)), n30d and float(np.log1p(n30d)),
                         float(np.mean(s7)) if s7 else 0.0,
                         float(np.nan_to_num(u120[i], nan=0.0)), float(np.nan_to_num(d120[i], nan=0.0)),
                         float(np.nan_to_num(u20[i], nan=0.0)), float(np.nan_to_num(d20[i], nan=0.0)),
                         float(np.log(max(c[i - 1], 1.0))), float(np.log1p(max(turn[i], 0.0) if np.isfinite(turn[i]) else 0.0)),
                         float(np.clip(open_pos, -1, 2)), co_prev, ush, lsh,
                         float((t, month) in small_tm), has_any))
            psq.append(pday[i - 30:i])
    cols = (["ticker", "date", "month",
             "UP2", "UP3", "UP5", "DN2", "DN3", "DN5",
             "gap_open", "hl20", "std20", "hl_prev", "absr1_prev",
             "nov_mean", "nov_max", "ln_n_over", "ln_n7", "simfrac", "has_over",
             "sent_mean", "sent_min", "sent_max",
             "ln_n7d", "ln_n30d", "sent7d",
             "u120", "d120", "u20", "d20", "ln_close", "ln_turn20",
             "open_pos", "co_prev", "ush", "lsh", "is_small", "has_any"])
    df = pd.DataFrame(rows, columns=cols)
    PSEQ = np.stack(psq).astype(np.float32)
    log(f"master rows={len(df)} PSEQ={PSEQ.shape}")
    # market-day context from FULL universe
    day = df.groupby("date").agg(u=("UP5", "mean"), d=("DN5", "mean"),
                                 mv=("hl20", "mean"), mg=("gap_open", "mean"),
                                 mr=("absr1_prev", "mean"))
    fgap = df.assign(a=(df.gap_open.abs() > 0.02)).groupby("date").a.mean()
    day["fg"] = fgap
    day = day.sort_index()
    mk = pd.DataFrame({
        "ur1": day.u.shift(1), "dr1": day.d.shift(1),
        "ur5": day.u.shift(1).rolling(5, min_periods=2).mean(),
        "dr5": day.d.shift(1).rolling(5, min_periods=2).mean(),
        "mvol1": day.mv.shift(1), "mret1": day.mr.shift(1),
        "mgap_t": day.mg, "fgap2_t": day.fg,
        "breadth1": day.u.shift(1).rolling(3, min_periods=1).mean()})
    df = df.merge(mk.reset_index(), on="date", how="left")
    for c_ in mk.columns: df[c_] = df[c_].fillna(0.0)
    return df, PSEQ

def pseq_aggs(P):
    feats = []
    for w in (5, 10, 30):
        seg = P[:, -w:, :]
        feats += [seg.mean(1), seg.std(1), seg.max(1), seg.min(1)]
    feats.append(P[:, -1, :])
    return np.concatenate(feats, 1).astype(np.float32)

TABCOLS = (["gap_open", "hl20", "std20", "hl_prev", "absr1_prev",
            "nov_mean", "nov_max", "ln_n_over", "ln_n7", "simfrac", "has_over",
            "sent_mean", "sent_min", "sent_max", "ln_n7d", "ln_n30d", "sent7d",
            "u120", "d120", "u20", "d20", "ln_close", "ln_turn20",
            "open_pos", "co_prev", "ush", "lsh", "is_small",
            "ur1", "dr1", "ur5", "dr5", "mvol1", "mret1", "mgap_t", "fgap2_t", "breadth1"])

def main():
    ohlcv, TD, over, info = build_buckets()
    df, PSEQ = build_master(ohlcv, TD, over, info)
    Y = df[NAMES].values.astype(np.float32)
    DT = df.date.values.astype("datetime64[ns]")
    Xtab = np.concatenate([df[TABCOLS].values.astype(np.float32), pseq_aggs(PSEQ)], 1)
    Xtab = np.nan_to_num(Xtab, nan=0.0, posinf=0.0, neginf=0.0)
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[DT[trN] >= dsplit]; trN0 = trN[DT[trN] < dsplit]
    log(f"X {Xtab.shape} train {len(trN0)} dev {len(dvN)} test {len(teN)}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    def gbm_fit(idx, y, w, seed=SEED, max_iter=500):
        m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08,
            max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=seed)
        m.fit(Xtab[idx], y[idx], sample_weight=np.where(y[idx] == 1, w, 1.0))
        return m
    # 1) primary GBM x 6
    g_dv = np.zeros((len(dvN), 6), np.float32); g_te = np.zeros((len(teN), 6), np.float32)
    models = []
    for j, nm in enumerate(NAMES):
        y = Y[:, j]; pos = y[trN0].mean()
        w = float(np.clip((1 - pos) / max(pos, 1e-4), 1, 25))
        t0 = time.time()
        m = gbm_fit(trN0, y, w)
        g_dv[:, j] = m.predict_proba(Xtab[dvN])[:, 1]
        g_te[:, j] = m.predict_proba(Xtab[teN])[:, 1]
        models.append(m)
        log(f"GBM {nm} iters={m.n_iter_} ({time.time()-t0:.0f}s)")
    th_g, mcc_gdv = best_thresholds(Y[dvN], g_dv)
    log("GBM dev MCC " + " ".join(f"{n}={m:+.3f}" for n, m in zip(NAMES, mcc_gdv)))

    DT_te = DT[teN]
    uniq, codes = date_codes(DT, teN); D = len(uniq)
    sm_te = df.is_small.values[teN] > 0
    ha_te = df.has_any.values[teN] > 0
    sub = np.where(sm_te & ha_te)[0]
    print(f"\n=== s48 GBM full-universe test (n={len(teN)}, days={D}) ===", flush=True)
    for j, nm in enumerate(NAMES):
        eval_label(f"GBM {nm}", Y[teN, j], g_te[:, j], th_g[j], DT_te, codes, D)
    print(f"\n=== s48 GBM small-cap event-window subset (n={len(sub)}, s47-comparable) ===", flush=True)
    uniq2, codes2 = np.unique(DT_te[sub], return_inverse=True)
    for j, nm in enumerate(NAMES):
        if nm in ("UP5", "DN5"):
            eval_label(f"GBMsub {nm}", Y[teN[sub], j], g_te[sub, j], th_g[j],
                       DT_te[sub], codes2, len(uniq2))
    print("\n--- GBM daily cross-sectional selection (full universe) ---", flush=True)
    for j, nm in enumerate(NAMES):
        if nm in ("UP5", "DN5"):
            daily_topk(f"GBM {nm}", Y[teN, j], g_te[:, j], DT_te)

    # 2) meta-labeling UP5/DN5
    print("\n=== meta-labeling (expanding OOF, top-20% candidates) ===", flush=True)
    m_te = {}
    for j, nm in ((2, "UP5"), (5, "DN5")):
        y = Y[:, j]
        tr_dates = np.unique(DT[trN])
        folds = np.array_split(tr_dates, 4)
        oof_idx, oof_p = [], []
        for fi in range(1, 4):
            fit_idx = trN[np.isin(DT[trN], np.concatenate(folds[:fi]))]
            pred_idx = trN[np.isin(DT[trN], folds[fi])]
            pos = y[fit_idx].mean(); w = float(np.clip((1 - pos) / max(pos, 1e-4), 1, 25))
            mm = gbm_fit(fit_idx, y, w, max_iter=300)
            oof_idx.append(pred_idx); oof_p.append(mm.predict_proba(Xtab[pred_idx])[:, 1])
        oof_idx = np.concatenate(oof_idx); oof_p = np.concatenate(oof_p)
        qcut = np.quantile(oof_p, 0.8)
        cand = oof_p >= qcut
        Xm = np.column_stack([oof_p[cand], Xtab[oof_idx[cand]]])
        ym = y[oof_idx[cand]]
        pos = ym.mean(); wm = float(np.clip((1 - pos) / max(pos, 1e-4), 1, 15))
        meta = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
            max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=SEED)
        meta.fit(Xm, ym, sample_weight=np.where(ym == 1, wm, 1.0))
        # test: primary trained on full train (reuse models[j] - trained on trN0; ok)
        p_prim = g_te[:, j]
        qte = np.quantile(p_prim, 0.8)
        ct = p_prim >= qte
        p_meta = p_prim.copy()
        p_meta[ct] = meta.predict_proba(np.column_stack([p_prim[ct], Xtab[teN[ct]]]))[:, 1]
        p_meta[~ct] = 0.0
        m_te[nm] = p_meta
        log(f"meta {nm}: cand oof n={cand.sum()} hit={ym.mean():.3f}")
        daily_topk(f"META {nm}", Y[teN, j], p_meta, DT_te)
        # top-decile compare within candidates
        n10 = len(teN) // 10
        base_sel = np.argsort(-p_prim)[:n10]; meta_sel = np.argsort(-p_meta)[:n10]
        print(f"   {nm} top10 primary={Y[teN[base_sel], j].mean():.3f} -> meta={Y[teN[meta_sel], j].mean():.3f}", flush=True)

    # 3) TF on small-cap event-window with enriched V
    if os.environ.get("S48_TF", "1") == "1":
        print("\n=== TF enriched-V (small-cap event-window) ===", flush=True)
        Z = np.load(os.path.join(ART, "kr46_features_bidirectional.npz"), allow_pickle=True)
        S46, M46, L46 = (Z[k].astype(np.float32) for k in ("S", "M", "L"))
        DT46 = Z["DT"].astype("datetime64[ns]")
        mask48 = (df.is_small.values > 0) & (df.has_any.values > 0)
        idx48 = np.where(mask48)[0]
        key48 = pd.MultiIndex.from_arrays([df.ticker.values[idx48], df.date.values[idx48]])
        if len(idx48) != len(DT46):
            log(f"row mismatch kr46={len(DT46)} vs s48 subset={len(idx48)} - aligning by position date-check")
        nmin = min(len(idx48), len(DT46))
        ok = (DT46[:nmin] == df.date.values[idx48[:nmin]]).mean()
        log(f"date alignment agreement {ok:.4f} on {nmin}")
        if ok > 0.999:
            sel = idx48[:nmin]
            MKCOLS = ["ur1", "dr1", "ur5", "dr5", "mvol1", "mret1", "mgap_t", "fgap2_t",
                      "breadth1", "u120", "d120", "u20", "d20", "ln_turn20", "open_pos", "co_prev"]
            Vold_cols = ["gap_open", "hl20", "std20", "hl_prev", "absr1_prev",
                         "nov_mean", "nov_max", "ln_n_over", "ln_n7", "simfrac", "has_over",
                         "sent_mean", "sent_min", "sent_max"]
            Vraw = df[Vold_cols + MKCOLS].values[sel].astype(np.float32)
            DTs = DT[sel]; Ys = Y[sel]
            dsel = np.sort(np.unique(DTs)); spl = dsel[int(len(dsel) * 0.6)]
            trS = np.where(DTs < spl)[0]; teS = np.where(DTs >= spl)[0]
            trdS = np.sort(np.unique(DTs[DTs < spl])); dspl = trdS[int(len(trdS) * 0.85)]
            dvS = trS[DTs[trS] >= dspl]; trS = trS[DTs[trS] < dspl]
            mu, sd = Vraw[trS].mean(0), Vraw[trS].std(0) + 1e-9
            Vz = ((Vraw - mu) / sd).astype(np.float32)
            T = (torch.from_numpy(S46[:nmin]), torch.from_numpy(M46[:nmin]),
                 torch.from_numpy(L46[:nmin]), torch.from_numpy(Vz),
                 torch.from_numpy(PSEQ[sel]))
            t_tr, t_dv, t_te = train_tf(T, Ys, trS, dvS, teS)
            th_t, mcc_tdv = best_thresholds(Ys[dvS], t_dv)
            log("TF dev MCC " + " ".join(f"{n}={m:+.3f}" for n, m in zip(NAMES, mcc_tdv)))
            # ensemble with GBM on same rows
            gsel_dv = np.zeros_like(t_dv); gsel_te = np.zeros_like(t_te)
            # map sel rows into dvN/teN spaces: recompute GBM probs directly
            for j in range(6):
                gsel_dv[:, j] = models[j].predict_proba(Xtab[sel[dvS]])[:, 1]
                gsel_te[:, j] = models[j].predict_proba(Xtab[sel[teS]])[:, 1]
            from sklearn.metrics import matthews_corrcoef
            uniqS, codesS = np.unique(DTs[teS], return_inverse=True)
            print(f"\n=== s48 FINAL ENS (small-cap event-window, n={len(teS)}) ===", flush=True)
            e_te = np.zeros_like(t_te)
            for j, nm in enumerate(NAMES):
                best_w, best_m = 0.0, -9.0
                for w in np.linspace(0, 1, 11):
                    bl = w * t_dv[:, j] + (1 - w) * gsel_dv[:, j]
                    th, _ = best_thresholds(Ys[dvS][:, [j]], bl[:, None])
                    pred = bl >= th[0]
                    if pred.any() and (~pred).any():
                        mm = matthews_corrcoef(Ys[dvS][:, j], pred.astype(int))
                        if mm > best_m: best_w, best_m = float(w), float(mm)
                bl_dv = best_w * t_dv[:, j] + (1 - best_w) * gsel_dv[:, j]
                e_te[:, j] = best_w * t_te[:, j] + (1 - best_w) * gsel_te[:, j]
                th, _ = best_thresholds(Ys[dvS][:, [j]], bl_dv[:, None])
                eval_label(f"ENS(w={best_w:.1f}) {nm}", Ys[teS, j], e_te[:, j], th[0],
                           DTs[teS], codesS, len(uniqS))
                if nm in ("UP5", "DN5"):
                    daily_topk(f"ENS {nm}", Ys[teS, j], e_te[:, j], DTs[teS])
            np.savez(os.path.join(ART, "kr48_probs.npz"), e_te=e_te, g_te=g_te,
                     y_sub=Ys[teS], dt_sub=DTs[teS].astype("datetime64[D]").astype(str),
                     y_full=Y[teN], dt_full=DT_te.astype("datetime64[D]").astype(str),
                     meta_up5=m_te.get("UP5"), meta_dn5=m_te.get("DN5"))
            log("saved kr48_probs.npz")
        else:
            log("TF SKIPPED - alignment failed")
    log("s48 done")

if __name__ == "__main__":
    main()
