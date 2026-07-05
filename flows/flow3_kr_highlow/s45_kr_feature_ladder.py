"""
Stage 45 - INPUT-side ladder on the winning FUSED-TF architecture (remote GPU):
what happens to accuracy AND overfitting as we add information?

  B0  FUSED-TF (s44 #4 replication)       tokens [CLS; VOL5; S; M7; L30]
  B1  + PRICE sequence                    +30 price-day tokens
        per past trading day d-30..d-1: [ret_1d, (H-L)/C, gap, ln vol_z, hit2%]
  B2  + NOVELTY / burst features          VOL token 5 -> 12 dims
        per overnight news vs SAME-ticker news in prior 7 days:
        nov = 1 - max cos(NTN emb); aggregates: mean/max nov, ln(1+n_overnight),
        mean ln(1+n_prev7d), frac(sim>0.8), has_overnight flag
  B3  + FinBERT sentiment (precomputed, news_id-keyed, 22.2M coverage)
        day tokens 100 -> 102 [emb; mean sent; max |sent|],
        VOL token 12 -> 15 [+ overnight mean/min/max sent]

Protocol identical to s43/s44 (multi-task k=2/3/5, AdamW wd .05 decoupled,
cosine+warmup, label smoothing .05, news day-dropout .10 + noise .03, dev-MCC
early stop patience 8, 2 seeds). Report per variant: test acc / MCC / GAP
(train-test) per k + top-10% selective cell at k=2%.
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
import os, sys, math, time, copy, numpy as np, pandas as pd, torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s43_kr_scale_remote import (build_corpus, build_w2v, build_event_emb, log,
                                 MID_DAYS, LONG_DAYS, KS, D, START, DATA, ART)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
FEAT = os.environ.get("KR45_FEAT", os.path.join(ART, "kr45_features.npz"))
SCORES = "/home/junyoung/bk_scores/bigkinds_finbert_scores.parquet"
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ features
def build_features():
    if os.path.exists(FEAT):
        z = np.load(FEAT, allow_pickle=True)
        return {k: z[k] for k in z.files}
    link, corpus, events_by_nid = build_corpus()
    vocab, W = build_w2v(corpus)
    nid2emb = build_event_emb(events_by_nid, vocab, W)
    del corpus, events_by_nid

    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    ohlcv = ohlcv[ohlcv.date >= pd.Timestamp(START)]
    ohlcv = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)
    capf = pd.read_parquet(os.path.join(DATA, "kr_market_cap_daily.parquet"))
    capf["date"] = pd.to_datetime(capf.trade_date)
    ov = capf.merge(ohlcv[["ticker", "date", "close"]], on=["ticker", "date"])
    ov = ov[(ov.market_cap > 0) & (ov.close > 0)]
    shares = (ov.market_cap / ov.close).groupby(ov.ticker).median()
    ohlcv["pcap"] = ohlcv.close * ohlcv.ticker.map(shares)
    cap = ohlcv[ohlcv.pcap.notna()][["ticker", "date", "pcap"]].copy()
    cap["month"] = cap.date.dt.strftime("%Y-%m")
    mcap = cap.groupby(["ticker", "month"]).pcap.median().reset_index()
    med = mcap.groupby("month").pcap.median().rename("xmed").reset_index()
    mcap = mcap.merge(med, on="month")
    small_tm = {(r.ticker, r.month) for r in mcap[mcap.pcap <= mcap.xmed].itertuples()}

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
    log(f"usable links {len(le)}")

    # ---- novelty vs same-ticker news in prior 7 days --------------------
    nov = np.ones(len(le), np.float32)
    n7 = np.zeros(len(le), np.float32)
    simfrac = np.zeros(len(le), np.float32)
    tsv = le.ts.values.astype("datetime64[ns]")
    week = np.timedelta64(7, "D")
    pos = 0
    for t, g in le.groupby("ticker", sort=False):
        n = len(g); base = pos
        gts = tsv[base:base + n]; gE = En[base:base + n]
        lo = 0
        for i in range(n):
            while gts[lo] < gts[i] - week:
                lo += 1
            if lo < i:
                sims = gE[lo:i] @ gE[i]
                nov[base + i] = 1.0 - float(sims.max())
                n7[base + i] = i - lo
                simfrac[base + i] = float((sims > 0.8).mean())
        pos += n
    le["nov"] = nov; le["n7"] = n7; le["simfrac"] = simfrac
    log(f"novelty done: mean={nov.mean():.3f} median n7={np.median(n7):.0f}")

    # ---- FinBERT sentiment (precomputed) --------------------------------
    sc = pd.read_parquet(SCORES, columns=["news_id", "finbert_sentiment"])
    sc = sc[sc.news_id.isin(set(le.news_id.unique()))]
    s_map = dict(zip(sc.news_id.values, sc.finbert_sentiment.values.astype(np.float32)))
    del sc
    le["sent"] = le.news_id.map(s_map).fillna(0.0).astype(np.float32)
    log(f"sentiment coverage {le.sent.ne(0).mean():.1%}")

    # ---- buckets with extras --------------------------------------------
    le["emb_i"] = np.arange(len(le))
    def bucket(df, key):
        out = {}
        for (t, d), g in df.dropna(subset=[key]).groupby(["ticker", key]):
            idx = g.emb_i.values
            emb = E[idx].mean(0)
            sent = g.sent.values
            out[(t, d)] = (emb, float(sent.mean()), float(np.abs(sent).max()),
                           float(g.nov.values.mean()), float(g.nov.values.max()),
                           float(len(g)), float(np.log1p(g.n7.values).mean()),
                           float(g.simfrac.values.mean()),
                           float(sent.min()), float(sent.max()))
        return out
    over = bucket(le[le.fresh], "eff"); info = bucket(le, "info")
    log(f"buckets over {len(over)} info {len(info)}")

    td_pos = {d: i for i, d in enumerate(TD)}
    Z102 = np.zeros(102, np.float32)
    recs = []
    for t, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        o = g.open.values.astype(float); h = g.high.values.astype(float)
        lo_ = g.low.values.astype(float); c = g.close.values.astype(float)
        v = g.volume.values.astype(float); gd = g.date.values
        hl = (h - lo_) / np.where(c > 0, c, np.nan)
        r1 = np.concatenate([[0.0], c[1:] / c[:-1] - 1.0])
        gap = np.concatenate([[0.0], o[1:] / c[:-1] - 1.0])
        hit2 = (h >= o * 1.02).astype(np.float32)
        vmed = pd.Series(v).rolling(60, min_periods=10).median().values
        volz = np.log(np.where((v > 0) & (vmed > 0), v / vmed, 1.0))
        pday = np.stack([r1, np.nan_to_num(hl), gap, np.clip(volz, -3, 3), hit2], 1).astype(np.float32)
        for i in range(31, len(g)):
            if o[i] <= 0 or v[i] <= 0 or c[i - 1] <= 0:
                continue
            di = gd[i]
            if (t, str(di)[:7]) not in small_tm:
                continue
            if o[i] / c[i - 1] - 1.0 >= 0.295:
                continue
            a = gd[i - 1]
            lo_p, hi_p = td_pos[a], td_pos[di]
            ods = [over[(t, TD[kk])] for kk in range(lo_p + 1, hi_p + 1) if (t, TD[kk]) in over]
            p = hi_p
            hasany = bool(ods) or any((t, TD[p - kk]) in info
                                      for kk in range(1, LONG_DAYS + 1) if p - kk >= 0)
            if not hasany:
                continue
            if ods:
                emb = np.mean([x[0] for x in ods], 0)
                sv = np.concatenate([emb, [np.mean([x[1] for x in ods]),
                                           np.max([x[2] for x in ods])]]).astype(np.float32)
                novf = (float(np.mean([x[3] for x in ods])), float(np.max([x[4] for x in ods])),
                        float(np.log1p(sum(x[5] for x in ods))),
                        float(np.mean([x[6] for x in ods])),
                        float(np.mean([x[7] for x in ods])), 1.0)
                sentf = (float(np.mean([x[1] for x in ods])),
                         float(np.min([x[8] for x in ods])),
                         float(np.max([x[9] for x in ods])))
            else:
                sv = Z102; novf = (0., 0., 0., 0., 0., 0.); sentf = (0., 0., 0.)
            hit = tuple(float(h[i] >= o[i] * (1 + k)) for k in KS)
            drop = tuple(float(lo_[i] <= o[i] * (1 - k)) for k in KS)
            volf = (float(o[i] / c[i - 1] - 1.0), float(np.nanmean(hl[max(0, i - 20):i])),
                    float(np.std(r1[max(0, i - 20):i])) if i >= 3 else 0.0,
                    float(hl[i - 1]) if np.isfinite(hl[i - 1]) else 0.0,
                    float(abs(r1[i - 1])))
            recs.append((t, di, sv, hit, drop, volf, novf, sentf, pday[i - 30:i]))
    log(f"samples {len(recs)}")
    S = np.stack([r[2] for r in recs])
    DT = np.array([r[1] for r in recs], dtype="datetime64[ns]")
    HIT = np.array([r[3] for r in recs], np.float32)
    DROP = np.array([r[4] for r in recs], np.float32)
    VOLF = np.nan_to_num(np.array([r[5] for r in recs], np.float32))
    NOVF = np.array([r[6] for r in recs], np.float32)
    SENTF = np.array([r[7] for r in recs], np.float32)
    PSEQ = np.stack([r[8] for r in recs]).astype(np.float32)
    hsk = NOVF[:, 5] > 0
    M_ = np.zeros((len(recs), MID_DAYS, 102), np.float32)
    L_ = np.zeros((len(recs), LONG_DAYS, 102), np.float32)
    for j, r in enumerate(recs):
        t, di = r[0], r[1]
        p = td_pos[di]
        for kk in range(1, LONG_DAYS + 1):
            if p - kk < 0: break
            x = info.get((t, TD[p - kk]))
            if x is not None:
                vec = np.concatenate([x[0], [x[1], x[2]]]).astype(np.float32)
                L_[j, LONG_DAYS - kk] = vec
                if kk <= MID_DAYS:
                    M_[j, MID_DAYS - kk] = vec
    np.savez(FEAT, S=S, M=M_, L=L_, VOLF=VOLF, NOVF=NOVF, SENTF=SENTF, PSEQ=PSEQ,
             HIT=HIT, DROP=DROP, DT=DT.astype("datetime64[D]").astype(str), hsk=hsk)
    log(f"features saved S{S.shape} L{L_.shape} PSEQ{PSEQ.shape}")
    return {k: v for k, v in [("S", S), ("M", M_), ("L", L_), ("VOLF", VOLF),
            ("NOVF", NOVF), ("SENTF", SENTF), ("PSEQ", PSEQ), ("HIT", HIT),
            ("DROP", DROP), ("DT", DT.astype("datetime64[D]").astype(str)), ("hsk", hsk)]}

# ------------------------------------------------------------------ model
class HeadX(nn.Module):
    def __init__(self, ddim, vdim, use_price, E=384, layers=4, heads=6, ff=1536, p=0.3):
        super().__init__()
        self.ddim, self.use_price = ddim, use_price
        self.proj = nn.Linear(ddim, E)
        self.vproj = nn.Linear(vdim, E)
        self.pproj = nn.Linear(5, E) if use_price else None
        n_tok = 1 + 1 + 1 + MID_DAYS + LONG_DAYS + (30 if use_price else 0)
        self.cls = nn.Parameter(torch.zeros(1, 1, E))
        self.pos = nn.Parameter(torch.zeros(1, n_tok, E))
        nn.init.normal_(self.pos, 0, 0.02); nn.init.normal_(self.cls, 0, 0.02)
        enc = nn.TransformerEncoderLayer(E, heads, ff, dropout=p, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(E)
        self.fc = nn.Linear(E, 3)
    def forward(self, s, m, l, v, pr):
        B = s.shape[0]
        seq = torch.cat([s[:, :self.ddim].unsqueeze(1), m[:, :, :self.ddim],
                         l[:, :, :self.ddim]], 1)
        empty = seq.abs().sum(-1) == 0
        toks = [self.cls.expand(B, -1, -1), self.vproj(v).unsqueeze(1), self.proj(seq)]
        mask = [torch.zeros(B, 2, dtype=torch.bool, device=s.device), empty]
        if self.use_price:
            toks.append(self.pproj(pr))
            mask.append(torch.zeros(B, 30, dtype=torch.bool, device=s.device))
        x = torch.cat(toks, 1) + self.pos
        x = self.enc(x, src_key_padding_mask=torch.cat(mask, 1))
        return self.fc(self.norm(x[:, 0]))

# ------------------------------------------------------------------ train/eval
def fit_eval(name, build, T, Y, trN, dvN, teN, seeds=(13, 14), lr=2e-4, wd=0.05,
             epochs=40, warmup=2, batch=256, patience=8, smooth=0.05,
             day_drop=0.10, noise=0.03):
    from sklearn.metrics import matthews_corrcoef, accuracy_score
    from scipy.stats import binomtest
    S, M, L, V, P = T
    Yt = torch.from_numpy(Y.astype(np.float32))
    def predict(m, idx, bs=2048):
        m.eval(); out = np.zeros((len(idx), 3), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(idx), bs):
                ii = idx[s0:s0 + bs]
                out[s0:s0 + bs] = torch.sigmoid(m(S[ii].to(DEV), M[ii].to(DEV),
                    L[ii].to(DEV), V[ii].to(DEV), P[ii].to(DEV))).cpu().numpy()
        return out
    def mcc3(y, p):
        from sklearn.metrics import matthews_corrcoef as mc
        return np.array([mc(y[:, j], (p[:, j] > .5).astype(int))
                         if len(np.unique((p[:, j] > .5))) > 1 else 0.0 for j in range(3)])
    ens = {k: [] for k in ("tr", "te")}
    t0 = time.time()
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        m = build().to(DEV)
        decay, nodecay = [], []
        for n_, p_ in m.named_parameters():
            (nodecay if p_.ndim <= 1 else decay).append(p_)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                 {"params": nodecay, "weight_decay": 0.0}], lr=lr)
        steps_per = math.ceil(len(trN) / batch)
        def lr_at(step):
            e = step / steps_per
            if e < warmup: return lr * (e / warmup + 1e-3)
            return lr * 0.5 * (1 + math.cos(math.pi * min((e - warmup) / max(epochs - warmup, 1), 1.0)))
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad, step = -9., None, 0, 0
        for ep in range(epochs):
            m.train()
            perm = np.random.default_rng(sd + ep).permutation(trN)
            for s0 in range(0, len(perm), batch):
                ii = perm[s0:s0 + batch]
                s_, m_, l_, v_, p_ = (X[ii].to(DEV) for X in (S, M, L, V, P))
                if day_drop > 0:
                    dm = (torch.rand(m_.shape[0], m_.shape[1], 1, device=DEV) > day_drop).float()
                    dl = (torch.rand(l_.shape[0], l_.shape[1], 1, device=DEV) > day_drop).float()
                    m_, l_ = m_ * dm, l_ * dl
                if noise > 0:
                    s_ = s_ + noise * torch.randn_like(s_) * (s_ != 0)
                    m_ = m_ + noise * torch.randn_like(m_) * (m_ != 0)
                    l_ = l_ + noise * torch.randn_like(l_) * (l_ != 0)
                for g in opt.param_groups: g["lr"] = lr_at(step)
                yb = Yt[ii].to(DEV) * (1 - smooth) + 0.5 * smooth
                loss = lossf(m(s_, m_, l_, v_, p_), yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); step += 1
            dv = mcc3(Y[dvN], predict(m, dvN)).mean()
            if dv > best:
                best, best_state, bad = dv, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
            if bad >= patience:
                break
        m.load_state_dict(best_state)
        ens["tr"].append(predict(m, trN)); ens["te"].append(predict(m, teN))
    p_tr, p_te = np.mean(ens["tr"], 0), np.mean(ens["te"], 0)
    n_par = sum(p.numel() for p in build().parameters())
    mtr, mte = mcc3(Y[trN], p_tr), mcc3(Y[teN], p_te)
    print(f"\n[{name}] params={n_par/1e6:.2f}M ({time.time()-t0:.0f}s)", flush=True)
    for j, k in enumerate(KS):
        yy = Y[teN, j].astype(int)
        acc = accuracy_score(yy, (p_te[:, j] > 0.5).astype(int))
        line = (f"   k={k:.0%}: test acc={acc:.4f} MCC={mte[j]:+.4f} "
                f"(train {mtr[j]:+.4f}, GAP {mtr[j]-mte[j]:+.4f})")
        if j == 0:
            order = np.argsort(-p_te[:, 0]); nsel = max(len(yy) // 10, 20)
            sel = order[:nsel]; base = yy.mean()
            pv = binomtest(int(yy[sel].sum()), nsel, base, alternative="greater").pvalue
            line += f"  top10%={yy[sel].mean():.3f} vs base {base:.3f} (p={pv:.0e})"
        print(line, flush=True)
    return p_te

# ------------------------------------------------------------------ main
def main():
    F = build_features()
    S, M_, L_, VOLF, NOVF, SENTF, PSEQ, HIT = (F[k] for k in
        ("S", "M", "L", "VOLF", "NOVF", "SENTF", "PSEQ", "HIT"))
    DT = F["DT"].astype("datetime64[ns]")
    log(f"features S{S.shape} P{PSEQ.shape} overnight {(NOVF[:,5]>0).mean():.1%}")
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[DT[trN] >= dsplit]; trN = trN[DT[trN] < dsplit]
    log(f"train {len(trN)} dev {len(dvN)} test {len(teN)}")

    def z(X):
        mu, sd = X[trN].mean(0), X[trN].std(0) + 1e-9
        return ((X - mu) / sd).astype(np.float32)
    Vz = z(VOLF)
    V12 = np.concatenate([Vz, z(NOVF[:, :5]), NOVF[:, 5:6]], 1).astype(np.float32)
    V15 = np.concatenate([V12, z(SENTF)], 1).astype(np.float32)
    Pz = PSEQ  # already scaled-ish per feature; leave raw (small magnitudes)

    St, Mt, Lt = torch.from_numpy(S), torch.from_numpy(M_), torch.from_numpy(L_)
    Pt = torch.from_numpy(Pz)
    T5 = lambda V: (St, Mt, Lt, torch.from_numpy(V), Pt)

    only = set(os.environ.get("S45_ONLY", "B0,B1,B2,B3").split(","))
    if "B0" in only:
        fit_eval("B0 FUSED-TF (news100+vol5)", lambda: HeadX(100, 5, False),
                 T5(Vz), HIT, trN, dvN, teN)
    if "B1" in only:
        fit_eval("B1 +PRICE seq (30d tokens)", lambda: HeadX(100, 5, True),
                 T5(Vz), HIT, trN, dvN, teN)
    if "B2" in only:
        fit_eval(f"B2 +NOVELTY (vol 5->{V12.shape[1]})",
                 lambda: HeadX(100, V12.shape[1], True), T5(V12), HIT, trN, dvN, teN)
    if "B3" in only:
        fit_eval(f"B3 +FinBERT (day 102, vol {V15.shape[1]})",
                 lambda: HeadX(102, V15.shape[1], True), T5(V15), HIT, trN, dvN, teN)

if __name__ == "__main__":
    main()
