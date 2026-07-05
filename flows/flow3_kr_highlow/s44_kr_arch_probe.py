"""
Stage 44 - architecture-vs-input disambiguation (remote GPU).

Question (user): "VOL-LR works - did the TRANSFORMER architecture fail?"
Confound: in s43 the transformer never saw the vol features; it consumed only
news embeddings. This probe separates input from architecture:

  1. VOL-LR      (control replication)         5 std. vol feats -> logistic
  2. VOL-MLP     64-64 GELU (~5k params)        same feats, nonlinear
  3. NEWS-TF+mask  s43 HeadL + key_padding_mask on all-zero day tokens
                   (most M/L tokens are empty days -> attention dilution fix)
  4. FUSED-TF+mask same encoder, tokens = [CLS; VOL(5->E); S; M(7); L(30)]
     -> if FUSED >= VOL-LR the architecture is fine and the news input is the
        bottleneck; if FUSED < VOL-LR the transformer/optimisation genuinely
        fails to carry even a known-good 5-feature signal.

Same protocol as s43 (multi-task k=2/3/5, AdamW wd .05, cosine+warmup, label
smoothing .05, day-dropout .10, noise .03, dev-MCC early stop, 2 seeds for the
transformers / 3 for small models). Features cached to kr43_features.npz.
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
                                 HeadS, MID_DAYS, LONG_DAYS, KS, D, START, DATA, ART)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
FEAT = os.path.join(ART, "kr43_features.npz")
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ features
def build_features():
    if os.path.exists(FEAT):
        z = np.load(FEAT, allow_pickle=True)
        return (z["S"], z["M"], z["L"], z["VOLF"], z["HIT"],
                z["DT"].astype("datetime64[ns]"), z["hsk"])
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
    le["emb_i"] = np.arange(len(le))
    E = np.stack([nid2emb[n] for n in le.news_id.values]).astype(np.float32)
    def bucket(df, key):
        g = df.dropna(subset=[key]).groupby(["ticker", key]).emb_i.apply(lambda s: E[s.values].mean(0))
        return {(t, d): v for (t, d), v in g.items()}
    over = bucket(le[le.fresh], "eff"); info = bucket(le, "info")
    td_pos = {d: i for i, d in enumerate(TD)}
    Z = np.zeros(D, np.float32)
    recs = []
    for t, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        o = g.open.values.astype(float); h = g.high.values.astype(float)
        lo = g.low.values.astype(float); c = g.close.values.astype(float)
        v = g.volume.values.astype(float); gd = g.date.values
        hl = (h - lo) / np.where(c > 0, c, np.nan)
        r1 = c[1:] / c[:-1] - 1.0
        for i in range(21, len(g)):
            if o[i] <= 0 or v[i] <= 0 or c[i - 1] <= 0:
                continue
            di = gd[i]
            if (t, str(di)[:7]) not in small_tm:
                continue
            if o[i] / c[i - 1] - 1.0 >= 0.295:
                continue
            a = gd[i - 1]
            lo_p, hi_p = td_pos[a], td_pos[di]
            vs = [over[(t, TD[kk])] for kk in range(lo_p + 1, hi_p + 1) if (t, TD[kk]) in over]
            sv = np.mean(vs, 0).astype(np.float32) if vs else None
            p = hi_p
            hasany = sv is not None or any((t, TD[p - kk]) in info
                                           for kk in range(1, LONG_DAYS + 1) if p - kk >= 0)
            if not hasany:
                continue
            hit = tuple(float(h[i] >= o[i] * (1 + k)) for k in KS)
            volf = (float(o[i] / c[i - 1] - 1.0), float(np.nanmean(hl[max(0, i - 20):i])),
                    float(np.std(r1[max(0, i - 21):i - 1])) if i >= 3 else 0.0,
                    float(hl[i - 1]) if np.isfinite(hl[i - 1]) else 0.0,
                    float(abs(r1[i - 2])) if i >= 2 else 0.0)
            recs.append((t, di, sv, hit, volf))
    S = np.stack([r[2] if r[2] is not None else Z for r in recs])
    DT = np.array([r[1] for r in recs], dtype="datetime64[ns]")
    HIT = np.array([r[3] for r in recs], np.float32)
    VOLF = np.nan_to_num(np.array([r[4] for r in recs], np.float32))
    hsk = np.array([r[2] is not None for r in recs])
    M_ = np.zeros((len(recs), MID_DAYS, D), np.float32)
    L_ = np.zeros((len(recs), LONG_DAYS, D), np.float32)
    for j, r in enumerate(recs):
        t, di = r[0], r[1]
        p = td_pos[di]
        for kk in range(1, LONG_DAYS + 1):
            if p - kk < 0: break
            v = info.get((t, TD[p - kk]))
            if v is not None:
                L_[j, LONG_DAYS - kk] = v
                if kk <= MID_DAYS:
                    M_[j, MID_DAYS - kk] = v
    np.savez(FEAT, S=S, M=M_, L=L_, VOLF=VOLF, HIT=HIT,
             DT=DT.astype("datetime64[D]").astype(str), hsk=hsk)
    log(f"features saved {S.shape} {M_.shape} {L_.shape}")
    return S, M_, L_, VOLF, HIT, DT, hsk

# ------------------------------------------------------------------ models
class VolMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(64, 3))
    def forward(self, s, m, l, v):
        return self.net(v)

class HeadLMask(nn.Module):
    """s43 HeadL + zero-day key-padding mask; optional VOL token."""
    def __init__(self, E=384, layers=4, heads=6, ff=1536, p=0.3, use_vol=False):
        super().__init__()
        self.use_vol = use_vol
        self.proj = nn.Linear(D, E)
        self.vproj = nn.Linear(5, E) if use_vol else None
        n_tok = 1 + int(use_vol) + 1 + MID_DAYS + LONG_DAYS
        self.cls = nn.Parameter(torch.zeros(1, 1, E))
        self.pos = nn.Parameter(torch.zeros(1, n_tok, E))
        nn.init.normal_(self.pos, 0, 0.02); nn.init.normal_(self.cls, 0, 0.02)
        enc = nn.TransformerEncoderLayer(E, heads, ff, dropout=p, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(E)
        self.fc = nn.Linear(E, 3)
    def forward(self, s, m, l, v=None):
        B = s.shape[0]
        seq = torch.cat([s.unsqueeze(1), m, l], 1)              # (B, 38, D)
        empty = seq.abs().sum(-1) == 0                          # zero-day tokens
        x = self.proj(seq)
        toks = [self.cls.expand(B, -1, -1)]
        mask = [torch.zeros(B, 1, dtype=torch.bool, device=s.device)]
        if self.use_vol:
            toks.append(self.vproj(v).unsqueeze(1))
            mask.append(torch.zeros(B, 1, dtype=torch.bool, device=s.device))
        toks.append(x); mask.append(empty)
        x = torch.cat(toks, 1) + self.pos
        kpm = torch.cat(mask, 1)
        x = self.enc(x, src_key_padding_mask=kpm)
        return self.fc(self.norm(x[:, 0]))

# ------------------------------------------------------------------ train/eval
def fit_eval(name, build, tensors, Y, trN, dvN, teN, seeds, lr=2e-4, wd=0.05,
             epochs=40, warmup=2, batch=256, patience=8, smooth=0.05,
             day_drop=0.10, noise=0.03):
    from sklearn.metrics import matthews_corrcoef
    S, M, L, V = tensors
    Yt = torch.from_numpy(Y.astype(np.float32))
    def predict(m, idx, bs=2048):
        m.eval(); out = np.zeros((len(idx), 3), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(idx), bs):
                ii = idx[s0:s0 + bs]
                out[s0:s0 + bs] = torch.sigmoid(
                    m(S[ii].to(DEV), M[ii].to(DEV), L[ii].to(DEV), V[ii].to(DEV))).cpu().numpy()
        return out
    def mcc3(y, p):
        vals = []
        for j in range(3):
            pr = (p[:, j] > 0.5).astype(int)
            vals.append(matthews_corrcoef(y[:, j], pr) if len(np.unique(pr)) > 1 else 0.0)
        return np.array(vals)
    ens = {k: [] for k in ("tr", "dv", "te")}
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
            t = (e - warmup) / max(epochs - warmup, 1)
            return lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad, step = -9., None, 0, 0
        for ep in range(epochs):
            m.train()
            perm = np.random.default_rng(sd + ep).permutation(trN)
            for s0 in range(0, len(perm), batch):
                ii = perm[s0:s0 + batch]
                s_, m_, l_, v_ = (X[ii].to(DEV) for X in (S, M, L, V))
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
                loss = lossf(m(s_, m_, l_, v_), yb)
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
        for k, idx in (("tr", trN), ("dv", dvN), ("te", teN)):
            ens[k].append(predict(m, idx))
    p_tr, p_dv, p_te = (np.mean(ens[k], 0) for k in ("tr", "dv", "te"))
    n_par = sum(p.numel() for p in build().parameters())
    mtr, mte = mcc3(Y[trN], p_tr), mcc3(Y[teN], p_te)
    from sklearn.metrics import accuracy_score
    print(f"\n[{name}] params={n_par/1e6:.3f}M seeds={len(seeds)} ({time.time()-t0:.0f}s)", flush=True)
    for j, k in enumerate(KS):
        yy = Y[teN, j].astype(int)
        acc = accuracy_score(yy, (p_te[:, j] > 0.5).astype(int))
        print(f"   k={k:.0%}: test acc={acc:.4f} MCC={mte[j]:+.4f} (train {mtr[j]:+.4f}, "
              f"GAP {mtr[j]-mte[j]:+.4f})", flush=True)
    return p_te

# ------------------------------------------------------------------ main
def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, accuracy_score
    S, M_, L_, VOLF, HIT, DT, hsk = build_features()
    log(f"features {S.shape} zero-day M rows {(M_.sum(-1) == 0).mean():.1%} "
        f"L rows {(L_.sum(-1) == 0).mean():.1%}")
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[DT[trN] >= dsplit]; trN = trN[DT[trN] < dsplit]
    log(f"train {len(trN)} dev {len(dvN)} test {len(teN)}")

    mu, sd = VOLF[trN].mean(0), VOLF[trN].std(0) + 1e-9
    Vz = ((VOLF - mu) / sd).astype(np.float32)

    print("\n=== 1) VOL-LR control ===", flush=True)
    for j, k in enumerate(KS):
        lr_ = LogisticRegression(max_iter=2000).fit(Vz[trN], HIT[trN, j].astype(int))
        pr = (lr_.predict_proba(Vz[teN])[:, 1] > 0.5).astype(int)
        yy = HIT[teN, j].astype(int)
        print(f"   k={k:.0%}: test acc={accuracy_score(yy, pr):.4f} "
              f"MCC={matthews_corrcoef(yy, pr):+.4f} base={yy.mean():.4f}", flush=True)

    St, Mt, Lt = torch.from_numpy(S), torch.from_numpy(M_), torch.from_numpy(L_)
    Vt = torch.from_numpy(Vz)
    tensors = (St, Mt, Lt, Vt)
    fit_eval("2) VOL-MLP 5k", VolMLP, tensors, HIT, trN, dvN, teN,
             seeds=(13, 14, 15), lr=1e-3, wd=0.01, day_drop=0.0, noise=0.0)
    fit_eval("3) NEWS-TF 7M + zero-day mask", lambda: HeadLMask(use_vol=False),
             tensors, HIT, trN, dvN, teN, seeds=(13, 14))
    fit_eval("4) FUSED-TF 7M (news + VOL token) + mask", lambda: HeadLMask(use_vol=True),
             tensors, HIT, trN, dvN, teN, seeds=(13, 14))

if __name__ == "__main__":
    main()
