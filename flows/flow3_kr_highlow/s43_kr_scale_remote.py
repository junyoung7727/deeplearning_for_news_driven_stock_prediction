"""
Stage 43 - model-capacity scaling for KR HIGH-prediction, WITH anti-overfitting
techniques and 10x data (remote GPU box).

Anti-overfit levers (vs the naive s13-style capacity sweep that collapsed):
  1. 10x DATA: news 2015+ (KR43_START), prices/caps 2015-2026 (s42 fetch;
     universe = 2024 survivors -> survivorship bias DOCUMENTED, fine for a
     capacity study).
  2. Multi-task: one model predicts k=2/3/5% jointly (shared representation).
  3. AdamW decoupled weight decay 0.05 (no decay on bias/norm), cosine LR with
     warmup, label smoothing 0.05.
  4. Train-time augmentation: whole-day dropout in M/L windows (p=0.10) +
     gaussian noise on non-zero features (sigma=0.03).
  5. Early stop on dev mean-MCC (patience 8), 3-seed ensembles.
  6. Explicit overfit accounting: train/dev/test MCC + train-test gap reported
     per model size.

Model ladder (news branch; NTN/w2v frozen upstream as before):
  S  ~0.10M  control  = s41 head (Conv64x2 -> FC100)
  M  ~0.9M   deep-CNN = Conv256+GELU+Conv256 per window -> FC384
  L  ~7M     transformer = proj 384 + 4-layer encoder (6 heads, ff 1536,
             norm_first, dropout 0.3) over [CLS; S; M(7); L(30)] = 39 tokens

Point-in-time small-cap membership: per month, ticker's median cap <= that
month's cross-sectional median (uses kr_cap_ext, no static 2024 split).
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
import os, re, sys, glob, time, math, pickle, numpy as np, pandas as pd, torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s41_kr_high_remote import kr_events, tok_content, NTN, log

HOME = os.path.expanduser("~")
BK = os.path.join(HOME, "bk_slim")
DATA = os.path.join(HOME, "dlfe", "data")
ART = os.path.join(HOME, "dlfe", "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
D = 100
MID_DAYS, LONG_DAYS = 7, 30
START = os.environ.get("KR43_START", "2015-01-01")
KS = (0.02, 0.03, 0.05)
COST = 0.0030
CUE = {"주가", "주식", "종목", "상장", "코스피", "코스닥", "실적", "영업이익", "매출",
       "인수", "합병", "지분", "배당", "수주", "공시", "대표", "회장", "부회장", "사장",
       "그룹", "계열", "IPO", "유상증자", "무상증자", "신저가", "신고가", "급등", "급락",
       "상한가", "하한가", "자사주", "전환사채", "블록딜"}
AMBIG_EXTRA = {"하이브", "신세계", "두산", "한샘", "동원", "빙그레"}
TIGHT = {
    "한화": re.compile(r"한화그룹|㈜\s*한화|한화\s*\(주\)|한화\s*(주가|주식|지주)|김승연"),
    "대상": re.compile(r"㈜\s*대상|대상\s*\(주\)|대상그룹|대상홀딩스|대상\s*(주가|주식)"),
}
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ stage 1
def build_corpus():
    p = os.path.join(ART, "kr43_corpus.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from kiwipiepy import Kiwi
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"), columns=["ticker"])
    tickers = set(ohlcv.ticker.unique())
    uni = pd.read_parquet(os.path.join(DATA, "kr_universe_enriched.parquet"))
    uni = uni[uni.ticker.isin(tickers)]
    name2t = {}
    for _, r in uni.iterrows():
        cand = {r["name"]} if isinstance(r["name"], str) and len(r["name"]) >= 2 else set()
        try:
            cand |= {a for a in (r["aliases"] or []) if isinstance(a, str) and len(a) >= 3}
        except Exception:
            pass
        for c in cand:
            name2t.setdefault(c, []).append(r.ticker)
    log(f"universe {len(tickers)} tickers, {len(name2t)} names")

    dfs = []
    for f in sorted(glob.glob(os.path.join(BK, "slim_0*.parquet"))):
        d = pd.read_parquet(f, columns=["news_id", "title"])
        dt = d.news_id.str.slice(9, 23)
        okm = dt.str.fullmatch(r"\d{14}")
        d = d[okm]; dt = dt[okm]
        d["ts"] = pd.to_datetime(dt, format="%Y%m%d%H%M%S", errors="coerce")
        d = d[d.ts.notna() & (d.ts >= pd.Timestamp(START))]
        dfs.append(d)
    news = pd.concat(dfs, ignore_index=True).drop_duplicates("news_id")
    log(f"titles since {START}: {len(news)}")

    ascii_names = {n for n in name2t if re.fullmatch(r"[0-9A-Za-z&.\- ]+", n)}
    kiwi = Kiwi(num_workers=24)
    ambig = set()
    for n in set(name2t) - ascii_names:
        toks = [(t.form, t.tag) for t in kiwi.tokenize(n)]
        if len(toks) == 1 and toks[0][1] == "NNG":
            ambig.add(n)
    ambig |= (AMBIG_EXTRA & set(name2t))
    big = re.compile("|".join(re.escape(k) for k in sorted(name2t, key=len, reverse=True)))
    cand = news[news.title.str.contains(big, na=False, regex=True)].reset_index(drop=True)
    del news
    log(f"substring candidates: {len(cand)}")
    toks_list = [[(t.form, t.tag) for t in s] for s in kiwi.tokenize(cand.title.tolist())]
    log("tokenized")

    def consec_concats(forms, maxlen=4):
        out = set()
        for i in range(len(forms)):
            s = ""
            for j in range(i, min(i + maxlen, len(forms))):
                s += forms[j]; out.add(s)
        return out
    ascii_pat = {n: re.compile(r"(?<![0-9A-Za-z&])" + re.escape(n) + r"(?![0-9A-Za-z&])")
                 for n in ascii_names}
    rows, corpus, events_by_nid = [], [], {}
    for (nid, title, ts), tks in zip(cand[["news_id", "title", "ts"]].itertuples(index=False),
                                     toks_list):
        raw = {m.group(0) for m in big.finditer(title)}
        if not raw:
            continue
        forms = [f for f, t in tks]
        concat = content = None
        matched = []
        for n in raw:
            if n in TIGHT:
                if not TIGHT[n].search(title): continue
            elif n in ascii_names:
                if not ascii_pat[n].search(title): continue
            else:
                if concat is None: concat = consec_concats(forms)
                if n not in concat: continue
            if n in ambig and n not in TIGHT:
                if content is None: content = set(forms)
                if not (content & CUE): continue
            matched.append(n)
        if not matched:
            continue
        corpus.append(tok_content(tks))
        ev = kr_events(tks)
        if ev:
            events_by_nid[nid] = ev
        for n in matched:
            for t in name2t[n]:
                rows.append((t, ts, nid))
    link = pd.DataFrame(rows, columns=["ticker", "ts", "news_id"]).drop_duplicates()
    log(f"precise links {len(link)} over {link.news_id.nunique()} titles; events {len(events_by_nid)}")
    obj = (link, corpus, events_by_nid)
    pickle.dump(obj, open(p, "wb"), protocol=4)
    return obj

# ------------------------------------------------------------------ stage 2+3
def build_w2v(corpus, n_pairs=12_000_000):
    p = os.path.join(ART, "kr43_w2v.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return list(z["vocab"]), z["W"]
    from collections import Counter
    rng = np.random.default_rng(SEED)
    cnt = Counter(w for s in corpus for w in s)
    vocab = [w for w, c in cnt.most_common() if c >= 5]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab); log(f"vocab {V}")
    centers, ctx = [], []
    for s in corpus:
        ids = [w2i[w] for w in s if w in w2i]
        for pos in range(len(ids)):
            w = rng.integers(1, 6)
            for j in range(max(0, pos - w), min(len(ids), pos + w + 1)):
                if j != pos: centers.append(ids[pos]); ctx.append(ids[j])
    centers = np.array(centers); ctx = np.array(ctx)
    if len(centers) > n_pairs:
        keep = rng.choice(len(centers), n_pairs, replace=False)
        centers, ctx = centers[keep], ctx[keep]
    log(f"w2v pairs {len(centers)}")
    inp = nn.Embedding(V, D).to(DEV); out = nn.Embedding(V, D).to(DEV)
    nn.init.uniform_(inp.weight, -.5 / D, .5 / D); nn.init.zeros_(out.weight)
    opt = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=.02)
    negp = np.array([cnt[w] for w in vocab], float) ** .75; negp /= negp.sum()
    negtab = torch.from_numpy(rng.choice(V, size=3_000_000, p=negp)).to(DEV)
    ct = torch.from_numpy(centers).to(DEV); ot = torch.from_numpy(ctx).to(DEV)
    for ep in range(4):
        perm = torch.randperm(len(centers), device=DEV); tot = 0.0; nb = 0
        for s in range(0, len(centers), 8192):
            idx = perm[s:s + 8192]
            vc, vo = inp(ct[idx]), out(ot[idx])
            ng = negtab[torch.randint(0, len(negtab), (len(idx), 5), device=DEV)]
            vn = out(ng)
            loss = -(torch.nn.functional.logsigmoid((vc * vo).sum(1)) +
                     torch.nn.functional.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        log(f"  w2v {ep+1}/4 loss={tot/nb:.4f}")
    W = inp.weight.detach().cpu().numpy().astype(np.float32)
    np.savez(p, vocab=np.array(vocab, object), W=W)
    return vocab, W

def build_event_emb(events_by_nid, vocab, W, cap=150_000):
    p = os.path.join(ART, "kr43_nid_emb.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return dict(zip(z["nids"], z["emb"]))
    rng = np.random.default_rng(SEED)
    w2i = {w: i for i, w in enumerate(vocab)}
    nids = list(events_by_nid)
    def avg(ws):
        idx = [w2i[w] for w in ws if w in w2i]
        return W[idx].mean(0) if idx else None
    O1 = np.zeros((len(nids), D), np.float32); P = np.zeros_like(O1); O2 = np.zeros_like(O1)
    ok = np.zeros(len(nids), bool)
    for i, nid in enumerate(nids):
        a, pp, b = events_by_nid[nid]
        va, vp, vb = avg(a), avg(pp), avg(b)
        if va is not None and vp is not None and vb is not None:
            O1[i], P[i], O2[i] = va, vp, vb; ok[i] = True
    O1t = torch.from_numpy(O1).to(DEV); Pt = torch.from_numpy(P).to(DEV)
    O2t = torch.from_numpy(O2).to(DEV); Wt = torch.from_numpy(W).to(DEV)
    tr_idx = np.where(ok)[0]
    ntn = NTN(D, D).to(DEV); nopt = torch.optim.Adam(ntn.parameters(), lr=0.01)
    tr = tr_idx if len(tr_idx) <= cap else rng.choice(tr_idx, cap, replace=False)
    for it in range(10):
        order = rng.permutation(tr); th = 0.0; nb = 0
        for s in range(0, len(order), 1024):
            ii = torch.from_numpy(order[s:s + 1024]).to(DEV)
            o1, pp, o2 = O1t[ii], Pt[ii], O2t[ii]
            ch = torch.randint(0, 2, (len(ii),), device=DEV)
            rv = Wt[torch.randint(0, len(W), (len(ii),), device=DEV)]
            co1 = torch.where((ch == 0).unsqueeze(1), rv, o1)
            co2 = torch.where((ch == 1).unsqueeze(1), rv, o2)
            hinge = torch.clamp(1 - ntn.score(o1, pp, o2) + ntn.score(co1, pp, co2), min=0)
            loss = hinge.mean() + 1e-4 * ntn.l2()
            nopt.zero_grad(); loss.backward(); nopt.step()
            th += hinge.mean().item(); nb += 1
        if it % 3 == 0 or it == 9:
            log(f"  NTN {it+1}/10 hinge={th/nb:.4f}")
    emb = np.zeros((len(nids), D), np.float32)
    with torch.no_grad():
        for s in range(0, len(tr_idx), 8192):
            ii = torch.from_numpy(tr_idx[s:s + 8192]).to(DEV)
            emb[tr_idx[s:s + 8192]] = ntn.embed(O1t[ii], Pt[ii], O2t[ii]).cpu().numpy()
    mu, sd = emb[ok].mean(0), emb[ok].std(0) + 1e-6
    emb = (emb - mu) / sd; emb[~ok] = 0
    keep = np.asarray(nids, object)[ok]
    np.savez(p, nids=keep, emb=emb[ok])
    log(f"event embs {int(ok.sum())}")
    return dict(zip(keep, emb[ok]))

# ------------------------------------------------------------------ models
class HeadS(nn.Module):                                   # ~0.10M (s41 control)
    def __init__(self):
        super().__init__()
        self.cl = nn.Conv1d(D, 64, 3); self.cm = nn.Conv1d(D, 64, 3)
        self.fc1 = nn.Linear(64 * 2 + D, 100); self.fc2 = nn.Linear(100, 3)
        self.drop = nn.Dropout(0.5)
    def forward(self, s, m, l):
        vl = torch.tanh(self.cl(l.transpose(1, 2))).max(2).values
        vm = torch.tanh(self.cm(m.transpose(1, 2))).max(2).values
        y = torch.sigmoid(self.fc1(torch.cat([vl, vm, s], 1)))
        return self.fc2(self.drop(y))

class HeadM(nn.Module):                                   # ~0.9M deep CNN
    def __init__(self, F=256):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Conv1d(D, F, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(F, F, 3, padding=1), nn.GELU())
        self.bl, self.bm = branch(), branch()
        self.fc1 = nn.Linear(F * 2 + D, 384); self.act = nn.GELU()
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(384, 3)
    def forward(self, s, m, l):
        vl = self.bl(l.transpose(1, 2)).max(2).values
        vm = self.bm(m.transpose(1, 2)).max(2).values
        y = self.act(self.fc1(torch.cat([vl, vm, s], 1)))
        return self.fc2(self.drop(y))

class HeadL(nn.Module):                                   # ~7M transformer
    def __init__(self, E=384, layers=4, heads=6, ff=1536, p=0.3):
        super().__init__()
        self.proj = nn.Linear(D, E)
        self.cls = nn.Parameter(torch.zeros(1, 1, E))
        self.pos = nn.Parameter(torch.zeros(1, 1 + 1 + MID_DAYS + LONG_DAYS, E) * 0.0)
        nn.init.normal_(self.pos, 0, 0.02); nn.init.normal_(self.cls, 0, 0.02)
        enc = nn.TransformerEncoderLayer(E, heads, ff, dropout=p, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(E)
        self.fc = nn.Linear(E, 3)
    def forward(self, s, m, l):
        B = s.shape[0]
        seq = torch.cat([s.unsqueeze(1), m, l], 1)          # (B, 38, D)
        x = self.proj(seq)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        x = self.enc(x)
        return self.fc(self.norm(x[:, 0]))

# ------------------------------------------------------------------ training
def fit_eval(name, build, S, M, L, Y, trN, dvN, teN, seeds=(13, 14, 15),
             lr=3e-4, wd=0.05, epochs=40, warmup=2, batch=256, patience=8,
             smooth=0.05, day_drop=0.10, noise=0.03):
    from sklearn.metrics import matthews_corrcoef
    Yt = torch.from_numpy(Y.astype(np.float32))
    def predict(m, idx, bs=2048):
        m.eval(); out = np.zeros((len(idx), 3), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(idx), bs):
                ii = idx[s0:s0 + bs]
                out[s0:s0 + bs] = torch.sigmoid(
                    m(S[ii].to(DEV), M[ii].to(DEV), L[ii].to(DEV))).cpu().numpy()
        return out
    def mcc3(y, p):
        vals = []
        for j in range(3):
            pr = (p[:, j] > 0.5).astype(int)
            vals.append(matthews_corrcoef(y[:, j], pr) if len(np.unique(pr)) > 1 else 0.0)
        return np.array(vals)
    ens_tr, ens_dv, ens_te = [], [], []
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
        import copy
        for ep in range(epochs):
            m.train()
            perm = np.random.default_rng(sd + ep).permutation(trN)
            for s0 in range(0, len(perm), batch):
                ii = perm[s0:s0 + batch]
                s_, m_, l_ = S[ii].to(DEV), M[ii].to(DEV), L[ii].to(DEV)
                if day_drop > 0:                       # whole-day dropout (train aug)
                    dm = (torch.rand(m_.shape[0], m_.shape[1], 1, device=DEV) > day_drop).float()
                    dl = (torch.rand(l_.shape[0], l_.shape[1], 1, device=DEV) > day_drop).float()
                    m_, l_ = m_ * dm, l_ * dl
                if noise > 0:                          # gaussian noise on non-zeros
                    s_ = s_ + noise * torch.randn_like(s_) * (s_ != 0)
                    m_ = m_ + noise * torch.randn_like(m_) * (m_ != 0)
                    l_ = l_ + noise * torch.randn_like(l_) * (l_ != 0)
                for g in opt.param_groups: g["lr"] = lr_at(step)
                yb = Yt[ii].to(DEV)
                yb = yb * (1 - smooth) + 0.5 * smooth   # label smoothing
                loss = lossf(m(s_, m_, l_), yb)
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
        ens_tr.append(predict(m, trN)); ens_dv.append(predict(m, dvN)); ens_te.append(predict(m, teN))
    p_tr, p_dv, p_te = (np.mean(x, 0) for x in (ens_tr, ens_dv, ens_te))
    n_par = sum(p.numel() for p in build().parameters())
    mtr, mdv, mte = mcc3(Y[trN], p_tr), mcc3(Y[dvN], p_dv), mcc3(Y[teN], p_te)
    print(f"\n[{name}] params={n_par/1e6:.2f}M  seeds={len(seeds)}  ({time.time()-t0:.0f}s)", flush=True)
    for j, k in enumerate(KS):
        print(f"   k={k:.0%}: MCC train={mtr[j]:+.4f} dev={mdv[j]:+.4f} test={mte[j]:+.4f}  "
              f"GAP(train-test)={mtr[j]-mte[j]:+.4f}", flush=True)
    return p_te

# ------------------------------------------------------------------ main
def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, accuracy_score
    from scipy.stats import binomtest
    link, corpus, events_by_nid = build_corpus()
    vocab, W = build_w2v(corpus)
    nid2emb = build_event_emb(events_by_nid, vocab, W)
    del corpus, events_by_nid

    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    ohlcv = ohlcv[ohlcv.date >= pd.Timestamp(START)]
    ohlcv = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)
    # point-in-time cap PROXY: KRX cap-history endpoint is down, so back out the
    # share count from actual caps (2023-12+) / adjusted close on overlapping
    # dates, then hist_cap = shares x adjusted close (splits are absorbed by the
    # price adjustment; new issues/buybacks are not - fine for a median SPLIT).
    capf = pd.read_parquet(os.path.join(DATA, "kr_market_cap_daily.parquet"))
    capf["date"] = pd.to_datetime(capf.trade_date)
    ov = capf.merge(ohlcv[["ticker", "date", "close"]], on=["ticker", "date"])
    ov = ov[(ov.market_cap > 0) & (ov.close > 0)]
    shares = (ov.market_cap / ov.close).groupby(ov.ticker).median()
    ohlcv["pcap"] = ohlcv.close * ohlcv.ticker.map(shares)
    cap = ohlcv[ohlcv.pcap.notna()][["ticker", "date", "pcap"]]
    cap["month"] = cap.date.dt.strftime("%Y-%m")
    mcap = cap.groupby(["ticker", "month"]).pcap.median().reset_index()
    med = mcap.groupby("month").pcap.median().rename("xmed").reset_index()
    mcap = mcap.merge(med, on="month")
    small_tm = {(r.ticker, r.month) for r in mcap[mcap.pcap <= mcap.xmed].itertuples()}
    log(f"ohlcv {len(ohlcv)} rows {ohlcv.date.min().date()}..{ohlcv.date.max().date()}; "
        f"small (ticker,month) pairs {len(small_tm)}")

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
    log(f"usable links {len(le)} fresh {le.fresh.mean():.1%}")
    def bucket(df, key):
        g = df.dropna(subset=[key]).groupby(["ticker", key]).emb_i.apply(lambda s: E[s.values].mean(0))
        return {(t, d): v for (t, d), v in g.items()}
    over = bucket(le[le.fresh], "eff"); info = bucket(le, "info")
    log(f"buckets over {len(over)} info {len(info)}")

    td_pos = {d: i for i, d in enumerate(TD)}
    Z = np.zeros(D, np.float32)
    # pass 1: enumerate samples + has_any WITHOUT materialising M/L
    recs = []           # (ticker, i, date, S_vec|None, hit3, pnl3, volf5, has_any)
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
            hasany = sv is not None or any((t, TD[p - kk]) in info for kk in range(1, LONG_DAYS + 1) if p - kk >= 0)
            hit = tuple(float(h[i] >= o[i] * (1 + k)) for k in KS)
            pnl = tuple(float((k if h[i] >= o[i] * (1 + k) else c[i] / o[i] - 1.0) - COST) for k in KS)
            volf = (float(o[i] / c[i - 1] - 1.0), float(np.nanmean(hl[max(0, i - 20):i])),
                    float(np.std(r1[max(0, i - 21):i - 1])) if i >= 3 else 0.0,
                    float(hl[i - 1]) if np.isfinite(hl[i - 1]) else 0.0,
                    float(abs(r1[i - 2])) if i >= 2 else 0.0)
            recs.append((t, di, sv, hit, pnl, volf, hasany))
    n_all = len(recs)
    has_any = np.array([r[6] for r in recs])
    log(f"samples {n_all}, event-window {has_any.sum()} ({has_any.mean():.1%})")

    keep = [r for r in recs if r[6]]
    del recs
    S = np.stack([r[2] if r[2] is not None else Z for r in keep])
    DT = np.array([r[1] for r in keep], dtype="datetime64[ns]")
    HIT = np.array([r[3] for r in keep], np.float32)
    PNL = np.array([r[4] for r in keep], np.float32)
    VOLF = np.nan_to_num(np.array([r[5] for r in keep], np.float32))
    hsk = np.array([r[2] is not None for r in keep])
    M_ = np.zeros((len(keep), MID_DAYS, D), np.float32)
    L_ = np.zeros((len(keep), LONG_DAYS, D), np.float32)
    for j, r in enumerate(keep):
        t, di = r[0], r[1]
        p = td_pos[di]
        for kk in range(1, LONG_DAYS + 1):
            if p - kk < 0: break
            v = info.get((t, TD[p - kk]))
            if v is not None:
                L_[j, LONG_DAYS - kk] = v
                if kk <= MID_DAYS:
                    M_[j, MID_DAYS - kk] = v
    log(f"tensors S{S.shape} M{M_.shape} L{L_.shape} "
        f"({(S.nbytes+M_.nbytes+L_.nbytes)/2**30:.2f} GiB)")

    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[DT[trN] >= dsplit]; trN = trN[DT[trN] < dsplit]
    log(f"train {len(trN)} dev {len(dvN)} test {len(teN)}  (test from {str(split)[:10]})")

    St = torch.from_numpy(S); Mt = torch.from_numpy(M_); Lt = torch.from_numpy(L_)
    Y = HIT

    # baselines
    print("\n=== VOL-LR baseline (extended data) ===", flush=True)
    for j, k in enumerate(KS):
        lr_ = LogisticRegression(max_iter=2000).fit(VOLF[trN], Y[trN, j].astype(int))
        p_ = lr_.predict_proba(VOLF[teN])[:, 1]
        pr = (p_ > 0.5).astype(int); yy = Y[teN, j].astype(int)
        print(f"   k={k:.0%}: acc={accuracy_score(yy, pr):.4f} "
              f"mcc={matthews_corrcoef(yy, pr):+.4f} base={yy.mean():.4f}", flush=True)

    results = {}
    results["S-0.1M"] = fit_eval("S-0.1M control", HeadS, St, Mt, Lt, Y, trN, dvN, teN,
                                 lr=1e-3, wd=1e-5, smooth=0.0, day_drop=0.0, noise=0.0)
    results["M-0.9M"] = fit_eval("M-0.9M deepCNN+reg", HeadM, St, Mt, Lt, Y, trN, dvN, teN)
    results["L-7M"] = fit_eval("L-7M transformer+reg", HeadL, St, Mt, Lt, Y, trN, dvN, teN, lr=2e-4)

    # selective cells + stack for the best (by test mean MCC)
    te_short = hsk[teN]
    for name, p_te in results.items():
        for j, k in enumerate(KS):
            yy = Y[teN, j].astype(int); pp = p_te[:, j]
            order = np.argsort(-pp); base = yy.mean()
            cells = []
            for kap in (0.1, 0.05):
                nsel = max(int(len(yy) * kap), 20); sel = order[:nsel]
                pv = binomtest(int(yy[sel].sum()), nsel, base, alternative="greater").pvalue
                cells.append(f"top{int(kap*100)}%={yy[sel].mean():.3f}(p={pv:.0e},n={nsel})")
            ys = yy[te_short]; ps = pp[te_short]
            so = np.argsort(-ps); nsel = max(int(len(ys) * 0.1), 20)
            cells.append(f"overnight-top10%={ys[so[:nsel]].mean():.3f}(n={nsel},base={ys.mean():.3f})")
            print(f"   [{name}] k={k:.0%} base={base:.3f} " + "  ".join(cells), flush=True)

if __name__ == "__main__":
    main()
