"""
Stage 41 - KR small-cap HIGH-prediction (runs on the remote GPU box).

Task change (user): predict whether the day's HIGH reaches open*(1+k) - the
paper's own market-sim entry (buy at open, take-profit at +k%). Unlike
close->close direction, this is (a) decided by information available at open
(overnight news, fresh mask) and (b) driven by volatility/attention, which news
demonstrably carries.

Also fixes the linking-precision bug found in s34/s36:
  * ASCII names ("KT","DB","SK"...) matched with word boundaries
    (no more SKT / KT&G / KTX / database-"DB" false links).
  * Hangul names validated against Kiwi token boundaries
    (no more HYBE<-"하이브리드").
  * Ambiguous common-noun names (Kiwi-tagged NNG alone, e.g. 대상/기아 +
    hand list 한화/하이브) additionally require a company-context cue token
    in the title.

Inputs (remote):
  ~/bk_slim/slim_0*.parquet                    news_id, title (ts encoded in id)
  ~/dlfe/data/kr_universe_enriched.parquet     ticker, name, aliases
  ~/dlfe/data/kr_market_cap_daily.parquet      small-cap split
  ~/dlfe/data/kr_ohlcv.parquet                 open/high/low/close/volume
Caches: ~/dlfe/artifacts/kr41_{corpus.pkl,w2v.npz,nid_emb.npz}

Models compared per k in {2%,3%,5%}, all trained on event-window samples,
time-based dev, 60/40 date split:
  BASE   base rate of y_k on the eval mask
  VOL-LR logistic on leak-free realized-vol features only
  EB-CNN paper head (fresh-overnight S + info-day mid/long NTN embeddings)
  STACK  logistic on [vol feats, p_cnn] fit on DEV, evaluated on TEST
Metrics: acc/MCC + precision@top-conf for class HIT + net-of-cost per-trade
profit (fees+tax 0.20% + slippage 0.10%; TP fill assumed when high>=target).
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
import os, re, glob, time, pickle, numpy as np, pandas as pd, torch
import torch.nn as nn

HOME = os.path.expanduser("~")
BK = os.path.join(HOME, "bk_slim")
DATA = os.path.join(HOME, "dlfe", "data")
ART = os.path.join(HOME, "dlfe", "artifacts")
os.makedirs(ART, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
D = 100
MID_DAYS, LONG_DAYS = 7, 30
START = os.environ.get("KR41_START", "2024-01-01")
KS = (0.02, 0.03, 0.05)
COST = 0.0030
NOUN = {"NNG", "NNP", "SL", "SN", "SH"}
CUE = {"주가", "주식", "종목", "상장", "코스피", "코스닥", "실적", "영업이익", "매출",
       "인수", "합병", "지분", "배당", "수주", "공시", "대표", "회장", "부회장", "사장",
       "그룹", "계열", "IPO", "유상증자", "무상증자", "신저가", "신고가", "급등", "급락",
       "상한가", "하한가", "자사주", "전환사채", "블록딜"}
AMBIG_EXTRA = {"하이브", "신세계", "두산", "한샘", "동원", "빙그레"}
# provably-toxic homonyms: currency "한화 N억", generic noun "대상" -> require a
# tight company pattern instead of a mere cue word (precision >> recall here)
TIGHT = {
    "한화": re.compile(r"한화그룹|㈜\s*한화|한화\s*\(주\)|한화\s*(주가|주식|지주)|김승연"),
    "대상": re.compile(r"㈜\s*대상|대상\s*\(주\)|대상그룹|대상홀딩스|대상\s*(주가|주식)"),
}
torch.manual_seed(SEED); np.random.seed(SEED)

def log(msg, t0=[time.time()]):
    print(f"{msg}  ({time.time()-t0[0]:.0f}s)", flush=True)

# ------------------------------------------------------------------ Korean SVO
def kr_events(tokens):
    nps, cur, marks, preds = [], [], [], []
    def flush():
        if cur: nps.append(cur.copy()); cur.clear()
    for i, (f, t) in enumerate(tokens):
        if t in NOUN:
            cur.append(f)
        else:
            flush()
            if t == "JKS" or (t == "JX" and f in ("은", "는", "도")):
                if nps: marks.append((len(nps) - 1, "S"))
            elif t == "JKO":
                if nps: marks.append((len(nps) - 1, "O"))
            elif t == "XSV" and nps and i > 0 and tokens[i-1][1] in NOUN:
                preds.append(nps[-1])
            elif t in ("VV", "VA"):
                preds.append([f])
    flush()
    if len(nps) < 2 or not preds:
        return None
    P = preds[-1]
    subj = [i for i, m in marks if m == "S"]; obj = [i for i, m in marks if m == "O"]
    o1i = subj[0] if subj else 0
    o2i = obj[-1] if obj else (len(nps) - 1)
    if o2i == o1i:
        o2i = len(nps) - 1 if o1i != len(nps) - 1 else 0
    O1, O2 = nps[o1i], nps[o2i]
    if O1 == P or O2 == P or O1 == O2:
        return None
    return (O1, P, O2)

def tok_content(tokens):
    return [f for f, t in tokens if t in NOUN or t in ("VV", "VA")]

# ------------------------------------------------------------------ stage 1
def build_corpus():
    p = os.path.join(ART, "kr41_corpus.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from kiwipiepy import Kiwi
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv.parquet"))
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
    log(f"universe {len(tickers)} tickers, {len(name2t)} names/aliases")

    # ---- titles 2024+ from bk_slim (ts encoded in news_id) ----
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

    # ---- precise matching ----
    ascii_names = {n for n in name2t if re.fullmatch(r"[0-9A-Za-z&.\- ]+", n)}
    hangul_names = set(name2t) - ascii_names
    kiwi = Kiwi(num_workers=16)
    # ambiguous = single-token common noun (NNG) or hand list
    ambig = set()
    for n in hangul_names:
        toks = [(t.form, t.tag) for t in kiwi.tokenize(n)]
        if len(toks) == 1 and toks[0][1] == "NNG":
            ambig.add(n)
    ambig |= (AMBIG_EXTRA & set(name2t))
    log(f"names: {len(ascii_names)} ascii, {len(hangul_names)} hangul, {len(ambig)} ambiguous")

    # prefilter: any-name substring (fast) then validate
    big = re.compile("|".join(re.escape(k) for k in
                              sorted(name2t, key=len, reverse=True)))
    cand = news[news.title.str.contains(big, na=False, regex=True)].reset_index(drop=True)
    log(f"substring candidates: {len(cand)}")

    toks_list = [[(t.form, t.tag) for t in s] for s in kiwi.tokenize(cand.title.tolist())]
    log("tokenized candidates")

    # per-title: set of concatenations of consecutive noun-ish token forms (for
    # Hangul boundary validation) + content-token set (for cue check)
    def consec_concats(forms, maxlen=4):
        out = set()
        for i in range(len(forms)):
            s = ""
            for j in range(i, min(i + maxlen, len(forms))):
                s += forms[j]
                out.add(s)
        return out

    ascii_pat = {n: re.compile(r"(?<![0-9A-Za-z&])" + re.escape(n) + r"(?![0-9A-Za-z&])")
                 for n in ascii_names}
    rows, corpus, events_by_nid = [], [], {}
    n_amb_reject = n_bound_reject = 0
    for (nid, title, ts), tks in zip(cand[["news_id", "title", "ts"]].itertuples(index=False),
                                     toks_list):
        # candidate names present in this title (longest-first, non-overlapping)
        raw = {m.group(0) for m in big.finditer(title)}
        if not raw:
            continue
        forms = [f for f, t in tks]
        concat = content = None                    # lazy
        matched = []
        for n in raw:
            if n in TIGHT:
                if not TIGHT[n].search(title):
                    n_amb_reject += 1
                    continue
            elif n in ascii_names:
                if not ascii_pat[n].search(title):
                    n_bound_reject += 1
                    continue
            else:
                if concat is None:
                    concat = consec_concats(forms)
                if n not in concat:
                    n_bound_reject += 1
                    continue
            if n in ambig and n not in TIGHT:
                if content is None:
                    content = set(forms)
                if not (content & CUE):
                    n_amb_reject += 1
                    continue
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
    log(f"precise links: {len(link)} over {link.news_id.nunique()} titles "
        f"(boundary-rejected {n_bound_reject}, ambiguity-rejected {n_amb_reject}); "
        f"events from {len(events_by_nid)}")
    obj = (link, corpus, events_by_nid)
    pickle.dump(obj, open(p, "wb"), protocol=4)
    return obj

# ------------------------------------------------------------------ stage 2 w2v
def build_w2v(corpus):
    p = os.path.join(ART, "kr41_w2v.npz")
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
    if len(centers) > 8_000_000:
        keep = rng.choice(len(centers), 8_000_000, replace=False)
        centers, ctx = centers[keep], ctx[keep]
    log(f"w2v pairs {len(centers)}")
    inp = nn.Embedding(V, D).to(DEV); out = nn.Embedding(V, D).to(DEV)
    nn.init.uniform_(inp.weight, -.5 / D, .5 / D); nn.init.zeros_(out.weight)
    opt = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=.02)
    negp = np.array([cnt[w] for w in vocab], float) ** .75; negp /= negp.sum()
    negtab = torch.from_numpy(rng.choice(V, size=3_000_000, p=negp)).to(DEV)
    ct = torch.from_numpy(centers).to(DEV); ot = torch.from_numpy(ctx).to(DEV)
    B = 8192
    for ep in range(4):
        perm = torch.randperm(len(centers), device=DEV); tot = 0.0; nb = 0
        for s in range(0, len(centers), B):
            idx = perm[s:s + B]
            c, o = ct[idx], ot[idx]
            ng = negtab[torch.randint(0, len(negtab), (len(idx), 5), device=DEV)]
            vc, vo, vn = inp(c), out(o), out(ng)
            loss = -(torch.nn.functional.logsigmoid((vc * vo).sum(1)) +
                     torch.nn.functional.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        log(f"  w2v epoch {ep+1}/4 loss={tot/nb:.4f}")
    W = inp.weight.detach().cpu().numpy().astype(np.float32)
    np.savez(p, vocab=np.array(vocab, object), W=W)
    return vocab, W

# ------------------------------------------------------------------ stage 3 NTN
class NTN(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        s = 1.0 / np.sqrt(d)
        self.T1 = nn.Parameter(torch.randn(d, d, k) * s)
        self.T2 = nn.Parameter(torch.randn(d, d, k) * s)
        self.T3 = nn.Parameter(torch.randn(k, k, k) * (1.0 / np.sqrt(k)))
        self.W1 = nn.Linear(2 * d, k); self.W2 = nn.Linear(2 * d, k)
        self.W3 = nn.Linear(2 * k, k)
        self.u = nn.Linear(k, 1, bias=False)

    @staticmethod
    def _bilinear(a, T, b):
        n = a.shape[0]; da, db, k = T.shape
        tmp = (a @ T.reshape(da, db * k)).view(n, db, k)
        return torch.einsum('ndk,nd->nk', tmp, b)

    def embed(self, O1, P, O2):
        R1 = torch.tanh(self._bilinear(O1, self.T1, P) + self.W1(torch.cat([O1, P], 1)))
        R2 = torch.tanh(self._bilinear(P, self.T2, O2) + self.W2(torch.cat([P, O2], 1)))
        return torch.tanh(self._bilinear(R1, self.T3, R2) + self.W3(torch.cat([R1, R2], 1)))

    def score(self, O1, P, O2):
        return self.u(self.embed(O1, P, O2)).squeeze(1)

    def l2(self):
        return sum((p ** 2).sum() for p in self.parameters())

def build_event_emb(events_by_nid, vocab, W):
    p = os.path.join(ART, "kr41_nid_emb.npz")
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
    tr = tr_idx if len(tr_idx) <= 80000 else rng.choice(tr_idx, 80000, replace=False)
    for it in range(10):
        order = rng.permutation(tr); th = 0.0; nb = 0; nsat = 0
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
            th += hinge.mean().item(); nb += 1; nsat += int((hinge <= 1e-6).sum())
        if it % 3 == 0 or it == 9:
            log(f"  NTN {it+1}/10 hinge={th/nb:.4f} sat={nsat/len(tr):.1%}")
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

# ------------------------------------------------------------------ head model
class Head(nn.Module):
    def __init__(self, d, n_filters=64, hidden=100, dropout=0.5):
        super().__init__()
        self.cl = nn.Conv1d(d, n_filters, 3)
        self.cm = nn.Conv1d(d, n_filters, 3)
        self.fc1 = nn.Linear(n_filters * 2 + d, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, short, mid, long):
        vl = torch.tanh(self.cl(long.transpose(1, 2))).max(dim=2).values
        vm = torch.tanh(self.cm(mid.transpose(1, 2))).max(dim=2).values
        y = torch.sigmoid(self.fc1(torch.cat([vl, vm, short], 1)))
        return torch.sigmoid(self.fc2(self.drop(y))).squeeze(1)

def fit_head(S, M, L, y01, trN, dvN, seed, epochs=120, patience=12):
    from sklearn.metrics import matthews_corrcoef
    torch.manual_seed(seed)
    m = Head(S.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCELoss()
    yt = torch.from_numpy(y01.astype(np.float32)).to(DEV)
    def pred(idx, batch=4096):
        m.eval(); out = np.zeros(len(idx), np.float32)
        with torch.no_grad():
            for s in range(0, len(idx), batch):
                ii = torch.from_numpy(idx[s:s + batch]).to(DEV)
                out[s:s + batch] = m(S[ii], M[ii], L[ii]).cpu().numpy()
        return out
    import copy
    best, best_state, bad = -2.0, copy.deepcopy(m.state_dict()), 0
    for ep in range(epochs):
        m.train()
        perm = np.random.default_rng(seed + ep).permutation(trN)
        for s in range(0, len(perm), 128):
            ii = torch.from_numpy(perm[s:s + 128]).to(DEV)
            out = m(S[ii], M[ii], L[ii])
            loss = lossf(out, yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
        dp = pred(dvN)
        pr = (dp > 0.5).astype(int)
        score = matthews_corrcoef(y01[dvN], pr) if len(np.unique(pr)) > 1 else 0.0
        if score > best:
            best, best_state, bad = score, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
        if bad >= patience:
            break
    m.load_state_dict(best_state)
    return m, pred, best

# ------------------------------------------------------------------ stage 4
def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, accuracy_score
    link, corpus, events_by_nid = build_corpus()
    vocab, W = build_w2v(corpus)
    nid2emb = build_event_emb(events_by_nid, vocab, W)

    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    ohlcv = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)
    cap = pd.read_parquet(os.path.join(DATA, "kr_market_cap_daily.parquet"))
    cap["date"] = pd.to_datetime(cap.trade_date)
    medcap = cap.groupby("ticker").market_cap.median()
    medcap = medcap[medcap.index.isin(set(ohlcv.ticker.unique()))]
    small = set(medcap[medcap <= medcap.median()].index)
    log(f"ohlcv {len(ohlcv)} rows, small-cap half {len(small)} tickers")

    # --- eff/info/fresh mapping over the global trading calendar
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
    log(f"usable links {len(le)}, fresh {le.fresh.mean():.1%}")

    def bucket(df, key):
        g = df.dropna(subset=[key]).groupby(["ticker", key]).emb_i.apply(
            lambda s: E[s.values].mean(0))
        return {(t, d): v for (t, d), v in g.items()}
    over = bucket(le[le.fresh], "eff")
    info = bucket(le, "info")
    log(f"buckets: overnight {len(over)}, info {len(info)}")

    # --- samples on small caps
    Z = np.zeros(D, np.float32)
    td_pos = {d: i for i, d in enumerate(TD)}
    S, M_, L, DT, meta = [], [], [], [], []
    g_ohlcv = ohlcv[ohlcv.ticker.isin(small)].groupby("ticker")
    for t, g in g_ohlcv:
        g = g.sort_values("date")
        o = g.open.values.astype(float); h = g.high.values.astype(float)
        lo = g.low.values.astype(float); c = g.close.values.astype(float)
        v = g.volume.values.astype(float); gd = g.date.values
        hl = (h - lo) / np.where(c > 0, c, np.nan)
        r1 = c[1:] / c[:-1] - 1.0
        for i in range(21, len(g)):
            if o[i] <= 0 or v[i] <= 0 or c[i - 1] <= 0:
                continue
            gap_open = o[i] / c[i - 1] - 1.0
            if gap_open >= 0.295:                 # locked limit-up open: not buyable
                continue
            a, di = gd[i - 1], gd[i]
            lo_p, hi_p = td_pos[a], td_pos[di]
            vs = [over[(t, TD[k])] for k in range(lo_p + 1, hi_p + 1) if (t, TD[k]) in over]
            S.append(np.mean(vs, 0) if vs else Z)
            p = td_pos[di]
            M_.append(np.stack([info.get((t, TD[p - k]), Z) if p - k >= 0 else Z
                                for k in range(MID_DAYS, 0, -1)]))
            L.append(np.stack([info.get((t, TD[p - k]), Z) if p - k >= 0 else Z
                               for k in range(LONG_DAYS, 0, -1)]))
            DT.append(di)
            hit = tuple(float(h[i] >= o[i] * (1 + k)) for k in KS)
            pnl = tuple(float((k if h[i] >= o[i] * (1 + k) else c[i] / o[i] - 1.0) - COST)
                        for k in KS)
            # stop-loss variant, SL=-3%, PESSIMISTIC fill: if both TP and SL are
            # touched intraday, assume SL fired first (OHLC cannot order them)
            SL = 0.03
            sl_hit = lo[i] <= o[i] * (1 - SL)
            pnl_sl = tuple(float((-SL if sl_hit else
                                  (k if h[i] >= o[i] * (1 + k) else c[i] / o[i] - 1.0)) - COST)
                           for k in KS)
            vol20 = float(np.nanmean(hl[max(0, i - 20):i]))
            std20 = float(np.std(r1[max(0, i - 21):i - 1])) if i >= 3 else 0.0
            prev_hl = float(hl[i - 1]) if np.isfinite(hl[i - 1]) else 0.0
            meta.append((t, hit, pnl, gap_open, vol20, std20, prev_hl,
                         float(abs(r1[i - 2])) if i >= 2 else 0.0, pnl_sl))
    S = np.asarray(S, np.float32); M_ = np.asarray(M_, np.float32); L = np.asarray(L, np.float32)
    DT = np.asarray(DT, dtype="datetime64[ns]")
    has_short = (S != 0).any(1)
    has_any = has_short | (M_ != 0).any((1, 2)) | (L != 0).any((1, 2))
    HIT = np.array([m[1] for m in meta], np.float32)
    PNL = np.array([m[2] for m in meta], np.float32)
    PNLSL = np.array([m[8] for m in meta], np.float32)
    VOLF = np.array([[m[3], m[4], m[5], m[6], m[7]] for m in meta], np.float32)
    VOLF = np.nan_to_num(VOLF)
    log(f"samples {len(S)}  overnight {has_short.mean():.1%}  any {has_any.mean():.1%}")
    np.savez(os.path.join(ART, "kr41_features.npz"), S=S, M=M_, L=L,
             DT=DT.astype("datetime64[D]").astype(str), HIT=HIT, PNL=PNL, PNLSL=PNLSL,
             VOLF=VOLF, has_short=has_short, has_any=has_any)

    # --- event-window training set
    keep = np.where(has_any)[0]
    St = torch.from_numpy(S[keep]).to(DEV)
    Mt = torch.from_numpy(M_[keep]).to(DEV)
    Lt = torch.from_numpy(L[keep]).to(DEV)
    dtk = DT[keep]; hsk = has_short[keep]
    hitk = HIT[keep]; pnlk = PNL[keep]; pnlslk = PNLSL[keep]; volk = VOLF[keep]
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(dtk < split)[0]; teN = np.where(dtk >= split)[0]
    trd = np.sort(np.unique(dtk[dtk < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[dtk[trN] >= dsplit]; trN = trN[dtk[trN] < dsplit]
    log(f"train {len(trN)} dev {len(dvN)} test {len(teN)}")

    from scipy.stats import binomtest
    def report(tag, y, prob, mask, pnl, pnlsl):
        yy = y[mask].astype(int); pp = prob[mask]; pn = pnl[mask]; ps = pnlsl[mask]
        if len(yy) < 30:
            print(f"    [{tag}] too few n={len(yy)}"); return
        base = yy.mean()
        pred = (pp > 0.5).astype(int)
        acc = accuracy_score(yy, pred)
        mcc = matthews_corrcoef(yy, pred) if len(np.unique(pred)) > 1 else 0.0
        print(f"    [{tag}] n={len(yy)} base(hit-rate)={base:.4f} acc={acc:.4f} mcc={mcc:+.4f}")
        order = np.argsort(-pp)
        for kap in (0.3, 0.2, 0.1, 0.05):
            nsel = max(int(len(yy) * kap), 20)
            sel = order[:nsel]
            hits = int(yy[sel].sum())
            pv = binomtest(hits, nsel, base, alternative="greater").pvalue
            print(f"       top{int(kap*100):2d}% conf: hit-rate={yy[sel].mean():.4f} "
                  f"(vs base {base:.4f}, p={pv:.1e})  pnl/trade={pn[sel].mean()*100:+.3f}% "
                  f"| SL-3%: {ps[sel].mean()*100:+.3f}%  n={nsel}")

    for ki, k in enumerate(KS):
        y01 = hitk[:, ki].astype(int)
        pnl = pnlk[:, ki]; pnsl = pnlslk[:, ki]
        print(f"\n=== HIGH target k={k:.0%}  (train hit-rate {y01[trN].mean():.4f}) ===", flush=True)
        te_mask = np.isin(np.arange(len(y01)), teN)
        te_short = te_mask & hsk

        # VOL-LR baseline
        lr = LogisticRegression(max_iter=2000).fit(volk[trN], y01[trN])
        p_vol = lr.predict_proba(volk)[:, 1]
        report("VOL-LR   | all-event test", y01, p_vol, te_mask, pnl, pnsl)
        report("VOL-LR   | overnight-news test", y01, p_vol, te_short, pnl, pnsl)

        # EB-CNN (2-seed ensemble)
        tps = []
        pred_fn = None
        for sd in (SEED, SEED + 1):
            m, pred_fn, bd = fit_head(St, Mt, Lt, y01, trN, dvN, seed=sd)
            tps.append(pred_fn(np.arange(len(y01))))
        p_cnn = np.mean(tps, 0)
        report("EB-CNN   | all-event test", y01, p_cnn, te_mask, pnl, pnsl)
        report("EB-CNN   | overnight-news test", y01, p_cnn, te_short, pnl, pnsl)

        # STACK: LR on [vol feats, p_cnn] fit on DEV
        Xs = np.column_stack([volk, p_cnn])
        sl = LogisticRegression(max_iter=2000).fit(Xs[dvN], y01[dvN])
        p_stack = sl.predict_proba(Xs)[:, 1]
        report("STACK    | all-event test", y01, p_stack, te_mask, pnl, pnsl)
        report("STACK    | overnight-news test", y01, p_stack, te_short, pnl, pnsl)

if __name__ == "__main__":
    main()
