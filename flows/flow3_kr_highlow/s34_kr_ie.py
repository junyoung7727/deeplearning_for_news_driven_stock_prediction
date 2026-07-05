"""
Stage 34 - KR small-cap event-driven prediction (paper's IE->NTN->CNN logic, NO
sentiment). Korean SVO events via Kiwi, skip-gram KR word vectors, NTN event
embeddings, long/mid/short CNN -> next-day direction on small-caps.
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
import os, glob, re, time, numpy as np, pandas as pd, torch
import config as C
from kiwipiepy import Kiwi
from s4_ntn import NTN
from s6_models import DenseModel, fit, predict, metrics
torch.manual_seed(C.SEED); np.random.seed(C.SEED)
BK = os.path.join(C.DATA_ROOT, "news", "bigkinds")
BASE = os.path.join(C.DATA_ROOT, "analysis_outputs", "kr_ff5_foreign_regression_20260619T105218Z")
D = C.WORD_DIM
NOUN = {"NNG", "NNP", "SL", "SN", "SH"}

def kr_events(tokens):
    """particle-based Korean SVO. tokens: list of (form,tag). returns (o1,p,o2) morph-lists."""
    nps, cur = [], []          # noun phrases (merged consecutive nouns)
    marks = []                 # (np_index, 'S'|'O') from following particle
    preds = []                 # predicate morphs (action noun before 하/되, or VV/VA stem)
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
                preds.append(nps[-1])           # action-noun (출시/인수/계약)
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

def main():
    t0 = time.time()
    # ---- prices + small-cap flag ----
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"),
                          columns=["ticker", "trade_date", "ret_1d"])
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"),
                          columns=["ticker", "trade_date", "market_cap"])
    sec["date"] = pd.to_datetime(sec.trade_date); cap["date"] = pd.to_datetime(cap.trade_date)
    px = sec.merge(cap[["ticker", "date", "market_cap"]], on=["ticker", "date"], how="left")
    px = px[px.ret_1d.notna()].sort_values(["ticker", "date"]).reset_index(drop=True)
    medcap = px.groupby("ticker").market_cap.median()
    small = set(medcap[medcap <= medcap.median()].index)
    all_tk = set(px.ticker.unique())

    # ---- universe names ----
    uni = pd.read_parquet(os.path.join(BK, "kr_universe_enriched.parquet"))
    uni = uni[uni.ticker.isin(all_tk)]
    name2t = {}
    for _, r in uni.iterrows():
        cand = {r["name"]} if isinstance(r["name"], str) and len(r["name"]) >= 2 else set()
        try:
            cand |= {a for a in (r["aliases"] or []) if isinstance(a, str) and len(a) >= 3}
        except Exception:
            pass
        for c in cand:
            name2t.setdefault(c, []).append(r.ticker)

    # ---- link 2024+ news to all priced tickers ----
    files = [f for f in glob.glob(os.path.join(BK, "econ_*.parquet"))
             if re.search(r"econ_(\d{4}-\d{2}-\d{2})_", f).group(1) >= "2024-01-01"]
    news = pd.concat([pd.read_parquet(f, columns=["news_id", "published_at", "title"]) for f in files], ignore_index=True)
    news["date"] = pd.to_datetime(news.published_at).dt.normalize()
    big = re.compile("|".join(sorted((re.escape(k) for k in name2t), key=len, reverse=True)))
    hit = news[news.title.str.contains(big, na=False)].drop_duplicates("news_id").reset_index(drop=True)
    print(f"linked-news titles: {len(hit)}  ({time.time()-t0:.0f}s)")

    # ---- Kiwi tokenize + events (once per title) ----
    kiwi = Kiwi(num_workers=4)
    titles = hit.title.tolist()
    toks_list = [[(t.form, t.tag) for t in s] for s in kiwi.tokenize(titles)]
    hit_tokens = toks_list
    events_by_nid = {}; corpus = []
    for nid, tk in zip(hit.news_id.values, hit_tokens):
        corpus.append(tok_content(tk))
        ev = kr_events(tk)
        if ev: events_by_nid[nid] = ev
    print(f"tokenized {len(titles)} titles, events from {len(events_by_nid)} "
          f"({len(events_by_nid)/len(titles):.1%})  ({time.time()-t0:.0f}s)")

    # map news->ticker(s)
    rows = []
    for name, tks in name2t.items():
        sub = hit[hit.title.str.contains(re.escape(name), na=False)]
        for t in tks:
            for nid, dt in zip(sub.news_id.values, sub.date.values):
                rows.append((t, dt, nid))
    link = pd.DataFrame(rows, columns=["ticker", "date", "news_id"]).drop_duplicates()
    link["date"] = pd.to_datetime(link.date)
    print(f"(ticker,news) links: {len(link)}")

    # ---- KR skip-gram word vectors ----
    from collections import Counter
    cnt = Counter(w for s in corpus for w in s)
    vocab = [w for w, c in cnt.most_common() if c >= 5]; w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab); print("KR vocab:", V)
    rng = np.random.default_rng(C.SEED)
    centers, ctx = [], []
    for s in corpus:
        ids = [w2i[w] for w in s if w in w2i]
        for pos in range(len(ids)):
            w = rng.integers(1, 6)
            for j in range(max(0, pos-w), min(len(ids), pos+w+1)):
                if j != pos: centers.append(ids[pos]); ctx.append(ids[j])
    centers = np.array(centers); ctx = np.array(ctx); print("w2v pairs:", len(centers))
    if len(centers) > 8_000_000:
        keep = rng.choice(len(centers), 8_000_000, replace=False)
        centers, ctx = centers[keep], ctx[keep]; print("subsampled w2v pairs to", len(centers))
    inp = torch.nn.Embedding(V, D); out = torch.nn.Embedding(V, D)
    torch.nn.init.uniform_(inp.weight, -.5/D, .5/D); torch.nn.init.zeros_(out.weight)
    opt = torch.optim.Adam(list(inp.parameters())+list(out.parameters()), lr=.02)
    negp = np.array([cnt[w] for w in vocab], float)**.75; negp /= negp.sum()
    negtab = rng.choice(V, size=3_000_000, p=negp)
    B = 4096
    for ep in range(4):
        perm = rng.permutation(len(centers)); tot = 0.0; nb = 0
        for s in range(0, len(centers), B):
            idx = perm[s:s+B]
            c = torch.from_numpy(centers[idx]); o = torch.from_numpy(ctx[idx])
            ng = torch.from_numpy(negtab[rng.integers(0, len(negtab), size=(len(idx), 5))])
            vc, vo, vn = inp(c), out(o), out(ng)
            loss = -(torch.nn.functional.logsigmoid((vc*vo).sum(1)) +
                     torch.nn.functional.logsigmoid(-(vn*vc.unsqueeze(1)).sum(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  w2v epoch {ep+1}/4 loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    W = inp.weight.detach().numpy().astype(np.float32)
    print(f"KR w2v done ({time.time()-t0:.0f}s)")

    # ---- events table + NTN ----
    ev = pd.DataFrame([(nid, *events_by_nid[nid]) for nid in events_by_nid],
                      columns=["news_id", "o1", "p", "o2"])
    def avg(ws):
        idx = [w2i[w] for w in ws if w in w2i]
        return W[idx].mean(0) if idx else None
    O1 = np.zeros((len(ev), D), np.float32); P = np.zeros_like(O1); O2 = np.zeros_like(O1); ok = np.zeros(len(ev), bool)
    for i, (a, p, b) in enumerate(zip(ev.o1, ev.p, ev.o2)):
        va, vp, vb = avg(a), avg(p), avg(b)
        if va is not None and vp is not None and vb is not None:
            O1[i], P[i], O2[i] = va, vp, vb; ok[i] = True
    O1t, Pt, O2t = map(torch.from_numpy, (O1, P, O2)); Wt = torch.from_numpy(W)
    tr_idx = np.where(ok)[0]
    ntn = NTN(D, C.NTN_K); nopt = torch.optim.Adam(ntn.parameters(), lr=C.NTN_LR)
    tr_ntn = tr_idx if len(tr_idx) <= 80000 else rng.choice(tr_idx, 80000, replace=False)
    for it in range(10):
        order = rng.permutation(tr_ntn); tot = 0.0; nb = 0
        for s in range(0, len(order), 512):
            ii = torch.from_numpy(order[s:s+512]); o1, p, o2 = O1t[ii], Pt[ii], O2t[ii]
            ch = torch.from_numpy(rng.integers(0, 2, len(ii)))
            rv = Wt[torch.from_numpy(rng.integers(0, len(W), len(ii)))]
            co1 = torch.where((ch == 0).unsqueeze(1), rv, o1); co2 = torch.where((ch == 1).unsqueeze(1), rv, o2)
            hinge = torch.clamp(1 - ntn.score(o1, p, o2) + ntn.score(co1, p, co2), min=0)
            loss = hinge.mean() + C.NTN_LAMBDA*ntn.l2(); nopt.zero_grad(); loss.backward(); nopt.step()
            tot += loss.item(); nb += 1
        if it % 3 == 0 or it == 9:
            print(f"  NTN epoch {it+1}/10 loss={tot/nb:.3f} ({time.time()-t0:.0f}s)", flush=True)
    emb = np.zeros((len(ev), C.NTN_K), np.float32)
    with torch.no_grad():
        for s in range(0, len(tr_idx), 4096):
            ii = torch.from_numpy(tr_idx[s:s+4096]); emb[tr_idx[s:s+4096]] = ntn.embed(O1t[ii], Pt[ii], O2t[ii]).numpy()
    mu, sd = emb[ok].mean(0), emb[ok].std(0)+1e-6; emb = (emb-mu)/sd; emb[~ok] = 0
    nid2emb = {nid: emb[i] for i, nid in enumerate(ev.news_id.values) if ok[i]}
    print(f"KR NTN done, event embs={len(nid2emb)} ({time.time()-t0:.0f}s)")

    # ---- daily event emb per ticker ----
    link["has"] = link.news_id.map(lambda n: n in nid2emb)
    le = link[link.has].copy()
    le["emb"] = le.news_id.map(nid2emb)
    day = le.groupby(["ticker", "date"]).emb.apply(lambda s: np.mean(np.stack(s.values), 0))
    dayd = {(t, d): v for (t, d), v in day.items()}

    # ---- build small-cap samples: short/mid/long event-emb + CNN ----
    pxs = px[px.ticker.isin(small)].reset_index(drop=True)
    def seq(t, end, n):
        return np.stack([dayd.get((t, end - pd.Timedelta(days=k)), np.zeros(C.NTN_K, np.float32)) for k in range(n-1, -1, -1)])
    S, M_, L, Y, DT = [], [], [], [], []
    for t, g in pxs.groupby("ticker"):
        g = g.sort_values("date")
        for i in range(1, len(g)):
            a = g.date.iloc[i-1]; di = g.date.iloc[i]
            S.append(dayd.get((t, a), np.zeros(C.NTN_K, np.float32)))
            M_.append(seq(t, a, C.MID_DAYS)); L.append(seq(t, a, C.LONG_DAYS))
            Y.append(int(g.ret_1d.iloc[i] > 0)); DT.append(di)
    S = torch.from_numpy(np.asarray(S, np.float32)); M_ = torch.from_numpy(np.asarray(M_, np.float32))
    L = torch.from_numpy(np.asarray(L, np.float32)); y = np.asarray(Y); DT = np.array(DT, dtype="datetime64[ns]")
    nz = (S.numpy() != 0).any(1)
    print(f"small-cap samples: {len(y)}  with short-event: {nz.mean():.1%}  ({time.time()-t0:.0f}s)")

    dates = np.sort(np.unique(DT)); split = dates[int(len(dates)*0.6)]
    tr = DT < split; te = DT >= split
    trN = np.where(tr)[0]; teN = np.where(te)[0]; dvN = trN[int(len(trN)*0.85):]; trN = trN[:int(len(trN)*0.85)]
    def inp(ix): return (S[ix], M_[ix], L[ix])
    m = DenseModel(C.NTN_K, nn_only=False)
    m, _ = fit(m, inp, y, trN, dvN, seed=C.SEED)
    p = predict(m, inp, teN)
    ypm = np.where(y == 1, 1, -1)
    Sn, Mn, Ln = S.numpy(), M_.numpy(), L.numpy()
    has_short = (Sn != 0).any(1)
    has_any = has_short | (Mn != 0).any((1, 2)) | (Ln != 0).any((1, 2))
    np.savez(os.path.join(C.ART, "kr_ie_features.npz"), S=Sn, M=Mn, L=Ln, y=y,
             DT=DT.astype("datetime64[D]").astype(str), has_short=has_short, has_any=has_any)
    print(f"\nKR SMALL-CAP EB-CNN (IE->NTN->CNN, no sentiment): "
          f"ALL-test acc={metrics(ypm[teN], p)[0]:.4f} n={len(teN)}")
    def rep(name, mask):
        mt = mask[teN]
        if mt.sum() < 30:
            print(f"  [{name}] too few (n={int(mt.sum())})"); return
        yy = ypm[teN][mt]; pp = p[mt]
        base = max((yy == 1).mean(), (yy == -1).mean())
        a, mc = metrics(yy, pp)
        print(f"  [{name}] n={int(mt.sum())} base={base:.4f} acc={a:.4f} mcc={mc:.4f}")
        cc = np.abs(pp - 0.5)
        for kap in [0.5, 0.3, 0.2]:
            thr = np.quantile(cc, 1 - kap); s = cc >= thr
            if s.sum() >= 10:
                print(f"     conf top {int(kap*100)}%: acc={metrics(yy[s], pp[s])[0]:.4f} (n={int(s.sum())})")
    rep("EVENT-days (event on prior trading day)", has_short)
    rep("ANY-event-in-window (30d) days", has_any)

if __name__ == "__main__":
    main()
