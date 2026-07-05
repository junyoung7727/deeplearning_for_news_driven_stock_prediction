"""
Stage 36 - KR small-cap event-driven prediction with CORRECT news-time alignment.

Root-cause fix over s34 (which keyed news by calendar date of the PRIOR trading
day):
  * KRX trades 09:00-15:30 KST. News published before 15:30 of day a is already
    in close(a): using it to predict ret(a+1) is stale dilution (~68% of volume).
  * News published 00:00-09:00 of the target day di (pre-open) and on weekends/
    holidays never reached the short-term slot at all - yet the US diagnosis
    (s20-s23) showed overnight news -> open gap is the ONLY measurable relation.

New alignment (all leak-free w.r.t. a decision at open(di), 09:00 KST):
  short(di) = mean emb of news in (15:30 of prev trading day a, 09:00 of di]
              == the overnight window, incl. weekends/halts.
  info-day(tau) = first trading day d with 15:30(d) >= tau  (i.e. the day by
              whose close the news is known).
  mid(di)  = per-info-day mean emb for the 7  trading days before di.
  long(di) = per-info-day mean emb for the 30 trading days before di.

Representation (Kiwi SVO -> skip-gram -> NTN) is IDENTICAL to s34 so the effect
of alignment is isolated. Stages are cached under artifacts/kr36_*.

Training protocol: event-window samples only (s35 finding: training on 96.6%
all-zero samples collapses the model to the base rate); time-based dev split.
NTN logging fixed: hinge and lambda*L2 reported separately + margin-satisfied%.
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
import os, glob, re, time, pickle, numpy as np, pandas as pd, torch
import config as C
from s4_ntn import NTN
from s6_models import DenseModel, fit, predict, metrics

torch.manual_seed(C.SEED); np.random.seed(C.SEED)
BK = r"D:\Github\homeserver\alphamale\data\news\bigkinds"
BASE = r"D:\Github\homeserver\alphamale\data\analysis_outputs\kr_ff5_foreign_regression_20260619T105218Z"
D = C.WORD_DIM
NOUN = {"NNG", "NNP", "SL", "SN", "SH"}
CACHE_CORPUS = os.path.join(C.ART, "kr36_corpus.pkl")
CACHE_W2V = os.path.join(C.ART, "kr36_w2v.npz")
CACHE_EMB = os.path.join(C.ART, "kr36_nid_emb.npz")
OPEN_T, CLOSE_T = pd.Timedelta(hours=9), pd.Timedelta(hours=15, minutes=30)

from s34_kr_ie import kr_events, tok_content   # identical extraction

def log(msg, t0=[time.time()]):
    print(f"{msg}  ({time.time()-t0[0]:.0f}s)", flush=True)

# ---------------------------------------------------------------- stage 1: corpus
def build_corpus():
    if os.path.exists(CACHE_CORPUS):
        with open(CACHE_CORPUS, "rb") as f:
            return pickle.load(f)
    sec = pd.read_parquet(os.path.join(BASE, "kr_ff5_security_factor_daily.parquet"),
                          columns=["ticker", "trade_date", "ret_1d"])
    cap = pd.read_parquet(os.path.join(BASE, "kr_market_cap_daily.parquet"),
                          columns=["ticker", "trade_date", "market_cap"])
    sec["date"] = pd.to_datetime(sec.trade_date); cap["date"] = pd.to_datetime(cap.trade_date)
    px = sec.merge(cap[["ticker", "date", "market_cap"]], on=["ticker", "date"], how="left")
    px = px[px.ret_1d.notna()].sort_values(["ticker", "date"]).reset_index(drop=True)
    px = px[["ticker", "date", "ret_1d", "market_cap"]]

    uni = pd.read_parquet(os.path.join(BK, "kr_universe_enriched.parquet"))
    uni = uni[uni.ticker.isin(set(px.ticker.unique()))]
    name2t = {}
    for _, r in uni.iterrows():
        cand = {r["name"]} if isinstance(r["name"], str) and len(r["name"]) >= 2 else set()
        try:
            cand |= {a for a in (r["aliases"] or []) if isinstance(a, str) and len(a) >= 3}
        except Exception:
            pass
        for c in cand:
            name2t.setdefault(c, []).append(r.ticker)

    files = [f for f in glob.glob(os.path.join(BK, "econ_*.parquet"))
             if re.search(r"econ_(\d{4}-\d{2}-\d{2})_", f).group(1) >= "2024-01-01"]
    news = pd.concat([pd.read_parquet(f, columns=["news_id", "published_at", "title"])
                      for f in files], ignore_index=True)
    news["ts"] = pd.to_datetime(news.published_at)          # KST, second precision
    big = re.compile("|".join(sorted((re.escape(k) for k in name2t), key=len, reverse=True)))
    hit = news[news.title.str.contains(big, na=False)].drop_duplicates("news_id").reset_index(drop=True)
    log(f"linked-news titles: {len(hit)}")

    from kiwipiepy import Kiwi
    kiwi = Kiwi(num_workers=4)
    toks_list = [[(t.form, t.tag) for t in s] for s in kiwi.tokenize(hit.title.tolist())]
    events_by_nid, corpus = {}, []
    for nid, tk in zip(hit.news_id.values, toks_list):
        corpus.append(tok_content(tk))
        ev = kr_events(tk)
        if ev: events_by_nid[nid] = ev
    log(f"tokenized {len(toks_list)} titles, events from {len(events_by_nid)} "
        f"({len(events_by_nid)/len(toks_list):.1%})")

    rows = []
    for name, tks in name2t.items():
        sub = hit[hit.title.str.contains(re.escape(name), na=False)]
        for t in tks:
            for nid, ts in zip(sub.news_id.values, sub.ts.values):
                rows.append((t, ts, nid))
    link = pd.DataFrame(rows, columns=["ticker", "ts", "news_id"]).drop_duplicates()
    log(f"(ticker,news) links: {len(link)}")
    obj = (px, link, corpus, events_by_nid)
    with open(CACHE_CORPUS, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return obj

# ---------------------------------------------------------------- stage 2: w2v
def build_w2v(corpus):
    if os.path.exists(CACHE_W2V):
        z = np.load(CACHE_W2V, allow_pickle=True)
        return list(z["vocab"]), z["W"]
    from collections import Counter
    rng = np.random.default_rng(C.SEED)
    cnt = Counter(w for s in corpus for w in s)
    vocab = [w for w, c in cnt.most_common() if c >= 5]; w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab); log(f"KR vocab: {V}")
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
    log(f"w2v pairs: {len(centers)}")
    inp = torch.nn.Embedding(V, D); out = torch.nn.Embedding(V, D)
    torch.nn.init.uniform_(inp.weight, -.5 / D, .5 / D); torch.nn.init.zeros_(out.weight)
    opt = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=.02)
    negp = np.array([cnt[w] for w in vocab], float) ** .75; negp /= negp.sum()
    negtab = rng.choice(V, size=3_000_000, p=negp)
    B = 4096
    for ep in range(4):
        perm = rng.permutation(len(centers)); tot = 0.0; nb = 0
        for s in range(0, len(centers), B):
            idx = perm[s:s + B]
            c = torch.from_numpy(centers[idx]); o = torch.from_numpy(ctx[idx])
            ng = torch.from_numpy(negtab[rng.integers(0, len(negtab), size=(len(idx), 5))])
            vc, vo, vn = inp(c), out(o), out(ng)
            loss = -(torch.nn.functional.logsigmoid((vc * vo).sum(1)) +
                     torch.nn.functional.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        log(f"  w2v epoch {ep+1}/4 loss={tot/nb:.4f}")
    W = inp.weight.detach().numpy().astype(np.float32)
    np.savez(CACHE_W2V, vocab=np.array(vocab, object), W=W)
    return vocab, W

# ---------------------------------------------------------------- stage 3: NTN
def build_event_emb(events_by_nid, vocab, W):
    if os.path.exists(CACHE_EMB):
        z = np.load(CACHE_EMB, allow_pickle=True)
        return dict(zip(z["nids"], z["emb"]))
    rng = np.random.default_rng(C.SEED)
    w2i = {w: i for i, w in enumerate(vocab)}
    ev = pd.DataFrame([(nid, *events_by_nid[nid]) for nid in events_by_nid],
                      columns=["news_id", "o1", "p", "o2"])
    def avg(ws):
        idx = [w2i[w] for w in ws if w in w2i]
        return W[idx].mean(0) if idx else None
    O1 = np.zeros((len(ev), D), np.float32); P = np.zeros_like(O1); O2 = np.zeros_like(O1)
    ok = np.zeros(len(ev), bool)
    for i, (a, p, b) in enumerate(zip(ev.o1, ev.p, ev.o2)):
        va, vp, vb = avg(a), avg(p), avg(b)
        if va is not None and vp is not None and vb is not None:
            O1[i], P[i], O2[i] = va, vp, vb; ok[i] = True
    O1t, Pt, O2t = map(torch.from_numpy, (O1, P, O2)); Wt = torch.from_numpy(W)
    tr_idx = np.where(ok)[0]
    ntn = NTN(D, C.NTN_K); nopt = torch.optim.Adam(ntn.parameters(), lr=C.NTN_LR)
    tr_ntn = tr_idx if len(tr_idx) <= 80000 else rng.choice(tr_idx, 80000, replace=False)
    for it in range(10):
        order = rng.permutation(tr_ntn); th = 0.0; nb = 0; nsat = 0
        for s in range(0, len(order), 512):
            ii = torch.from_numpy(order[s:s + 512]); o1, p, o2 = O1t[ii], Pt[ii], O2t[ii]
            ch = torch.from_numpy(rng.integers(0, 2, len(ii)))
            rv = Wt[torch.from_numpy(rng.integers(0, len(W), len(ii)))]
            co1 = torch.where((ch == 0).unsqueeze(1), rv, o1)
            co2 = torch.where((ch == 1).unsqueeze(1), rv, o2)
            hinge = torch.clamp(1 - ntn.score(o1, p, o2) + ntn.score(co1, p, co2), min=0)
            loss = hinge.mean() + C.NTN_LAMBDA * ntn.l2()
            nopt.zero_grad(); loss.backward(); nopt.step()
            th += hinge.mean().item(); nb += 1; nsat += int((hinge <= 1e-6).sum())
        if it % 3 == 0 or it == 9:
            l2 = C.NTN_LAMBDA * ntn.l2().item()
            log(f"  NTN epoch {it+1}/10 hinge={th/nb:.4f} lam*L2={l2:.3f} "
                f"margin-satisfied={nsat/len(tr_ntn):.1%}")
    emb = np.zeros((len(ev), C.NTN_K), np.float32)
    with torch.no_grad():
        for s in range(0, len(tr_idx), 4096):
            ii = torch.from_numpy(tr_idx[s:s + 4096])
            emb[tr_idx[s:s + 4096]] = ntn.embed(O1t[ii], Pt[ii], O2t[ii]).numpy()
    mu, sd = emb[ok].mean(0), emb[ok].std(0) + 1e-6
    emb = (emb - mu) / sd; emb[~ok] = 0
    nids = ev.news_id.values[ok]
    np.savez(CACHE_EMB, nids=nids, emb=emb[ok])
    log(f"KR NTN done, event embs={int(ok.sum())}")
    return dict(zip(nids, emb[ok]))

# ---------------------------------------------------------------- stage 4: features
def main():
    px, link, corpus, events_by_nid = build_corpus()
    vocab, W = build_w2v(corpus)
    nid2emb = build_event_emb(events_by_nid, vocab, W)

    # --- effective-day mapping (vectorised searchsorted over global trading days)
    TD = np.sort(px.date.unique())                       # datetime64[ns] trading days
    open_ts = TD + np.timedelta64(9, "h")               # 09:00 KST
    close_ts = TD + np.timedelta64(15 * 60 + 30, "m")   # 15:30 KST
    le = link[link.news_id.isin(nid2emb)].copy()
    ts = le.ts.values.astype("datetime64[ns]")
    # overnight bucket: first trading day whose OPEN is >= ts  (news at 09:00:00
    # sharp counts for that open)
    eff_i = np.searchsorted(open_ts, ts, side="left")
    # info day: first trading day whose CLOSE is >= ts (known by that close)
    info_i = np.searchsorted(close_ts, ts, side="left")
    valid = eff_i < len(TD)
    le = le[valid].reset_index(drop=True)
    le["eff"] = TD[eff_i[valid]]
    ii = info_i[valid]
    le["info"] = TD[np.minimum(ii, len(TD) - 1)]
    le.loc[ii >= len(TD), "info"] = pd.NaT               # after last close: unusable
    le["fresh"] = eff_i[valid] == ii                     # post-close/weekend/pre-open:
                                                         # not yet priced into ANY close
    le["emb_i"] = np.arange(len(le))
    E = np.stack([nid2emb[n] for n in le.news_id.values]).astype(np.float32)
    FRESH = os.environ.get("KR36_FRESH", "0") == "1"
    log(f"usable links: {len(le)}  fresh share: {le.fresh.mean():.1%}  FRESH_ONLY={FRESH}")

    # per (ticker, eff-day) overnight mean emb; per (ticker, info-day) daily mean emb
    def bucket(df, key):
        g = df.dropna(subset=[key]).groupby(["ticker", key]).emb_i.apply(
            lambda s: E[s.values].mean(0))
        return {(t, d): v for (t, d), v in g.items()}
    over = bucket(le[le.fresh] if FRESH else le, "eff")
    info = bucket(le, "info")
    log(f"overnight buckets: {len(over)}  info buckets: {len(info)}")

    # --- small-cap universe
    medcap = px.groupby("ticker").market_cap.median()
    small = set(medcap[medcap <= medcap.median()].index)
    pxs = px[px.ticker.isin(small)].reset_index(drop=True)
    td_pos = {d: i for i, d in enumerate(TD)}
    Z = np.zeros(C.NTN_K, np.float32)

    S, M_, L, Y, DTs = [], [], [], [], []
    for t, g in pxs.groupby("ticker"):
        g = g.sort_values("date")
        gd = g.date.values
        for i in range(1, len(g)):
            a, di = gd[i - 1], gd[i]
            # overnight window (a 15:30, di 09:00]: eff days in (a, di] global TDs
            # (halt gaps: aggregate all eff buckets in between)
            lo, hi = td_pos[a], td_pos[di]
            vs = [over[(t, TD[k])] for k in range(lo + 1, hi + 1) if (t, TD[k]) in over]
            S.append(np.mean(vs, 0) if vs else Z)
            p = td_pos[di]
            M_.append(np.stack([info.get((t, TD[p - k]), Z) if p - k >= 0 else Z
                                for k in range(C.MID_DAYS, 0, -1)]))
            L.append(np.stack([info.get((t, TD[p - k]), Z) if p - k >= 0 else Z
                               for k in range(C.LONG_DAYS, 0, -1)]))
            Y.append(int(g.ret_1d.values[i] > 0)); DTs.append(di)
    S = np.asarray(S, np.float32); M_ = np.asarray(M_, np.float32); L = np.asarray(L, np.float32)
    y = np.asarray(Y); DT = np.asarray(DTs, dtype="datetime64[ns]")
    has_short = (S != 0).any(1)
    has_any = has_short | (M_ != 0).any((1, 2)) | (L != 0).any((1, 2))
    log(f"small-cap samples: {len(y)}  overnight-news: {has_short.mean():.1%}  "
        f"any-in-window: {has_any.mean():.1%}")
    sfx = "_fresh" if FRESH else ""
    np.savez(os.path.join(C.ART, f"kr36_features{sfx}.npz"), S=S, M=M_, L=L, y=y,
             DT=DT.astype("datetime64[D]").astype(str), has_short=has_short, has_any=has_any)
    keep = np.where(has_any)[0]
    Sk, Mk, Lk = S[keep], M_[keep], L[keep]
    yk, dtk, hsk = y[keep], DT[keep], has_short[keep]
    del S, M_, L
    St, Mt, Lt = torch.from_numpy(Sk), torch.from_numpy(Mk), torch.from_numpy(Lk)
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(dtk < split)[0]; teN = np.where(dtk >= split)[0]
    trd = np.sort(np.unique(dtk[dtk < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[dtk[trN] >= dsplit]; trN = trN[dtk[trN] < dsplit]
    log(f"train {len(trN)}  dev {len(dvN)}  test {len(teN)}")

    def inp(ix): return (St[ix], Mt[ix], Lt[ix])
    m = DenseModel(C.NTN_K, nn_only=False)
    m, best_dev = fit(m, inp, yk, trN, dvN, seed=C.SEED, verbose=True)
    p = predict(m, inp, teN)
    ypm = np.where(yk == 1, 1, -1)

    def rep(name, mask_te):
        yy = ypm[teN][mask_te]; pp = p[mask_te]
        if len(yy) < 30:
            print(f"  [{name}] too few (n={len(yy)})"); return
        base = max((yy == 1).mean(), (yy == -1).mean())
        a, mc = metrics(yy, pp)
        print(f"  [{name}] n={len(yy)} base={base:.4f} acc={a:.4f} mcc={mc:+.4f}", flush=True)
        cc = np.abs(pp - 0.5)
        for kap in (0.5, 0.3, 0.2, 0.1):
            thr = np.quantile(cc, 1 - kap); s = cc >= thr
            if s.sum() >= 20:
                aa, mm = metrics(yy[s], pp[s])
                bb = max((yy[s] == 1).mean(), (yy[s] == -1).mean())
                print(f"     conf top {int(kap*100)}%: acc={aa:.4f} mcc={mm:+.4f} "
                      f"base={bb:.4f} (n={int(s.sum())})", flush=True)

    print(f"\nKR SMALL-CAP EB-CNN, OVERNIGHT ALIGNMENT{sfx} (dev_mcc={best_dev:.4f}):")
    rep("ANY-event-in-window", np.ones(len(teN), bool))
    rep("OVERNIGHT-news days", hsk[teN])

if __name__ == "__main__":
    main()
