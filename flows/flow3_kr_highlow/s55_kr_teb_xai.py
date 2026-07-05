"""
Stage 55 - KR TEB-NN (transformer news embeddings) + per-title XAI (remote GPU).

Answers: (a) does a transformer/KR-FinBERT news embedding beat NTN(EB) on the GAP
target (direction / size)?  (b) which individual NEWS TITLES drove the biggest
correct vs wrong gap predictions (gradient x embedding attribution)?

Inputs (remote):
  ~/dlfe/artifacts/kr43_corpus.pkl   link[ticker,ts,news_id] 2.08M (2015-2026)
  ~/bk_slim/slim_*.parquet           news_id,title  (22M; join for linked titles)
  ~/dlfe/data/kr_ohlcv_ext.parquet   date,open,high,low,close,volume,ticker
Caches title embeddings at ~/dlfe/artifacts/kr55_title_emb.npz (skip re-encode).
Saves ~/dlfe/artifacts/s55_kr_teb_xai.json.
Memory-lean: only has-any samples kept, float16 features, lazy title lookup.
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
import os, json, time, glob, gc, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HOME = os.path.expanduser("~")
ART = os.path.join(HOME, "dlfe", "artifacts")
DATA = os.path.join(HOME, "dlfe", "data")
BK = os.path.join(HOME, "bk_slim")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
MID_DAYS, LONG_DAYS = 7, 30
MODELS = ["snunlp/KR-FinBert", "klue/roberta-base", "beomi/kcbert-base",
          "bert-base-multilingual-cased"]
torch.manual_seed(SEED); np.random.seed(SEED)


def log(m, t0=[time.time()]):
    print(f"{m}  ({time.time()-t0[0]:.0f}s)", flush=True)


def _spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def _mcc(yb, pb):
    yb = yb.astype(bool); pb = pb.astype(bool)
    tp = (yb & pb).sum(); tn = (~yb & ~pb).sum(); fp = (~yb & pb).sum(); fn = (yb & ~pb).sum()
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


def encode_titles(need_ids):
    cache = os.path.join(ART, "kr55_title_emb.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        log(f"loaded title-emb cache: {len(z['ids'])} titles dim {z['emb'].shape[1]}")
        return {n: i for i, n in enumerate(z["ids"])}, z["emb"], str(z["model"])
    from transformers import AutoTokenizer, AutoModel
    need = set(need_ids); parts = []
    for f in sorted(glob.glob(os.path.join(BK, "slim_*.parquet"))):
        d = pd.read_parquet(f, columns=["news_id", "title"])
        parts.append(d[d.news_id.isin(need)])
    tt = pd.concat(parts, ignore_index=True).drop_duplicates("news_id").reset_index(drop=True)
    log(f"linked titles to encode: {len(tt)}")
    name = tok = model = None
    for cand in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            model = AutoModel.from_pretrained(cand).half().to(DEV).eval(); name = cand; break
        except Exception as e:
            log(f"model {cand} load failed: {repr(e)[:120]}")
    if name is None:
        raise RuntimeError("no transformer model could be loaded")
    log(f"encoder = {name}")
    titles = tt.title.astype(str).tolist(); ids = tt.news_id.values
    embs = np.zeros((len(titles), model.config.hidden_size), np.float32); B = 256
    for s in range(0, len(titles), B):
        enc = tok(titles[s:s + B], padding=True, truncation=True, max_length=48, return_tensors="pt").to(DEV)
        with torch.no_grad():
            h = model(**enc).last_hidden_state; mask = enc.attention_mask.unsqueeze(-1)
            e = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs[s:s + B] = e.float().cpu().numpy()
        if s % (B * 200) == 0:
            log(f"  encoded {s}/{len(titles)}")
    np.savez(cache, ids=ids, emb=embs, model=name); log(f"encoded {len(titles)} dim {embs.shape[1]}")
    return {n: i for i, n in enumerate(ids)}, embs, name


def build_buckets(link, nid2row, emb, TD):
    le = link[link.news_id.isin(nid2row)].copy()
    ts = le.ts.values.astype("datetime64[ns]")
    open_ts = TD + np.timedelta64(9, "h"); close_ts = TD + np.timedelta64(15 * 60 + 30, "m")
    info_i = np.searchsorted(close_ts, ts, side="left"); valid = info_i < len(TD)
    le = le[valid].reset_index(drop=True); info_i = info_i[valid]; ts = ts[valid]
    le["day"] = TD[info_i]; le["row"] = np.array([nid2row[n] for n in le.news_id.values])
    le["sess"] = ts >= open_ts[info_i]

    def agg(sub):
        g = sub.groupby(["ticker", "day"]).row.apply(lambda s: emb[s.values].astype(np.float32).mean(0).astype(np.float16))
        return {(t, d): v for (t, d), v in g.items()}
    return agg(le[le.sess]), agg(le[~le.sess]), agg(le)


def build_samples(ohlcv, sess, over, daily, TD, H):
    tdpos = {d: i for i, d in enumerate(TD)}; Z = np.zeros(H, np.float16)
    Ss, So_, M_, L_, G, DT, PV, TK = [], [], [], [], [], [], [], []
    for t, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date"); d = g.date.values
        op = g.open.values.astype(float); cl = g.close.values.astype(float)
        for i in range(1, len(g)):
            a, D_ = d[i - 1], d[i]; pc = cl[i - 1]
            if pc <= 0 or op[i] <= 0:
                continue
            p = tdpos.get(pd.Timestamp(D_)); pa = tdpos.get(pd.Timestamp(a))
            if p is None or pa is None:
                continue
            ss = sess.get((t, a), Z)
            ov = [over[(t, TD[k])] for k in range(pa + 1, p + 1) if (t, TD[k]) in over]
            so = np.mean(ov, 0).astype(np.float16) if ov else Z
            md = np.mean([daily.get((t, TD[p - k]), Z) for k in range(1, MID_DAYS + 1)], 0).astype(np.float16)
            lg = np.mean([daily.get((t, TD[p - k]), Z) for k in range(1, LONG_DAYS + 1)], 0).astype(np.float16)
            if not (ss.any() or so.any() or md.any() or lg.any()):
                continue                                        # has_any filter (keep memory small)
            Ss.append(ss); So_.append(so); M_.append(md); L_.append(lg)
            G.append(op[i] / pc - 1.0); DT.append(D_); PV.append(a); TK.append(t)
    out = dict(sess=np.asarray(Ss, np.float16), over=np.asarray(So_, np.float16),
               mid=np.asarray(M_, np.float16), long=np.asarray(L_, np.float16),
               gap=np.asarray(G, np.float32), dt=np.asarray(DT, "datetime64[ns]"),
               prev=np.asarray(PV, "datetime64[ns]"), tk=np.asarray(TK))
    out["has_over"] = (out["over"] != 0).any(1)
    log(f"kept samples {len(out['gap'])}  has_over {out['has_over'].mean():.1%}")
    return out


class MLP(nn.Module):
    def __init__(self, din, reg=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.reg = reg

    def forward(self, x):
        o = self.net(x).squeeze(-1)
        return o if self.reg else torch.sigmoid(o)


def train(X, y, tr, dv, te, reg, mu, sd, epochs=60):
    mt = torch.tensor(mu, device=DEV); st = torch.tensor(sd, device=DEV)
    m = MLP(X.shape[1], reg=reg).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    lf = nn.MSELoss() if reg else nn.BCELoss()
    rng = np.random.default_rng(SEED)

    def predict(idx):
        outs = []
        for s in range(0, len(idx), 16384):
            b = idx[s:s + 16384]
            xb = (torch.tensor(X[b], device=DEV).float() - mt) / st
            with torch.no_grad():
                outs.append(m(xb).cpu().numpy())
        return np.concatenate(outs)
    best, bstate, bad = -1e9, None, 0
    for ep in range(epochs):
        m.train(); perm = tr.copy(); rng.shuffle(perm)
        for s in range(0, len(perm), 4096):
            b = perm[s:s + 4096]
            xb = (torch.tensor(X[b], device=DEV).float() - mt) / st
            yb = torch.tensor(y[b], dtype=torch.float32, device=DEV)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        m.eval(); pv = predict(dv)
        sc = _spear(pv, y[dv]) if reg else _mcc(y[dv] > 0.5, pv > 0.5)
        sc = 0.0 if sc != sc else sc
        if sc > best:
            best, bstate, bad = sc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    if bstate:
        m.load_state_dict(bstate)
    m.eval(); pte = predict(te)
    torch.cuda.empty_cache()
    return m, pte, float(best)


def rep_dir(name, y, p):
    yb = y > 0; base = max(yb.mean(), 1 - yb.mean())
    acc = float(((p > 0.5) == yb).mean()); mcc = _mcc(yb, p > 0.5)
    print(f"  [{name}] DIR n={len(y)} base={base:.4f} acc={acc:.4f} mcc={mcc:+.4f}", flush=True)
    r = {"n": int(len(y)), "base": float(base), "acc": acc, "mcc": mcc, "conf": {}}
    c = np.abs(p - 0.5)
    for k in (0.5, 0.2, 0.1, 0.05):
        s = c >= np.quantile(c, 1 - k)
        if s.sum() >= 20:
            r["conf"][f"top{int(k*100)}"] = {"n": int(s.sum()), "acc": float(((p[s] > 0.5) == yb[s]).mean()),
                                              "mcc": _mcc(yb[s], p[s] > 0.5), "base": float(max(yb[s].mean(), 1 - yb[s].mean()))}
    return r


def rep_size(name, y, p):
    ic = _spear(p, y); da = float(((p > 0) == (y > 0)).mean())
    print(f"  [{name}] SIZE n={len(y)} IC={ic:+.4f} dir_acc={da:.4f}", flush=True)
    r = {"n": int(len(y)), "spearman_ic": ic, "dir_acc": da, "conf": {}}
    ap = np.abs(p)
    for k in (0.5, 0.2, 0.1, 0.05):
        s = ap >= np.quantile(ap, 1 - k)
        if s.sum() >= 20:
            r["conf"][f"top{int(k*100)}"] = {"n": int(s.sum()), "mean_abs_gap": float(np.abs(y[s]).mean()),
                                             "dir_acc": float(((p[s] > 0) == (y[s] > 0)).mean()), "ic": _spear(p[s], y[s])}
    return r


def main():
    link, corpus, ev = pickle.load(open(os.path.join(ART, "kr43_corpus.pkl"), "rb"))
    del corpus, ev; gc.collect()
    nid2row, emb, enc_name = encode_titles(link.news_id.unique()); H = emb.shape[1]
    emb = emb.astype(np.float16)
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet")); ohlcv["date"] = pd.to_datetime(ohlcv.date)
    TD = np.sort(ohlcv.date.unique()).astype("datetime64[ns]")
    sess, over, daily = build_buckets(link, nid2row, emb, TD)
    log(f"buckets session {len(sess)} overnight {len(over)} daily {len(daily)}")
    S = build_samples(ohlcv, sess, over, daily, TD, H)
    del sess, over, daily; gc.collect()

    gap = S["gap"]; dt = S["dt"]; prev = S["prev"]; tk = S["tk"]; has_over = S["has_over"]
    sesf, ovf, mid, lng = S["sess"], S["over"], S["mid"], S["long"]
    dates = np.sort(np.unique(dt)); split = dates[int(len(dates) * 0.6)]
    tr_all = np.where(dt < split)[0]; te = np.where(dt >= split)[0]
    trd = np.sort(np.unique(dt[dt < split])); dsplit = trd[int(len(trd) * 0.85)]
    dv = tr_all[dt[tr_all] >= dsplit]; tr = tr_all[dt[tr_all] < dsplit]
    log(f"H={H} train {len(tr)} dev {len(dv)} test {len(te)} test_gap_up={float((gap[te]>0).mean()):.4f}")

    variant_arrs = {"V1_session": [sesf, mid, lng], "V2_overnight": [ovf, mid, lng],
                    "V3_split": [sesf, ovf, mid, lng]}
    results = {"encoder": enc_name, "H": int(H), "n_links": int(len(link)), "n_samples": int(len(gap)),
               "test_gap_up_rate": float((gap[te] > 0).mean()), "variants": {}}
    keep_v3 = {}
    for name, arrs in variant_arrs.items():
        X = np.concatenate(arrs, 1)
        samp = tr[np.random.default_rng(SEED).choice(len(tr), min(120000, len(tr)), replace=False)]
        mu = X[samp].astype(np.float32).mean(0); sd = X[samp].astype(np.float32).std(0) + 1e-6
        print(f"\n===== TEB {name} dim={X.shape[1]} =====", flush=True)
        _, pdir, bd = train(X, (gap > 0).astype(np.float32), tr, dv, te, False, mu, sd)
        msz, psz, bs = train(X, gap.astype(np.float32), tr, dv, te, True, mu, sd)
        tem = has_over[te]
        results["variants"][name] = {"dim": int(X.shape[1]), "dev_dir_mcc": bd, "dev_size_ic": bs,
                                     "direction": rep_dir("ALL", gap[te], pdir), "size": rep_size("ALL", gap[te], psz),
                                     "direction_overnight": rep_dir("OVERNIGHT", gap[te][tem], pdir[tem]) if tem.sum() >= 30 else None,
                                     "size_overnight": rep_size("OVERNIGHT", gap[te][tem], psz[tem]) if tem.sum() >= 30 else None}
        if name == "V3_split":
            keep_v3 = {"msz": msz, "mu": mu, "sd": sd, "pdir": pdir}
        del X; gc.collect()

    # ---------------- XAI on V3_split size model: best/worst BIG-gap overnight days
    msz, mu, sd, pdir = keep_v3["msz"], keep_v3["mu"], keep_v3["sd"], keep_v3["pdir"]
    Xv3 = np.concatenate([sesf, ovf, mid, lng], 1)
    over_slice = slice(H, 2 * H); sd_over = sd[over_slice]
    te_gap = gap[te]; correct = (pdir > 0.5) == (te_gap > 0); conf = np.abs(pdir - 0.5)
    big = (np.abs(te_gap) >= 0.03) & has_over[te]
    lk = link[["ticker", "ts", "news_id"]]

    def gather(mask, n=12):
        idx = np.where(mask)[0]; idx = idx[np.argsort(-conf[idx])][:n]
        rows = []
        for j in idx:
            gi = te[j]; t = str(tk[gi]); D_ = pd.Timestamp(dt[gi]); a = pd.Timestamp(prev[gi])
            lo = a + pd.Timedelta(hours=15, minutes=30); hi = D_ + pd.Timedelta(hours=9)
            sub = lk[(lk.ticker == t) & (lk.ts > lo) & (lk.ts <= hi) & (lk.news_id.isin(nid2row))].drop_duplicates("news_id")
            if len(sub) == 0:
                continue
            rows.append({"j": int(j), "gi": int(gi), "ticker": t, "date": str(D_.date()),
                         "gap_pct": round(float(te_gap[j]) * 100, 3), "pred_up": bool(pdir[j] > 0.5),
                         "correct": bool(correct[j]), "nids": list(sub.news_id.values)})
        return rows

    best = gather(big & correct); worst = gather(big & (~correct))
    need = set()
    for r in best + worst:
        need.update(r["nids"])
    id2title = {}
    for f in sorted(glob.glob(os.path.join(BK, "slim_*.parquet"))):
        d = pd.read_parquet(f, columns=["news_id", "title"]); d = d[d.news_id.isin(need)]
        id2title.update(dict(zip(d.news_id.values, d.title.values)))
    log(f"XAI titles resolved {len(id2title)}/{len(need)}")

    def finalize(rows):
        out = []
        for r in rows:
            erows = np.stack([emb[nid2row[n]] for n in r["nids"]]).astype(np.float32)
            xs = torch.tensor(((Xv3[r["gi"]] - mu) / sd).astype(np.float32)[None, :], device=DEV, requires_grad=True)
            o = msz(xs); o.backward(); g = xs.grad[0].cpu().numpy()
            graw = g[over_slice] / sd_over; nn_ = len(erows)
            contribs = [float((graw * e).sum() / nn_) for e in erows]
            order = np.argsort(-np.abs(contribs))[:6]
            titles = [{"title": str(id2title.get(r["nids"][k], "")), "contrib_pct": round(contribs[k] * 100, 4)} for k in order]
            out.append({"ticker": r["ticker"], "date": r["date"], "gap_pct": r["gap_pct"],
                        "pred_up": r["pred_up"], "correct": r["correct"], "n_overnight_news": len(r["nids"]),
                        "top_titles": titles})
        return out

    results["xai_best_days"] = finalize(best)
    results["xai_worst_days"] = finalize(worst)
    log(f"XAI best {len(results['xai_best_days'])} worst {len(results['xai_worst_days'])}")
    with open(os.path.join(ART, "s55_kr_teb_xai.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    log("saved artifacts/s55_kr_teb_xai.json")


if __name__ == "__main__":
    main()
