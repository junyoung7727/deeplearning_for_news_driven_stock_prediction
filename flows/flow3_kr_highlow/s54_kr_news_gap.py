"""
Stage 54 - KR paper-repro NEWS model, GAP target, 1-day-bucket ablation (remote GPU).

Reuses the FULL BigKinds (2015-2026) already linked + NTN event-embedded:
  ~/dlfe/artifacts/kr43_corpus.pkl   (link[ticker,ts,news_id] 2.08M, events_by_nid)
  ~/dlfe/artifacts/kr43_nid_emb.npz  (nids, emb  523k NTN event embeddings, D=100)
  ~/dlfe/data/kr_ohlcv_ext.parquet   (date,open,high,low,close,volume,ticker 2015-2026)

Model = EB-NN (the paper's event-embedding, feed-forward head over 1/7/30 context).
NOTE: transformers is not installed on the box, so a FinBERT/TEB variant is a
separate follow-up; NTN is the reproduction's actual event embedding.

Question: for predicting day D's OPENING GAP  g(D) = open(D)/close(D-1) - 1,
how does the definition of the "1-day" (short) news bucket change performance?
  V1 SESSION   : prev trading day's intraday news [09:00,15:30] -> known at
                 close(D-1)  (TRADABLE at prev close)
  V2 OVERNIGHT : (15:30 of D-1, 09:00 of D] -> the gap-driving news
                 (NOT tradable at prev close; bounds predictable signal)
  V3 SPLIT     : session and overnight as two separate channels
  mid  = per-info-day mean emb over the 7  trading days before D
  long = per-info-day mean emb over the 30 trading days before D
Targets: (a) gap DIRECTION (sign) and (b) gap SIZE (regression). Plus confidence-
stratified performance (top-k%).
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
import os, json, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HOME = os.path.expanduser("~")
ART = os.path.join(HOME, "dlfe", "artifacts")
DATA = os.path.join(HOME, "dlfe", "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
D = 100
MID_DAYS, LONG_DAYS = 7, 30
torch.manual_seed(SEED); np.random.seed(SEED)


def log(msg, t0=[time.time()]):
    print(f"{msg}  ({time.time()-t0[0]:.0f}s)", flush=True)


def _spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _mcc(yb, pb):
    yb = yb.astype(bool); pb = pb.astype(bool)
    tp = (yb & pb).sum(); tn = (~yb & ~pb).sum(); fp = (~yb & pb).sum(); fn = (yb & ~pb).sum()
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


# ---------------------------------------------------------------- load caches
def load():
    link, corpus, ev = pickle.load(open(os.path.join(ART, "kr43_corpus.pkl"), "rb"))
    z = np.load(os.path.join(ART, "kr43_nid_emb.npz"), allow_pickle=True)
    nids, emb = z["nids"], z["emb"].astype(np.float32)
    nid2row = {n: i for i, n in enumerate(nids)}
    ohlcv = pd.read_parquet(os.path.join(DATA, "kr_ohlcv_ext.parquet"))
    ohlcv["date"] = pd.to_datetime(ohlcv.date)
    log(f"link {len(link)} rows  events {len(nid2row)}  ohlcv {len(ohlcv)} rows "
        f"{ohlcv.ticker.nunique()} tickers")
    return link, nid2row, emb, ohlcv


# ---------------------------------------------------------------- bucketing
def build_buckets(link, nid2row, emb, TD):
    le = link[link.news_id.isin(nid2row)].copy()
    ts = le.ts.values.astype("datetime64[ns]")
    open_ts = TD + np.timedelta64(9, "h")
    close_ts = TD + np.timedelta64(15 * 60 + 30, "m")
    info_i = np.searchsorted(close_ts, ts, side="left")     # first close >= ts
    valid = info_i < len(TD)
    le = le[valid].reset_index(drop=True); info_i = info_i[valid]; ts = ts[valid]
    le["day"] = TD[info_i]
    le["row"] = np.array([nid2row[n] for n in le.news_id.values])
    le["sess"] = ts >= open_ts[info_i]                      # else overnight/pre-open

    def agg(sub):
        g = sub.groupby(["ticker", "day"]).row.apply(lambda s: emb[s.values].mean(0))
        return {(t, d): v for (t, d), v in g.items()}

    sess = agg(le[le.sess]); over = agg(le[~le.sess]); daily = agg(le)
    log(f"buckets  session {len(sess)}  overnight {len(over)}  daily {len(daily)}")
    return sess, over, daily


# ---------------------------------------------------------------- samples
def build_samples(ohlcv, sess, over, daily, TD):
    tdpos = {d: i for i, d in enumerate(TD)}
    Z = np.zeros(D, np.float32)
    Ssess, Sover, M_, L_, G, DT = [], [], [], [], [], []
    for t, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        d = g.date.values
        op = g.open.values.astype(float); cl = g.close.values.astype(float)
        for i in range(1, len(g)):
            a, D_ = d[i - 1], d[i]
            pc = cl[i - 1]
            if pc <= 0 or op[i] <= 0:
                continue
            p = tdpos.get(pd.Timestamp(D_)); pa = tdpos.get(pd.Timestamp(a))
            if p is None or pa is None:
                continue
            Ss = sess.get((t, a), Z)
            ov = [over[(t, TD[k])] for k in range(pa + 1, p + 1) if (t, TD[k]) in over]
            So = np.mean(ov, 0) if ov else Z
            mid = np.mean([daily.get((t, TD[p - k]), Z) for k in range(1, MID_DAYS + 1)], 0)
            lng = np.mean([daily.get((t, TD[p - k]), Z) for k in range(1, LONG_DAYS + 1)], 0)
            Ssess.append(Ss); Sover.append(So); M_.append(mid); L_.append(lng)
            G.append(op[i] / pc - 1.0); DT.append(D_)
    out = dict(
        sess=np.asarray(Ssess, np.float32), over=np.asarray(Sover, np.float32),
        mid=np.asarray(M_, np.float32), long=np.asarray(L_, np.float32),
        gap=np.asarray(G, np.float32), dt=np.asarray(DT, dtype="datetime64[ns]"))
    hs = (out["sess"] != 0).any(1); ho = (out["over"] != 0).any(1)
    ha = hs | ho | (out["mid"] != 0).any(1) | (out["long"] != 0).any(1)
    out.update(has_sess=hs, has_over=ho, has_any=ha)
    log(f"samples {len(out['gap'])}  has_session {hs.mean():.1%}  "
        f"has_overnight {ho.mean():.1%}  has_any {ha.mean():.1%}")
    return out


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, din, reg=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
        self.reg = reg

    def forward(self, x):
        o = self.net(x).squeeze(1)
        return o if self.reg else torch.sigmoid(o)


def zfit(X, tr):
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    return (X - mu) / sd


def train(X, y, tr, dv, te, reg, epochs=60, lr=1e-3):
    Xt = torch.tensor(zfit(X, tr), device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    m = MLP(X.shape[1], reg=reg).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.MSELoss() if reg else nn.BCELoss()
    tri = torch.tensor(tr, device=DEV)
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = tri[torch.randperm(len(tri), device=DEV)]
        for s in range(0, len(perm), 4096):
            b = perm[s:s + 4096]
            opt.zero_grad(); loss = lossf(m(Xt[b]), yt[b]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xt[torch.tensor(dv, device=DEV)]).cpu().numpy()
        yv = y[dv]
        score = _spear(pv, yv) if reg else _mcc(yv > 0.5, pv > 0.5)
        score = 0.0 if score != score else score
        if score > best:
            best, best_state, bad = score, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pte = m(Xt[torch.tensor(te, device=DEV)]).cpu().numpy()
    return pte, float(best)


# ---------------------------------------------------------------- reports
def rep_dir(name, y, p):
    yb = y > 0
    base = max(yb.mean(), 1 - yb.mean())
    acc = float(((p > 0.5) == yb).mean()); mcc = _mcc(yb, p > 0.5)
    r = {"n": int(len(y)), "base": float(base), "acc": acc, "mcc": mcc, "conf": {}}
    print(f"  [{name}] DIR n={len(y)} base={base:.4f} acc={acc:.4f} mcc={mcc:+.4f}", flush=True)
    conf = np.abs(p - 0.5)
    for kap in (0.5, 0.2, 0.1, 0.05):
        thr = np.quantile(conf, 1 - kap); s = conf >= thr
        if s.sum() >= 20:
            aa = float(((p[s] > 0.5) == yb[s]).mean()); mm = _mcc(yb[s], p[s] > 0.5)
            bb = float(max(yb[s].mean(), 1 - yb[s].mean()))
            r["conf"][f"top{int(kap*100)}"] = {"n": int(s.sum()), "acc": aa, "mcc": mm, "base": bb}
            print(f"     conf top {int(kap*100)}%: n={int(s.sum())} acc={aa:.4f} mcc={mm:+.4f} base={bb:.4f}", flush=True)
    return r


def rep_size(name, y, p):
    ic = _spear(p, y); pear = float(np.corrcoef(p, y)[0, 1])
    diracc = float(((p > 0) == (y > 0)).mean())
    r = {"n": int(len(y)), "spearman_ic": ic, "pearson": pear, "dir_acc": diracc, "conf": {}}
    print(f"  [{name}] SIZE n={len(y)} IC(spearman)={ic:+.4f} pearson={pear:+.4f} dir_acc={diracc:.4f}", flush=True)
    ap = np.abs(p)
    for kap in (0.5, 0.2, 0.1, 0.05):
        thr = np.quantile(ap, 1 - kap); s = ap >= thr
        if s.sum() >= 20:
            mg = float(np.abs(y[s]).mean()); da = float(((p[s] > 0) == (y[s] > 0)).mean()); ii = _spear(p[s], y[s])
            r["conf"][f"top{int(kap*100)}"] = {"n": int(s.sum()), "mean_abs_gap": mg, "dir_acc": da, "ic": ii}
            print(f"     conf top {int(kap*100)}%: n={int(s.sum())} mean|gap|={mg*100:.2f}% dir_acc={da:.4f} ic={ii:+.4f}", flush=True)
    return r


# ---------------------------------------------------------------- main
def main():
    link, nid2row, emb, ohlcv = load()
    TD = np.sort(ohlcv.date.unique()).astype("datetime64[ns]")
    sess, over, daily = build_buckets(link, nid2row, emb, TD)
    S = build_samples(ohlcv, sess, over, daily, TD)

    keep = np.where(S["has_any"])[0]
    gap = S["gap"][keep]; dt = S["dt"][keep]
    mid = S["mid"][keep]; lng = S["long"][keep]
    sesf = S["sess"][keep]; ovf = S["over"][keep]; has_over = S["has_over"][keep]

    dates = np.sort(np.unique(dt)); split = dates[int(len(dates) * 0.6)]
    tr_all = np.where(dt < split)[0]; te = np.where(dt >= split)[0]
    trd = np.sort(np.unique(dt[dt < split])); dsplit = trd[int(len(trd) * 0.85)]
    dv = tr_all[dt[tr_all] >= dsplit]; tr = tr_all[dt[tr_all] < dsplit]
    log(f"train {len(tr)} dev {len(dv)} test {len(te)} test_gap_up={float((gap[te]>0).mean()):.4f}")

    variants = {
        "V1_session":   np.concatenate([sesf, mid, lng], 1),
        "V2_overnight": np.concatenate([ovf, mid, lng], 1),
        "V3_split":     np.concatenate([sesf, ovf, mid, lng], 1),
    }
    tradable = {"V1_session": True, "V2_overnight": False, "V3_split": False}
    results = {"n_links": int(len(link)), "n_events": int(len(nid2row)),
               "n_samples": int(len(gap)), "test_gap_up_rate": float((gap[te] > 0).mean()),
               "note": "EB=NTN event embeddings on full BigKinds 2015-2026; transformers unavailable on box (TEB is a follow-up).",
               "variants": {}}
    for name, X in variants.items():
        print(f"\n===== {name}  (tradable={tradable[name]})  dim={X.shape[1]} =====", flush=True)
        pdir, bd = train(X, (gap > 0).astype(np.float32), tr, dv, te, reg=False)
        psz, bs = train(X, gap.astype(np.float32), tr, dv, te, reg=True)
        rd = rep_dir("ALL", gap[te], pdir)
        rs = rep_size("ALL", gap[te], psz)
        tem = has_over[te]
        rd_ov = rep_dir("OVERNIGHT-present", gap[te][tem], pdir[tem]) if tem.sum() >= 30 else None
        rs_ov = rep_size("OVERNIGHT-present", gap[te][tem], psz[tem]) if tem.sum() >= 30 else None
        results["variants"][name] = {"tradable": tradable[name], "dim": int(X.shape[1]),
                                     "dev_dir_mcc": bd, "dev_size_ic": bs,
                                     "direction": rd, "size": rs,
                                     "direction_overnight": rd_ov, "size_overnight": rs_ov}

    with open(os.path.join(ART, "s54_kr_gap.json"), "w") as f:
        json.dump(results, f, indent=1)
    log("saved artifacts/s54_kr_gap.json")


if __name__ == "__main__":
    main()
