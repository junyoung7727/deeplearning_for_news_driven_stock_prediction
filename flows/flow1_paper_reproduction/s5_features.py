"""
Stage 5 - Build per-day prediction samples for NVDA (paper Sec 3, Fig.3).

For each NVDA trading day D_i we build, using ONLY news dated <= D_{i-1}
(previous trading day; strictly leak-free for the close-to-close target):

  short-term : the single most recent trading-day news bucket   (paper "past day")
  mid-term   : MID_DAYS (=7)  calendar-day sequence ending D_{i-1}  (paper "week")
  long-term  : LONG_DAYS(=30) calendar-day sequence ending D_{i-1}  (paper "month")

Dense representations (daily unit = mean of the day's item vectors):
  WB  : skip-gram word-embedding of the day's titles            (d=100)
  EB  : NTN event-embedding of the day's events                 (d=100)
  TWB : finance-transformer embedding of the day's titles       (d=768, optional)
  TEB : finance-transformer embedding of the day's event triples(d=768, optional)
Discrete representation:
  E   : set of discrete event-ids (subject-head, verb, object-head), vocab from
        TRAIN events only (unseen -> UNK), embedded in-model.

Outputs (artifacts/): samples.parquet, feats_{WB,EB[,TWB,TEB]}.npz, feats_E.npz
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
import os, numpy as np, pandas as pd
import config as C

D = C.WORD_DIM

def _event_key(o1, p, o2):
    h1 = o1[-1] if len(o1) else ""; hv = p[0] if len(p) else ""; h2 = o2[-1] if len(o2) else ""
    return f"{h1}|{hv}|{h2}"

def _std_train(E, dts):
    """Standardise per-dim on train-period items (date < DEV_START); no leakage."""
    tr = dts < pd.Timestamp(C.DEV_START)
    mu = E[tr].mean(0); sd = E[tr].std(0) + 1e-6
    return (E - mu) / sd

def main():
    prices = pd.read_parquet(os.path.join(C.ART, "prices.parquet")).sort_values("date").reset_index(drop=True)
    news   = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    ev     = pd.read_parquet(os.path.join(C.ART, "events.parquet"))
    emb    = np.load(os.path.join(C.ART, "event_emb.npy"))
    ok     = np.load(os.path.join(C.ART, "event_ok.npy"))
    z = np.load(os.path.join(C.ART, "word_vectors.npz"), allow_pickle=True)
    vocab = list(z["vocab"]); W = z["vectors"].astype(np.float32)
    w2i = {w: i for i, w in enumerate(vocab)}

    # ---- NVDA-only daily aggregates keyed by calendar date -----------------
    nv_news = news[news.ticker == C.TARGET].reset_index(drop=True)
    ev = ev.reset_index(drop=True)
    ev_nv_mask = (ev.ticker.values == C.TARGET) & ok
    ev_nv = ev[ev_nv_mask].reset_index(drop=True)
    ev_nv_emb = emb[ev_nv_mask.nonzero()[0]]

    wb_day = {}                                            # skip-gram WB
    for date, grp in nv_news.groupby("date"):
        idxs = [w2i[w] for toks in grp.tokens for w in toks if w in w2i]
        if idxs:
            wb_day[date] = W[idxs].mean(0)

    eb_day = {}                                            # NTN EB
    for date, sub in ev_nv.groupby("date"):
        eb_day[date] = ev_nv_emb[sub.index.values].mean(0)

    text_day = {d: " . ".join(g.title_clean) for d, g in nv_news.groupby("date")}

    ev_nv["key"] = [_event_key(a, p, b) for a, p, b in zip(ev_nv.o1, ev_nv.p, ev_nv.o2)]
    train_keys = ev_nv[ev_nv.date < pd.Timestamp(C.DEV_START)].key.value_counts()
    key2id = {k: i + 1 for i, k in enumerate(train_keys.index)}      # 0 = UNK/pad
    n_ids = len(key2id) + 1
    eid_day = {}
    for date, sub in ev_nv.groupby("date"):
        eid_day[date] = [key2id.get(k, 0) for k in sub.key]

    # ---- optional finance-transformer daily aggregates --------------------
    dense_reps = {"WB": (wb_day, D), "EB": (eb_day, D)}
    tf_t = os.path.join(C.ART, "tf_title_emb.npy")
    tf_e = os.path.join(C.ART, "tf_event_emb.npy")
    if os.path.exists(tf_t) and os.path.exists(tf_e):
        from sklearn.decomposition import PCA
        def prep(path, dts):
            E = _std_train(np.load(path).astype(np.float32), dts)
            if C.TF_PCA_DIM and C.TF_PCA_DIM < E.shape[1]:
                tr = dts < pd.Timestamp(C.DEV_START)
                pca = PCA(n_components=C.TF_PCA_DIM, random_state=C.SEED).fit(E[tr])
                E = pca.transform(E).astype(np.float32)
            return E
        Et = prep(tf_t, pd.to_datetime(nv_news.date).values)
        twb_day = {date: Et[sub.index.values].mean(0) for date, sub in nv_news.groupby("date")}
        nv_ev_all = ev[ev.ticker.values == C.TARGET].reset_index(drop=True)
        Ee = prep(tf_e, pd.to_datetime(nv_ev_all.date).values)
        teb_day = {date: Ee[sub.index.values].mean(0) for date, sub in nv_ev_all.groupby("date")}
        tf_dim = Et.shape[1]
        dense_reps["TWB"] = (twb_day, tf_dim)
        dense_reps["TEB"] = (teb_day, tf_dim)
        print(f"transformer features enabled (dim={tf_dim}):", list(dense_reps.keys()))

    # ---- window builders ---------------------------------------------------
    dates = prices.date.tolist()
    def daily_vec(dct, d, dim):
        v = dct.get(d); return v if v is not None else np.zeros(dim, np.float32)
    def seq(dct, end_date, ndays, dim):
        days = [end_date - pd.Timedelta(days=k) for k in range(ndays - 1, -1, -1)]
        return np.stack([daily_vec(dct, d, dim) for d in days])       # (ndays, dim)
    def short_mean(dct, sdates, dim):
        vs = [dct[d] for d in sdates if d in dct]
        return np.mean(vs, 0) if vs else np.zeros(dim, np.float32)
    def id_seq(end_date, ndays, maxe):
        out = np.zeros((ndays, maxe), np.int64)
        days = [end_date - pd.Timedelta(days=k) for k in range(ndays - 1, -1, -1)]
        for j, d in enumerate(days):
            ids = eid_day.get(d, [])[:maxe]
            out[j, :len(ids)] = ids
        return out

    MAXE = 64
    rows = []
    feats = {name: {"short": [], "mid": [], "long": []} for name in dense_reps}
    e_s, e_m, e_l = [], [], []
    for i in range(len(prices)):
        if i < 2:
            continue
        a, pp, d_i = dates[i - 1], dates[i - 2], dates[i]
        if d_i < pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.LONG_DAYS + 5):
            continue
        short_dates = [pp + pd.Timedelta(days=k) for k in range(1, (a - pp).days + 1)]

        rows.append({
            "date": d_i, "label": prices.label.iloc[i], "split": prices.split.iloc[i],
            "ret": prices.ret.iloc[i], "open": prices.open.iloc[i],
            "high": prices.high.iloc[i], "low": prices.low.iloc[i],
            "close": prices.close.iloc[i], "prev_close": prices.prev_close.iloc[i],
            "doc_text": text_day.get(a, ""),
        })
        for name, (dct, dim) in dense_reps.items():
            feats[name]["short"].append(short_mean(dct, short_dates, dim))
            feats[name]["mid"].append(seq(dct, a, C.MID_DAYS, dim))
            feats[name]["long"].append(seq(dct, a, C.LONG_DAYS, dim))
        e_ids = []
        for d in short_dates:
            e_ids += eid_day.get(d, [])
        arr = np.zeros(MAXE, np.int64); arr[:min(len(e_ids), MAXE)] = e_ids[:MAXE]
        e_s.append(arr); e_m.append(id_seq(a, C.MID_DAYS, MAXE)); e_l.append(id_seq(a, C.LONG_DAYS, MAXE))

    samp = pd.DataFrame(rows)
    samp.to_parquet(os.path.join(C.ART, "samples.parquet"))
    for name in dense_reps:
        np.savez(os.path.join(C.ART, f"feats_{name}.npz"),
                 short=np.asarray(feats[name]["short"], np.float32),
                 mid=np.asarray(feats[name]["mid"], np.float32),
                 long=np.asarray(feats[name]["long"], np.float32))
    np.savez(os.path.join(C.ART, "feats_E.npz"),
             short=np.asarray(e_s, np.int64), mid=np.asarray(e_m, np.int64),
             long=np.asarray(e_l, np.int64), n_ids=np.int64(n_ids))

    print("SAMPLES", samp.shape, "| reps:", list(dense_reps.keys()) + ["E"])
    print(samp.split.value_counts().to_dict())
    for name in dense_reps:
        nz = (np.asarray(feats[name]["short"], np.float32) != 0).any(1).mean()
        print(f"  {name}: short-term coverage {nz:.1%}")
    print("E event-id vocab (train):", n_ids)

if __name__ == "__main__":
    main()
