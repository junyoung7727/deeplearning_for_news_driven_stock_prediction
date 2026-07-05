"""
Stage 37 - US (NVDA) re-alignment fix. Two bugs in the flagship s1->s5->s7 chain:

  1. s1 dated news by the UTC calendar day of `published_at` (s22 PROVED the
     timestamps are UTC), never applying the s23 correction: evening-ET news
     (>= ~19:00 ET) was pushed one calendar day forward.
  2. s5's short-term window is "news dated (D_{i-2}, D_{i-1}]": pre-open news of
     the target day (00:00-09:30 ET, the most actionable) never reaches the
     short slot, weekend news reaches Monday's slot a day late, and intraday
     news already priced into close(D_{i-1}) dilutes what remains.

Fix, mirroring s36 (KR): convert to ET; assign each news item to
  eff  = first trading day whose 09:30 OPEN  >= ts   (actionable-at-open day)
  info = first trading day whose 16:00 CLOSE >= ts   (known-by-close day)
  fresh = (eff == info)  <=> published post-close/weekend/pre-open, i.e. NOT
          yet priced into any close.
short(D_i) = mean of FRESH items with eff == D_i; mid/long = per-info-day means
over the prior 7/30 trading days. Representation caches are reused unchanged
(word_vectors.npz, event_emb.npy, tf_*.npy - all position-aligned); the spaCy
parse is replayed only to recover each event's source-news timestamp (row
alignment asserted).

Protocol identical to s7: 4-seed ensemble probabilities, dev = 2024, test = 2025+.
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
import os, time, numpy as np, pandas as pd, torch
import config as C
from s3_events import extract
from s6_models import DenseModel, fit, predict, metrics

torch.manual_seed(C.SEED); np.random.seed(C.SEED)
D = C.WORD_DIM
N_SEEDS = 4
OPEN_TD, CLOSE_TD = pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=16)
CACHE_TS = os.path.join(C.ART, "us_event_ts.npy")


def log(msg, t0=[time.time()]):
    print(f"{msg}  ({time.time()-t0[0]:.0f}s)", flush=True)


def event_source_ts(news):
    """Replay s3's extraction to map each events.parquet row to its source news
    row (order-preserving), returning the source published_at per event."""
    if os.path.exists(CACHE_TS):
        return np.load(CACHE_TS)
    import spacy
    ev_old = pd.read_parquet(os.path.join(C.ART, "events.parquet"))
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    src = []
    for i, doc in enumerate(nlp.pipe(news.title_clean.tolist(), batch_size=256)):
        for _ in extract(doc):
            src.append(i)
        if (i + 1) % 20000 == 0:
            log(f"  re-parsed {i+1}/{len(news)}")
    assert len(src) == len(ev_old), f"event count mismatch {len(src)} vs {len(ev_old)}"
    # row-alignment spot check on tickers
    tick = news.ticker.values[np.asarray(src)]
    assert (tick == ev_old.ticker.values).all(), "event/news row alignment broken"
    ts = news.published_at.values[np.asarray(src)]
    np.save(CACHE_TS, ts)
    return ts


def main():
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    ev = pd.read_parquet(os.path.join(C.ART, "events.parquet"))
    emb = np.load(os.path.join(C.ART, "event_emb.npy"))
    ok = np.load(os.path.join(C.ART, "event_ok.npy"))
    z = np.load(os.path.join(C.ART, "word_vectors.npz"), allow_pickle=True)
    vocab = list(z["vocab"]); W = z["vectors"].astype(np.float32)
    w2i = {w: i for i, w in enumerate(vocab)}
    prices = pd.read_parquet(os.path.join(C.ART, "prices.parquet")).sort_values("date").reset_index(drop=True)

    # ---- ET timestamps ------------------------------------------------------
    def to_et(ts):
        s = pd.Series(pd.to_datetime(ts))
        return s.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).values
    news_ts_et = to_et(news.published_at.values)
    ev_ts_et = to_et(event_source_ts(news))
    log(f"events with recovered ts: {len(ev_ts_et)}")

    # ---- eff / info / fresh mapping ----------------------------------------
    TD = prices.date.values.astype("datetime64[ns]")
    open_ts = TD + np.timedelta64(9 * 60 + 30, "m")
    close_ts = TD + np.timedelta64(16, "h")

    def assign(ts):
        eff = np.searchsorted(open_ts, ts, side="left")
        info = np.searchsorted(close_ts, ts, side="left")
        return eff, info, eff == info

    # NVDA-only item tables
    nvn = news.ticker.values == C.TARGET
    nve = (ev.ticker.values == C.TARGET)
    n_eff, n_info, n_fresh = assign(news_ts_et)
    e_eff, e_info, e_fresh = assign(ev_ts_et)

    # ---- bucket builders ----------------------------------------------------
    def buckets(item_mask, eff, info, fresh, vec_of_item):
        """vec_of_item(i) -> vector or None. Returns (over, info_) dicts keyed by
        trading-day index."""
        over, infod = {}, {}
        for i in np.where(item_mask)[0]:
            v = vec_of_item(i)
            if v is None:
                continue
            if fresh[i] and eff[i] < len(TD):
                over.setdefault(int(eff[i]), []).append(v)
            if info[i] < len(TD):
                infod.setdefault(int(info[i]), []).append(v)
        return ({k: np.mean(v, 0) for k, v in over.items()},
                {k: np.mean(v, 0) for k, v in infod.items()})

    # WB: day vector = mean over token vectors (mirror s5: pool tokens)
    tok_cache = news.tokens.values
    def wb_vec(i):
        idxs = [w2i[w] for w in tok_cache[i] if w in w2i]
        return W[idxs].mean(0) if idxs else None
    wb_over, wb_info = buckets(nvn, n_eff, n_info, n_fresh, wb_vec)

    # EB: NTN event embeddings (ok rows only)
    def eb_vec(i):
        return emb[i] if ok[i] else None
    eb_over, eb_info = buckets(nve, e_eff, e_info, e_fresh, eb_vec)

    reps = {"WB": (wb_over, wb_info, D), "EB": (eb_over, eb_info, D)}

    # TWB/TEB: cached transformer embeddings, standardise+PCA on train (ET dates)
    tf_t, tf_e = (os.path.join(C.ART, f) for f in ("tf_title_emb.npy", "tf_event_emb.npy"))
    if os.path.exists(tf_t) and os.path.exists(tf_e):
        from sklearn.decomposition import PCA
        dev_start = pd.Timestamp(C.DEV_START).to_datetime64()
        def prep(path, ts_all):
            E = np.load(path).astype(np.float32)
            tr = ts_all < dev_start
            mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6
            E = (E - mu) / sd
            pca = PCA(n_components=C.TF_PCA_DIM, random_state=C.SEED).fit(E[tr])
            return pca.transform(E).astype(np.float32)
        Et = prep(tf_t, news_ts_et[nvn])
        Ee = prep(tf_e, ev_ts_et[nve])
        nv_news_rows = np.where(nvn)[0]; nv_ev_rows = np.where(nve)[0]
        twb = {i: Et[j] for j, i in enumerate(nv_news_rows)}
        teb = {i: Ee[j] for j, i in enumerate(nv_ev_rows)}
        twb_over, twb_info = buckets(nvn, n_eff, n_info, n_fresh, lambda i: twb[i])
        teb_over, teb_info = buckets(nve, e_eff, e_info, e_fresh, lambda i: teb[i])
        reps["TWB"] = (twb_over, twb_info, C.TF_PCA_DIM)
        reps["TEB"] = (teb_over, teb_info, C.TF_PCA_DIM)
    log(f"reps: {list(reps)}; fresh share of NVDA news: {n_fresh[nvn].mean():.1%}")

    # ---- samples (mirror s5 skip rules -> same sample set as report.md) -----
    Z = {name: np.zeros(dim, np.float32) for name, (_, _, dim) in reps.items()}
    idxs = []
    for i in range(len(prices)):
        if i < 2:
            continue
        if prices.date.iloc[i] < pd.Timestamp(C.START_DATE) + pd.Timedelta(days=C.LONG_DAYS + 5):
            continue
        idxs.append(i)
    y01 = (prices.label.values[idxs] == 1).astype(int)
    split = prices.split.values[idxs]
    feats = {}
    for name, (over, infod, dim) in reps.items():
        S = np.stack([over.get(i, Z[name]) for i in idxs])
        Mw = np.stack([np.stack([infod.get(i - k, Z[name]) for k in range(C.MID_DAYS, 0, -1)]) for i in idxs])
        Lw = np.stack([np.stack([infod.get(i - k, Z[name]) for k in range(C.LONG_DAYS, 0, -1)]) for i in idxs])
        feats[name] = (S.astype(np.float32), Mw.astype(np.float32), Lw.astype(np.float32))
    tr = np.where(split == "train")[0]; dv = np.where(split == "dev")[0]; te = np.where(split == "test")[0]
    log(f"samples {len(idxs)}  train/dev/test = {len(tr)}/{len(dv)}/{len(te)}")

    # ---- train (s7 protocol: N_SEEDS ensemble) ------------------------------
    y_pm1 = np.where(y01 == 1, 1, -1)
    base = max((y_pm1[te] == 1).mean(), (y_pm1[te] == -1).mean())
    print(f"\ntest majority baseline = {base:.4f}  (n={len(te)})")
    for name, (S, Mw, Lw) in feats.items():
        St, Mt, Lt = map(torch.from_numpy, (S, Mw, Lw))
        def inp(ix): return (St[ix], Mt[ix], Lt[ix])
        tps = []
        for s in range(N_SEEDS):
            torch.manual_seed(C.SEED + s); np.random.seed(C.SEED + s)
            m = DenseModel(S.shape[1], nn_only=False)
            m, dev_mcc = fit(m, inp, y01, tr, dv, seed=C.SEED + s)
            tps.append(predict(m, inp, te))
        p = np.mean(tps, 0)
        a, mc = metrics(y_pm1[te], p)
        fresh_te = (S[te] != 0).any(1)
        line = f"{name}-CNN (realigned): test acc={a:.4f} mcc={mc:+.4f}"
        if fresh_te.sum() >= 30:
            af, mf = metrics(y_pm1[te][fresh_te], p[fresh_te])
            bf = max((y_pm1[te][fresh_te] == 1).mean(), (y_pm1[te][fresh_te] == -1).mean())
            line += f" | fresh-overnight days n={int(fresh_te.sum())} base={bf:.4f} acc={af:.4f} mcc={mf:+.4f}"
        print(line, flush=True)

if __name__ == "__main__":
    main()
