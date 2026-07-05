"""
Stage 14 - enriched, leak-free, COMPACT feature engineering for NVDA next-day
direction (financial-ML techniques; anti-overfit = few, robust features).

Per trading day D_i (target = sign(close_i/close_{i-1}-1)), using ONLY info at
close of D_{i-1}:
  price/technical (s10)  + cross-asset/market (s12)
  + NEWS (spam-filtered):
      cnt_short/week/month  (news volume)
      abn_vol               (short vs monthly baseline)
      sent_short/week       (FinBERT net sentiment, count-weighted)
      novelty               (1 - cos(short-emb, month-emb))
      npc1..K               (top-K PCA of short-window title embedding, train-fit)

Output: artifacts/features_ff.parquet  (date, split, label, ohlc + all features)
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
import os, re, numpy as np, pandas as pd
from sklearn.decomposition import PCA
import config as C
from s10_boost import price_features
from s12_crossasset import xasset_features

K_PC = 8
SPAM = re.compile(r"law firm|law offices|llp|class action|investor alert|investigation|"
                  r"lawsuit|securities fraud|deadline|rosen|pomerantz|glancy|schall|"
                  r"howard g\.? smith|bragar|levi & korsinsky|kirby mcinerney|"
                  r"reminds investors|encourages investors|shareholder rights|"
                  r"portnoy|robbins|kessler|hagens|bronstein", re.I)

def _wmean(vals, wts, dim):
    wts = np.asarray(wts, float)
    if wts.sum() <= 0:
        return np.zeros(dim, np.float32)
    return (np.asarray(vals, np.float32) * wts[:, None]).sum(0) / wts.sum()

def main():
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    daily = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close", "volume"])
    d = daily[daily.ticker == C.TARGET].copy()
    d["date"] = pd.to_datetime(d.trade_date); d = d.sort_values("date").reset_index(drop=True)
    d["close"] = d.close.astype(float); d["volume"] = d.volume.astype(float)
    d["ret"] = d.close / d.close.shift(1) - 1
    feat = samp.merge(price_features(d), on="date", how="left").merge(xasset_features(), on="date", how="left")

    # ---- news: per-title emb + sentiment + spam filter ----
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    emb = np.load(os.path.join(C.ART, "tf_title_emb.npy")).astype(np.float32)
    sent_path = os.path.join(C.ART, "tf_title_sent.npy")
    sent = np.load(sent_path).astype(np.float32) if os.path.exists(sent_path) else np.zeros(len(nv), np.float32)
    keep = ~nv.title_clean.str.contains(SPAM)
    print(f"news total={len(nv)} kept(non-spam)={int(keep.sum())} ({keep.mean():.1%})")

    nv = nv[keep].reset_index(drop=True); emb = emb[keep.values]; sent = sent[keep.values]
    nv["_i"] = np.arange(len(nv))
    dcol = pd.to_datetime(nv.date)

    # PCA of title embeddings fit on TRAIN titles only (leak-free)
    trm = (dcol < pd.Timestamp(C.DEV_START)).values
    pca = PCA(n_components=K_PC, random_state=C.SEED).fit(emb[trm] if trm.sum() > K_PC else emb)

    # per-calendar-date aggregates
    by_date = {}
    for dt, g in nv.groupby("date"):
        idx = g._i.values
        by_date[dt] = (emb[idx].mean(0), float(sent[idx].mean()), len(idx))

    dates = samp.date.tolist()
    def collect(win_dates):
        embs, sents, wts = [], [], []
        for x in win_dates:
            if x in by_date:
                e, s, c = by_date[x]; embs.append(e); sents.append(s); wts.append(c)
        cnt = int(sum(wts))
        if cnt == 0:
            return np.zeros(emb.shape[1], np.float32), 0.0, 0
        me = _wmean(embs, wts, emb.shape[1]); sm = float(np.average(sents, weights=wts))
        return me, sm, cnt

    rows = []
    for i in range(len(samp)):
        di = pd.Timestamp(dates[i])
        # locate a=prev trading day, pp=prev-prev via samp order (samp is a trading-day subset)
        a = pd.Timestamp(dates[i - 1]) if i >= 1 else di - pd.Timedelta(days=1)
        pp = pd.Timestamp(dates[i - 2]) if i >= 2 else a - pd.Timedelta(days=1)
        short_d = [pp + pd.Timedelta(days=k) for k in range(1, (a - pp).days + 1)]
        week_d = [a - pd.Timedelta(days=k) for k in range(0, 7)]
        month_d = [a - pd.Timedelta(days=k) for k in range(0, 30)]
        se, ss, sc = collect(short_d)
        we, ws, wc = collect(week_d)
        me, ms, mc = collect(month_d)
        nov = 0.0
        if sc > 0 and mc > 0:
            nov = 1.0 - float(se @ me / (np.linalg.norm(se) * np.linalg.norm(me) + 1e-9))
        pcs = pca.transform(se[None, :])[0] if sc > 0 else np.zeros(K_PC, np.float32)
        rows.append({"cnt_short": sc, "cnt_week": wc, "cnt_month": mc,
                     "abn_vol": sc / (mc / 30.0 * 2 + 1.0),
                     "sent_short": ss, "sent_week": ws, "novelty": nov,
                     **{f"npc{j+1}": float(pcs[j]) for j in range(K_PC)}})
    news_feat = pd.DataFrame(rows)
    out = pd.concat([feat.reset_index(drop=True), news_feat], axis=1)
    out.to_parquet(os.path.join(C.ART, "features_ff.parquet"))

    fcols = [c for c in out.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
             "dist", "logvol", "streak", "peer_", "mkt_", "cnt_", "abn_", "sent_", "novelty", "npc"))]
    print("features_ff.parquet", out.shape, "| n_features:", len(fcols))
    print("news feature sample (last test row):")
    print(out[["date"] + [c for c in fcols if c.startswith(("cnt", "sent", "nov", "npc"))]].tail(1).T)

if __name__ == "__main__":
    main()
