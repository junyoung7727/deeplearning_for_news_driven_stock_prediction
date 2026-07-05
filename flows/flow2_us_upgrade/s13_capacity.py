"""
Stage 13 - Does a BIGGER model reach 70%?  Capacity sweep (honest, leak-free).

Hypothesis: the bottleneck is signal, not capacity.  So as model size grows,
TRAIN accuracy should rise toward ~1.0 while DEV/TEST accuracy stays ~0.5-0.56
(overfitting), never approaching 0.70.

Uses the strongest leak-free feature set (price/technical + cross-asset + news).
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
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import config as C
from s10_boost import price_features
from s12_crossasset import xasset_features

def build():
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    daily = pd.read_parquet(C.DAILY_PARQUET, columns=["ticker", "trade_date", "close", "volume"])
    daily = daily[daily.ticker == C.TARGET].copy()
    daily["date"] = pd.to_datetime(daily.trade_date); daily = daily.sort_values("date").reset_index(drop=True)
    daily["close"] = daily.close.astype(float); daily["volume"] = daily.volume.astype(float)
    daily["ret"] = daily.close / daily.close.shift(1) - 1
    samp = samp.merge(price_features(daily), on="date", how="left").merge(xasset_features(), on="date", how="left")
    cols = [c for c in samp.columns if c.startswith(("lagret", "mom", "vol", "maratio", "rsi",
            "dist", "logvol", "streak", "peer_", "mkt_"))]
    TEB = np.load(os.path.join(C.ART, "feats_TEB.npz"))
    news = np.concatenate([TEB["short"], TEB["mid"].mean(1), TEB["long"].mean(1)], axis=1)
    X = np.concatenate([samp[cols].values.astype(np.float32), news], 1)
    y = ((samp.label.values + 1) // 2).astype(int)
    sp = samp.split.values
    return X, y, sp == "train", sp == "dev", sp == "test"

def ev(name, model, X, y, tr, dv, te, needs_impute=False):
    if needs_impute:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    model.fit(X[tr], y[tr])
    a = lambda m: accuracy_score(y[m], model.predict(X[m]))
    return name, a(tr), a(dv), a(te)

def main():
    X, y, tr, dv, te = build()
    print(f"train={tr.sum()} dev={dv.sum()} test={te.sum()} features={X.shape[1]} "
          f"| samples/feature = {tr.sum()/X.shape[1]:.1f}")
    print(f"test majority baseline = {max(y[te].mean(), 1-y[te].mean()):.4f}\n")

    rows = []
    # GBM capacity ladder (trees x depth)
    for it, dep, leaf in [(100, 3, 40), (400, 3, 30), (400, 6, 20), (1500, 6, 10),
                          (3000, 10, 5), (5000, 12, 2)]:
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=it, max_depth=dep,
                                           min_samples_leaf=leaf, l2_regularization=0.0,
                                           random_state=C.SEED)
        rows.append(ev(f"GBM it={it} d={dep} leaf={leaf}", m, X, y, tr, dv, te))
    # MLP capacity ladder (width/depth) - bigger = more params
    for hl in [(64,), (256,), (1024,), (1024, 512), (2048, 1024, 512), (4096, 2048, 1024, 512)]:
        m = MLPClassifier(hidden_layer_sizes=hl, alpha=1e-4, max_iter=300,
                          early_stopping=False, random_state=C.SEED)
        n_params = sum(a*b for a, b in zip((X.shape[1],)+hl, hl+(1,)))
        rows.append(ev(f"MLP {hl} (~{n_params/1000:.0f}k params)", m, X, y, tr, dv, te, needs_impute=True))

    print(f"{'model':40s} {'train':>7s} {'dev':>7s} {'test':>7s}")
    for n, ta, da, te_ in rows:
        print(f"{n:40s} {ta:7.4f} {da:7.4f} {te_:7.4f}")

    best_test = max(r[3] for r in rows)
    max_train = max(r[1] for r in rows)
    print(f"\nAs capacity grows: max TRAIN acc = {max_train:.4f}, best TEST acc = {best_test:.4f}")
    print(f"=> bigger models fit train (up to {max_train:.2f}) but TEST stays ~"
          f"{np.median([r[3] for r in rows]):.2f}; capacity is NOT the bottleneck, signal is.")
    json.dump([{"model": n, "train": ta, "dev": da, "test": te_} for n, ta, da, te_ in rows],
              open(os.path.join(C.ART, "results_capacity.json"), "w"), indent=2)
    print("saved results_capacity.json")

if __name__ == "__main__":
    main()
