"""
Stage 7 - Train, tune (on dev) and test all models, then run the market
simulation.  Reproduces, for NVDA, the model matrix of Ding et al. (2015):

   Luss [2012]   bag-of-words + SVM                       (baseline)
   E-NN          discrete structured events + NN          (Ding 2014 baseline)
   WB-NN         word embeddings        + NN
   WB-CNN        word embeddings        + CNN
   E-CNN         discrete events        + CNN
   EB-NN         NTN event embeddings   + NN
   EB-CNN        NTN event embeddings   + CNN              (proposed, main)

Metrics: Accuracy + Matthews Correlation Coefficient (paper Sec 4.1).
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
import os, json, numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import config as C
from s6_models import DenseModel, EventModel, fit, predict, metrics
from s8_simulate import simulate, randomization_dist

N_SEEDS = 4

def load():
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    dense = {}
    for name in ["WB", "EB", "TWB", "TEB"]:
        p = os.path.join(C.ART, f"feats_{name}.npz")
        if os.path.exists(p):
            dense[name] = np.load(p)
    E = np.load(os.path.join(C.ART, "feats_E.npz"))
    return samp, dense, E

def idx_of(samp, split):
    return np.where(samp.split.values == split)[0].astype(np.int64)

def torch_dense(d):
    return (torch.from_numpy(d["short"]), torch.from_numpy(d["mid"]),
            torch.from_numpy(d["long"]))

def run_torch(build, inputs, y01, tr, dv, te):
    """Train N_SEEDS models and return the ENSEMBLE-averaged dev/test
    probabilities.  Averaging is an unbiased, variance-reduced estimate of the
    method - unlike best-dev-seed selection, which cherry-picks the seed most
    overfit to the small dev set."""
    import time
    dps, tps = [], []
    for s in range(N_SEEDS):
        t0 = time.time()
        torch.manual_seed(C.SEED + s); np.random.seed(C.SEED + s)
        model = build()
        model, dev_sel = fit(model, inputs, y01, tr, dv, seed=C.SEED + s)
        dps.append(predict(model, inputs, dv))
        tps.append(predict(model, inputs, te))
        print(f"    seed{s} dev_mcc={dev_sel:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return np.mean(dps, 0), np.mean(tps, 0)        # ensemble mean probability

def luss(samp, tr, dv, te, y01):
    txt = samp.doc_text.fillna("").values
    from sklearn.metrics import matthews_corrcoef
    best_dev, best = -2.0, None
    for Cc in [0.1, 0.5, 1.0, 2.0]:
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
        Xtr = vec.fit_transform(txt[tr])
        svm = LinearSVC(C=Cc).fit(Xtr, y01[tr])
        def prob(ix):
            s = svm.decision_function(vec.transform(txt[ix]))
            return 1.0 / (1.0 + np.exp(-s))
        dp = prob(dv)
        pr = (dp > 0.5).astype(int)
        mcc = matthews_corrcoef(y01[dv], pr) if len(np.unique(pr)) > 1 else 0.0
        if mcc > best_dev:
            best_dev, best = mcc, (dp, prob(te))
    return best

def main():
    samp, dense, E = load()
    y_pm1 = samp.label.values.astype(int)
    y01 = ((y_pm1 + 1) // 2).astype(int)
    tr, dv, te = idx_of(samp, "train"), idx_of(samp, "dev"), idx_of(samp, "test")
    print(f"samples: train={len(tr)} dev={len(dv)} test={len(te)}  "
          f"test up-rate={(y01[te]==1).mean():.3f}")

    dense_t = {name: torch_dense(d) for name, d in dense.items()}
    dense_dim = {name: int(d["short"].shape[1]) for name, d in dense.items()}
    es = torch.from_numpy(E["short"]); em = torch.from_numpy(E["mid"]); el = torch.from_numpy(E["long"])
    n_ids = int(E["n_ids"])

    def dense_inputs(t):
        return lambda ix: (t[0][ix], t[1][ix], t[2][ix])
    def event_inputs():
        return lambda ix: (es[ix], em[ix], el[ix])

    def dense_spec(t, dim, nn_only):
        return lambda: run_torch(lambda: DenseModel(dim, nn_only), dense_inputs(t), y01, tr, dv, te)

    # order mirrors the paper's table, transformer reps appended
    specs = {}
    for name in ["WB", "E", "EB", "TWB", "TEB"]:
        if name == "E":
            specs["E-NN"]  = lambda: run_torch(lambda: EventModel(n_ids, C.WORD_DIM, True),  event_inputs(), y01, tr, dv, te)
            specs["E-CNN"] = lambda: run_torch(lambda: EventModel(n_ids, C.WORD_DIM, False), event_inputs(), y01, tr, dv, te)
        elif name in dense_t:
            specs[f"{name}-NN"]  = dense_spec(dense_t[name], dense_dim[name], True)
            specs[f"{name}-CNN"] = dense_spec(dense_t[name], dense_dim[name], False)

    results = {}
    # Luss baseline
    dp, tp = luss(samp, tr, dv, te, y01)
    results["Luss[2012] (BoW+SVM)"] = (dp, tp)
    for name, fn in specs.items():
        print("training", name, "...", flush=True)
        results[name] = fn()
        dp_, tp_ = results[name]
        ta_, tm_ = metrics(y_pm1[te], tp_)
        print(f"  -> {name}: test_acc={ta_:.4f} test_mcc={tm_:.4f}", flush=True)

    # ---- metrics + simulation table ----
    o, h, l, c = (samp.open.values, samp.high.values, samp.low.values, samp.close.values)
    rand = randomization_dist(o[te], h[te], l[te], c[te])
    rmean = float(rand.mean())
    rows = []
    sim_rows = []
    for name, (dp, tp) in results.items():
        d_acc, d_mcc = metrics(y_pm1[dv], dp)
        t_acc, t_mcc = metrics(y_pm1[te], tp)
        rows.append((name, d_acc, d_mcc, t_acc, t_mcc))
        basic = simulate(tp, o[te], h[te], l[te], c[te]).sum()
        thr = simulate(tp, o[te], h[te], l[te], c[te], threshold=C.SIM_THRESHOLD).sum()
        pval = float((rand >= basic).mean())
        sim_rows.append((name, basic, thr, rmean, pval))

    print("\n================ NVDA prediction (Acc / MCC) ================")
    print(f"{'Model':24s} {'Dev Acc':>8s} {'Dev MCC':>8s} {'Test Acc':>9s} {'Test MCC':>9s}")
    for name, da, dm, ta, tm in rows:
        print(f"{name:24s} {da:8.4f} {dm:8.4f} {ta:9.4f} {tm:9.4f}")

    print("\n================ Market simulation on TEST (per $10,000) ================")
    print(f"{'Model':24s} {'Always':>12s} {'beta=0.70':>12s} {'Rand-mean':>12s} {'p-value':>8s}")
    for name, b, t, rm, pv in sim_rows:
        print(f"{name:24s} {b:12,.0f} {t:12,.0f} {rm:12,.0f} {pv:8.3f}")

    out = {
        "metrics": [{"model": n, "dev_acc": da, "dev_mcc": dm, "test_acc": ta, "test_mcc": tm}
                    for n, da, dm, ta, tm in rows],
        "simulation": [{"model": n, "always_profit": b, "threshold_profit": t,
                        "rand_mean": rm, "p_value": pv} for n, b, t, rm, pv in sim_rows],
        "n_test": int(len(te)), "test_up_rate": float((y01[te] == 1).mean()),
    }
    with open(os.path.join(C.ART, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved artifacts/results.json")

if __name__ == "__main__":
    main()
