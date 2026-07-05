"""
Stage 47 - push UP5/DN5 to a statistically defensible level (remote GPU).

Upgrades over s46:
  1. GBM tabular learner (HistGradientBoosting) on compact features:
     V15 (vol/gap + novelty + FinBERT) + 65 price-window aggregates from PSEQ.
     Decorrelated from the transformer -> ensemble candidate.
  2. TF retrain (identical s46 protocol, 2 seeds) but test/dev/train probs SAVED.
  3. Per-label ensemble weight w_j (dev-MCC grid search, thresholds recalibrated).
  4. Significance machinery (the point of this stage):
       - date-cluster bootstrap 95% CI for MCC and top-decile hit-rate
         (resample test DATES, not rows - respects cross-sectional correlation)
       - daily cross-sectional top-k selection (k=1,3): the tradable statistic
       - 4 contiguous test-period chunks: temporal stability of MCC
Outputs: ~/dlfe/artifacts/kr47_probs.npz + s47.log
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
import os, sys, math, time, copy, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("KR45_FEAT", os.path.join(os.path.expanduser("~"), "dlfe",
                      "artifacts", "kr46_features_bidirectional.npz"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s46_kr_bidirectional_rare import HeadBi, best_thresholds, NAMES, FOCUS
from s45_kr_feature_ladder import build_features, DEV, log

SEED = 13
ART = os.path.join(os.path.expanduser("~"), "dlfe", "artifacts")
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------ GBM
def pseq_aggs(P):
    feats = []
    for w in (5, 10, 30):
        seg = P[:, -w:, :]
        feats += [seg.mean(1), seg.std(1), seg.max(1), seg.min(1)]
    feats.append(P[:, -1, :])
    return np.concatenate(feats, 1).astype(np.float32)  # 5*(4*3+1) = 65

def fit_gbm(Xtab, Y, trN, dvN, teN):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p_tr = np.zeros((len(trN), Y.shape[1]), np.float32)
    p_dv = np.zeros((len(dvN), Y.shape[1]), np.float32)
    p_te = np.zeros((len(teN), Y.shape[1]), np.float32)
    for j, name in enumerate(NAMES):
        y = Y[:, j]
        pos = y[trN].mean()
        wpos = float(np.clip((1 - pos) / max(pos, 1e-4), 1.0, 25.0))
        sw = np.where(y[trN] == 1, wpos, 1.0)
        t0 = time.time()
        gbm = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            random_state=SEED)
        gbm.fit(Xtab[trN], y[trN], sample_weight=sw)
        p_tr[:, j] = gbm.predict_proba(Xtab[trN])[:, 1]
        p_dv[:, j] = gbm.predict_proba(Xtab[dvN])[:, 1]
        p_te[:, j] = gbm.predict_proba(Xtab[teN])[:, 1]
        log(f"GBM {name} iters={gbm.n_iter_} wpos={wpos:.1f} ({time.time()-t0:.0f}s)")
    return p_tr, p_dv, p_te

# ------------------------------------------------------------------ TF
def train_tf(T, Y, trN, dvN, teN, seeds=(13, 14), epochs=45, patience=9,
             batch=256, lr=2e-4, wd=0.05, gamma=2.0, smooth=0.01,
             news_drop=0.10, noise=0.03):
    S, M, L, V, P = T
    pos = Y[trN].mean(0)
    posw = torch.from_numpy(np.clip((1 - pos) / np.maximum(pos, 1e-4), 1.0, 25.0)
                            .astype(np.float32)).to(DEV)
    Yt = torch.from_numpy(Y.astype(np.float32))
    def predict(model, idx, bs=2048):
        model.eval(); out = np.zeros((len(idx), Y.shape[1]), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(idx), bs):
                ii = idx[s0:s0 + bs]
                out[s0:s0 + bs] = torch.sigmoid(model(
                    S[ii].to(DEV), M[ii].to(DEV), L[ii].to(DEV),
                    V[ii].to(DEV), P[ii].to(DEV))).cpu().numpy()
        return out
    ens = {"tr": [], "dv": [], "te": []}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        model = HeadBi(ddim=S.shape[1], vdim=V.shape[1]).to(DEV)
        decay, nodecay = [], []
        for n_, p_ in model.named_parameters():
            (nodecay if p_.ndim <= 1 else decay).append(p_)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                                 {"params": nodecay, "weight_decay": 0.0}], lr=lr)
        steps_per = math.ceil(len(trN) / batch)
        def lr_at(step):
            e = step / steps_per
            if e < 2: return lr * (e / 2 + 1e-3)
            return lr * 0.5 * (1 + math.cos(math.pi * min((e - 2) / max(epochs - 2, 1), 1.0)))
        best, best_state, bad, step = -9.0, None, 0, 0
        rng = np.random.default_rng(sd)
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(trN)
            for s0 in range(0, len(perm), batch):
                ii = perm[s0:s0 + batch]
                s, m, l, v, p = (X[ii].to(DEV) for X in (S, M, L, V, P))
                m = m * (torch.rand(m.shape[0], m.shape[1], 1, device=DEV) > news_drop).float()
                l = l * (torch.rand(l.shape[0], l.shape[1], 1, device=DEV) > news_drop).float()
                s = s + noise * torch.randn_like(s) * (s != 0)
                m = m + noise * torch.randn_like(m) * (m != 0)
                l = l + noise * torch.randn_like(l) * (l != 0)
                yb = Yt[ii].to(DEV) * (1 - smooth) + 0.5 * smooth
                logits = model(s, m, l, v, p)
                bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none",
                                                         pos_weight=posw)
                prob = torch.sigmoid(logits)
                pt = prob * yb + (1 - prob) * (1 - yb)
                loss = (((1 - pt).clamp_min(1e-4) ** gamma) * bce).mean()
                for g in opt.param_groups: g["lr"] = lr_at(step)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); step += 1
            pdv = predict(model, dvN)
            _, mdv = best_thresholds(Y[dvN], pdv)
            score = float(mdv[FOCUS].mean())
            if score > best:
                best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if bad >= patience: break
        model.load_state_dict(best_state)
        for k, idx in (("tr", trN), ("dv", dvN), ("te", teN)):
            ens[k].append(predict(model, idx))
        log(f"TF seed {sd} best dev focus MCC={best:+.4f}")
    return tuple(np.mean(ens[k], 0) for k in ("tr", "dv", "te"))

# ------------------------------------------------------------------ stats
def date_codes(DT, idx):
    d = DT[idx]
    uniq, codes = np.unique(d, return_inverse=True)
    return uniq, codes

def boot_mcc_ci(y, p, th, codes, D, B=2000, seed=0):
    pred = (p >= th).astype(np.int64); yy = y.astype(np.int64)
    cls = yy * 2 + pred  # 0=TN,1=FP,2=FN,3=TP
    counts = np.zeros((D, 4), np.float64)
    np.add.at(counts, (codes, cls), 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, (B, D))
    c = counts[idx].sum(1)  # B x 4
    tn, fp, fn, tp = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    tot = counts.sum(0)
    den0 = math.sqrt((tot[3] + tot[1]) * (tot[3] + tot[2]) * (tot[0] + tot[1]) * (tot[0] + tot[2]))
    point = (tot[3] * tot[0] - tot[1] * tot[2]) / den0 if den0 > 0 else 0.0
    return point, float(np.percentile(mcc, 2.5)), float(np.percentile(mcc, 97.5))

def boot_rate_ci(hits, sel, codes, D, B=2000, seed=0):
    """hits/sel: per-row selected-hit indicator and selected indicator."""
    num = np.zeros(D); den = np.zeros(D)
    np.add.at(num, codes, hits); np.add.at(den, codes, sel)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, (B, D))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    point = num.sum() / max(den.sum(), 1)
    return point, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

def eval_label(name, y, p, th, DT_te, codes, D):
    from sklearn.metrics import accuracy_score, average_precision_score
    pred = p >= th
    acc = accuracy_score(y, pred.astype(int))
    ap = average_precision_score(y, p)
    m, mlo, mhi = boot_mcc_ci(y, p, th, codes, D)
    order = np.argsort(-p); n10 = len(y) // 10
    sel = np.zeros(len(y)); sel[order[:n10]] = 1
    r, rlo, rhi = boot_rate_ci(sel * y, sel, codes, D, seed=1)
    print(f"{name}: base={y.mean():.4f} th={th:.3f} acc={acc:.4f} AP={ap:.4f} "
          f"MCC={m:+.4f} [{mlo:+.4f},{mhi:+.4f}] top10={r:.3f} [{rlo:.3f},{rhi:.3f}]",
          flush=True)
    # temporal stability: 4 contiguous chunks
    uniq = np.unique(DT_te); chunks = np.array_split(uniq, 4)
    parts = []
    for ch in chunks:
        mask = np.isin(DT_te, ch)
        if mask.sum() < 100: parts.append("n/a"); continue
        yy, pp = y[mask], p[mask]
        prd = pp >= th
        if prd.any() and (~prd).any():
            from sklearn.metrics import matthews_corrcoef
            parts.append(f"{matthews_corrcoef(yy, prd.astype(int)):+.3f}")
        else:
            parts.append("deg")
    print(f"   quarter MCC: {' | '.join(parts)}", flush=True)
    return m, mlo, mhi

def daily_topk(name, y, p, DT_te, ks=(1, 3)):
    import pandas as pd
    df = pd.DataFrame({"d": DT_te, "y": y, "p": p})
    for k in ks:
        g = df.sort_values("p", ascending=False).groupby("d")
        top = g.head(k)
        per_day = top.groupby("d").y.mean()
        base = df.groupby("d").y.mean()
        rng = np.random.default_rng(2)
        vals = per_day.values; B = 2000
        boot = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(1)
        print(f"   {name} daily top{k}: hit={vals.mean():.3f} "
              f"[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}] "
              f"vs day-base {base.mean():.3f} (days={len(vals)})", flush=True)

# ------------------------------------------------------------------ main
def main():
    Fd = build_features()
    S, M, L = (Fd[k].astype(np.float32) for k in ("S", "M", "L"))
    HIT, DROP = Fd["HIT"].astype(np.float32), Fd["DROP"].astype(np.float32)
    Y = np.concatenate([HIT, DROP], 1)
    VOLF, NOVF, SENTF, PSEQ = (Fd[k].astype(np.float32) for k in
                               ("VOLF", "NOVF", "SENTF", "PSEQ"))
    DT = Fd["DT"].astype("datetime64[ns]")
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[DT[trN] >= dsplit]; trN = trN[DT[trN] < dsplit]
    def z(X):
        mu, sd = X[trN].mean(0), X[trN].std(0) + 1e-9
        return ((X - mu) / sd).astype(np.float32)
    V15 = np.concatenate([z(VOLF), z(NOVF[:, :5]), NOVF[:, 5:6], z(SENTF)], 1)
    Xtab = np.concatenate([V15, pseq_aggs(PSEQ)], 1)
    log(f"tabular X {Xtab.shape}; train/dev/test {len(trN)}/{len(dvN)}/{len(teN)}")

    # 1) GBM
    g_tr, g_dv, g_te = fit_gbm(Xtab, Y, trN, dvN, teN)
    th_g, mcc_g = best_thresholds(Y[dvN], g_dv)
    log("GBM dev MCC " + " ".join(f"{n}={m:+.3f}" for n, m in zip(NAMES, mcc_g)))

    # 2) TF (saved probs)
    T = tuple(torch.from_numpy(x) for x in (S, M, L, V15, PSEQ))
    t_tr, t_dv, t_te = train_tf(T, Y, trN, dvN, teN)
    th_t, mcc_t = best_thresholds(Y[dvN], t_dv)
    log("TF  dev MCC " + " ".join(f"{n}={m:+.3f}" for n, m in zip(NAMES, mcc_t)))

    # 3) per-label ensemble weight
    W = np.zeros(Y.shape[1], np.float32)
    e_dv = np.zeros_like(t_dv); e_te = np.zeros_like(t_te)
    from sklearn.metrics import matthews_corrcoef
    for j in range(Y.shape[1]):
        best_w, best_m = 0.0, -9.0
        for w in np.linspace(0, 1, 11):
            bl = w * t_dv[:, j] + (1 - w) * g_dv[:, j]
            th, _ = best_thresholds(Y[dvN][:, [j]], bl[:, None])
            pred = bl >= th[0]
            if pred.any() and (~pred).any():
                mm = matthews_corrcoef(Y[dvN][:, j], pred.astype(int))
                if mm > best_m: best_w, best_m = float(w), float(mm)
        W[j] = best_w
        e_dv[:, j] = best_w * t_dv[:, j] + (1 - best_w) * g_dv[:, j]
        e_te[:, j] = best_w * t_te[:, j] + (1 - best_w) * g_te[:, j]
    th_e, mcc_e = best_thresholds(Y[dvN], e_dv)
    log("ENS w(TF) " + " ".join(f"{n}={w:.1f}" for n, w in zip(NAMES, W)))

    # 4) evaluation with significance
    DT_te = DT[teN]
    uniq, codes = date_codes(DT, teN); D = len(uniq)
    print(f"\n=== s47 test evaluation (n={len(teN)}, days={D}) ===", flush=True)
    for tag, pte, th in (("GBM", g_te, th_g), ("TF ", t_te, th_t), ("ENS", e_te, th_e)):
        print(f"\n--- {tag} ---", flush=True)
        for j, name in enumerate(NAMES):
            eval_label(f"{tag} {name}", Y[teN, j], pte[:, j], th[j], DT_te, codes, D)
    print("\n--- daily cross-sectional selection (ENS) ---", flush=True)
    for j, name in enumerate(NAMES):
        if name in ("UP5", "DN5"):
            daily_topk(name, Y[teN, j], e_te[:, j], DT_te)
    np.savez(os.path.join(ART, "kr47_probs.npz"),
             g_te=g_te, t_te=t_te, e_te=e_te, y_te=Y[teN], th_e=th_e, W=W,
             dates_te=DT_te.astype("datetime64[D]").astype(str))
    log("saved kr47_probs.npz")

if __name__ == "__main__":
    main()
