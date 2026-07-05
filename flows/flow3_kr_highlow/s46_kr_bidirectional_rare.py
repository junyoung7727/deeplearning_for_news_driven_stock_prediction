"""
Stage 46 - bidirectional rare-event learning for KR HIGH/LOW exceedance.

User correction: do not learn only 5% upward hits. Learn downward tail events too.
Targets are six multi-label heads:
  UP2/UP3/UP5   = high(d) >= open(d) * (1+k)
  DN2/DN3/DN5   = low(d)  <= open(d) * (1-k)

Technique stack for rare classes:
  - symmetric up/down multi-task sharing
  - FinBERT+novelty+price-token B3 feature set from s45
  - focal BCE with clipped positive-class weights from train prevalence
  - dev-calibrated per-label thresholds for MCC instead of fixed 0.5
  - report AP plus top-10% precision/hit-rate for UP5 and DN5
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

# Force a fresh bidirectional feature cache with DROP labels.
os.environ.setdefault("KR45_FEAT", os.path.join(os.path.expanduser("~"), "dlfe", "artifacts", "kr46_features_bidirectional.npz"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "dlfe", "code"))
from s45_kr_feature_ladder import build_features, MID_DAYS, LONG_DAYS, KS, DEV, log

SEED = 13
NAMES = ["UP2", "UP3", "UP5", "DN2", "DN3", "DN5"]
FOCUS = [2, 5]
torch.manual_seed(SEED); np.random.seed(SEED)

class HeadBi(nn.Module):
    def __init__(self, ddim, vdim, E=384, layers=4, heads=6, ff=1536, p=0.35, n_out=6):
        super().__init__()
        self.ddim = ddim
        self.proj = nn.Linear(ddim, E)
        self.vproj = nn.Linear(vdim, E)
        self.pproj = nn.Linear(5, E)
        n_tok = 1 + 1 + 1 + MID_DAYS + LONG_DAYS + 30
        self.cls = nn.Parameter(torch.zeros(1, 1, E))
        self.pos = nn.Parameter(torch.zeros(1, n_tok, E))
        nn.init.normal_(self.cls, 0, 0.02); nn.init.normal_(self.pos, 0, 0.02)
        enc = nn.TransformerEncoderLayer(E, heads, ff, dropout=p, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(E)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(E, n_out)
    def forward(self, s, m, l, v, pr):
        B = s.shape[0]
        seq = torch.cat([s[:, :self.ddim].unsqueeze(1), m[:, :, :self.ddim], l[:, :, :self.ddim]], 1)
        empty = seq.abs().sum(-1) == 0
        x = torch.cat([
            self.cls.expand(B, -1, -1),
            self.vproj(v).unsqueeze(1),
            self.proj(seq),
            self.pproj(pr),
        ], 1) + self.pos
        mask = torch.cat([
            torch.zeros(B, 2, dtype=torch.bool, device=s.device),
            empty,
            torch.zeros(B, 30, dtype=torch.bool, device=s.device),
        ], 1)
        x = self.enc(x, src_key_padding_mask=mask)
        return self.fc(self.drop(self.norm(x[:, 0])))

def best_thresholds(y, p):
    from sklearn.metrics import matthews_corrcoef
    th, mcc = [], []
    for j in range(y.shape[1]):
        cand = np.unique(np.concatenate([
            np.linspace(0.02, 0.98, 49),
            np.quantile(p[:, j], np.linspace(0.50, 0.995, 80)),
        ]))
        best_t, best_m = 0.5, -1.0
        for t in cand:
            pred = p[:, j] >= t
            if pred.any() and (~pred).any():
                mm = matthews_corrcoef(y[:, j], pred.astype(int))
                if mm > best_m:
                    best_t, best_m = float(t), float(mm)
        th.append(best_t); mcc.append(best_m if best_m > -1 else 0.0)
    return np.array(th, np.float32), np.array(mcc, np.float32)

def mcc_at(y, p, th):
    from sklearn.metrics import matthews_corrcoef
    out = []
    for j in range(y.shape[1]):
        pred = p[:, j] >= th[j]
        out.append(matthews_corrcoef(y[:, j], pred.astype(int)) if pred.any() and (~pred).any() else 0.0)
    return np.array(out, np.float32)

def fit_eval(T, Y, trN, dvN, teN, seeds=(13, 14), epochs=45, patience=9, batch=256,
             lr=2e-4, wd=0.05, gamma=2.0, smooth=0.01, news_drop=0.10, noise=0.03):
    from sklearn.metrics import accuracy_score, average_precision_score
    S, M, L, V, P = T
    pos = Y[trN].mean(0)
    posw_np = np.clip((1.0 - pos) / np.maximum(pos, 1e-4), 1.0, 25.0).astype(np.float32)
    log("train prevalence " + " ".join(f"{n}={p:.3f}" for n, p in zip(NAMES, pos)))
    log("pos_weight " + " ".join(f"{n}={w:.1f}" for n, w in zip(NAMES, posw_np)))
    Yt = torch.from_numpy(Y.astype(np.float32))
    posw = torch.from_numpy(posw_np).to(DEV)

    def predict(model, idx, bs=2048):
        model.eval(); out = np.zeros((len(idx), Y.shape[1]), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(idx), bs):
                ii = idx[s0:s0 + bs]
                out[s0:s0 + bs] = torch.sigmoid(model(S[ii].to(DEV), M[ii].to(DEV), L[ii].to(DEV), V[ii].to(DEV), P[ii].to(DEV))).cpu().numpy()
        return out

    ens_tr, ens_dv, ens_te = [], [], []
    t0 = time.time()
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        model = HeadBi(ddim=S.shape[1], vdim=V.shape[1]).to(DEV)
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            (nodecay if p.ndim <= 1 else decay).append(p)
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
                if news_drop:
                    m = m * (torch.rand(m.shape[0], m.shape[1], 1, device=DEV) > news_drop).float()
                    l = l * (torch.rand(l.shape[0], l.shape[1], 1, device=DEV) > news_drop).float()
                if noise:
                    s = s + noise * torch.randn_like(s) * (s != 0)
                    m = m + noise * torch.randn_like(m) * (m != 0)
                    l = l + noise * torch.randn_like(l) * (l != 0)
                yb = Yt[ii].to(DEV) * (1 - smooth) + 0.5 * smooth
                logits = model(s, m, l, v, p)
                bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none", pos_weight=posw)
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
            if bad >= patience:
                break
        model.load_state_dict(best_state)
        ens_tr.append(predict(model, trN)); ens_dv.append(predict(model, dvN)); ens_te.append(predict(model, teN))
        log(f"seed {sd} best dev focus MCC={best:+.4f}")
    p_tr, p_dv, p_te = map(lambda xs: np.mean(xs, 0), [ens_tr, ens_dv, ens_te])
    th, dev_mcc = best_thresholds(Y[dvN], p_dv)
    tr_mcc, te_mcc = mcc_at(Y[trN], p_tr, th), mcc_at(Y[teN], p_te, th)
    print(f"\n[BIDIR rare focal+calibrated] params={sum(p.numel() for p in HeadBi(S.shape[1], V.shape[1]).parameters())/1e6:.2f}M ({time.time()-t0:.0f}s)", flush=True)
    for j, name in enumerate(NAMES):
        yy = Y[teN, j].astype(int); pred = p_te[:, j] >= th[j]
        ap = average_precision_score(yy, p_te[:, j]) if yy.any() else float("nan")
        line = (f"{name}: base={yy.mean():.4f} th={th[j]:.3f} acc={accuracy_score(yy, pred):.4f} "
                f"MCC={te_mcc[j]:+.4f} (train {tr_mcc[j]:+.4f}, GAP {tr_mcc[j]-te_mcc[j]:+.4f}) AP={ap:.4f}")
        if name in ("UP5", "DN5"):
            order = np.argsort(-p_te[:, j]); n10 = max(len(yy) // 10, 20); n5 = max(len(yy) // 20, 20)
            line += f" top10={yy[order[:n10]].mean():.3f} top5={yy[order[:n5]].mean():.3f}"
        print(line, flush=True)
    # directional selector: among UP5/DN5, pick whichever model is more confident.
    up, dn = p_te[:, 2], p_te[:, 5]
    margin = np.maximum(up - th[2], dn - th[5])
    for frac in (0.10, 0.05):
        n = max(int(len(teN) * frac), 20)
        idx = np.argsort(-margin)[:n]
        side_up = up[idx] >= dn[idx]
        ysel = np.where(side_up, Y[teN[idx], 2], Y[teN[idx], 5])
        print(f"DIR5 top{int(frac*100)}%: hit={ysel.mean():.3f} n={n} long_frac={side_up.mean():.3f}", flush=True)

def main():
    Fd = build_features()
    S, M, L = Fd["S"].astype(np.float32), Fd["M"].astype(np.float32), Fd["L"].astype(np.float32)
    HIT, DROP = Fd["HIT"].astype(np.float32), Fd["DROP"].astype(np.float32)
    Y = np.concatenate([HIT, DROP], axis=1)
    VOLF, NOVF, SENTF, PSEQ = (Fd[k].astype(np.float32) for k in ("VOLF", "NOVF", "SENTF", "PSEQ"))
    DT = Fd["DT"].astype("datetime64[ns]")
    dates = np.sort(np.unique(DT)); split = dates[int(len(dates) * 0.6)]
    trN = np.where(DT < split)[0]; teN = np.where(DT >= split)[0]
    trd = np.sort(np.unique(DT[DT < split])); dsplit = trd[int(len(trd) * 0.85)]
    dvN = trN[DT[trN] >= dsplit]; trN = trN[DT[trN] < dsplit]
    def z(X):
        mu, sd = X[trN].mean(0), X[trN].std(0) + 1e-9
        return ((X - mu) / sd).astype(np.float32)
    Vz = z(VOLF)
    V12 = np.concatenate([Vz, z(NOVF[:, :5]), NOVF[:, 5:6]], 1).astype(np.float32)
    V15 = np.concatenate([V12, z(SENTF)], 1).astype(np.float32)
    T = tuple(torch.from_numpy(x) for x in (S, M, L, V15, PSEQ))
    log(f"samples={len(Y)} train={len(trN)} dev={len(dvN)} test={len(teN)} labels " + " ".join(f"{n}={Y[:,j].mean():.3f}" for j,n in enumerate(NAMES)))
    fit_eval(T, Y, trN, dvN, teN)

if __name__ == "__main__":
    main()
