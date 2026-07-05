"""
Stage 35a - controlled experiment on SAVED s34 features (identical representation,
identical split): retrain the EB-CNN head ONLY on event-window samples (has_any),
instead of all 147k samples (96.6% of which have all-zero features).

Isolates root-cause #2 (zero-feature training dilution) from #1 (news-time
alignment, which needs a pipeline re-run and is handled in s36).
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
import numpy as np, torch, time
import config as C
from s6_models import DenseModel, fit, predict, metrics

torch.manual_seed(C.SEED); np.random.seed(C.SEED)
t0 = time.time()

z = np.load(_ROOT / "artifacts" / "kr_ie_features.npz")
y = z["y"]; hs = z["has_short"]; ha = z["has_any"]
DT = z["DT"].astype("datetime64[D]")

keep = np.where(ha)[0]                       # event-in-30d-window samples only
S = torch.from_numpy(z["S"][keep])
M = torch.from_numpy(z["M"][keep])
L = torch.from_numpy(z["L"][keep])           # loads full L then subsets
yk = y[keep]; hsk = hs[keep]; dtk = DT[keep]
del z
print(f"kept {len(keep)} / {len(y)} samples  ({time.time()-t0:.0f}s)", flush=True)

# same date split as s34: 60% of ALL unique dates
dates = np.sort(np.unique(DT)); split = dates[int(len(dates)*0.6)]
tr = dtk < split; te = dtk >= split
trN = np.where(tr)[0]; teN = np.where(te)[0]

# time-based dev: last 15% of TRAIN dates (s34 used a ticker-ordered slice)
trd = np.sort(np.unique(dtk[tr])); dsplit = trd[int(len(trd)*0.85)]
dvN = trN[dtk[trN] >= dsplit]; trN = trN[dtk[trN] < dsplit]
print(f"train {len(trN)}  dev {len(dvN)}  test {len(teN)}", flush=True)

def inp(ix): return (S[ix], M[ix], L[ix])
m = DenseModel(C.NTN_K, nn_only=False)
m, best_dev = fit(m, inp, yk, trN, dvN, seed=C.SEED, verbose=True)
p = predict(m, inp, teN)
ypm = np.where(yk == 1, 1, -1)

def rep(name, mask_te):
    yy = ypm[teN][mask_te]; pp = p[mask_te]
    base = max((yy == 1).mean(), (yy == -1).mean())
    a, mc = metrics(yy, pp)
    print(f"  [{name}] n={int(mask_te.sum())} base={base:.4f} acc={a:.4f} mcc={mc:+.4f}")
    cc = np.abs(pp - 0.5)
    for kap in (0.5, 0.3, 0.2, 0.1):
        thr = np.quantile(cc, 1 - kap); s = cc >= thr
        if s.sum() >= 20:
            aa, mm = metrics(yy[s], pp[s])
            bb = max((yy[s] == 1).mean(), (yy[s] == -1).mean())
            print(f"     conf top {int(kap*100)}%: acc={aa:.4f} mcc={mm:+.4f} base={bb:.4f} (n={int(s.sum())})")

print(f"\nKR SMALL-CAP EB-CNN, EVENT-DAY-ONLY TRAINING (dev_mcc={best_dev:.4f}):")
rep("ANY-event-in-window", np.ones(len(teN), bool))
rep("EVENT-days (prior trading day)", hsk[teN])
