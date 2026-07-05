"""
Stage 4 - Neural Tensor Network for event embeddings (paper Sec 2.2, Fig.2,
Eq.1, Algorithm 1).

Architecture (k = d = 100, f = tanh):
    R1 = f( O1^T T1^[1:k] P  + W1 [O1;P] + b1 )      R1 in R^k
    R2 = f( P^T  T2^[1:k] O2 + W2 [P;O2] + b2 )      R2 in R^k
    U  = f( R1^T T3^[1:k] R2 + W3 [R1;R2] + b3 )      U  in R^k   (event embedding)
    score(E) = u . U                                  (scalar, for the margin loss)

Each argument vector (O1,P,O2 in R^d) is the average of the skip-gram word
embeddings of the words composing it (paper Sec 2.2).

Training (Algorithm 1):
    - corrupt each event once by replacing one argument's words with random
      dictionary words  -> E^r  (fixed corruption).
    - margin loss  L = max(0, 1 - score(E) + score(E^r)) + lambda ||Phi||^2
    - events whose hinge reaches 0 are removed from the active set; iterate
      until the set is empty or N=500 iterations are reached.

Outputs:
    artifacts/event_emb.npy   (M x d, float32) event embeddings aligned 1:1 with
                              the rows of artifacts/events.parquet
    artifacts/ntn.pt          trained parameters
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
import os, numpy as np, pandas as pd, torch
import config as C

torch.manual_seed(C.SEED); np.random.seed(C.SEED)
DEV = "cpu"

# ---------------------------------------------------------------- word lookup
def load_word_emb():
    z = np.load(os.path.join(C.ART, "word_vectors.npz"), allow_pickle=True)
    vocab = list(z["vocab"]); vectors = z["vectors"].astype(np.float32)
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i, vectors

def avg_vec(words, w2i, W):
    idx = [w2i[w] for w in words if w in w2i]
    if not idx:
        return None
    return W[idx].mean(0)

def build_arg_matrices(ev, w2i, W):
    """Return O1,P,O2 (M x d) and a validity mask (all three args in-vocab)."""
    d = W.shape[1]; M = len(ev)
    O1 = np.zeros((M, d), np.float32); P = np.zeros((M, d), np.float32)
    O2 = np.zeros((M, d), np.float32); ok = np.zeros(M, bool)
    for i, (a, p, b) in enumerate(zip(ev.o1.values, ev.p.values, ev.o2.values)):
        va, vp, vb = avg_vec(a, w2i, W), avg_vec(p, w2i, W), avg_vec(b, w2i, W)
        if va is not None and vp is not None and vb is not None:
            O1[i], P[i], O2[i] = va, vp, vb; ok[i] = True
    return O1, P, O2, ok

# ---------------------------------------------------------------- NTN module
class NTN(torch.nn.Module):
    def __init__(self, d, k):
        super().__init__()
        s = 1.0 / np.sqrt(d)
        self.T1 = torch.nn.Parameter(torch.randn(d, d, k) * s)
        self.T2 = torch.nn.Parameter(torch.randn(d, d, k) * s)
        self.T3 = torch.nn.Parameter(torch.randn(k, k, k) * (1.0 / np.sqrt(k)))
        self.W1 = torch.nn.Linear(2 * d, k); self.W2 = torch.nn.Linear(2 * d, k)
        self.W3 = torch.nn.Linear(2 * k, k)
        self.u  = torch.nn.Linear(k, 1, bias=False)
        self.d, self.k = d, k

    @staticmethod
    def _bilinear(a, T, b):                      # a:(n,da) T:(da,db,k) b:(n,db)
        n = a.shape[0]; da, db, k = T.shape
        tmp = (a @ T.reshape(da, db * k)).view(n, db, k)
        return torch.einsum('ndk,nd->nk', tmp, b)

    def embed(self, O1, P, O2):
        R1 = torch.tanh(self._bilinear(O1, self.T1, P) + self.W1(torch.cat([O1, P], 1)))
        R2 = torch.tanh(self._bilinear(P, self.T2, O2) + self.W2(torch.cat([P, O2], 1)))
        U  = torch.tanh(self._bilinear(R1, self.T3, R2) + self.W3(torch.cat([R1, R2], 1)))
        return U

    def score(self, O1, P, O2):
        return self.u(self.embed(O1, P, O2)).squeeze(1)

    def l2(self):
        return sum((p ** 2).sum() for p in self.parameters())

# ---------------------------------------------------------------- corruption
def corrupt(O1, P, O2, ok, w2i, W, ev):
    """Fixed corruption: replace one argument (O1 or O2) of each event with the
    average of |arg| random dictionary words."""
    rng = np.random.default_rng(C.SEED)
    V = W.shape[0]; M = len(ev)
    cO1, cO2 = O1.copy(), O2.copy()
    which = rng.integers(0, 2, size=M)           # 0 -> corrupt O1, 1 -> corrupt O2
    for i in range(M):
        if not ok[i]:
            continue
        arg = ev.o1.values[i] if which[i] == 0 else ev.o2.values[i]
        n = max(1, len(arg))
        rnd = W[rng.integers(0, V, size=n)].mean(0)
        if which[i] == 0:
            cO1[i] = rnd
        else:
            cO2[i] = rnd
    return cO1, cO2, which

# ---------------------------------------------------------------- training
def main():
    ev = pd.read_parquet(os.path.join(C.ART, "events.parquet"))
    vocab, w2i, W = load_word_emb()
    Wt = torch.from_numpy(W)
    O1, P, O2, ok = build_arg_matrices(ev, w2i, W)
    V = W.shape[0]

    # training set = emb-eligible events with all args in vocab
    train_mask = ok & ev.emb_eligible.values
    tr = np.where(train_mask)[0]
    print(f"events={len(ev)}  in-vocab={int(ok.sum())}  NTN-train={len(tr)}")

    O1t, Pt, O2t = map(torch.from_numpy, (O1, P, O2))

    model = NTN(C.WORD_DIM, C.NTN_K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=C.NTN_LR)

    rng = np.random.default_rng(C.SEED)
    B = C.NTN_BATCH
    import sys
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else C.NTN_ITERS
    for it in range(n_iters):
        order = rng.permutation(tr)
        tot = 0.0; nb = 0; nsat = 0
        for s in range(0, len(order), B):
            idx = order[s:s + B]; ii = torch.from_numpy(idx)
            o1, p, o2 = O1t[ii], Pt[ii], O2t[ii]
            b = len(idx)
            # fresh corruption (Alg.1): replace one argument with a random word
            choose = torch.from_numpy(rng.integers(0, 2, size=b))
            rvec = Wt[torch.from_numpy(rng.integers(0, V, size=b))]
            co1 = torch.where((choose == 0).unsqueeze(1), rvec, o1)
            co2 = torch.where((choose == 1).unsqueeze(1), rvec, o2)
            sp = model.score(o1, p, o2)
            sn = model.score(co1, p, co2)
            hinge = torch.clamp(C.NTN_MARGIN - sp + sn, min=0)
            loss = hinge.mean() + C.NTN_LAMBDA * model.l2()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            nsat += int((hinge <= 1e-6).sum())
        if (it + 1) % 5 == 0 or it < 5:
            print(f"iter {it+1}/{n_iters}  loss={tot/max(nb,1):.4f}  "
                  f"margin-satisfied={nsat/len(tr):.1%}", flush=True)

    # encode ALL in-vocab events (for downstream features, incl. test period)
    model.eval()
    M = len(ev); emb = np.zeros((M, C.NTN_K), np.float32)
    with torch.no_grad():
        allidx = np.where(ok)[0]
        for s in range(0, len(allidx), 4096):
            idx = allidx[s:s + 4096]; ii = torch.from_numpy(idx)
            emb[idx] = model.embed(O1t[ii], Pt[ii], O2t[ii]).numpy()
    # standardise on TRAIN in-vocab events (no leakage): zero-mean/unit-var per
    # dim - de-correlates the shared bias the tanh+L2 path induces and helps the
    # downstream linear classifier.
    tr_idx = np.where(ok & ev.emb_eligible.values)[0]
    mu = emb[tr_idx].mean(0); sd = emb[tr_idx].std(0) + 1e-6
    emb = (emb - mu) / sd
    emb[~ok] = 0.0
    np.save(os.path.join(C.ART, "event_emb.npy"), emb)
    np.save(os.path.join(C.ART, "event_ok.npy"), ok)
    torch.save(model.state_dict(), os.path.join(C.ART, "ntn.pt"))
    print("saved event_emb.npy", emb.shape, "| ok events:", int(ok.sum()))

    # qualitative check: nearest event neighbours by cosine
    nv = np.where(ok & (ev.ticker.values == C.TARGET))[0]
    if len(nv) > 50:
        E = emb[nv]; En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        q = 0
        sims = En @ En[q]
        top = sims.argsort()[::-1][1:4]
        def s(i): return f"({' '.join(ev.o1.values[i])}|{' '.join(ev.p.values[i])}|{' '.join(ev.o2.values[i])})"
        print("query:", s(nv[q]))
        for t in top:
            print("   ~", s(nv[t]), round(float(sims[t]), 3))

if __name__ == "__main__":
    main()
