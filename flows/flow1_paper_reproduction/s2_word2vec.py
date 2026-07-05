"""
Stage 2 - Word embeddings (paper Sec 2.2): skip-gram, d = 100.

Faithful skip-gram with negative sampling (Mikolov et al. 2013), implemented in
PyTorch.  Trained ONLY on emb-eligible news (date < TEST_START), all corpus
tickers, so there is no look-ahead into the test period.

Output: artifacts/word_vectors.npz  { vocab(list[str]), vectors(V x 100) }
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
from collections import Counter
import config as C

torch.manual_seed(C.SEED); np.random.seed(C.SEED)
DEV = "cpu"

def load_sentences() -> list[list[str]]:
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    news = news[news.emb_eligible]
    return list(news.tokens.values)

def build_vocab(sents, min_count):
    cnt = Counter(w for s in sents for w in s)
    vocab = [w for w, c in cnt.most_common() if c >= min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    freq = np.array([cnt[w] for w in vocab], dtype=np.float64)
    return vocab, w2i, freq

def make_pairs(sents, w2i, freq, window, t=1e-3):
    # subsampling keep-prob per Mikolov
    total = freq.sum()
    z = freq / total
    keep = (np.sqrt(z / t) + 1) * (t / z)
    keep = np.clip(keep, 0, 1)
    rng = np.random.default_rng(C.SEED)
    centers, contexts = [], []
    for s in sents:
        ids = [w2i[w] for w in s if w in w2i]
        ids = [i for i in ids if rng.random() < keep[i]]
        n = len(ids)
        for pos in range(n):
            w = rng.integers(1, window + 1)          # dynamic window
            lo, hi = max(0, pos - w), min(n, pos + w + 1)
            for j in range(lo, hi):
                if j != pos:
                    centers.append(ids[pos]); contexts.append(ids[j])
    return np.asarray(centers, np.int64), np.asarray(contexts, np.int64)

class SGNS(torch.nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.inp = torch.nn.Embedding(V, d)
        self.out = torch.nn.Embedding(V, d)
        torch.nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        torch.nn.init.zeros_(self.out.weight)

    def forward(self, c, o, neg):
        vc = self.inp(c)                       # B x d
        vo = self.out(o)                       # B x d
        vn = self.out(neg)                     # B x K x d
        pos = torch.nn.functional.logsigmoid((vc * vo).sum(1))
        negs = torch.nn.functional.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)
        return -(pos + negs).mean()

def main():
    sents = load_sentences()
    vocab, w2i, freq = build_vocab(sents, C.W2V_MIN_COUNT)
    V = len(vocab)
    print(f"sentences={len(sents)}  vocab={V}")
    centers, contexts = make_pairs(sents, w2i, freq, C.W2V_WINDOW)
    print(f"training pairs={len(centers):,}")

    neg_p = freq ** 0.75; neg_p /= neg_p.sum()
    neg_table = np.random.default_rng(C.SEED).choice(
        V, size=10_000_000, p=neg_p).astype(np.int64)

    model = SGNS(V, C.WORD_DIM).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=C.W2V_LR)

    N = len(centers); B = C.W2V_BATCH
    rng = np.random.default_rng(C.SEED)
    for ep in range(C.W2V_EPOCHS):
        perm = rng.permutation(N)
        tot = 0.0; nb = 0
        for s in range(0, N, B):
            idx = perm[s:s + B]
            c = torch.from_numpy(centers[idx])
            o = torch.from_numpy(contexts[idx])
            ni = rng.integers(0, len(neg_table), size=(len(idx), C.W2V_NEG))
            neg = torch.from_numpy(neg_table[ni])
            loss = model(c, o, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{C.W2V_EPOCHS}  loss={tot/nb:.4f}")

    vectors = model.inp.weight.detach().cpu().numpy().astype(np.float32)
    np.savez(os.path.join(C.ART, "word_vectors.npz"),
             vocab=np.array(vocab, dtype=object), vectors=vectors)
    print("saved word_vectors.npz", vectors.shape)

    # sanity: nearest neighbours of a few finance words
    def nn(word, k=6):
        if word not in w2i: return f"{word}: OOV"
        v = vectors[w2i[word]]
        sims = vectors @ v / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(v) + 1e-9)
        top = sims.argsort()[::-1][1:k + 1]
        return f"{word}: " + ", ".join(vocab[i] for i in top)
    for w in ["nvidia", "chip", "earnings", "lawsuit", "ai"]:
        print("  NN", nn(w))

if __name__ == "__main__":
    main()
