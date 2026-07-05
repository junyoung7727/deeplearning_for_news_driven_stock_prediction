"""단어/이벤트 임베딩 + 사용자 학습 NTN 가중치 로드/추론."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import ART, bootstrap


@lru_cache(maxsize=1)
def load_word_vectors():
    z = np.load(ART / "word_vectors.npz", allow_pickle=True)
    vocab = list(z["vocab"])
    W = z["vectors"].astype(np.float32)
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i, W


def nearest_words(word: str, k: int = 8) -> list[tuple[str, float]]:
    vocab, w2i, W = load_word_vectors()
    if word not in w2i:
        raise KeyError(f"'{word}' not in vocab (size {len(vocab)})")
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sims = Wn @ Wn[w2i[word]]
    order = np.argsort(sims)[::-1]
    out = []
    for i in order:
        if vocab[i] == word:
            continue
        out.append((vocab[i], float(sims[i])))
        if len(out) >= k:
            break
    return out


def load_event_embeddings() -> np.ndarray:
    return np.load(ART / "event_emb.npy")


def load_ntn():
    """사용자가 학습시킨 artifacts/ntn.pt 가중치를 실은 실제 NTN 모듈."""
    bootstrap()
    import torch

    import config as C
    from s4_ntn import NTN

    model = NTN(C.WORD_DIM, C.NTN_K)
    state = torch.load(ART / "ntn.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def embed_events(ntn, events: pd.DataFrame, idx_list) -> np.ndarray:
    """주어진 이벤트 행들을 학습된 NTN으로 인코딩해 (n, k) 벡터 반환."""
    bootstrap()
    import torch

    from s4_ntn import build_arg_matrices

    _, w2i, W = load_word_vectors()
    sub = events.iloc[list(idx_list)]
    O1, P, O2, ok = build_arg_matrices(sub, w2i, W)
    with torch.no_grad():
        U = ntn.embed(torch.from_numpy(O1), torch.from_numpy(P), torch.from_numpy(O2))
    out = U.numpy()
    out[~ok] = 0.0
    return out


def nearest_events(events: pd.DataFrame, emb: np.ndarray, query_idx: int, k: int = 5,
                   dedupe: bool = True):
    """query 이벤트의 cosine 이웃. dedupe=True면 동일 triple 문자열은 한 번만."""
    from .data import triple_str

    En = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = En @ En[query_idx]
    order = np.argsort(sims)[::-1]
    seen = {triple_str(events, int(query_idx))} if dedupe else set()
    out = []
    for i in order:
        if int(i) == int(query_idx):
            continue
        trip = triple_str(events, int(i))
        if dedupe:
            if trip in seen:
                continue
            seen.add(trip)
        out.append((int(i), float(sims[i]), trip))
        if len(out) >= k:
            break
    return out


def pca_2d(X: np.ndarray) -> np.ndarray:
    """numpy SVD 기반 2D PCA (sklearn 불필요)."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:2].T).astype(np.float32)
