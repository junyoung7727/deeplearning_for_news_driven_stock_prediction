"""
Stage 6 - Prediction models (paper Sec 3).

Head: feedforward classifier over the (short, mid, long) daily-unit tensors.
  - nn_only=True  -> uses only the short-term unit  (paper's NN models: WB-NN,
    E-NN, EB-NN).
  - nn_only=False -> narrow convolution (kernel l=3) + max-over-time pooling on
    the long- and mid-term sequences, concatenated with the short-term unit, then
    the feedforward layers  (paper's CNN models: WB-CNN, E-CNN, EB-CNN; Eq.3-4).

Feedforward (paper Eq. after 4):  Y = sigma(W2 . V^C),  y = sigma(W3 . Y).

DenseModel  : WB / EB inputs are pre-computed dense daily vectors.
EventModel  : E (discrete events) - trainable per-event-id embedding table
              (random init, no NTN pre-training); daily unit = masked mean of the
              day's event-id embeddings.  Unseen test ids -> id 0 (UNK), which
              reproduces the sparsity disadvantage of discrete events.
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
import copy, numpy as np, torch, torch.nn as nn
from sklearn.metrics import matthews_corrcoef, accuracy_score
import config as C


class Head(nn.Module):
    def __init__(self, d, nn_only, n_filters=C.N_FILTERS, hidden=C.HIDDEN,
                 dropout=C.CLF_DROPOUT):
        super().__init__()
        self.nn_only = nn_only
        if not nn_only:
            self.cl = nn.Conv1d(d, n_filters, C.CONV_L)
            self.cm = nn.Conv1d(d, n_filters, C.CONV_L)
            feat = n_filters * 2 + d
        else:
            feat = d
        self.fc1 = nn.Linear(feat, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, short, mid, long):
        if self.nn_only:
            vc = short
        else:
            vl = torch.tanh(self.cl(long.transpose(1, 2))).max(dim=2).values
            vm = torch.tanh(self.cm(mid.transpose(1, 2))).max(dim=2).values
            vc = torch.cat([vl, vm, short], dim=1)
        y = torch.sigmoid(self.fc1(vc))
        y = self.drop(y)
        return torch.sigmoid(self.fc2(y)).squeeze(1)


class DenseModel(nn.Module):
    def __init__(self, d, nn_only):
        super().__init__()
        self.head = Head(d, nn_only)

    def forward(self, short, mid, long):
        return self.head(short, mid, long)


class EventModel(nn.Module):
    def __init__(self, n_ids, d, nn_only):
        super().__init__()
        self.emb = nn.Embedding(n_ids, d, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, 0.1)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.nn_only = nn_only
        self.head = Head(d, nn_only)

    def _daily(self, ids):                       # (B, E) -> (B, d) masked mean
        e = self.emb(ids)
        m = (ids > 0).float().unsqueeze(-1)
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

    def _daily_seq(self, ids):                   # (B, T, E) -> (B, T, d)
        B, T, E = ids.shape
        return self._daily(ids.reshape(B * T, E)).reshape(B, T, -1)

    def forward(self, short_ids, mid_ids, long_ids):
        short = self._daily(short_ids)
        if self.nn_only:
            return self.head(short, None, None)
        return self.head(short, self._daily_seq(mid_ids), self._daily_seq(long_ids))


# ---------------------------------------------------------------- train / eval
def predict(model, inputs, idx, batch=256):
    model.eval()
    out = np.zeros(len(idx), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sub = idx[s:s + batch]
            p = model(*inputs(torch.from_numpy(sub)))
            out[s:s + batch] = p.numpy()
    return out


def fit(model, inputs, y01, train_idx, dev_idx,
        epochs=C.CLF_EPOCHS, lr=C.CLF_LR, batch=C.CLF_BATCH,
        patience=C.CLF_PATIENCE, seed=C.SEED, verbose=False):
    """y01 in {0,1}; selects the epoch with best dev MCC (the paper's headline
    metric; robust to the class imbalance that makes accuracy collapse to the
    majority predictor)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCELoss()
    yt = torch.from_numpy(y01.astype(np.float32))
    best, best_state, bad = -2.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(train_idx)
        for s in range(0, len(perm), batch):
            sub = perm[s:s + batch]
            out = model(*inputs(torch.from_numpy(sub)))
            loss = lossf(out, yt[torch.from_numpy(sub)])
            opt.zero_grad(); loss.backward(); opt.step()
        dev_p = predict(model, inputs, dev_idx)
        pred = (dev_p > 0.5).astype(int)
        score = matthews_corrcoef(y01[dev_idx], pred) if len(np.unique(pred)) > 1 else 0.0
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or bad == 0):
            print(f"    ep{ep:3d} dev_mcc={score:.4f} best={best:.4f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best


def metrics(y_true_pm1, prob):
    """y_true_pm1 in {-1,+1}; prob = P(up)."""
    pred = np.where(prob > 0.5, 1, -1)
    acc = (pred == y_true_pm1).mean()
    mcc = matthews_corrcoef(y_true_pm1, pred) if len(set(pred)) > 1 else 0.0
    return float(acc), float(mcc)
