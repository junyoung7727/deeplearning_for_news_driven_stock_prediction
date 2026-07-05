"""
Stage T2 - FinBERT sentiment for NVDA titles (news feature engineering input).

Saves per-title net sentiment  p(positive) - p(negative)  aligned by position to
news[ticker==NVDA] (same order as t1_embed / s5).
Output: artifacts/tf_title_sent.npy  (n_nvda_titles,) float32 in [-1, 1]
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
import os, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config as C

@torch.no_grad()
def main():
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nv = news[news.ticker == C.TARGET].reset_index(drop=True)
    texts = nv.title_clean.astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(C.TF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(C.TF_MODEL).eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    pos_i, neg_i = lab.get("positive", 0), lab.get("negative", 1)
    print("id2label:", model.config.id2label, "| pos_i", pos_i, "neg_i", neg_i)

    net = np.zeros(len(texts), np.float32)
    B = C.TF_BATCH; t0 = time.time()
    for s in range(0, len(texts), B):
        enc = tok(texts[s:s + B], padding=True, truncation=True,
                  max_length=C.TF_MAXLEN, return_tensors="pt")
        prob = torch.softmax(model(**enc).logits, dim=1).numpy()
        net[s:s + B] = prob[:, pos_i] - prob[:, neg_i]
        if (s // B) % 50 == 0:
            print(f"  {s+B}/{len(texts)} ({(s+B)/max(time.time()-t0,1e-9):.0f}/s)", flush=True)
    np.save(os.path.join(C.ART, "tf_title_sent.npy"), net)
    print("saved tf_title_sent.npy", net.shape, "mean", float(net.mean()))

if __name__ == "__main__":
    main()
