"""
Stage T1 - Finance-specialized transformer embeddings.

Replaces the paper's skip-gram word vectors + NTN event embeddings with a
pretrained finance-domain transformer (default: ProsusAI/finbert).  The model is
already trained on a large financial corpus, so no NTN training is needed.

Encodes NVDA news:
  * each title            -> mean-pooled hidden state (D=768)   -> TWB
  * each event triple     "O1 P O2"                             -> TEB

Outputs (aligned by position to the NVDA-filtered rows of news.parquet /
events.parquet, in file order):
  artifacts/tf_title_emb.npy   (n_nvda_titles x D)
  artifacts/tf_event_emb.npy   (n_nvda_events x D)
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
from transformers import AutoTokenizer, AutoModel
import config as C

torch.manual_seed(C.SEED)

def load_model():
    tok = AutoTokenizer.from_pretrained(C.TF_MODEL)
    model = AutoModel.from_pretrained(C.TF_MODEL)
    model.eval()
    return tok, model

@torch.no_grad()
def encode(texts, tok, model, tag=""):
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    B = C.TF_BATCH; t0 = time.time()
    for s in range(0, len(texts), B):
        chunk = texts[s:s + B]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=C.TF_MAXLEN, return_tensors="pt")
        hs = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[s:s + B] = ((hs * m).sum(1) / m.sum(1).clamp(min=1)).numpy()
        if (s // B) % 50 == 0:
            done = s + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {tag} {done}/{len(texts)}  ({rate:.0f}/s)", flush=True)
    return out

def main():
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    ev = pd.read_parquet(os.path.join(C.ART, "events.parquet"))
    nv_news = news[news.ticker == C.TARGET].reset_index(drop=True)
    nv_ev = ev[ev.ticker == C.TARGET].reset_index(drop=True)

    title_texts = nv_news.title_clean.astype(str).tolist()
    event_texts = [f"{' '.join(o1)} {' '.join(p)} {' '.join(o2)}".strip()
                   for o1, p, o2 in zip(nv_ev.o1, nv_ev.p, nv_ev.o2)]

    print(f"model={C.TF_MODEL}  titles={len(title_texts)}  events={len(event_texts)}")
    tok, model = load_model()
    E_title = encode(title_texts, tok, model, "title")
    E_event = encode(event_texts, tok, model, "event")

    np.save(os.path.join(C.ART, "tf_title_emb.npy"), E_title)
    np.save(os.path.join(C.ART, "tf_event_emb.npy"), E_event)
    print("saved tf_title_emb", E_title.shape, "| tf_event_emb", E_event.shape)

if __name__ == "__main__":
    main()
