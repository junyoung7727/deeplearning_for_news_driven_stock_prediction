"""
Stage 3 - Structured event extraction (paper Sec 2.1).

The paper extracts E = (O1, P, O2, T) from each news *title* using Open IE
(ReVerb) + dependency parsing (ZPar):  O1 = actor/subject, P = action/predicate,
O2 = object, T = timestamp.

Modern faithful equivalent: spaCy dependency parsing.  For every verb we read
its grammatical subject (nsubj/nsubjpass) as O1, the verb (lemma, with negation
and particle) as P, and the direct / prepositional object as O2.  Multi-word
arguments are kept as token lists - the paper represents each argument as the
*average of its word embeddings*.

Output: artifacts/events.parquet
   columns: ticker, date, o1(list[str]), p(list[str]), o2(list[str]), emb_eligible
   one row per extracted event (a title can yield 0..n events).
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
import os, pandas as pd, spacy
import config as C

_DROP_POS = {"DET", "PUNCT", "PART", "SPACE", "ADP"}

def _phrase(tok, chunk_of):
    ch = chunk_of.get(tok.i)
    toks = list(ch) if ch is not None else [tok]
    return [t.text.lower() for t in toks
            if t.is_alpha and len(t) >= 2 and t.pos_ not in _DROP_POS]

def extract(doc):
    chunk_of = {}
    for ch in doc.noun_chunks:
        for t in ch:
            chunk_of[t.i] = ch
    events = []
    for v in doc:
        if v.pos_ not in ("VERB", "AUX"):
            continue
        subs = [c for c in v.children if c.dep_ in ("nsubj", "nsubjpass")]
        objs = [c for c in v.children if c.dep_ in ("dobj", "attr", "dative", "oprd")]
        for c in v.children:
            if c.dep_ == "prep":
                objs += [g for g in c.children if g.dep_ == "pobj"]
        if not subs or not objs:
            continue
        action = []
        if any(c.dep_ == "neg" for c in v.children):
            action.append("not")
        action.append((v.lemma_ or v.text).lower())
        action += [c.text.lower() for c in v.children if c.dep_ == "prt"]
        action = [a for a in action if a.replace("'", "").isalpha()]
        o1, o2 = _phrase(subs[0], chunk_of), _phrase(objs[0], chunk_of)
        if o1 and action and o2:
            events.append((o1, action, o2))
    return events

def main():
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    titles = news.title_clean.tolist()
    meta = news[["ticker", "date", "emb_eligible"]].to_dict("records")

    rows = []
    n_titles_with_event = 0
    for i, doc in enumerate(nlp.pipe(titles, batch_size=256)):
        evs = extract(doc)
        if evs:
            n_titles_with_event += 1
        m = meta[i]
        for o1, p, o2 in evs:
            rows.append((m["ticker"], m["date"], o1, p, o2, m["emb_eligible"]))
        if (i + 1) % 10000 == 0:
            print(f"  parsed {i+1}/{len(titles)} titles, events so far={len(rows)}")

    ev = pd.DataFrame(rows, columns=["ticker", "date", "o1", "p", "o2", "emb_eligible"])
    ev.to_parquet(os.path.join(C.ART, "events.parquet"))

    print("EVENTS", ev.shape)
    print(f"titles with >=1 event: {n_titles_with_event}/{len(titles)} "
          f"({n_titles_with_event/len(titles):.1%})")
    print("events per ticker:\n", ev.ticker.value_counts())
    print("emb-eligible events:", int(ev.emb_eligible.sum()))
    nv = ev[ev.ticker == C.TARGET]
    print("\nNVDA example events:")
    for _, r in nv.head(6).iterrows():
        print(f"  ({' '.join(r.o1)} | {' '.join(r.p)} | {' '.join(r.o2)})")

if __name__ == "__main__":
    main()
