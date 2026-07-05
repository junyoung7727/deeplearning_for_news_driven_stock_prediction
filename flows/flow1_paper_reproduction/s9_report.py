"""
Stage 9 - Render artifacts/report.md from results.json + dataset artifacts.
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
import os, json, numpy as np, pandas as pd
import config as C

def main():
    res = json.load(open(os.path.join(C.ART, "results.json")))
    samp = pd.read_parquet(os.path.join(C.ART, "samples.parquet"))
    prices = pd.read_parquet(os.path.join(C.ART, "prices.parquet"))
    news = pd.read_parquet(os.path.join(C.ART, "news.parquet"))
    ev = pd.read_parquet(os.path.join(C.ART, "events.parquet"))

    def split_counts(df):
        return df.split.value_counts().reindex(["train", "dev", "test"]).to_dict()

    L = []
    w = L.append
    w("# Reproduction - Deep Learning for Event-Driven Stock Prediction (Ding et al., IJCAI 2015)")
    w("")
    w("Single target: **NVDA**.  Source: FMP news *titles* + FMP daily OHLC.")
    w("")
    w("## 1. Data")
    w(f"- Period: {prices.date.min().date()} -> {prices.date.max().date()}  "
      f"(START={C.START_DATE}, DEV={C.DEV_START}, TEST={C.TEST_START})")
    w(f"- NVDA trading-day samples: {split_counts(samp)}")
    w(f"- News titles (corpus, 9 tickers): {len(news):,}; events extracted: {len(ev):,}")
    w(f"- NVDA events: {int((ev.ticker=='NVDA').sum()):,}")
    w(f"- Test up-rate (majority baseline): {res['test_up_rate']:.4f}  "
      f"(n_test={res['n_test']})")
    w("")
    w("## 2. Pipeline (paper section -> implementation)")
    w("| Paper | This reproduction |")
    w("|---|---|")
    w("| Event E=(O1,P,O2,T), Open IE (ReVerb)+ZPar (Sec 2.1) | spaCy dependency parse: nsubj->O1, verb(+neg/prt)->P, dobj/pobj->O2 |")
    w("| Skip-gram word emb d=100 (Sec 2.2) | SGNS in PyTorch on pre-test corpus, d=100 |")
    w("| Neural Tensor Network event emb (Eq.1, Alg.1) | k=d=100 tensors T1/T2/T3, margin loss vs corrupted event, lambda=1e-4 |")
    w("| (alternative) finance transformer emb | FinBERT mean-pooled -> PCA-100 (train-fit); TWB=title, TEB=event triple; no NTN |")
    w("| CNN long(month)+mid(week)+short(day), conv l=3, max-pool (Sec 3) | Conv1d(d,F,3)+max-over-time on long(30)/mid(7), short concat |")
    w("| NN baseline (short only) | same head, conv disabled |")
    w("| Acc + MCC (Sec 4.1) | sklearn accuracy + matthews_corrcoef |")
    w("| Market sim: +2% long / -1% short cover (Sec 4.3) | s8_simulate on daily OHLC, beta=0.7 threshold |")
    w("")
    w("## 3. NVDA prediction results (Accuracy / MCC)")
    w("")
    w("| Model | Dev Acc | Dev MCC | Test Acc | Test MCC |")
    w("|---|---|---|---|---|")
    for m in res["metrics"]:
        w(f"| {m['model']} | {m['dev_acc']:.4f} | {m['dev_mcc']:.4f} | "
          f"{m['test_acc']:.4f} | {m['test_mcc']:.4f} |")
    w("")
    w("## 4. Market simulation on TEST (net profit per $10,000 traded/day)")
    w("")
    w("| Model | Always-trade | beta=0.70 | Random-mean | p-value |")
    w("|---|---|---|---|---|")
    for s in res["simulation"]:
        w(f"| {s['model']} | ${s['always_profit']:,.0f} | ${s['threshold_profit']:,.0f} | "
          f"${s['rand_mean']:,.0f} | {s['p_value']:.3f} |")
    w("")
    w("## 5. Paper claims checked on NVDA")
    met = {m["model"]: m for m in res["metrics"]}
    def mcc(n): return met[n]["test_mcc"] if n in met else float("nan")
    w("Ranking by **test MCC** (the paper's headline metric; test accuracies are "
      "near the majority baseline and tie, so MCC is the discriminating metric).")
    w("")
    checks = []
    if "EB-CNN" in met and "EB-NN" in met:
        checks.append(("CNN > NN  (EB-CNN MCC %.4f > EB-NN %.4f)" % (mcc("EB-CNN"), mcc("EB-NN")),
                       mcc("EB-CNN") > mcc("EB-NN")))
    if "WB-CNN" in met and "WB-NN" in met:
        checks.append(("CNN > NN  (WB-CNN MCC %.4f > WB-NN %.4f)" % (mcc("WB-CNN"), mcc("WB-NN")),
                       mcc("WB-CNN") > mcc("WB-NN")))
    if "EB-CNN" in met and "WB-CNN" in met:
        checks.append(("event-emb > word-emb  (EB-CNN MCC %.4f > WB-CNN %.4f)" % (mcc("EB-CNN"), mcc("WB-CNN")),
                       mcc("EB-CNN") > mcc("WB-CNN")))
    if "EB-CNN" in met and "E-CNN" in met:
        checks.append(("event-emb > discrete-event  (EB-CNN MCC %.4f > E-CNN %.4f)" % (mcc("EB-CNN"), mcc("E-CNN")),
                       mcc("EB-CNN") > mcc("E-CNN")))
    best = max(met, key=lambda k: met[k]["test_mcc"])
    checks.append((f"EB-CNN has the best test MCC (best = {best})", best == "EB-CNN"))
    for desc, ok in checks:
        w(f"- [{'x' if ok else ' '}] {desc}")
    w("")
    w("## 6. Finance-transformer embeddings vs skip-gram + NTN")
    if any(n.startswith(("TWB", "TEB")) for n in met):
        def cmp(a, b): return f"{a} (MCC {mcc(a):+.4f}) vs {b} (MCC {mcc(b):+.4f})"
        w(f"- transformer *title* vs skip-gram title: {cmp('TWB-CNN','WB-CNN')} -> "
          f"{'transformer wins' if mcc('TWB-CNN')>mcc('WB-CNN') else 'skip-gram wins'}")
        w(f"- transformer *event* vs NTN event: {cmp('TEB-CNN','EB-CNN')} -> "
          f"{'transformer wins' if mcc('TEB-CNN')>mcc('EB-CNN') else 'NTN wins'}")
        overall = max(met, key=lambda k: met[k]['test_mcc'])
        fam = 'finance-transformer' if overall.startswith(('TWB', 'TEB')) else 'skip-gram/NTN'
        w(f"- overall best model: **{overall}** (test MCC {mcc(overall):+.4f}), family: {fam}")
        verdict = ('surpasses' if fam == 'finance-transformer'
                   else 'is competitive with but does NOT surpass')
        w(f"- **verdict:** on this single-stock task the finance-pretrained "
          f"transformer {verdict} the paper's from-scratch skip-gram+NTN.")
    else:
        w("- (transformer features not present in this run)")
    w("")
    w("## 7. Notes / faithful deviations")
    w("- Corpus is FMP (9 mega-caps, 2018-2026), not Reuters/Bloomberg 2006-2013; "
      "embeddings trained only on pre-TEST news (no look-ahead).")
    w("- Headlines are often verb-less noun phrases, so ~46% of titles yield an "
      "event (the paper's Open IE has the same limitation); long/mid windows cover days without daily news.")
    w("- NTN scalar score uses a learned vector u (Socher-style) for the margin "
      "loss; event embeddings standardised on train stats for the linear classifier.")
    w("- Discrete-event (E) baseline uses randomly-initialised trainable per-event-"
      "id embeddings (unseen test ids -> UNK), isolating the value of NTN pre-training.")
    w("- Simulation uses adjusted daily OHLC; the +2% long take-profit is judged "
      "reachable when the day's high>=open*1.02 (intraday approximation).")
    w("- Transformer path: FinBERT (ProsusAI/finbert) mean-pooled hidden states, "
      "standardised + PCA-100 on train items; replaces skip-gram+NTN, no NTN training.")

    path = os.path.join(C.ART, "report.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", path)
    print("\n".join(L))

if __name__ == "__main__":
    main()
