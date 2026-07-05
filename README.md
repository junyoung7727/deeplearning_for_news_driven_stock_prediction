# Deep Learning for Event-Driven Stock Prediction — NVDA reproduction

Faithful reproduction of

> Xiao Ding, Yue Zhang, Ting Liu, Junwen Duan.
> **Deep Learning for Event-Driven Stock Prediction.** IJCAI 2015.

applied to a **single stock, NVDA**, using locally available **FMP news titles**
and **FMP daily OHLC** (parquet).

## What it reproduces

The full method, end to end:

1. **Event extraction** `E = (O1, P, O2, T)` from news *titles* — the paper's
   Open IE (ReVerb) + dependency parser (ZPar) is reproduced with **spaCy**
   dependency parsing (subject → O1, verb (+neg/particle) → P, object → O2).
2. **Skip-gram word embeddings**, `d = 100` (paper Sec 2.2), trained with
   negative sampling on the pre-test news corpus.
3. **Neural Tensor Network** event embeddings (paper Eq. 1, Algorithm 1):
   `R1 = f(O1ᵀ T1 P + W1[O1;P] + b1)`, `R2` likewise, `U = f(R1ᵀ T3 R2 + …)`,
   trained with the margin loss `max(0, 1 − s(E) + s(Eʳ)) + λ‖Φ‖²`, `λ = 1e‑4`.
4. **Deep CNN prediction model** (paper Sec 3, Fig. 3): long‑term (30 d),
   mid‑term (7 d) and short‑term (1 d) event sequences; narrow convolution
   (`l = 3`) + max‑over‑time pooling on long/mid; feed‑forward classifier.
5. **Evaluation**: Accuracy + Matthews Correlation Coefficient, and the
   **market simulation** (Lavrenko 2000 strategy: +2 % long take‑profit,
   −1 % short cover) with the confidence‑threshold variant (β = 0.7).

It also reproduces the paper's **model matrix** so the qualitative claims can be
checked on NVDA: `Luss[2012] (BoW+SVM)`, `E‑NN`, `WB‑NN`, `WB‑CNN`, `E‑CNN`,
`EB‑NN`, `EB‑CNN`.

## Data sources (read-only)

| Role | File |
|---|---|
| News titles (9 mega-caps) | `…/alphamale/data/news/us_fmp_news_rich.parquet` |
| Daily OHLC (adjusted) | `…/alphamale/data/price/us_daily_data.parquet` |

Word/event embeddings are trained on all 9 tickers' titles (NVDA, AAPL, MSFT,
JPM, V, BRK-B, CAT, RTX, GE) **before** the test period; prediction is NVDA-only.

## Pipeline

```
flows/flow1_paper_reproduction/s1_data.py        -> artifacts/prices.parquet, news.parquet
flows/flow1_paper_reproduction/s2_word2vec.py    -> artifacts/word_vectors.npz         (skip-gram, d=100)
flows/flow1_paper_reproduction/s3_events.py      -> artifacts/events.parquet           (spaCy SVO triples)
flows/flow1_paper_reproduction/s4_ntn.py [iters] -> artifacts/event_emb.npy, ntn.pt    (NTN event embeddings)
flows/flow2_us_upgrade/t1_embed.py               -> artifacts/tf_{title,event}_emb.npy (FinBERT embeddings)
flows/flow1_paper_reproduction/s5_features.py    -> artifacts/samples.parquet, feats_{WB,EB,TWB,TEB,E}.npz
flows/flow1_paper_reproduction/s7_train_eval.py  -> artifacts/results.json             (Acc/MCC + simulation)
flows/flow1_paper_reproduction/s9_report.py      -> artifacts/report.md
run_all.py                                       -> runs the whole chain
```

## Repository layout

```
.
├── flows/
│   ├── flow1_paper_reproduction/
│   ├── flow2_us_upgrade/
│   └── flow3_kr_highlow/
├── src/
│   └── dlfe_lab/
├── notebooks/
│   └── three_flow_lab.ipynb
├── tests/
│   └── test_lab.py
└── artifacts/
```

### Accuracy campaign, diagnostics & KR replication (beyond the paper)

`s10`–`s27`: US accuracy push + diagnosis (see `artifacts/FINAL_REPORT.md`) —
price/tech features, selective prediction, cross-asset, capacity sweep,
walk-forward, volatility/confidence targets, timezone probe (s22: FMP
`published_at` is tz-naive **UTC**), event-window intraday studies.
`s28`–`s34`: Korean-market replication (BigKinds news, KRX prices, foreign
flows; Kiwi SVO events → NTN → CNN on small-caps).
`s35`–`s37`: alignment-bug audit — `s35` event-day-only head retrain;
`s36` KR timestamp-correct overnight alignment (09:00/15:30 KST session
cutoffs, fresh-news mask, cached stages `artifacts/kr36_*`); `s37` US
ET-correct + overnight realignment reusing all embedding caches.
**Verdict** (`artifacts/FINAL_REPORT.md` §4c): the bugs were real, fixing them
does not move any model off the majority baseline — the next-day-direction
null is a property of the data, not a pipeline artifact.
`s39`–`s41`: **HIGH-prediction task** (user-redefined target:
`high ≥ open·(1+k)`, decided at open) — `s39` pykrx OHLCV fetch;
`s41` runs on the remote GPU box (`~/dlfe`, bk_slim titles, ts parsed from
`news_id`) with the precise linker (ASCII word boundaries, Kiwi token-boundary
validation, homonym rules for 대상/한화). Result (`FINAL_REPORT.md` §4d): acc
0.71–0.88, MCC +0.26…+0.45 OOS; top-decile selective hit-rates 0.70–0.87 vs
bases 0.16–0.37; naked TP-at-open trading remains negative-EV after costs —
execution research is the open item.

## Finance-transformer variant (alternative to skip-gram + NTN)

`t1_embed.py` encodes NVDA titles and event triples with a finance-specialised
transformer (`config.TF_MODEL`, default **FinBERT** `ProsusAI/finbert`),
mean-pooled and PCA-reduced to 100-d (train-fit).  This yields two extra
representations that feed the same NN/CNN heads:
  * **TWB** = transformer(title)        (analogue of WB)
  * **TEB** = transformer(event triple) (analogue of EB; no NTN training needed)
Any HF encoder id (e.g. a finance RoBERTa) is a one-line swap in `config.py`.
Models are ensembled over `N_SEEDS` seeds (unbiased vs best-dev-seed selection).

## Running

The interpreter is the alphamale venv (has torch/pandas/pyarrow/sklearn/spaCy):

```bat
D:\Github\homeserver\alphamale\.venv\Scripts\python.exe run_all.py
```

Practice notebook: `notebooks/three_flow_lab.ipynb` (run it with the alphamale venv Jupyter kernel).

(On Git Bash, invoke through `MSYS_NO_PATHCONV=1 cmd /c '…python.exe run_all.py'`.)

Splits and all hyperparameters live in `config.py`. Paper-specified values are
tagged `(paper)`; values the paper leaves unspecified are tagged `(paper-silent)`.

## Results

See `artifacts/report.md` (generated) for the NVDA Accuracy/MCC table, the market
simulation table, and the automatically-checked paper claims (CNN > NN,
event-embeddings > word-embeddings > discrete-events, EB-CNN best).
