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
| News titles (9 mega-caps) | `data/news/us_fmp_news_rich.parquet` (또는 `DLFE_NEWS_PARQUET`) |
| Daily OHLC (adjusted) | `data/price/us_daily_data.parquet` (또는 `DLFE_DAILY_PARQUET`) |

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

## Setup (설치)

```bash
git clone https://github.com/junyoung7727/deeplearning_for_news_driven_stock_prediction.git
cd deeplearning_for_news_driven_stock_prediction

# 1) 가상환경 (Python 3.11+ 권장, 3.14에서 검증됨)
python -m venv .venv
#    Windows:      .venv\Scripts\activate
#    macOS/Linux:  source .venv/bin/activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) Jupyter 커널 등록 후 실습 노트북 열기
python -m ipykernel install --user --name dlfe --display-name "Python (dlfe)"
jupyter lab notebooks/three_flow_lab.ipynb   # 커널을 "Python (dlfe)"로 선택
```

실습 노트북은 저장소에 포함된 `artifacts/`(데이터·학습된 가중치·결과)만 읽으므로
**추가 데이터나 환경변수 없이 바로 전체 셀이 실행됩니다.**
테스트: `python -m unittest tests.test_lab`

### Environment variables (환경변수) — 원본 파이프라인 재실행 시에만

`run_all.py`(s1→…→s9 전체 재학습)와 KR 스크립트는 비공개 원천 데이터가 필요합니다.
**모든 경로는 저장소 기준 상대경로가 기본값**입니다: 데이터를 `<repo>/data/` 아래에
두면 환경변수 없이 그대로 동작하고, 다른 곳에 있다면 아래 변수로 지정하세요:

| 변수 | 기본값 (repo 기준) | 용도 |
|---|---|---|
| `DLFE_DATA_ROOT` | `data/` | 모든 원천 데이터의 루트 (아래 변수들의 부모) |
| `DLFE_NEWS_PARQUET` | `data/news/us_fmp_news_rich.parquet` | FMP 뉴스 제목 parquet |
| `DLFE_DAILY_PARQUET` | `data/price/us_daily_data.parquet` | 미국 일봉 parquet |
| `DLFE_MIN5_NVDA` | `data/prices/fmp_5min_us/NVDA.parquet` | NVDA 5분봉 parquet |
| `DLFE_MIN5_KR_DIR` | `data/prices/fmp_5min/` | KR 5분봉 폴더 (s50/s51) |
| `DLFE_BK_SCORES` | `~/bk_scores/bigkinds_finbert_scores.parquet` | BigKinds FinBERT 점수 (원격 s45/s48) |

설정 방법:

```powershell
# Windows PowerShell — 현재 세션만
$env:DLFE_NEWS_PARQUET  = "C:\data\us_fmp_news_rich.parquet"
$env:DLFE_DAILY_PARQUET = "C:\data\us_daily_data.parquet"

# Windows — 영구 등록 (새 터미널부터 적용)
setx DLFE_NEWS_PARQUET "C:\data\us_fmp_news_rich.parquet"
```

```bash
# macOS / Linux — 현재 세션만 (영구 등록은 ~/.bashrc 나 ~/.zshrc 에 추가)
export DLFE_NEWS_PARQUET=/data/us_fmp_news_rich.parquet
export DLFE_DAILY_PARQUET=/data/us_daily_data.parquet
```

## Running (원본 파이프라인)

```bash
python run_all.py    # s1 → s2 → s3 → s4 → t1 → s5 → s7 → s9 (수 시간, GPU 없이 CPU 가능)
```

GitHub 100 MB 제한으로 제외된 대용량 캐시(`kr36_features*.npz`, `kr_ie_features.npz`,
`kr_titles_2024p.parquet`)는 `.gitignore`에 적힌 생성 스크립트로 재생성할 수 있습니다.

Splits and all hyperparameters live in `config.py`. Paper-specified values are
tagged `(paper)`; values the paper leaves unspecified are tagged `(paper-silent)`.

## Results

See `artifacts/report.md` (generated) for the NVDA Accuracy/MCC table, the market
simulation table, and the automatically-checked paper claims (CNN > NN,
event-embeddings > word-embeddings > discrete-events, EB-CNN best).
