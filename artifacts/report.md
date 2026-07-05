# Reproduction - Deep Learning for Event-Driven Stock Prediction (Ding et al., IJCAI 2015)

Single target: **NVDA**.  Source: FMP news *titles* + FMP daily OHLC.

## 1. Data
- Period: 2018-01-03 -> 2026-06-18  (START=2018-01-01, DEV=2024-01-01, TEST=2025-01-01)
- NVDA trading-day samples: {'train': 1458, 'dev': 251, 'test': 366}
- News titles (corpus, 9 tickers): 87,254; events extracted: 44,130
- NVDA events: 10,797
- Test up-rate (majority baseline): 0.5355  (n_test=366)

## 2. Pipeline (paper section -> implementation)
| Paper | This reproduction |
|---|---|
| Event E=(O1,P,O2,T), Open IE (ReVerb)+ZPar (Sec 2.1) | spaCy dependency parse: nsubj->O1, verb(+neg/prt)->P, dobj/pobj->O2 |
| Skip-gram word emb d=100 (Sec 2.2) | SGNS in PyTorch on pre-test corpus, d=100 |
| Neural Tensor Network event emb (Eq.1, Alg.1) | k=d=100 tensors T1/T2/T3, margin loss vs corrupted event, lambda=1e-4 |
| (alternative) finance transformer emb | FinBERT mean-pooled -> PCA-100 (train-fit); TWB=title, TEB=event triple; no NTN |
| CNN long(month)+mid(week)+short(day), conv l=3, max-pool (Sec 3) | Conv1d(d,F,3)+max-over-time on long(30)/mid(7), short concat |
| NN baseline (short only) | same head, conv disabled |
| Acc + MCC (Sec 4.1) | sklearn accuracy + matthews_corrcoef |
| Market sim: +2% long / -1% short cover (Sec 4.3) | s8_simulate on daily OHLC, beta=0.7 threshold |

## 3. NVDA prediction results (Accuracy / MCC)

| Model | Dev Acc | Dev MCC | Test Acc | Test MCC |
|---|---|---|---|---|
| Luss[2012] (BoW+SVM) | 0.5219 | 0.0106 | 0.5219 | -0.0049 |
| WB-NN | 0.5538 | 0.0000 | 0.5355 | 0.0000 |
| WB-CNN | 0.5657 | 0.1575 | 0.5137 | -0.0028 |
| E-NN | 0.5777 | 0.1169 | 0.5246 | -0.0150 |
| E-CNN | 0.5737 | 0.1429 | 0.5027 | -0.0270 |
| EB-NN | 0.5498 | 0.0132 | 0.5437 | 0.0607 |
| EB-CNN | 0.5618 | 0.0778 | 0.5410 | 0.0796 |
| TWB-NN | 0.5578 | 0.0705 | 0.5355 | 0.0053 |
| TWB-CNN | 0.5857 | 0.1447 | 0.5383 | 0.0495 |
| TEB-NN | 0.4701 | 0.1172 | 0.4727 | 0.0624 |
| TEB-CNN | 0.5618 | 0.1117 | 0.5546 | 0.0842 |

## 4. Market simulation on TEST (net profit per $10,000 traded/day)

| Model | Always-trade | beta=0.70 | Random-mean | p-value |
|---|---|---|---|---|
| Luss[2012] (BoW+SVM) | $-1,860 | $-80 | $-892 | 0.640 |
| WB-NN | $-926 | $0 | $-892 | 0.506 |
| WB-CNN | $69 | $200 | $-892 | 0.380 |
| E-NN | $-1,836 | $0 | $-892 | 0.637 |
| E-CNN | $-2,351 | $1,703 | $-892 | 0.695 |
| EB-NN | $41 | $0 | $-892 | 0.384 |
| EB-CNN | $-586 | $0 | $-892 | 0.464 |
| TWB-NN | $-1,191 | $0 | $-892 | 0.546 |
| TWB-CNN | $-1,609 | $-1,749 | $-892 | 0.604 |
| TEB-NN | $-818 | $0 | $-892 | 0.491 |
| TEB-CNN | $1,260 | $968 | $-892 | 0.234 |

## 5. Paper claims checked on NVDA
Ranking by **test MCC** (the paper's headline metric; test accuracies are near the majority baseline and tie, so MCC is the discriminating metric).

- [x] CNN > NN  (EB-CNN MCC 0.0796 > EB-NN 0.0607)
- [ ] CNN > NN  (WB-CNN MCC -0.0028 > WB-NN 0.0000)
- [x] event-emb > word-emb  (EB-CNN MCC 0.0796 > WB-CNN -0.0028)
- [x] event-emb > discrete-event  (EB-CNN MCC 0.0796 > E-CNN -0.0270)
- [ ] EB-CNN has the best test MCC (best = TEB-CNN)

## 6. Finance-transformer embeddings vs skip-gram + NTN
- transformer *title* vs skip-gram title: TWB-CNN (MCC +0.0495) vs WB-CNN (MCC -0.0028) -> transformer wins
- transformer *event* vs NTN event: TEB-CNN (MCC +0.0842) vs EB-CNN (MCC +0.0796) -> transformer wins
- overall best model: **TEB-CNN** (test MCC +0.0842), family: finance-transformer
- **verdict:** on this single-stock task the finance-pretrained transformer surpasses the paper's from-scratch skip-gram+NTN.

## 7. Notes / faithful deviations
- Corpus is FMP (9 mega-caps, 2018-2026), not Reuters/Bloomberg 2006-2013; embeddings trained only on pre-TEST news (no look-ahead).
- Headlines are often verb-less noun phrases, so ~46% of titles yield an event (the paper's Open IE has the same limitation); long/mid windows cover days without daily news.
- NTN scalar score uses a learned vector u (Socher-style) for the margin loss; event embeddings standardised on train stats for the linear classifier.
- Discrete-event (E) baseline uses randomly-initialised trainable per-event-id embeddings (unseen test ids -> UNK), isolating the value of NTN pre-training.
- Simulation uses adjusted daily OHLC; the +2% long take-profit is judged reachable when the day's high>=open*1.02 (intraday approximation).
- Transformer path: FinBERT (ProsusAI/finbert) mean-pooled hidden states, standardised + PCA-100 on train items; replaces skip-gram+NTN, no NTN training.