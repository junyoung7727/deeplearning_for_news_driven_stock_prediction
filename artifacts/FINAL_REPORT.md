# NVDA accuracy investigation — final report

**Objective:** raise NVDA next-day direction prediction accuracy to ≥70%.
**Verdict:** NOT achievable honestly with the available data (news titles + OHLC).
Robust, leak-free, walk-forward ceiling ≈ **0.54–0.56** (majority baseline ≈0.54).

> Note on a misleading artifact: a naive scan of `results_*.json` finds a `0.846`
> value (`results_confidence.json:recent_acc`). That is the selective point at
> **10% coverage on the 2024–26 sub-window = n≈13 days** (95% CI ≈ ±0.27) and it
> is **not robust** — an equally-valid config (s19, recency-weighted+calibrated)
> gives ~0.52 at the same operating point. It is NOT evidence of ≥70%.

## 1. What was built (deliverables, all runnable)
- **Paper reproduction** (Ding et al., IJCAI 2015): `s1`–`s9` — skip-gram word
  vectors, spaCy SVO events, Neural Tensor Network event embeddings, long/mid/
  short CNN, Acc/MCC, market simulation. (`report.md`)
- **Finance-transformer variant**: `t1_embed`,`t2_sentiment` + `s5`–`s9` — FinBERT
  title/event embeddings → TWB/TEB models.
- **Accuracy campaign**: `s10` price/tech+GBM, `s11` selective, `s12` cross-asset,
  `s13` capacity sweep, `s14` enriched features, `s15` walk-forward+meta-labeling,
  `s16` divergent model sweep.
- **Volatility / confidence**: `s17`,`s18`,`s19`.
- **Diagnosis**: `s20`–`s25` (leading/lagging, timezone, extended-hours pricing).
- **Intraday post-news research**: `s26`,`s27` (5-min event windows).

## 2. Honest accuracy ceilings (walk-forward OOS 2021–2026)
| target | baseline | best honest OOS acc | real edge |
|---|---|---|---|
| price direction (next-day) | 0.537 | 0.51–0.54 | ~0 |
| volatility direction (balanced) | 0.500 | 0.544 | **+0.04** |
| overnight gap direction | 0.552 | 0.529 | <0 |
| post-news intraday 5–120 min | ~0.50 | 0.50–0.56 (vol-spike news max 0.556 @60m) | ~0 |
| model capacity 21k→12.4M params | — | test flat 0.52–0.55 (train→1.00) | overfits |

Every target/horizon sits at its baseline ±0.04. Only volatility shows a small
genuine edge (+0.04), still far from 0.70.

## 3. Why (concrete mechanism, measured — not "EMH")
- **News is coincident/reactive**: corr(sentiment, SAME-day ret)=+0.215;
  NEXT-day=−0.003.
- **News is priced into the OPEN via extended-hours trading**: overnight news →
  opening GAP corr +0.061 (direction), +0.122 (magnitude); → regular-session
  (open→close) ≈ 0. Heavy-news nights have ~30% bigger gaps.
- **FMP timestamps are rounded/batched** (37.9% at :00 sec; clustered at
  :00/:15/:30/:45) → ingestion/publisher times, not precise release → by the
  timestamp the move is largely done.
- **Post-news intraday direction ~50–56%** at all horizons, even for the 617
  genuine volume-spike news items.
- **Model size is not the bottleneck** — bigger models overfit (train→1.00,
  test flat). The bottleneck is *signal*, and the signal is gone by the time we
  can act.

## 4. What a genuine ≥70% would require (not in this repo)
1. **Precise, low-latency news** (sub-second release timestamps, no ingestion lag).
2. **Tick / second-level trade data** (5-min is too coarse for the immediate move).
3. **Order-flow / limit-order-book microstructure**, options IV/skew, or
   alt-data (web traffic, supply chain) that *leads* price.
4. A **latency edge** to act before the initial jump.

## 4b. Korean-market replication (BigKinds news + KRX prices + foreign flows)
Ran the same experiment on Korea using much richer data (`s28`–`s32`):
- **News**: BigKinds 22M articles (2006–2026, precise second-level timestamps,
  pre-computed FinBERT sentiment) — superior to the US FMP feed.
- **Prices**: 626 KRX stocks (2024–2026, incl. small/mid caps).
- **Foreign/institution FLOWS**: `kr_foreign_flow_daily` — the documented Korean
  "smart-money" signal.

Results (walk-forward OOS, leak-free):
- KR large-cap next-day direction from news: **0.4995** (≈ baseline); same
  coincident-news pattern as US (corr same-day +0.18 / next-day +0.06).
- KR 626-stock next-day direction from **foreign/institution flows + price**:
  full-coverage **~0.53** (≈ baseline), selective ~0.55–0.57 @10%, 0.64 @1%.
- **Foreign-flow event study**: sign(foreign flow_{t-1})→day-t direction = 0.51;
  heavy-BUY decile next-day up-rate 0.462 (BELOW the 0.456 overall) → the
  buy→up hypothesis does not hold at daily horizon.
- Horizons 1/5/10-day all ≈ baseline; `excess_ret`≈`ret_1d` (corr 1.0, no
  separate market-neutral alpha in this data).

**Conclusion**: even Korea's strongest documented signal (foreign flows) gives
only a tiny real edge (MCC ~0.05) — not ≥70%. Confirms the limit is cross-market,
not a US/NVDA quirk.

## 4c. Alignment-bug audit and controlled re-test (s35-s37, 2026-07-02)
Three REAL pipeline bugs were found and fixed, then the experiments were re-run
with the representation held constant to test whether the null results above
were bug artifacts. They were not.

**Bugs found:**
1. **US `s1_data.py` dated news by UTC calendar day** - the s22-proven UTC bug
   was fixed only in the s23 diagnostic, never in the flagship chain, so every
   number in `report.md` used news shifted +1 day for evening-ET items. Fixed
   at source (ET conversion) + `s37` realigns cached artifacts by timestamp.
2. **Overnight window discarded (US `s5`, KR `s34`)** - short-term slot keyed
   by calendar date of the prior trading day: pre-open news of the target day
   (~19-20% of volume) never entered features, weekend news arrived a day
   late, and intraday news already priced into the prior close (~53-68% of
   volume) diluted the slot. Fixed via eff/info-day mapping with exact
   session cutoffs (09:00/15:30 KST; 09:30/16:00 ET) and a `fresh` mask
   (= published post-close/weekend/pre-open).
3. **KR trained on 96.6% all-zero samples (s34)** - only 3.4% of small-cap
   samples had a prior-day event, so BCE collapsed the model to the base rate
   (constant confidence on 75% of test). Fixed: event-window-only training,
   time-based dev split (s34's dev was ticker-ordered).
   Also fixed: NTN logging - the "diverging" loss (2.9->3.8) was the
   lambda*L2 term over ~3M params; hinge is actually stable (~0.24,
   margin-satisfied ~75%).

**Re-test results (identical embeddings, corrected alignment):**
| experiment | n(test) | base | acc | MCC |
|---|---|---|---|---|
| KR s36 event-window training, since-prev-open bucket | 18,597 | 0.5700 | 0.5422 | +0.009 |
| KR s36 fresh-overnight-only bucket | 18,597 | 0.5700 | 0.5463 | +0.002 |
| KR s36 fresh, overnight-news days only | 1,539 | 0.5452 | 0.5328 | +0.018 |
| KR linear probe on overnight emb (3 reg. levels) | 1,539 | 0.5452 | 0.525-0.534 | - |
| US s37 WB-CNN realigned (was -0.003 MCC) | 366 | 0.5355 | 0.5410 | +0.060 |
| US s37 EB-CNN realigned (was +0.080 MCC) | 366 | 0.5355 | 0.5164 | -0.021 |
| US s37 TWB-CNN realigned (was +0.050 MCC) | 366 | 0.5355 | 0.5355 | +0.068 |
| US s37 TEB-CNN realigned (was +0.084 MCC) | 366 | 0.5355 | 0.5383 | +0.036 |

The best selective cell (KR fresh, top-10% confidence: 0.5844 vs base 0.5584,
n=154) is not significant (binomial p=0.285). Model ranking reshuffles inside
the noise band when alignment changes - i.e. the previous per-model MCC
differences (incl. "TEB-CNN best") were seed/alignment noise, not method skill.
Protocol note: KR w2v/NTN train on the full 2024-26 corpus (unsupervised, no
labels; US embeddings remain pre-test-only) - this can only inflate, not hide,
test skill, so the null stands a fortiori.

## 4d. Task redefinition -> HIGH prediction (user-directed, s39-s41, remote GPU)
Target changed from close->close direction to **intraday-high exceedance**:
y_k = [high(d) >= open(d)*(1+k)], decided at open(d) - the paper's own
market-sim entry (buy at open, +k% take-profit). This is a volatility/attention
target, which IS what news carries. Pipeline runs on the remote GPU box
(RTX 5060, 32 cores; bk_slim titles already on the server; ts parsed from
news_id; OHLCV fetched via pykrx per-ticker, s39).

**Linking precision fixed first (s41)**: ASCII word boundaries (KT !< SKT/KT&G),
Kiwi token-boundary validation (하이브 !< 하이브리드), NNG-homonym cue rules +
tight patterns for 대상/한화 (currency). 671,081 raw links -> 461,071 precise
(72,789 boundary-rejected + 109,994 ambiguity-rejected pairs).

**Results** (KR small-caps, event-window samples, test 2025-08->2026-06,
n=12,833 all-event / 749 overnight-news days; VOL-LR = leak-free realized-vol+
gap logistic; STACK = vol + EB-CNN news prob):
| k | model | mask | acc | MCC | top-10% conf hit-rate vs base | p |
|---|---|---|---|---|---|---|
| 2% | VOL-LR | all-event | 0.714 | +0.26 | 0.656 vs 0.319 | 7e-135 |
| 2% | STACK | overnight | 0.684 | +0.29 | **0.865 vs 0.371** | 2e-18 |
| 3% | VOL-LR | overnight | **0.809** | **+0.43** | 0.784 vs 0.252 | 2e-21 |
| 3% | STACK | overnight | 0.764 | +0.21 | 0.811 vs 0.252 | 1e-23 |
| 5% | VOL-LR | overnight | 0.880 | +0.45 | 0.703 vs 0.155 | 7e-26 |
| 5% | EB-CNN (news only) | overnight | 0.845 | 0.00 | 0.365 vs 0.155 | 8e-06 |

The >=70%-accuracy goal is MET on this task with real class separation
(MCC +0.26..+0.45, not majority-class artifacts); news carries significant
incremental signal (pure-news cells p<=1e-3; STACK 0.865 top-decile at k=2%),
though most predictability comes from vol/gap features.

**Honest trading caveat**: hit-prediction quality != profit. Naked
buy-open/TP-at-+k%/else-sell-close is negative-EV after 0.30% costs on most
selected sets (miss-day close losses dominate; SL-3% with pessimistic fill
ordering worsens it). Positive cells exist only in small-n news-selected sets
(k=5% EB-CNN top-10%: +0.85%/trade, n=74). Converting prediction into a
strategy needs intraday data (barrier-order resolution) and asymmetric exits -
that is execution research, not signal absence.

## 4e. Capacity scaling with anti-overfit suite (s42-s43, 10x data, remote GPU)
User-directed: scale the model while controlling overfitting. Data first:
news 2015-01+ (18.2M titles -> 2.08M precise links -> 523k events) + OHLCV
2015-2026 via pykrx (s42; cap-history endpoint down -> share-count proxy from
2023-12+ caps / adjusted close; universe = 2024 survivors, survivorship
DOCUMENTED). 808,500 samples, 188,100 event-window (10x s41); point-in-time
monthly small-cap membership; test = 2021-11..2026-06 (4.5y OOS).

Ladder (multi-task k=2/3/5; AdamW wd 0.05 decoupled, cosine+warmup, label
smoothing 0.05, day-dropout 0.10 + gaussian noise 0.03, dev-MCC early stop,
3-seed ensembles):
| model | params | k=2% MCC train/test | GAP | top-10% hit (base 0.305) |
|---|---|---|---|---|
| S control (s41 head) | 0.06M | +0.114 / -0.001 | +0.115 | 0.320 (p=4e-3) |
| M deep-CNN + reg | 0.78M | +0.111 / -0.002 | +0.113 | 0.320 (p=3e-3) |
| L transformer + reg | 7.15M | +0.113 / -0.005 | +0.118 | 0.300 (p=0.8) |

**Findings**: (1) the anti-overfit suite WORKS - train MCC stays pinned at
~0.11 across a 120x parameter range (no s13-style train->1.0 collapse), gap
bounded. (2) Capacity buys NOTHING: test MCC ~0 at every size; the small
selective news edge (+1.5pp @top-10%, p~3e-3) survives at 0.06-0.78M and is
LOST at 7M. The news branch is signal-bound, not capacity-bound. (3) VOL-LR
stays strong and stable on 4.5y OOS (acc 0.72/0.82/0.91, MCC +0.24/+0.20/+0.18
for k=2/3/5) - the task's predictability is real and durable, carried by
vol/gap features. (4) The strong 2025-08+ overnight-news cells (4d) do NOT
extend back through 2021-24 (overnight top-10% ~ base on the long window):
recency-specific or small-n - treat 4d selective cells as unconfirmed until
intraday-era data arrives. Next efficient axis = input information, not
parameters: article BODY encodings (bigkinds_finbert_scores.parquet already on
the box), news novelty/density features, minute bars.

## 4f. Architecture-vs-input probe (s44): the transformer is exonerated
Question: VOL-LR works - did the TRANSFORMER fail? Confound: s43's transformer
never saw the vol features (news-only input). Probe on identical data/protocol
(188k samples, 4.5y OOS; zero-day key-padding mask fixes the 91.2%-empty-token
attention dilution):
| model | input | params | k=2% MCC | k=3% | k=5% | GAP |
|---|---|---|---|---|---|---|
| VOL-LR | vol 5 feats | ~0 | +0.240 | +0.199 | +0.171 | - |
| VOL-MLP | vol 5 feats | 0.005M | +0.303 | +0.248 | +0.172 | +0.03 |
| NEWS-TF + mask | news only | 7.15M | -0.004 | -0.003 | 0.000 | +0.14 |
| **FUSED-TF** | news + vol token | 7.16M | **+0.304** | **+0.267** | **+0.209** | +0.03-0.06 |

**Verdict**: the architecture is NOT the failure - the same 7M transformer,
handed one vol token, beats VOL-LR at every k and beats VOL-MLP at k=3/5%
(+0.267 vs +0.248, +0.209 vs +0.172; n_test=72,139). The news-only branch
stays ~0 even after the masking fix -> the news EMBEDDING INPUT, not the
model, is the bottleneck. Caveat: FUSED > VOL-MLP may partly reflect a better
nonlinear vol fit rather than news contribution (VOL-only-TF ablation not
run). Best model to date: FUSED-TF - acc 0.72/0.82/0.91, MCC
+0.30/+0.27/+0.21, overfit-controlled. Evidence: `s44.log`.

## 4g. Bidirectional rare-event model (s46): learn 5% drops too
User correction: do not learn only upward 5% moves. s46 converts the task to
six multi-label heads: UP2/UP3/UP5 and DN2/DN3/DN5. It keeps the s45 B3 input
set (news embeddings + novelty + FinBERT + 30d price tokens + vol/gap token)
and adds rare-class technique: clipped positive-class weights, focal BCE, and
per-label dev-threshold calibration for MCC instead of fixed 0.5.

| label | test base | threshold | acc | MCC | train MCC | GAP | AP | top cells |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UP2 | 0.305 | 0.529 | 0.720 | +0.313 | +0.340 | +0.027 | 0.542 | - |
| UP3 | 0.183 | 0.540 | 0.786 | +0.314 | +0.335 | +0.021 | 0.424 | - |
| **UP5** | 0.085 | 0.555 | 0.852 | **+0.281** | +0.316 | +0.035 | 0.294 | top10=0.317, top5=0.402 |
| DN2 | 0.333 | 0.495 | 0.707 | +0.348 | +0.340 | -0.008 | 0.598 | - |
| DN3 | 0.183 | 0.522 | 0.792 | +0.364 | +0.351 | -0.013 | 0.485 | - |
| **DN5** | 0.063 | 0.627 | 0.924 | **+0.339** | +0.337 | -0.002 | 0.351 | top10=0.315, top5=0.417 |

Directional selector at 5% (choose UP5 vs DN5 by calibrated margin) gives
top10%=0.332 and top5%=0.429 hit-rate on n=7,183/3,591. This is a real
rare-class lift: prior UP5 fixed-threshold MCC was ~+0.18; s46 raises UP5 to
+0.281 and adds DN5 at +0.339 with negligible overfit. Evidence: `s46.log`.

## 4h. Ensemble + significance (s47), then the scaled max-push (s48)
**s47** added a GBM tabular learner (79 feats), per-label TF+GBM ensembling and
the significance machinery (date-cluster bootstrap 95% CIs, daily cross-
sectional top-k, 4-chunk stability). ENS test: UP5 MCC +0.287 [+0.274,+0.300],
DN5 +0.354 [+0.338,+0.372]; daily top-1 hit 44.7% (UP5, base 8.4%) / 51.3%
(DN5, base 6.1%). GBM alone matched the 7M TF. Evidence: `s47.log`.

**s48 exploration** (`s48e.log`): market-day regime is the biggest unexploited
axis (day-rate lag-1 autocorr +0.41, p10->p90 = 3%->17%); per-ticker hit-rate
persistence corr +0.56; big-cap base rates similar to small-cap (scaling
headroom 2x); top-decile TP/FP separable by gap/volz/nov/sent (meta-labeling).

**s48** scaled the tabular learner to the FULL universe (626 tickers, 1.62M
ticker-days, no news-window requirement), added 37 engineered features
(market-day context, rolling ticker priors, liquidity/candle shape, news
windows), expanding-OOF meta-labeling, and an enriched-V TF (14->30 dims)
ensembled per label. Results (test 2021-11..2026-06):

Small-cap event-window subset (s46/s47-comparable, n=71,838):
| label | s46 | s47 ENS | s48 ENS | 95% CI | top10 hit |
|---|---:|---:|---:|---|---:|
| UP2 | +0.313 | +0.328 | **+0.349** | [+0.334,+0.362] | 0.699 |
| UP3 | +0.314 | +0.315 | **+0.332** | [+0.316,+0.347] | 0.538 |
| UP5 | +0.281 | +0.287 | **+0.292** | [+0.278,+0.306] | 0.325 |
| DN2 | +0.348 | +0.357 | **+0.413** | [+0.401,+0.424] | 0.788 |
| DN3 | +0.364 | +0.373 | **+0.419** | [+0.407,+0.432] | 0.626 |
| DN5 | +0.339 | +0.354 | **+0.383** | [+0.367,+0.400] | 0.347 |
All 4 temporal chunks positive for every label (min +0.24).

Full-universe daily cross-sectional picks (626 tickers, 1,113 days, the
tradable statistic; META = OOF meta-labeled GBM):
| pick | hit-rate [95% CI] | day-base | lift |
|---|---|---:|---:|
| UP5 top-1/day | 0.562 [0.534,0.590] | 0.075 | 7.5x |
| UP5 top-3/day | 0.508 [0.490,0.526] | 0.075 | 6.8x |
| DN5 top-1/day | 0.692 [0.665,0.719] | 0.055 | 12.6x |
| DN5 top-3/day | 0.637 [0.620,0.655] | 0.055 | 11.6x |

Attribution: universe scaling + market-day/prior features drove the jump
(GBM daily top-1 44.7%->54.7% UP5, 51.3%->68.5% DN5 vs s47); meta-labeling
+0.7-1.5pp on top-1; enriched-V TF lifted DN-side ensemble (w_TF 0.6-0.8).
Evidence: `s48.log`, `s48e.log`; probs `kr48_probs.npz` (remote).

## 4i. Tradability: daily-bar then 5-min intraday backtest (s49, s50)
**Question**: does the UP5 signal become a profitable strategy (buy at open,
+5% limit TP, stop, risk-sized so a stop-out ~= 1% of equity)?

**s49 daily-bar portfolio backtest** exposed the core ambiguity: on daily OHLC
the same names touch both +5% and the stop, and bar-order is unknown. Fill
bracket on the optimized config (train->OOS): PESSIMISTIC (stop-first) CAGR
-19.8%, OPTIMISTIC (tp-first) CAGR +9.4%. The result is entirely determined by
the unknowable intraday order -> daily bars cannot decide tradability.

**s50 5-min intraday backtest** (KR 5-min bars, 65 tickers, 2022-11..2026-06;
first-touch resolved bar-by-bar; only same-5-min-bar straddles ambiguous, and
those are ~0.0% of trades). Risk sizing verified: worst single-trade hit
-0.87% of equity, ZERO trades lost >1% (the 1% rule holds). SL n% optimized on
the first-half, tested OOS on the second-half:
| window | config | CAGR | Sharpe | MDD | TP-rate | win |
|---|---|---:|---:|---:|---:|---:|
| TRAIN (22Q4-24Q3) | SL4% k3 score>=.8 gap<=2% | +5.7% | 0.59 | -9.9% | 0.34 | - |
| TEST OOS (24Q3-26Q2) | same | -8.8% | -0.63 | -20.9% | 0.30 | 0.46 |
| TEST OOS | k=1 (least bad) | -2.9% | -0.31 | -12.4% | 0.32 | 0.47 |

**Verdict**: NOT tradable as specified. With unambiguous 5-min fills the
strategy is slightly positive in-sample but NEGATIVE out-of-sample (CAGR -3%
to -11% across SL 2-5%, all k). Mechanism: buying at the open captures no edge
- only ~30% of high-score names reach +5% intraday while ~36% hit the stop
first, and the 0.33% round-trip cost erodes the rest (EV ~ -0.26%/trade). The
UP5 *ranking* is real (8.3% base -> 30% TP-hit at score>=0.8) but does not
survive entry-at-open execution + costs OOS. Evidence: `s49.log`, `s50.log`,
`kr50_equity_oos.csv`.

## 4j. Divergent realistic-entry search (s51): no entry rescues it OOS
If buy-at-open has no edge, does a better entry? Tested 6 realistic entry
families on 5-min bars, each with SL / k / score-threshold optimized on the
first-half and evaluated OOS on the second-half (TP = open*1.05, same 1%-risk
sizing). OOS leaderboard (train-selected configs):
| entry family | OOS CAGR | Sharpe | TP-rate | logic |
|---|---:|---:|---:|---|
| A market-at-open | -8.8% | -0.63 | 0.30 | baseline |
| D first-bar close | -6.2% | -0.76 | 0.30 | skip auction noise |
| B limit dip d% | -15.8% | -1.41 | 0.21 | buying weakness keeps falling |
| C delay >=09:30 | -11.7% | -2.10 | 0.30 | later entry, no help |
| E opening-range breakout | -14.7% | -2.97 | **0.51** | momentum hits TP more but gives back |
| F dip-then-reclaim | -39.0% | -5.57 | 0.09 | worst |

**All six are negative OOS.** The most logical (ORB momentum) maximizes TP-rate
(0.51: confirmed strength reaches +5% more often) yet still loses - it buys
higher, so the stop is farther and the give-back + cost dominate. The signal
identifies which names will be VOLATILE (UP5 and DN5 fire on the same high-vol
names, §4g co-occurrence), not a tradable DIRECTION from any fixed entry. A
directional long+TP+stop is structurally net-negative here regardless of entry
timing. Evidence: `s51.log`, `kr51_equity_champion.csv`.

## 4k. The EXACT test (s52, user-challenged): no-stop limit-TP, full universe
User challenge: "~50% of top picks touch open*1.05 - proper entry/exit/sizing
MUST monetize that." Valid criticisms of 4i/4j were identified and removed:
(a) s49's pessimistic same-bar SL-first ordering (that assumption alone spanned
-19.8%..+9.4% CAGR); (b) s50/s51 ran on the 65-name 5-min subset where top-k
hit collapses to 0.30-0.34 (vs 0.49-0.55 full-universe). Fix: **no-stop**
policy - buy top-k at open, limit-sell at open*1.05, else exit at close/next
open. On daily OHLC this is EXACT (a resting limit fills iff high>=tp; no
path/ordering assumption). Full 626-ticker universe; GBM scores are model-OOS
(fit < 2021-11-29); knobs tuned 2021-11..2024-06, frozen, validated
2024-06..2026-06. Consistency: top-1 unfiltered TP-touch = 0.547 = s48's
reported 0.547.

**Result: the EV surface is flat at ~0 everywhere.** Hit-rate and miss-day
severity move in lockstep (both are the same volatility):
| config (k=1, VAL) | hit | E[net & miss] | EV/trade |
|---|---:|---:|---:|
| no gap filter | 0.571 | -7.03% | -0.34% |
| gap <= 5% | 0.530 | -5.35% | -0.02% |
| gap <= 2% (frozen) | 0.491 | -4.74% | -0.11% [CI -0.61,+0.39] |
| gap-down only (<=0) | 0.490 | -4.23% | +0.14% (TUNE +0.04%) |
| gap <= -2% | 0.416 | -3.69% | -0.20% (TUNE +0.15% - sign flips) |

Portfolio (frozen, EW): CAGR -49.9%, Sharpe -0.29 [CI -1.61,+1.14]; deploy 1/3
scales MDD -85%->-38% but EV/trade unchanged. Breakeven needs hit 50.2% at the
observed miss mean; we sit 1.1pp short - i.e. **the open already prices the
50/50 +-5% distribution of these rockets**. No (filter, exit, k) cell shows a
replicating positive EV (gap-down flips sign between windows = noise). Sizing
cannot flip EV~0; it only shapes the path. Conclusion unchanged but now exact:
the ranking skill is real and confirmed at full strength, and it is fully
priced at the open. Monetization would require selling/buying the VOLATILITY
(options; absent from data) or pre-open execution (impossible).
Evidence: `s52.log`, `s52b.log`, `kr52_equity_val.csv`; code
`s52_dump_scores.py`, `s52_daily_nostop.py`, `s52b_cells.py`.

## 4l. TEB-CNN/news-only survival audit (s53): selection overfit
The old TEB-CNN headline was basically one lucky seed: in `report.md` the
seed-3-like run posted test MCC **+0.0842** and raw always-trade profit
**+$1,260** (randomization p=0.234), but s37 realignment already cut that to
**+0.0355**, and on truly fresh overnight-only days to **+0.0168**. s53 asks
the survival question directly: across 4 seeds, always-trade profits are
**+$3,562.94 / +$262.21 / -$925.52 / +$1,260.46** - dispersion이 signal보다
크다. The 4-seed ensemble lands at test acc **0.5273**, test MCC **+0.0089**,
always-trade profit only **+$202.38**; randomization p=**0.35339**,
bootstrap **P(profit<=0)=0.47126**, 95% CI **[-$7,109, +$266, +$7,230]**.
Thresholding does not rescue it: dev-best is **0.50** and honestly gives the
same **+$202.38** on test, while the test-peek best **0.57** shows **+$3,192**
on 104 trades but that is ex-post selection, not validation. High-confidence
beta=**0.70** is worse: **7 trades, -$604.81**, bootstrap CI
**[-$1,336, -$575, -$35]**, **P(profit<=0)=0.98446**. Multiple-testing closes
the loop: the original raw p=**0.234** becomes Bonferroni **1.0** over 11
models (and still **1.0** over 22 model×strategy variants). 결론: this is
research/selection overfit — seed + alignment + threshold-pick noise — not a
robust news-only edge, and not tradable evidence.

## 5. Status
- **Original target (close->close next-day direction >=70%)**: not achievable
  with this data class - measured mechanism (news coincident, overnight
  component priced at the open, no drift), confirmed bug-free by the 4c audit.
- **Redefined target (intraday high/low exceedance, user-directed)**: measured
  strong, CI-bounded separability on logged KR walk-forward tests - s48
  ensemble MCC +0.29..+0.42 across UP/DN 2/3/5% (all CI lower bounds > +0.27),
  full-universe daily top-1 hit-rates 56%/69% (UP5/DN5) vs bases 7.5%/5.5%,
  stable across 4 temporal chunks. News adds incremental signal on
  overnight-news days; vol/gap/market-regime structure is the dominant source.
- Tradability (s49-s52): closed with the EXACT no-stop test (s52) after fixing
  the pessimistic-fill and subset-handicap artifacts the user correctly
  challenged: top-k hit 0.49-0.57 is real, but E[net|miss] (-4.2..-7.0%)
  cancels it in lockstep across every filter - EV/trade ~ 0 minus costs,
  CI straddles zero, no replicating positive cell. The signal is a VOLATILITY
  selector (UP5/DN5 co-fire) that the open fully prices; a directional long
  cannot extract it. Monetization would need options-style volatility trades
  (no options data) or pre-open fills.
- **News-only survival / robustness (s53)**: closed. The prior TEB-CNN win was
  a seed/alignment artifact: s37 cuts MCC +0.0842 -> +0.0355 (fresh +0.0168),
  the 4-seed ensemble falls to acc 0.5273 / MCC +0.0089 with always-profit
  +$202.38 [rand p=0.35339; bootstrap CI -$7,109..+$7,230;
  P(profit<=0)=0.47126], and the beta=0.70 slice loses -$604.81 on 7 trades
  [P<=0=0.98446]. Dev-picked 0.50 does not improve test; the +$3,192 at 0.57
  is test-peek only. Conclusion: news-only selection overfit, no robust
  tradable edge.

Evidence: `results_*.json`, `accuracy_push.md`, `diagnosis.md`,
`s3[5-9]*.out`, `s41b.log`, `s42.out`, `s43.log`, `s44.log`, `s46.log`,
`s47.log`, `s48.log`, `s48e.log`, `s49.log`, `s50.log`, `s51.log`, `s52.log`,
`s52b.log`, `s53_teb_survival.json`, `s53_teb_daily_profit.csv`,
`s53_teb_dev_threshold_curve.csv`, `s53_teb_test_threshold_curve.csv`;
code `s1`-`s53`, `s53_teb_cnn_survival.py`, `t1`-`t2`
(US = s1-s27+s37, KR = s28-s36 + s39-s53 remote/local).
