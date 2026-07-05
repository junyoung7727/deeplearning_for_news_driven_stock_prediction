# Accuracy push — target ≥70% test accuracy (NVDA next-day direction)

Task (fixed): sign of NVDA close-to-close return on day `D_i`, using ONLY info at
close of `D_{i-1}` (leak-free). Majority ("up") baseline ≈ 0.54.

## Techniques applied (financial ML, anti-overfit, no test-peeking)
- **News**: spam/legal-wire filtering, FinBERT **sentiment**, **novelty**
  (embedding distance), **news-volume / abnormal-volume**, PCA of title embeddings.
- **Price/technical**: lagged returns, momentum, MA-ratios, realized vol, RSI-14,
  dist-from-high/low, volume z-score, streak.
- **Cross-asset**: 8 peer mega-caps' prior-day returns + market proxy.
- **Models**: Logistic (L2 / ElasticNet), HistGBM, RandomForest, ExtraTrees,
  ensemble, |return|-weighted training.
- **Confidence / meta-labeling** (López de Prado): secondary model for P(correct),
  confidence-selective with thresholds fixed on PRIOR data.
- **Evaluation**: expanding-window **walk-forward** (retrain quarterly, embargo 1d)
  → long OOS span 2021-2026 (1364 days), regime-adaptive, fully leak-free.

## Walk-forward OOS results (2021-2026, 1364 days; majority baseline = 0.5374)
| method | OOS acc | OOS mcc |
|---|---|---|
| always-up (majority) | **0.5374** | 0.000 |
| ExtraTrees | 0.5330 | -0.007 |
| RandomForest | 0.5242 | -0.001 |
| Logit-ElasticNet | 0.5183 | 0.023 |
| HGB | 0.5103 | 0.007 |
| HGB price-only | 0.5081 | -0.004 |
| HGB |ret|-weighted | 0.5081 | 0.001 |
| momentum rule | 0.5059 | 0.000 |
| ENSEMBLE(LE+HGB+RF) | 0.5022 | -0.017 |

**No learned model beats the majority baseline out-of-sample.** HGB overfits
badly (train 0.90 → OOS 0.51, gap 0.39): the features have no generalizable
signal, so capacity only memorizes noise. Confidence-selective (ensemble, WF
threshold) tops out at 0.59 @ 3% coverage — not 70%.

## Earlier single-split numbers were regime-luck
A fixed 2018-23 train → 2025-26 test gave 0.56 (price+HGB); under proper
walk-forward across 2021-26 that edge disappears (≈ baseline). The "0.699/0.750"
selective seen once was test-peeking and does not survive honest validation.

## Verdict (rigorously verified, 4 independent ways)
1. Feature/model sweep (single split): ceiling ≈ 0.56.
2. Capacity sweep (21k → 12.4M params): train → 1.00, test flat ~0.52-0.55.
3. **Walk-forward divergent sweep (gold standard): best OOS = majority (0.5374);
   no method has an edge.**
4. **Changed the target to an *easier* one, done fairly**: next-day VOLATILITY
   direction with a BALANCED trailing-median label (baseline 0.50) → OOS 0.544
   (walk-forward). Real skill (+0.04, MCC 0.09) but still nowhere near 0.70. The
   "70% volatility" seen in literature comes from imbalanced labels, not skill.

**≥70% is not achievable by any honest, balanced, walk-forward-validated target
on this data** — price direction ~0.51-0.54, volatility direction ~0.54, both
with only tiny genuine skill (MCC 0.05-0.09). This is the Efficient Market limit
(Fama 1965, the paper's own premise): Bayes error ≈ 0.5. Displaying 0.70 requires
leakage, test-peeking, or an **imbalance-driven relabel** (e.g. "predict NOT a
>3% down day" ≈ 90% trivially) — accuracy WITHOUT skill. None was done.

Evidence: results_boost.json, results_selective.json, results_xasset.json,
results_capacity.json, results_walkforward.json, results_divergent.json,
results_volatility.json; code s10-s17, features_ff.parquet.
