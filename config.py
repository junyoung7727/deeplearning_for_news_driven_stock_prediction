"""
Configuration for reproducing:
  Ding, Zhang, Liu, Duan (IJCAI 2015)
  "Deep Learning for Event-Driven Stock Prediction"

Target: single stock NVDA, using FMP news (titles) + FMP daily OHLC.

All hyperparameters below are taken directly from the paper where stated.
Where the paper is silent, the value is marked  # (paper-silent) with the
chosen default and the reason.
"""
from __future__ import annotations
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
ART  = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

# Source data (FMP, already on disk as parquet)
NEWS_PARQUET  = r"D:\Github\homeserver\alphamale\data\news\us_fmp_news_rich.parquet"
DAILY_PARQUET = r"D:\Github\homeserver\alphamale\data\price\us_daily_data.parquet"
MIN5_NVDA     = r"D:\Github\homeserver\alphamale\data\prices\fmp_5min_us\NVDA.parquet"

# ----------------------------------------------------------------------------
# Target & universe
# ----------------------------------------------------------------------------
TARGET = "NVDA"                      # individual-stock prediction (paper Sec 4.3)
# Titles from these tickers form the corpus for word2vec + NTN event embeddings
# (analogue of the paper's "large-scale financial news corpus").
CORPUS_TICKERS = ["NVDA", "AAPL", "MSFT", "JPM", "V", "BRK-B", "CAT", "RTX", "GE"]

# ----------------------------------------------------------------------------
# Chronological splits (paper uses time-ordered train/dev/test, Table 1)
# NVDA news becomes dense from 2019 onward.
# ----------------------------------------------------------------------------
START_DATE = "2018-01-01"
DEV_START  = "2024-01-01"           # train  = [START, DEV_START)
TEST_START = "2025-01-01"           # dev    = [DEV_START, TEST_START)
END_DATE   = "2026-06-30"           # test   = [TEST_START, END_DATE]

# Embeddings (word2vec, NTN) are trained ONLY on news strictly before TEST_START
# (train+dev period) to avoid any look-ahead into the test set.
EMB_TRAIN_END = TEST_START

# ----------------------------------------------------------------------------
# Word embeddings  (paper Sec 2.2: skip-gram, d = 100)
# ----------------------------------------------------------------------------
WORD_DIM      = 100                  # d = 100  (paper)
W2V_WINDOW    = 5                    # (paper-silent) standard skip-gram window
W2V_MIN_COUNT = 5                    # (paper-silent) drop very rare tokens
W2V_NEG       = 5                    # (paper-silent) negative samples (SGNS)
W2V_EPOCHS    = 10                   # (paper-silent)
W2V_BATCH     = 4096
W2V_LR        = 0.025                # standard w2v initial lr

# ----------------------------------------------------------------------------
# Neural Tensor Network for event embeddings (paper Sec 2.2, Eq.1; Alg.1)
# ----------------------------------------------------------------------------
NTN_K        = WORD_DIM             # tensor slices k; paper states R1 in R^d => k=d
NTN_MARGIN   = 1.0                  # margin in max(0, 1 - f(E) + f(E^r))   (paper)
NTN_LAMBDA   = 1e-4                 # L2 reg lambda = 0.0001                 (paper)
NTN_ITERS    = 40                   # epochs over the event set.  Paper sets N=500
                                    #  for its 10M-event corpus with Alg.1 example
                                    #  removal; at NVDA scale (~28k events, full-set
                                    #  updates + fresh negatives) the margin
                                    #  objective converges in well under 40 epochs.
NTN_BATCH    = 512
NTN_LR       = 0.01                 # (paper-silent) Adam lr for back-prop

# ----------------------------------------------------------------------------
# Deep prediction model (paper Sec 3, Fig.3, Eq.3-4)
# ----------------------------------------------------------------------------
LONG_DAYS    = 30                   # long-term  = past month  (paper)
MID_DAYS     = 7                    # mid-term   = past week   (paper)
SHORT_DAYS   = 1                    # short-term = past day    (paper)
CONV_L       = 3                    # narrow convolution combines l=3 events (paper)
N_FILTERS    = 64                   # (paper-silent) conv feature maps; paper's
                                    #  single-filter Eq.3 generalised to N filters
HIDDEN       = 100                  # (paper-silent) feedforward hidden layer size
CLF_EPOCHS   = 120
CLF_LR       = 1e-3
CLF_BATCH    = 128
CLF_DROPOUT  = 0.5                  # (paper-silent) regularisation
CLF_PATIENCE = 12                   # early stop on dev (tuning set)

# ----------------------------------------------------------------------------
# Market simulation (paper Sec 4.3 "Market Simulation", Lavrenko et al. 2000)
# ----------------------------------------------------------------------------
SIM_CAPITAL   = 10000.0            # invest $10,000 per signal (paper)
SIM_TAKEPROFIT= 0.02              # sell once +2% intraday gain reachable (paper)
SIM_COVER     = 0.01              # cover short if price 1% lower than shorted (paper)
SIM_THRESHOLD = 0.70             # best threshold beta = 0.7 (paper Fig.5)

# ----------------------------------------------------------------------------
# Finance-specialized transformer embeddings (drop-in replacement for the
# skip-gram word vectors + NTN event embeddings).  The pretrained model already
# encodes a large financial corpus, so no NTN training is needed:
#   TWB = transformer encoding of the news title   (analogue of WB)
#   TEB = transformer encoding of the event triple (analogue of EB)
# ----------------------------------------------------------------------------
TF_MODEL   = "ProsusAI/finbert"    # finance-domain BERT (Reuters TRC2 +
                                   #  Financial PhraseBank).  Any HF encoder id
                                   #  works (e.g. a finance RoBERTa) - one line.
TF_MAXLEN  = 64                    # titles/triples are short
TF_BATCH   = 64
TF_DIM     = 768                   # model hidden size (mean-pooled)
TF_PCA_DIM = 100                   # PCA-reduce transformer emb to this dim (train-fit)
                                   #  -> equal dim to skip-gram/NTN (fair, less overfit);
                                   #  set None to keep the full 768-d embedding.

SEED = 13
