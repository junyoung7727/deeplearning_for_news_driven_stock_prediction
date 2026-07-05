# Concrete diagnosis — why NVDA next-day direction can't reach 70% (not "EMH")

All numbers measured on the local data (news = FinBERT sentiment on NVDA titles;
prices = FMP daily OHLC + 5-min bars). Code: s20-s23.

## 1. News is priced in INSTANTLY (into the open) with NO drift
`published_at` is tz-naive **UTC** (proven in s22; a real pipeline bug — s1 dated
news by UTC, mis-assigning evening-ET news by a day). After converting to ET:

| relationship | corr | dir. accuracy |
|---|---|---|
| sentiment(today) ↔ **same-day** return | **+0.215** | (coincident) |
| sentiment(today) → **next-day** return | −0.003 | 52.2% |
| sentiment(prev) → today return (model uses) | −0.003 | — |
| **pre-open news → overnight GAP** (close→open) | **+0.073** | **58.8%** |
| pre-open news → **open→close** (after open) | +0.007 | 48.9% |

Reading: overnight news DOES move the price — but the move is the **opening gap**;
by the time the market is open, the news is fully in the price and there is **no
intraday or next-day drift** (+0.007 / −0.003 ≈ 0). The +0.215 same-day figure is
**coincidence** (news and price react to the same event at the same time), not a
signal you can act on ahead of the move.

## 2. Consequence
Every horizon that begins AFTER the news timestamp (open→close, next-day
close→close = our task) has ~zero signal. The one leading relation (overnight
news → gap, 58.8%) is **not tradable** (the news arrives after the prior close, so
you can't position before the gap) and is not 70% anyway.

## 3. Secondary, concrete issues (fixable but don't create signal)
- **Timezone bug**: news dated in UTC not ET (fixed/verified in s23).
- **Signal dilution**: median 5, up to 165 articles/day averaged into one vector.
- **Noise floor**: 31% of days have |ret| < 1% (near-random sign).
- **Model size is NOT the cause**: 21k→12.4M params all overfit (train→1.0, test flat ~0.52-0.55).

## 4. What a 70% would actually require (data we don't have)
Signal that LEADS price: (a) news faster than the market (latency edge),
(b) order-flow / limit-order-book microstructure, (c) options-implied vol/skew,
(d) alternative data (web traffic, supply chain) not yet in prices. With
news-titles + OHLC alone, the information is already in the price by the time we
can act — measured, not assumed.
