"""노트북에서 import 해서 쓰는 시각화 모음 (모든 함수는 fig 반환)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def news_overview(news: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    monthly = news.groupby(pd.Grouper(key="date", freq="MS")).size()
    axes[0].plot(monthly.index, monthly.values, color="#2563eb")
    axes[0].set_title("월별 뉴스 기사 수")
    counts = news.ticker.value_counts()
    axes[1].bar(counts.index, counts.values, color="#7c3aed")
    axes[1].set_title("티커별 기사 수")
    axes[1].tick_params(axis="x", rotation=45)
    tok = news.tokens.map(len)
    axes[2].hist(tok, bins=range(2, 30), color="#0ea5e9")
    axes[2].set_title(f"제목 토큰 길이 (mean {tok.mean():.1f})")
    fig.tight_layout()
    return fig


def price_overview(prices: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    colors = {"train": "#dbeafe", "dev": "#fde68a", "test": "#fecaca"}
    for split, g in prices.groupby("split"):
        axes[0].axvspan(g.date.min(), g.date.max(), color=colors.get(split, "#eee"), alpha=0.5)
    axes[0].plot(prices.date, prices.close, color="#0f172a", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("NVDA 종가 (log) + train/dev/test 구간")
    axes[1].hist(prices.ret * 100, bins=60, color="#16a34a")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title(f"일간 수익률 % (상승비율 {(prices.label == 1).mean():.3f})")
    fig.tight_layout()
    return fig


def embedding_scatter(xy: np.ndarray, labels=None, highlight_mask=None, title: str = ""):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.35, color="#64748b")
    if highlight_mask is not None:
        hm = np.asarray(highlight_mask, bool)
        ax.scatter(xy[hm, 0], xy[hm, 1], s=14, alpha=0.8, color="#dc2626", label="highlight")
        ax.legend()
    if labels is not None:
        for i, lab in enumerate(labels):
            if lab:
                ax.annotate(str(lab), (xy[i, 0], xy[i, 1]), fontsize=8, alpha=0.85)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def training_curves(hist: dict):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    ax1.plot(ep, hist["train_loss"], "o-", color="#2563eb", label="train BCE loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(ep, hist["dev_mcc"], "s--", color="#dc2626", label="dev MCC")
    ax2.set_ylabel("dev MCC", color="#dc2626")
    ax1.set_title(f"{hist['rep']}-{'NN' if hist['nn_only'] else 'CNN'} 짧은 학습: loss vs dev MCC")
    fig.tight_layout()
    return fig


def results_bar(results: dict):
    rows = results["metrics"]
    names = [r["model"] for r in rows]
    acc = [r["test_acc"] for r in rows]
    mcc = [r["test_mcc"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(rows) + 1.8))
    axes[0].barh(names, acc, color="#2563eb")
    axes[0].axvline(results.get("test_up_rate", 0.5355), color="k", ls="--", lw=1, label="majority")
    axes[0].set_title("Test Accuracy")
    axes[0].legend()
    axes[1].barh(names, mcc, color="#dc2626")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("Test MCC")
    for ax in axes:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def equity_curve(dates, daily, title: str = ""):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    eq = np.cumsum(np.asarray(daily, float))
    ax.plot(pd.to_datetime(dates), eq, color="#0f172a")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(title or f"누적 손익 (final ${eq[-1]:,.0f})")
    fig.tight_layout()
    return fig


def randomization_hist(dist, model_profit, p):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(dist, bins=50, color="#94a3b8")
    ax.axvline(model_profit, color="#dc2626", lw=2, label=f"model ${model_profit:,.0f}")
    ax.set_title(f"랜덤 롱/숏 1000회 분포 vs 모델 손익 (p={p:.3f})")
    ax.legend()
    fig.tight_layout()
    return fig


def sentiment_vs_returns(sent, next_ret):
    sent = np.asarray(sent, float)
    next_ret = np.asarray(next_ret, float)
    corr = float(np.corrcoef(sent, next_ret)[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(sent, next_ret * 100, s=10, alpha=0.4, color="#7c3aed")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("FinBERT title sentiment")
    ax.set_ylabel("다음날 수익률 %")
    ax.set_title(f"감성 vs 다음날 수익률 (corr {corr:+.3f})")
    fig.tight_layout()
    return fig


def threshold_curves(dev_df: pd.DataFrame, test_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(dev_df.threshold, dev_df.profit, "o-", color="#2563eb", label="dev profit")
    ax.plot(test_df.threshold, test_df.profit, "s-", color="#dc2626", label="test profit")
    best = dev_df.loc[dev_df.profit.idxmax()]
    ax.axvline(best.threshold, color="#2563eb", ls="--", lw=1, label=f"dev-best {best.threshold:.2f}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("confidence threshold β")
    ax.set_ylabel("profit ($)")
    ax.set_title("임계값별 손익: dev로 고르고 test에서 확인해야 하는 이유")
    ax.legend()
    fig.tight_layout()
    return fig


def upgrade_ladder(mapping: dict, baseline: float = 0.5355):
    names = list(mapping.keys())
    vals = [mapping[k] for k in names]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(names) + 1.6))
    ax.barh(names, vals, color="#0ea5e9")
    ax.axvline(baseline, color="k", ls="--", lw=1.2, label=f"majority {baseline:.3f}")
    ax.set_xlim(0.4, max(0.62, max(vals) + 0.02))
    ax.set_title("업그레이드 사다리: 다음날 방향 정확도")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def exceedance_bars(rates: dict):
    names = list(rates.keys())
    vals = [rates[k] * 100 for k in names]
    colors = ["#dc2626" if n.startswith("UP") else "#2563eb" for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("발생 비율 %")
    ax.set_title("KR intraday HIGH/LOW exceedance base rate (2015–2026, 626종목)")
    fig.tight_layout()
    return fig


def kr_equity_panels(curves: dict):
    titles = {
        "kr50": "s50: 5분봉 OOS (buy-open/TP/SL)",
        "kr51": "s51: 엔트리 챔피언 OOS",
        "kr52": "s52: EXACT no-stop (frozen VAL)",
    }
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.6))
    if len(curves) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, curves.items()):
        ax.plot(df.date, df.equity, color="#0f172a")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_title(titles.get(name, name))
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def score_lift(topk: dict):
    names = list(topk.keys())
    hit = [topk[n]["hit_rate"] * 100 for n in names]
    base = [topk[n]["base_rate"] * 100 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, base, width=0.4, color="#94a3b8", label="base rate")
    ax.bar(x + 0.2, hit, width=0.4, color="#dc2626", label="top-k hit rate")
    ax.set_xticks(x, names)
    for i in range(len(names)):
        ax.text(i + 0.2, hit[i] + 0.6, f"{hit[i]:.1f}%", ha="center", fontsize=9)
        ax.text(i - 0.2, base[i] + 0.6, f"{base[i]:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("TP(+5%) 도달 비율 %")
    ax.set_title("사용자 학습 GBM 점수의 daily top-k lift (모델-OOS)")
    ax.legend()
    fig.tight_layout()
    return fig
