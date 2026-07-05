import os
from pathlib import Path
import math
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
OUT = ART / 'KR_UP5_signal_research_report_20260703_s53.pdf'
FIGDIR = ART / 'kr_research_report_figures'
FIGDIR.mkdir(exist_ok=True)

FONT = Path('C:/Windows/Fonts/malgun.ttf')
BOLD = Path('C:/Windows/Fonts/malgunbd.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = 'Malgun Gothic'
if BOLD.exists():
    font_manager.fontManager.addfont(str(BOLD))
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COL = {
    'blue': '#2F5D9B',
    'blue2': '#7EA6D8',
    'green': '#2E8B57',
    'red': '#B54434',
    'orange': '#D9822B',
    'purple': '#6F4AA8',
    'grey': '#64748B',
    'light': '#F4F7FB',
    'dark': '#1F2937',
    'grid': '#E5E7EB',
    'yellow': '#F7C948',
}

A4L = (11.69, 8.27)


def save_page(pdf, fig, name):
    fig.savefig(FIGDIR / f'{name}.png', dpi=170, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page(title, subtitle=None):
    fig = plt.figure(figsize=A4L, facecolor='white')
    fig.text(0.04, 0.955, title, fontsize=22, fontweight='bold', color=COL['dark'], va='top')
    if subtitle:
        fig.text(0.04, 0.915, subtitle, fontsize=10.5, color=COL['grey'], va='top')
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.895, 0.895], color=COL['blue'], lw=2))
    return fig


def add_wrapped(fig, x, y, text, width=92, size=10.5, color=None, line=0.032, weight='normal'):
    color = color or COL['dark']
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=False, replace_whitespace=False))
    fig.text(x, y, '\n'.join(lines), fontsize=size, color=color, va='top', ha='left', linespacing=1.35, fontweight=weight)
    return y - line * max(1, len(lines))


def add_box(fig, x, y, w, h, text, fc, ec, size=11.5, title=None):
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    rect = plt.Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor=ec, lw=1.2, transform=ax.transAxes)
    ax.add_patch(rect)

    def wrapped(s, font_size, top_y):
        box_width_pts = max(1, w * fig.get_figwidth() * 72 * 0.88)
        char_width_pts = max(4.5, font_size * 0.62)
        width_chars = max(18, int(box_width_pts / char_width_pts))
        lines = []
        for para in s.split('\n'):
            if para.strip():
                lines.extend(textwrap.wrap(para, width=width_chars, break_long_words=False, replace_whitespace=False))
            else:
                lines.append('')
        line_step = (font_size * 1.35) / max(1, h * fig.get_figheight() * 72)
        available = max(0.08, top_y - 0.06)
        if len(lines) * line_step > available and font_size > 8.5:
            font_size = max(8.5, font_size * available / max(available, len(lines) * line_step) * 0.96)
            return wrapped(s, font_size, top_y)
        return '\n'.join(lines), font_size, line_step

    if title:
        ax.text(0.04, 0.84, title, fontsize=size + 1, fontweight='bold', color=ec, va='top', clip_on=True)
        body, fsize, _ = wrapped(text, size, 0.66)
        ax.text(0.04, 0.66, body, fontsize=fsize, color=COL['dark'], va='top', linespacing=1.25, clip_on=True)
    else:
        body, fsize, _ = wrapped(text, size, 0.88)
        ax.text(0.04, 0.88, body, fontsize=fsize, color=COL['dark'], va='top', linespacing=1.25, clip_on=True)
    return ax


def barh(ax, labels, values, color, xlabel=None, xlim=None, zero=True):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.92)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    if zero:
        ax.axvline(0, color='#334155', lw=0.8)
    ax.grid(axis='x', color=COL['grid'], lw=0.8)
    ax.spines[['top', 'right', 'left']].set_visible(False)


def annotate_bars(ax, values, fmt='{:.1f}', pct=False):
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for i, v in enumerate(values):
        txt = (fmt.format(v * 100) + '%') if pct else fmt.format(v)
        if v >= 0:
            x = v + span * 0.01
            ha = 'left'
        else:
            x = v - span * 0.01
            ha = 'right'
        ax.text(x, i, txt, va='center', ha=ha, fontsize=9, color=COL['dark'])


def read_equity(path, date_col='date', value_col='equity'):
    p = ART / path
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if date_col not in df.columns:
        df = df.rename(columns={df.columns[0]: 'date'})
        date_col = 'date'
    if value_col not in df.columns:
        value_col = 'net' if 'net' in df.columns else df.columns[-1]
    df[date_col] = pd.to_datetime(df[date_col])
    return df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: 'equity'}).dropna()


# Data grounded in artifacts/FINAL_REPORT.md and logs s43-s53.
us_acc = [
    ('majority baseline', 0.5374, 0.000),
    ('ExtraTrees', 0.5330, -0.007),
    ('RandomForest', 0.5242, -0.001),
    ('Logit-EN', 0.5183, 0.023),
    ('HGB', 0.5103, 0.007),
    ('HGB price-only', 0.5081, -0.004),
    ('momentum rule', 0.5059, 0.000),
    ('ensemble', 0.5022, -0.017),
]
capacity = [
    ('GBM 100', 0.8512, 0.5191), ('GBM 400 d3', 0.9534, 0.5410), ('GBM 400 d6', 1.0000, 0.5546),
    ('GBM 1500', 1.0000, 0.5437), ('MLP 21k', 0.9979, 0.5082), ('MLP 3.3M', 0.9973, 0.5273), ('MLP 12.4M', 0.9911, 0.5191)
]
mechanism_corr = [
    ('same-day ret', 0.215), ('next-day ret', -0.003), ('pre-open gap', 0.073), ('open-close after open', 0.007)
]
mechanism_acc = [('pre-open news -> gap', 0.588), ('pre-open news -> open-close', 0.489)]
kr_bug_audit = [
    ('KR event-window', 0.5700, 0.5422, 0.009),
    ('KR fresh overnight bucket', 0.5700, 0.5463, 0.002),
    ('KR overnight-news days', 0.5452, 0.5328, 0.018),
    ('US WB realigned', 0.5355, 0.5410, 0.060),
    ('US EB realigned', 0.5355, 0.5164, -0.021),
    ('US TWB realigned', 0.5355, 0.5355, 0.068),
    ('US TEB realigned', 0.5355, 0.5383, 0.036),
]
arch_mcc = {
    'VOL-LR': [0.2404, 0.1985, 0.1711],
    'VOL-MLP': [0.3025, 0.2479, 0.1719],
    'NEWS-TF': [-0.0044, -0.0029, 0.0000],
    'FUSED-TF': [0.3043, 0.2674, 0.2088],
}
rare_mcc_s46 = [('UP2', .3132), ('UP3', .3138), ('UP5', .2813), ('DN2', .3479), ('DN3', .3637), ('DN5', .3385)]
ens_mcc_s48 = [('UP2', .3485), ('UP3', .3320), ('UP5', .2915), ('DN2', .4125), ('DN3', .4194), ('DN5', .3830)]
full_daily = [('UP5 top-1', .562, .075), ('UP5 top-3', .508, .075), ('DN5 top-1', .692, .055), ('DN5 top-3', .637, .055)]
trade_fill = [('daily OHLC stop-first', -19.8), ('daily OHLC TP-first', 9.4), ('5-min OOS open', -8.8), ('5-min OOS k=1', -2.9)]
entries = [('open', -8.8, -0.63, .304), ('first-bar close', -6.2, -0.76, .301), ('limit dip', -15.8, -1.41, .207), ('delay', -11.7, -2.10, .301), ('ORB', -14.7, -2.97, .506), ('dip-reclaim', -39.0, -5.57, .085)]
s52_configs = [
    ('no gap filter', .571, -7.03, -0.34),
    ('gap <= 5%', .530, -5.35, -0.02),
    ('gap <= 2% frozen', .491, -4.74, -0.11),
    ('gap-down only', .490, -4.23, +0.14),
    ('gap <= -2%', .416, -3.69, -0.20),
]
s52_quarters = [('24Q2', -0.81), ('24Q3', -0.81), ('24Q4', 0.43), ('25Q1', -0.22), ('25Q2', -0.22), ('25Q3', 0.14), ('25Q4', -0.56), ('26Q1', 0.69), ('26Q2', -0.08)]
s53_seed_rows = [
    ('seed 0', 0.09306848942014956, 3562.93972293662),
    ('seed 1', 0.09269624493721504, 262.20756144992345),
    ('seed 2', 0.09984210138718569, -925.5222961609145),
    ('seed 3', 0.11171909538752513, 1260.4605170970194),
]
s53_survival = {
    'headline_seed3_profit': 1260.4605170970194,
    'headline_raw_p': 0.234,
    'ensemble_test_acc': 0.5273224043715847,
    'ensemble_test_mcc': 0.008873782988341588,
    'always_profit': 202.37619959860905,
    'always_trades': 366,
    'always_randomization_p': 0.35339,
    'always_bootstrap_p_le_0': 0.47126,
    'always_bootstrap_ci': (-7109.014129050125, 266.2442831050298, 7229.8857568630665),
    'beta_threshold': 0.70,
    'beta_profit': -604.8074212284249,
    'beta_trades': 7,
    'beta_bootstrap_p_le_0': 0.98446,
    'beta_bootstrap_ci': (-1336.2299792527592, -575.1675805183786, -35.266802195244026),
    'bonferroni_p': 1.0,
    's37_original_mcc': 0.08415567657858812,
    's37_realigned_mcc': 0.0355,
    's37_fresh_mcc': 0.0168,
}
s53_threshold_checks = [
    ('dev-best frozen', 0.50, 6482.085766056616, 202.37619959860905, 366),
    ('test-peek best', 0.57, 2091.0447865214455, 3192.1543326568817, 104),
]

with PdfPages(OUT) as pdf:
    # 1 Cover
    fig = page('KR/US 이벤트 기반 주가 예측 연구 보고서', '뉴스/가격/수급 신호 → 고가·저가 돌파 예측 → 실제 진입·청산 가능성 검증. Evidence: artifacts/FINAL_REPORT.md, s1-s53 logs.')
    fig.text(0.04, 0.80, '핵심 결론', fontsize=18, fontweight='bold', color=COL['blue'])
    add_box(fig, 0.04, 0.55, 0.44, 0.20,
            'UP5/DN5 “5% 고가/저가 터치” 랭킹 신호는 진짜다. s48 full-universe daily top-1 hit: UP5 56.2%, DN5 69.2% vs day-base 7.5%/5.5%.',
            '#EAF2FC', COL['blue'], title='Signal exists')
    add_box(fig, 0.52, 0.55, 0.44, 0.20,
            '하지만 buy-at-open → +5% limit TP → otherwise close/next-open의 방향성 롱 전략은 EV가 0 근처다. miss-day 손실이 hit-rate 상승과 같이 커진다.',
            '#FCEAE8', COL['red'], title='Not a directional long edge')
    add_box(fig, 0.04, 0.30, 0.92, 0.18,
            '사용자 지적 후 s52에서 stop/동일봉 순서 가정과 65종목 subset 문제를 제거했다. full 626종목, no-stop, daily OHLC exact test에서도 VAL EV/trade = -0.11% [95% CI -0.61%, +0.39%], Sharpe = -0.29 [CI -1.61, +1.14]. gap-down only는 +0.14%지만 TUNE +0.04%, gap<=-2%는 부호 반전으로 재현성이 없다.',
            '#FFF7E6', COL['orange'], size=11.0, title='Final adjudication')
    fig.text(0.04, 0.18, '보고서 생성일: 2026-07-03', fontsize=10, color=COL['grey'])
    fig.text(0.04, 0.14, '생성 산출물: artifacts/KR_UP5_signal_research_report_20260703_s53.pdf', fontsize=10, color=COL['grey'])
    save_page(pdf, fig, '01_cover')

    # 2 Roadmap
    fig = page('1. 실험 로드맵과 산출물', '전체 과정은 “원 논문 재현 → 미국/NVDA 진단 → 한국시장 확장 → high/low exceedance → tradability” 순서로 진행됐다.')
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.58])
    ax.axis('off')
    phases = [
        ('s1-s9', 'Ding et al. 재현\nNVDA 뉴스 제목+OHLC'),
        ('s10-s19', '정확도 캠페인\nGBM/선택/용량/vol'),
        ('s20-s27', '원인 진단\n타임존/갭/5분'),
        ('s28-s37', 'KR 복제/버그감사\nBigKinds+KRX+수급'),
        ('s39-s48', '고가/저가 돌파\nUP/DN 2/3/5%'),
        ('s49-s52', '매매 가능성\ndaily/5min/no-stop'),
    ]
    xs = np.linspace(0.06, 0.94, len(phases))
    y = 0.55
    for i, (code, label) in enumerate(phases):
        ax.add_patch(plt.Circle((xs[i], y), 0.055, color=COL['blue'] if i < 5 else COL['orange'], alpha=0.95))
        ax.text(xs[i], y, code, ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.text(xs[i], y - 0.16, label, ha='center', va='top', fontsize=9.5, color=COL['dark'])
        if i < len(phases) - 1:
            ax.annotate('', xy=(xs[i+1]-0.065, y), xytext=(xs[i]+0.065, y), arrowprops=dict(arrowstyle='->', color=COL['grey'], lw=1.5))
    add_box(fig, 0.06, 0.06, 0.88, 0.10,
            '핵심 evidence: report.md, accuracy_push.md, diagnosis.md, FINAL_REPORT.md, s43/s44/s46/s47/s48/s49/s50/s51/s52/s52b logs, kr50/kr51/kr52 equity CSV.',
            COL['light'], COL['grey'], size=10.3)
    save_page(pdf, fig, '02_roadmap')

    # 3 US ceiling
    fig = page('2. 미국/NVDA: next-day 방향성의 정직한 ceiling', '목표: close(D-1)까지의 정보로 close(D) 방향을 예측. Walk-forward OOS 2021-2026.')
    ax1 = fig.add_axes([0.07, 0.46, 0.42, 0.34])
    labels = [x[0] for x in us_acc]
    vals = [x[1] for x in us_acc]
    barh(ax1, labels, vals, COL['blue'], xlabel='OOS accuracy', xlim=(0.48, 0.72), zero=False)
    ax1.axvline(0.5374, color=COL['orange'], lw=2, label='majority 0.537')
    ax1.axvline(0.70, color=COL['red'], lw=2, ls='--', label='70% target')
    ax1.legend(fontsize=8, loc='lower right')
    annotate_bars(ax1, vals, fmt='{:.1f}', pct=True)
    ax1.set_title('모델이 majority baseline을 안정적으로 넘지 못함', fontsize=11, fontweight='bold')

    ax2 = fig.add_axes([0.56, 0.46, 0.38, 0.34])
    cx = np.arange(len(capacity))
    ax2.plot(cx, [x[1] for x in capacity], '-o', color=COL['red'], label='train acc')
    ax2.plot(cx, [x[2] for x in capacity], '-o', color=COL['blue'], label='test acc')
    ax2.axhline(0.5374, color=COL['orange'], lw=1.5, label='baseline')
    ax2.set_xticks(cx, [x[0] for x in capacity], rotation=35, ha='right')
    ax2.set_ylim(0.48, 1.02)
    ax2.grid(axis='y', color=COL['grid'])
    ax2.set_title('용량 증가는 memorization만 만든다', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)

    add_box(fig, 0.07, 0.16, 0.87, 0.18,
            '결론: next-day 방향성은 0.51~0.54 accuracy, MCC는 거의 0. Only volatility direction은 balanced target에서 0.544/MCC 0.088로 작은 skill을 보였지만 70%와는 거리가 멀다. 이 단계에서 “모델 크기 부족” 가설은 기각됐다.',
            '#EEF6ED', COL['green'], size=11.0)
    save_page(pdf, fig, '03_us_ceiling')

    # 4 mechanism
    fig = page('3. 왜 방향성 신호가 사라지는가: 가격 반영 시점', '진단 핵심: 뉴스는 가격과 같이 움직이지만, open 이후 drift가 없다.')
    ax1 = fig.add_axes([0.07, 0.43, 0.42, 0.37])
    labs = [x[0] for x in mechanism_corr]
    vals = [x[1] for x in mechanism_corr]
    barh(ax1, labs, vals, [COL['green'] if v > 0.02 else COL['grey'] for v in vals], xlabel='correlation', xlim=(-0.05, 0.25))
    annotate_bars(ax1, vals, fmt='{:.3f}', pct=False)
    ax1.set_title('sentiment/뉴스와 수익률의 관계', fontsize=11, fontweight='bold')

    ax2 = fig.add_axes([0.58, 0.43, 0.34, 0.37])
    ax2.bar([x[0] for x in mechanism_acc], [x[1]*100 for x in mechanism_acc], color=[COL['orange'], COL['grey']])
    ax2.axhline(50, color=COL['dark'], lw=1, ls='--')
    ax2.set_ylabel('directional accuracy (%)')
    ax2.set_ylim(45, 62)
    ax2.set_title('pre-open 뉴스는 gap에는 반영되나 open 이후는 random', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', color=COL['grid'])
    ax2.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate([x[1]*100 for x in mechanism_acc]):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

    add_box(fig, 0.07, 0.14, 0.87, 0.18,
            '해석: overnight/news event가 실제로 가격을 움직인다. 단, 그 수익은 open gap에 이미 들어가 있고, 우리가 open에서 진입하면 남은 open→close drift는 거의 0이다. 따라서 next-day close 방향성 예측은 본질적으로 signal이 아니라 timing 문제에 막힌다.',
            '#FFF7E6', COL['orange'], size=11.0)
    save_page(pdf, fig, '04_mechanism')

    # 5 KR bug audit
    fig = page('4. 한국시장 복제와 alignment-bug audit', 'UTC/세션 버킷/zero-event collapse 같은 실제 버그를 찾고 고친 뒤 재검증했다.')
    ax = fig.add_axes([0.07, 0.35, 0.86, 0.43])
    labels = [x[0] for x in kr_bug_audit]
    base = np.array([x[1] for x in kr_bug_audit])
    acc = np.array([x[2] for x in kr_bug_audit])
    yy = np.arange(len(labels))
    ax.barh(yy - 0.18, base, height=0.34, color=COL['grey'], alpha=0.45, label='base')
    ax.barh(yy + 0.18, acc, height=0.34, color=COL['blue'], alpha=0.9, label='acc')
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.49, 0.59)
    ax.grid(axis='x', color=COL['grid'])
    ax.legend(fontsize=9)
    ax.set_xlabel('accuracy')
    ax.set_title('버그 수정 후에도 next-day 방향성은 baseline 근처', fontsize=11, fontweight='bold')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    add_box(fig, 0.07, 0.10, 0.86, 0.17,
            '수정한 버그: (1) UTC calendar-day 배정, (2) overnight/pre-open news window 누락, (3) KR all-zero sample collapse. 결과는 개선된 것이 아니라 “기존 null 결과가 bug artifact가 아니었음”을 확인했다. best selective KR fresh top-10%도 n=154, p=0.285로 불충분.',
            COL['light'], COL['grey'], size=10.8)
    save_page(pdf, fig, '05_kr_audit')

    # 6 architecture / input
    fig = page('5. 타깃 전환: intraday high/low exceedance는 예측 가능했다', 'y_k = high(d) >= open(d)*(1+k) 또는 low(d) <= open(d)*(1-k). 이 타깃은 방향성보다 volatility/attention에 가깝다.')
    ax = fig.add_axes([0.07, 0.43, 0.52, 0.35])
    ks = ['2%', '3%', '5%']
    x = np.arange(len(ks))
    width = 0.18
    colors = [COL['grey'], COL['blue2'], COL['red'], COL['green']]
    for i, (name, vals) in enumerate(arch_mcc.items()):
        ax.bar(x + (i-1.5)*width, vals, width, label=name, color=colors[i])
    ax.axhline(0, color=COL['dark'], lw=0.8)
    ax.set_xticks(x, ks)
    ax.set_ylabel('test MCC')
    ax.set_title('s44: transformer 실패가 아니라 input 문제', fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=COL['grid'])
    ax.legend(fontsize=8, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    add_box(fig, 0.64, 0.45, 0.28, 0.30,
            'NEWS-TF는 MCC가 0 근처다. 하지만 같은 7M transformer에 vol token을 주면 k=2/3/5% MCC +0.304/+0.267/+0.209. 구조가 아니라 입력 정보(뉴스 embedding만으로 부족)가 병목이었다.',
            '#EAF2FC', COL['blue'], size=10.5, title='Architecture finding')
    add_box(fig, 0.07, 0.13, 0.86, 0.18,
            's43 capacity ladder도 같은 결론: 0.06M→7.15M으로 키워도 news-only test MCC는 0 근처. 반면 vol/gap 기반 baseline은 k=2/3/5에서 MCC +0.24/+0.20/+0.18로 안정적이었다.',
            '#EEF6ED', COL['green'], size=10.8)
    save_page(pdf, fig, '06_architecture')

    # 7 rare and full universe
    fig = page('6. UP/DN 2/3/5% rare-event 모델과 full-universe cross-section', 's46-s48: 양방향 rare labels, GBM/TF ensemble, full 626종목 daily selection.')
    ax1 = fig.add_axes([0.06, 0.46, 0.42, 0.34])
    labs = [x[0] for x in ens_mcc_s48]
    vals = [x[1] for x in ens_mcc_s48]
    ax1.bar(labs, vals, color=[COL['blue'], COL['blue'], COL['blue'], COL['red'], COL['red'], COL['red']])
    ax1.set_ylim(0, 0.46)
    ax1.set_ylabel('MCC')
    ax1.set_title('s48 small-cap event-window ENS MCC', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', color=COL['grid'])
    ax1.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(vals): ax1.text(i, v+0.012, f'{v:.3f}', ha='center', fontsize=8.5)

    ax2 = fig.add_axes([0.56, 0.46, 0.38, 0.34])
    labels = [x[0] for x in full_daily]
    hit = [x[1] for x in full_daily]
    base = [x[2] for x in full_daily]
    x = np.arange(len(labels))
    ax2.bar(x-0.18, base, width=0.36, color=COL['grey'], alpha=0.55, label='day-base')
    ax2.bar(x+0.18, hit, width=0.36, color=[COL['blue'], COL['blue2'], COL['red'], '#D88C7A'], label='selected hit')
    ax2.set_xticks(x, labels, rotation=20, ha='right')
    ax2.set_ylim(0, 0.78)
    ax2.set_ylabel('hit-rate')
    ax2.set_title('s48 full-universe daily cross-sectional picks', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', color=COL['grid'])
    ax2.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(hit): ax2.text(i+0.18, v+0.018, f'{v*100:.1f}%', ha='center', fontsize=8.5)

    add_box(fig, 0.06, 0.13, 0.88, 0.18,
            '여기까지는 “랭킹/분류 신호”의 성공이다. 특히 DN5 top-1/day 69.2%는 강하다. 그러나 hit-rate는 손익 함수가 아니다. 다음 단계(s49-s52)는 이 신호가 open에서 실제 돈으로 바뀌는지 별도로 검증했다.',
            '#FFF7E6', COL['orange'], size=11.0)
    save_page(pdf, fig, '07_rare_full')

    # 8 tradability daily/intraday
    fig = page('7. Tradability 1: daily-bar ambiguity와 5분봉 entry search', '질문: UP5 top picks를 open에 사서 +5% limit/stop으로 수익화할 수 있는가?')
    ax1 = fig.add_axes([0.07, 0.48, 0.38, 0.31])
    labels = [x[0] for x in trade_fill]
    vals = [x[1] for x in trade_fill]
    barh(ax1, labels, vals, [COL['red'], COL['green'], COL['red'], COL['orange']], xlabel='CAGR (%)', xlim=(-25, 15))
    annotate_bars(ax1, vals, fmt='{:.1f}', pct=False)
    ax1.set_title('s49/s50: fill assumption and 5-min reality', fontsize=11, fontweight='bold')

    ax2 = fig.add_axes([0.56, 0.42, 0.38, 0.37])
    labels = [x[0] for x in entries]
    sharpe = [x[2] for x in entries]
    barh(ax2, labels, sharpe, [COL['red'] if s < -1 else COL['orange'] for s in sharpe], xlabel='OOS Sharpe', xlim=(-6.0, 0.4))
    annotate_bars(ax2, sharpe, fmt='{:.2f}', pct=False)
    ax2.set_title('s51: 6개 realistic entry 모두 OOS 음수', fontsize=11, fontweight='bold')

    add_box(fig, 0.07, 0.12, 0.87, 0.18,
            's49 daily OHLC는 stop-first -19.8% vs TP-first +9.4%로 순서 가정이 결과를 지배했다. s50 5분봉은 first-touch를 해소했지만 OOS Sharpe -0.63. s51에서는 open, dip, delay, first-bar close, ORB, dip-reclaim 모두 음수였다. ORB는 TP-rate 0.51까지 올렸지만 더 높은 진입가/손실폭 때문에 Sharpe -2.97로 악화됐다.',
            '#FCEAE8', COL['red'], size=10.8)
    save_page(pdf, fig, '08_tradability')

    # 9 s52 exact
    fig = page('8. 사용자 지적 반영: s52 exact no-stop full-universe 검증', 'Stop 순서 가정 제거. 65종목 subset handicap 제거. Full 626종목에서 buy-open, +5% limit TP, 아니면 EOD/next-open exit.')
    ax1 = fig.add_axes([0.06, 0.44, 0.42, 0.34])
    labels = [x[0] for x in s52_configs]
    ev = [x[3] for x in s52_configs]
    barh(ax1, labels, ev, [COL['green'] if v > 0 else COL['red'] for v in ev], xlabel='EV/trade (%)', xlim=(-0.7, 0.35))
    ax1.axvspan(-0.61, 0.39, color=COL['blue2'], alpha=0.12, label='frozen CI range')
    annotate_bars(ax1, ev, fmt='{:.2f}', pct=False)
    ax1.set_title('EV surface: 모든 filter가 0 근처', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')

    ax2 = fig.add_axes([0.57, 0.44, 0.36, 0.34])
    x = np.arange(len(labels))
    hit = np.array([x[1] for x in s52_configs]) * 100
    miss = np.array([abs(x[2]) for x in s52_configs])
    ax2.plot(x, hit, '-o', color=COL['blue'], label='TP-touch hit (%)')
    ax2.plot(x, miss*10, '-o', color=COL['red'], label='|E[net|miss]| x10')
    ax2.set_xticks(x, labels, rotation=25, ha='right')
    ax2.set_ylabel('scaled value')
    ax2.set_title('hit-rate와 miss severity가 같이 움직임', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', color=COL['grid'])
    ax2.legend(fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)

    add_box(fig, 0.06, 0.12, 0.88, 0.20,
            '핵심 통계: no gap filter는 hit 57.1%지만 miss 평균 -7.03%라 EV -0.34%. gap<=2% frozen은 hit 49.1%, miss -4.74%, EV -0.11% [CI -0.61,+0.39]. gap-down only는 VAL +0.14%지만 튜닝에서도 +0.04%로 작고, gap<=-2%는 TUNE +0.15% → VAL -0.20%로 부호가 뒤집혔다. 즉 사용자가 지적한 “stop 때문에 Sharpe가 음수로 보였을 수 있음”은 맞았지만, stop을 제거해도 수익 edge는 재현되지 않았다.',
            '#FFF7E6', COL['orange'], size=10.5)
    save_page(pdf, fig, '09_s52_exact')

    # 10 quarterly and equity curves
    fig = page('9. Equity path와 regime stability', '손실은 한 구간의 사고가 아니라 여러 분기에서 반복되는 EV 0 근처-minus-cost 구조다.')
    ax1 = fig.add_axes([0.06, 0.50, 0.38, 0.29])
    qlabels = [x[0] for x in s52_quarters]
    qvals = [x[1] for x in s52_quarters]
    ax1.bar(qlabels, qvals, color=[COL['green'] if v > 0 else COL['red'] for v in qvals])
    ax1.axhline(0, color=COL['dark'], lw=0.9)
    ax1.set_ylabel('mean net/trade (%)')
    ax1.set_title('s52 VAL quarterly EV/trade', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', color=COL['grid'])
    ax1.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(qvals):
        ax1.text(i, v + (0.06 if v >= 0 else -0.12), f'{v:+.2f}', ha='center', fontsize=8.5)

    ax2 = fig.add_axes([0.52, 0.25, 0.42, 0.54])
    eqs = [
        ('s50 5-min open OOS', read_equity('kr50_equity_oos.csv')),
        ('s51 champion open OOS', read_equity('kr51_equity_champion.csv')),
        ('s52 exact no-stop VAL', read_equity('kr52_equity_val.csv', value_col='net')),
    ]
    for name, df in eqs:
        if df is None or df.empty:
            continue
        ax2.plot(df['date'], df['equity'], lw=1.6, label=name)
    ax2.axhline(1.0, color=COL['dark'], lw=0.8, ls='--')
    ax2.set_ylabel('equity')
    ax2.set_title('검증 equity curves', fontsize=11, fontweight='bold')
    ax2.grid(color=COL['grid'])
    ax2.legend(fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)

    add_box(fig, 0.06, 0.20, 0.38, 0.16,
            's52 frozen portfolio: CAGR -49.9%, Sharpe -0.29, MDD -84.6%, final 0.243. deploy 1/3은 MDD를 -38.3%로 줄이지만 EV/trade는 그대로 -0.11%다. sizing은 경로를 바꿀 뿐 기대값을 바꾸지 못한다.',
            '#FCEAE8', COL['red'], size=10.2)
    save_page(pdf, fig, '10_equity')

    # 11 Final decision and next work
    fig = page('10. 최종 판정과 다음 연구축', '통계적으로 놓쳤던 부분과 최종적으로 닫힌 부분을 분리한다.')
    add_box(fig, 0.06, 0.63, 0.40, 0.19,
            '놓쳤던 부분: daily-bar stop-first 가정과 65종목 5분봉 subset은 실제 신호를 과소평가했다. 이 지적은 타당했고 s52로 제거했다.',
            '#FFF7E6', COL['orange'], title='Corrected criticism')
    add_box(fig, 0.54, 0.63, 0.40, 0.19,
            '남은 사실: full-universe no-stop exact test에서도 EV는 0 근처다. hit-rate 상승분은 miss-day severity 상승분에 의해 상쇄된다.',
            '#EAF2FC', COL['blue'], title='Final statistic')
    y = 0.51
    y = add_wrapped(fig, 0.07, y,
        '운용 결론: 이 신호는 방향성 롱 edge라기보다 “다음날 크게 움직일 종목”을 고르는 volatility selector다. UP5와 DN5가 같은 고변동 이름에서 함께 강해지는 것도 이 해석과 일치한다.', width=96, size=12)
    y = add_wrapped(fig, 0.07, y - 0.03,
        '따라서 다음 실험이 의미 있으려면 payoff를 바꿔야 한다: (1) 옵션 IV/realized-vol spread, straddle/strangle 같은 volatility trade, (2) pre-open/tick/order-book/초단위 뉴스로 open 전에 잡는 latency trade, (3) DN5 short/hedged pair까지 포함한 symmetric execution. 현재 보유 데이터(일봉/부분 5분봉, 옵션 없음)에서는 directional long +5% TP로는 edge가 재현되지 않는다.', width=96, size=11.5)
    add_box(fig, 0.06, 0.08, 0.88, 0.18,
            '주요 산출물: FINAL_REPORT.md(393 lines), s52.log/s52b.log, kr52_equity_val.csv, s49/s50/s51 tradability logs, s48 full-universe signal log, s53_teb_survival.json, s53_teb_dev_threshold_curve.csv, s53_teb_test_threshold_curve.csv. 본 PDF는 이 파일들의 숫자만 사용해 생성됐다.',
            COL['light'], COL['grey'], size=10.5)
    save_page(pdf, fig, '11_final')

    # 12 s53 TEB-CNN survival / overfitting audit
    fig = page('11. TEB-CNN 뉴스-only 생존 검증: 과적합 audit', 's53는 기존 headline(+$1,260, raw p=0.234)이 seed·alignment·threshold 선택에 얼마나 민감했는지 마지막으로 닫는다.')
    ax1 = fig.add_axes([0.06, 0.47, 0.32, 0.32])
    seed_labels = [x[0] for x in s53_seed_rows]
    seed_profit = [x[2] for x in s53_seed_rows]
    ax1.bar(seed_labels, seed_profit, color=[COL['green'] if v > 0 else COL['red'] for v in seed_profit])
    ax1.axhline(0, color=COL['dark'], lw=0.9)
    ax1.set_ylabel('test profit ($)')
    ax1.set_title('4개 seed 결과: sign이 뒤집힌다', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', color=COL['grid'])
    ax1.spines[['top', 'right']].set_visible(False)
    for i, (_, mcc, profit) in enumerate(s53_seed_rows):
        ax1.text(i, profit + (180 if profit >= 0 else -220), f'{profit:+.0f}\nMCC {mcc:.4f}',
                 ha='center', va='bottom' if profit >= 0 else 'top', fontsize=8.3)

    add_box(fig, 0.42, 0.47, 0.24, 0.32,
            '기존 headline은 seed3-like 단일 run profit +$1,260, raw p=0.234였다.\n\n'
            f'4-seed ensemble TEST acc={s53_survival["ensemble_test_acc"]:.4f}, MCC={s53_survival["ensemble_test_mcc"]:.4f}, '
            f'always profit=+${s53_survival["always_profit"]:.2f}.\n\n'
            f'stored p-values만 표기: randomization p={s53_survival["always_randomization_p"]:.5f}, '
            f'bootstrap P(profit<=0)={s53_survival["always_bootstrap_p_le_0"]:.5f}, '
            f'Bonferroni(11)={s53_survival["bonferroni_p"]:.1f}.\n\n'
            f'alignment audit: s37 MCC {s53_survival["s37_original_mcc"]:.4f} → {s53_survival["s37_realigned_mcc"]:.4f} '
            f'/ fresh {s53_survival["s37_fresh_mcc"]:.4f}.',
            '#EAF2FC', COL['blue'], size=9.8, title='Selection-noise summary')

    ax2 = fig.add_axes([0.70, 0.47, 0.24, 0.32])
    comp_labels = [f'always\n{int(s53_survival["always_trades"])} trades', f'beta=0.70\n{int(s53_survival["beta_trades"])} trades']
    obs = np.array([s53_survival['always_profit'], s53_survival['beta_profit']])
    med = np.array([s53_survival['always_bootstrap_ci'][1], s53_survival['beta_bootstrap_ci'][1]])
    lo = np.array([med[0] - s53_survival['always_bootstrap_ci'][0], med[1] - s53_survival['beta_bootstrap_ci'][0]])
    hi = np.array([s53_survival['always_bootstrap_ci'][2] - med[0], s53_survival['beta_bootstrap_ci'][2] - med[1]])
    ypos = np.arange(len(comp_labels))
    ax2.errorbar(med, ypos, xerr=np.vstack([lo, hi]), fmt='o', color=COL['blue'], ecolor=COL['grey'], capsize=4, label='bootstrap 95% CI')
    ax2.scatter(obs, ypos, s=70, color=[COL['green'], COL['red']], zorder=3, label='observed profit')
    ax2.axvline(0, color=COL['dark'], lw=0.9, ls='--')
    ax2.set_yticks(ypos, comp_labels)
    ax2.set_xlabel('profit ($)')
    ax2.set_xlim(-8000, 8000)
    ax2.set_title('always vs beta filter', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', color=COL['grid'])
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=7.5, loc='lower right')
    ax2.text(5600, 0, f'P<=0 {s53_survival["always_bootstrap_p_le_0"]:.5f}', fontsize=8.3, color=COL['dark'], va='center')
    ax2.text(5600, 1, f'P<=0 {s53_survival["beta_bootstrap_p_le_0"]:.5f}', fontsize=8.3, color=COL['dark'], va='center')

    ax3 = fig.add_axes([0.06, 0.12, 0.40, 0.22])
    thr_labels = [f'{label}\nθ={thr:.2f}' for label, thr, _, _, _ in s53_threshold_checks]
    x = np.arange(len(thr_labels))
    dev_profit = [x[2] for x in s53_threshold_checks]
    test_profit = [x[3] for x in s53_threshold_checks]
    width = 0.32
    ax3.bar(x - width/2, dev_profit, width=width, color=COL['blue2'], label='DEV profit ($)')
    ax3.bar(x + width/2, test_profit, width=width, color=[COL['green'], COL['orange']], label='TEST profit ($)')
    ax3.axhline(0, color=COL['dark'], lw=0.9)
    ax3.set_xticks(x, thr_labels)
    ax3.set_ylabel('profit ($)')
    ax3.set_title('threshold validation: 0.50 frozen vs 0.57 ex-post', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', color=COL['grid'])
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.legend(fontsize=8, loc='upper right')
    for i, (_, _, _, profit, trades) in enumerate(s53_threshold_checks):
        ax3.text(i + width/2, profit + 140, f'{profit:+.0f}\n{trades} trades', ha='center', fontsize=8.1)
    ax3.text(1 + width/2, test_profit[1] + 700, 'test-peek\nnot honest', ha='center', fontsize=8.5, color=COL['red'], fontweight='bold')

    add_box(fig, 0.52, 0.10, 0.42, 0.24,
            '해석: 0.50은 DEV에서 고른 frozen threshold라 TEST +$202.38이 honest 검증이다. 반면 0.57의 +$3,192는 TEST 곡선을 보고 고른 ex-post 선택이라 그대로 쓰면 안 된다. '
            '특히 “test-peek corrected p≈0.12” 같은 수치는 저장 산출물에 남아 있지 않아 이 PDF에서는 의도적으로 쓰지 않았다. '
            '여기서는 stored p-value들(raw 0.234, randomization 0.35339, bootstrap P<=0 0.47126)만 사용한다. 결론적으로 TEB-CNN 뉴스-only 생존 가설은 연구/선택 overfit이며 robust tradable edge가 아니다.',
            '#FCEAE8', COL['red'], size=9.9, title='Audit conclusion')
    fig.text(0.06, 0.05, 'evidence: report.md, s37.out, s53_teb_survival.json, s53_teb_dev_threshold_curve.csv, s53_teb_test_threshold_curve.csv', fontsize=9.3, color=COL['grey'])
    save_page(pdf, fig, '12_s53_teb_survival')

print(f'WROTE {OUT}')
print(f'FIGDIR {FIGDIR}')
