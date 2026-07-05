# Korean report outline for s53

## Sections
1. 연구 목표와 최종 판정
2. 실험 흐름 요약
3. 원래 목표(익일 방향성) 실패의 정량 근거
4. 한국시장 복제와 버그 감사
5. 과제 재정의(intraday high/low exceedance) 성과
6. 스케일링·아키텍처 분석
7. full-universe ranking skill
8. tradability 검증
9. 최종 exact no-stop 결과(s52)
10. 통계적 caveat와 최종 해석

## Must-include numbers
- OOS ceiling 0.54–0.56; majority baseline ≈0.54. (FINAL_REPORT.md:3-5)
- Next-day direction baseline 0.537, best honest OOS 0.51–0.54. (FINAL_REPORT.md:26-30)
- same-day corr +0.215 vs next-day corr -0.003. (FINAL_REPORT.md:39-40)
- KR next-day news acc 0.4995; flow+price full coverage ~0.53. (FINAL_REPORT.md:68-72)
- Best selective bug-audit cell 0.5844 vs 0.5584, n=154, p=0.285. (FINAL_REPORT.md:120-123)
- k=3 VOL-LR overnight acc 0.809, MCC +0.43; k=5 acc 0.880, MCC +0.45. (FINAL_REPORT.md:146-151)
- FUSED-TF MCC +0.304/+0.267/+0.209. (FINAL_REPORT.md:202-216)
- UP5 top-1/day 0.562 [0.534,0.590] vs base 0.075; DN5 top-1/day 0.692 [0.665,0.719] vs base 0.055. (FINAL_REPORT.md:268-275)
- s50 OOS CAGR -8.8%, Sharpe -0.63; least-bad k=1 CAGR -2.9%. (FINAL_REPORT.md:297-301)
- s52 frozen VAL: n=495, hit 0.491, EV/trade -0.11%, E[net|miss] -4.74%, CAGR -49.9%, Sharpe -0.29, MDD -84.6%, final 0.243. (s52.log:17-18)
- EV/trade 95% CI [-0.61%, +0.39%]; breakeven hit 50.2% vs observed 49.1%. (s52.log:31-32; FINAL_REPORT.md:357-360)
- s52b sign-flip warning: gap<=-2% TUNE +0.15% -> VAL -0.20%. (s52b.log:13-15,23-25)

## Tables
- 실험 단계별 질문·데이터·결론
- 원래 목표 baseline 대비 성능 상한
- 버그 감사 후 재실험 결과
- 재정의 과제 최고 성능 셀
- 스케일링/입력 아블레이션
- full-universe top-k hit-rate와 base/lift
- tradability 단계별 OOS 결과(s49→s52)
- s52b gap filter 민감도 표

## Figures
- 연구 전체 흐름도
- 원래 목표 vs 재정의 과제 성능 비교 막대그래프
- 뉴스가 시가에 반영되는 메커니즘 도식
- 모델 크기 증가 vs test MCC
- VOL-LR / VOL-MLP / NEWS-TF / FUSED-TF 비교
- exact no-stop EV surface

## Exact no-stop conclusion
- full-universe exact no-stop 테스트에서도 EV는 0 부근.
- ranking skill은 실재하지만 시가에서 이미 가격 반영이 끝나 open 이후 방향성 롱 전략으로는 수익화되지 않음.
- 핵심 근거: top-1 TP-touch 0.547, frozen VAL EV/trade -0.11%, CAGR -49.9%, Sharpe -0.29, EV CI [-0.61%, +0.39%].

## Caveats
- 0.846 수치는 selective artifact(n≈13, CI ±0.27). (FINAL_REPORT.md:7-11)
- 버그 수정 후 최고 selective cell도 p=0.285로 비유의. (FINAL_REPORT.md:120-123)
- 2025-08+ overnight 강세 셀은 긴 OOS에서 재현되지 않음. (FINAL_REPORT.md:190-193)
- FUSED-TF 우위는 일부가 nonlinear vol fit일 수 있음. (FINAL_REPORT.md:213-214)
- 일봉 tradability는 TP/SL 순서 미관측 때문에 단독 증거로 약함. (FINAL_REPORT.md:286-290)
- s52의 EV/Sharpe CI가 모두 0을 가로지름. (s52.log:31-32)
- gap-down positive EV는 재현되지 않음. (s52b.log:13-15,23-25)
- 65종목 subset은 full-universe 강도를 대표하지 않음(hit 0.340, EV/trade -0.23%). (s52.log:43)
