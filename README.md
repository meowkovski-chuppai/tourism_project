# Korea Inbound Tourism Demand Forecasting

**BUS964 Team 5** — Korea University Graduate School of Business

25개국 방한 외래관광객 수요 예측 시스템.  
V7~V9 앙상블 메타러닝 파이프라인을 통해 2025년 월별 국가별 방문객 수를 예측하고, 2026년 분기별 갱신 예측을 수행합니다.

---

## Project Structure

```
tourism_project/
├── src/
│   ├── loader.py          # 25개국 데이터 로딩 + ETC 국가 산출
│   ├── engineer.py         # 피처 엔지니어링 (Lag, MA, YoY, Tier2)
│   ├── models.py           # ML 모델 (GBM, Ridge, WeightedLag, Ensemble)
│   ├── run_v7.py           # V7: 3-Model (A/B/C) 독립 학습
│   ├── run_v8.py           # V8: Information Diversity Stacking
│   └── run_v9.py           # V9 Final: Bootstrap×10 + BayesianRidge(log) + 2026 Forecast
├── data/
│   ├── country/            # 25개국 관광 데이터 (한국관광데이터랩)
│   ├── processed/          # 통합 데이터셋
│   ├── exchange/           # USD/KRW 환율
│   ├── holiday/            # 지역별 행사/축제 데이터
│   ├── tourism/            # 입국자 통계 (xls)
│   └── weather/            # 기상 데이터
├── outputs/                # 모델 예측 결과 CSV
├── dashboards/             # 인터랙티브 HTML 대시보드
└── reports/                # 분석 보고서 (DOCX)
```

## Model Evolution

| Version | Approach | Key Innovation | 2025 OOF MAPE |
|---------|----------|----------------|---------------|
| **V7** | 3-Model A/B/C 독립 | Missing value 전략 3가지 (제거/0대체/보간) | ~19% |
| **V8** | Information Diversity Stacking | 완전히 다른 feature set으로 error 다양성 확보 (corr 0.99→0.61) | ~17% |
| **V9 Final** | Bootstrap×10 + BayesianRidge(log) | 정보 누출 제거 + bootstrap variance 감소 + log-space meta | **12.6%** |

### V9 Final — 최종 모델 아키텍처

```
Level 1 (내부 스태킹):
  Model A (과거 추세)  — 17개 Lag/MA/YoY 피처
  Model B (경제 환경)  — 21개 환율/유가/관심도 피처
  Model C (문화 트렌드) — 40개 한류/이미지/행사 피처
  각 모델: GBM + Ridge + WLag → 역RMSE 가중평균

Bootstrap: 10회 복원추출 → 10개 Stacked 모델 → 예측 평균 (variance 감소)

Level 2 (메타러너):
  BayesianRidge on log1p(predictions) → 최종 결합
  Coefficients: A=0.4415, B=0.0636, C=0.4947
```

### V9 Final Performance

| Metric | Value |
|--------|-------|
| OOF MAPE (2~12월, 보정 후) | 12.6% |
| Weighted MAPE | 10.2% |
| 총 예측 오차 (보정 후) | -0.13% (-24,867명) |
| 2026 1~2월 실제 검증 오차 | +1.3% (+35,812명) |
| 2026 1~2월 24국 MAPE | 14.5% |

**2025년 실적:**
- 총 예측: 18,911,695명 vs 실제: 18,936,562명 (오차 -0.13%)
- 25개국 중 18개국 MAPE 15% 이하, 21개국 20% 이하
- 2025년 1월 계엄 영향 제외 시 모델 자체 정확도: MAPE 13.8%

**2026년 예측:** 21,410,731명 (전년 대비 +13.1%)
- Dampened Rolling Prediction 적용 (alpha=0.7)
- 2026년 1~2월 실제 데이터로 검증 완료: 합산 오차 +1.3%

### 분기 갱신형 Rolling Forecast (실무 활용)

| 시점 | 방식 | 예측 구간 | 기대 정확도 |
|------|------|-----------|------------|
| 1월 (연초) | 2025 실제 기반 → 전체 예측 | 1~12월 | MAPE ~15% |
| 4월 | 1~3월 실제 교체 → 재예측 | 4~12월 | MAPE ~12% |
| 7월 | 1~6월 실제 교체 → 재예측 | 7~12월 | MAPE ~10% |
| 10월 | 1~9월 실제 교체 → 재예측 | 10~12월 | MAPE ~8% |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run V9 Final pipeline
cd src
python run_v9.py    # → outputs/v9_final_*.csv, outputs/v9_forecast_2026_dampened.csv
```

### Output Files

| File | Description |
|------|-------------|
| `v9_final_2025_predictions.csv` | 2025년 25개국×12개월 예측 (보정 전) |
| `v9_final_2025_corrected.csv` | 2025년 1월 계엄 보정 적용 |
| `v9_final_2024_test.csv` | 2024년 테스트셋 예측 |
| `v9_forecast_2026_dampened.csv` | 2026년 Dampened Rolling 예측 |

## Data Sources

| Source | Description |
|--------|-------------|
| 한국관광데이터랩 | 25개국 방한 외래관광객 추이, 한류 관심도, 여행 행태 |
| 한국관광공사 | 월별 입국자 통계 (2018~2025) |
| 한국은행 | USD/KRW 환율 |
| 기상청 | 월별 기온, 강수량 |
| 문화체육관광부 | 지역별 행사/축제 데이터 |

## Key Concepts

- **Information Diversity**: 같은 알고리즘 + 다른 전처리 ≠ 앙상블 효과. 완전히 다른 feature set이 핵심
- **Error Correlation**: V7 0.983 → V8 0.612 → V9 0.498 (낮을수록 앙상블 효과 ↑)
- **Information Leakage**: V8은 2024 test 정답으로 meta weight 학습 → V9 OOF로 해결
- **Bootstrap×10**: 복원추출로 sub-model variance 감소, 10회가 비용 대비 최적
- **BayesianRidge(log)**: log-space에서 메타러닝 → 소규모 국가 예측 개선
- **Dampened Rolling**: 2026 예측 시 lag에 예측값 70% + 작년 실제 30% 혼합 → 오차 누적 억제
- **COVID Lag Contamination**: 2020~2023 lag feature가 COVID 이상치 전파 → 구간 제외
- **Structural Break (THAAD)**: 2015~2017 중국 데이터 이상 → 학습 기간 2018~ 시작

## 25 Countries (4 Groups)

- **A1 Western Surge**: 미국, 영국, 프랑스, 독일, 캐나다, 호주, 멕시코, 튀르키예, 사우디, 카자흐스탄
- **A2 Asia Surge**: 싱가포르, 대만, 인도, 인도네시아, 필리핀, 몽골
- **B Recovering**: 일본, 중국, 베트남, 홍콩, 말레이시아
- **C Underperform**: 러시아, 태국, 아랍에미리트
- **ETC**: 나머지 국가 합산

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, openpyxl
- See `requirements.txt` for full list

## Team

BUS964 Team 5 — Korea University
