# Korea Inbound Tourism Demand Forecasting

**BUS964 Team 5** — Korea University Graduate School of Business

25개국 방한 외래관광객 수요 예측 시스템.  
V7~V10 앙상블 메타러닝 파이프라인을 통해 2025년 월별 국가별 방문객 수를 예측합니다.

---

## Model Version Mapping

This project was developed iteratively through multiple versions (V7–V10).

For presentation clarity, these versions are abstracted into Model 1–4.

* Model 1 ≈ V7 (Baseline ensemble with same feature space)
* Model 2 ≈ V8 (Information diversity-based ensemble)
* Model 3 ≈ V9 (OOF validation to prevent data leakage)
* Model 4 ≈ V10 (Final hybrid ensemble: blending, constrained meta, Bayesian update)

This mapping ensures consistency between implementation (code) and presentation (slides).

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
│   ├── run_v9.py           # V9: Rolling Window OOF Meta-Learning
│   └── run_v10.py          # V10: Hybrid Meta-Stacking (Ultimate)
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

| Version | Approach | Key Innovation | 2025 MAPE |
|---------|----------|----------------|-----------|
| **V7** | 3-Model A/B/C 독립 | Missing value 전략 3가지 (제거/0대체/보간) | ~19% |
| **V8** | Information Diversity Stacking | 완전히 다른 feature set으로 error 다양성 확보 (corr 0.99→0.61) | ~17% |
| **V9** | Rolling Window OOF | 정보 누출 제거 — expanding window cross-validation | ~18% |
| **V10** | Hybrid Meta-Stacking | V8 예측력 + V9 방법론 결합 (3 strategies) | **15.3%** |

### V10 Ultimate — 3 Strategies

1. **Prediction Blending**: V8 Ridge + V9 Simple Average를 50:50 결합
2. **Constrained Meta**: V9 OOF 가중치 범위 내에서 2024 데이터로 최적화
3. **Bayesian Update**: V9 OOF = Prior, V8 2024 성능 = Likelihood → Posterior

### V10 Performance Breakdown (2025)

MAPE 계산 기준: 25개국 × 12개월 country-month level

| Metric | Value |
|--------|-------|
| 단순 평균 MAPE | 16.8% |
| 방문객 가중 평균 MAPE | 13.5% |
| Grid Optimized MAPE (S1:60/S2:35/S3:5) | 15.3% |

**그룹별 성능:**

| Group | MAPE | 2025 Visitors | 비고 |
|-------|------|---------------|------|
| A2 Asia Surge | 11.0% | 3,637,964 | 가장 정확 |
| ETC | 11.3% | 1,308,834 | 합산 효과로 안정 |
| A1 Western Surge | 16.8% | 2,792,560 | 중간 |
| B Recovering | 17.3% | 10,629,507 | 최대 비중 (일본·중국 포함) |
| C Underperform | 29.7% | 567,697 | 소규모·불규칙 패턴 |

**국가별 Best / Worst:**

| Best 5 | MAPE | Worst 5 | MAPE |
|--------|------|---------|------|
| 인도네시아 | 7.0% | 아랍에미리트 | 40.2% |
| 몽골 | 9.3% | 말레이시아 | 29.5% |
| 카자흐스탄 | 9.7% | 멕시코 | 27.7% |
| 필리핀 | 10.0% | 태국 | 26.0% |
| 인도 | 10.8% | 러시아 | 23.0% |

> 모든 수치는 `outputs/v10_pred.csv`에서 직접 산출. Constrained Meta (MAPE-optimized weights) 기준.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline (순서대로)
cd src
python run_v7.py    # → outputs/v7_*.csv
python run_v8.py    # → outputs/v8_*.csv
python run_v9.py    # → outputs/v9_*.csv
python run_v10.py   # → outputs/v10_*.csv
```

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
