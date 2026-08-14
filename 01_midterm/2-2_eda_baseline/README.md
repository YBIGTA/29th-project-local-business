# 2-2. EDA & 정형 피처 베이스라인

`product_month_labeled.parquet`을 검증하고, 리뷰 텍스트 없이 정형 피처만으로 `is_low_rating_surge`를 예측하는 베이스라인 모델을 만든다. 이 결과는 2-4에서 2-3의 NLP 피처를 추가한 모델과 비교할 기준점으로 사용된다.

## 폴더 구성

```
01_midterm/2-2_eda_baseline/
├── sanity_check.py       # 핸드오프 데이터 검증
├── eda.py                # EDA 8개 함수 (분포/시계열/타당성/VIF/신뢰도/이상치)
├── eda_baseline.ipynb    # eda.py 실행 노트북
├── baseline_model.ipynb  # 베이스라인 모델링 노트북
└── README.md
```

## 실행 순서

1. 의존성 설치 (`requirements.txt`에 이미 반영됨: `matplotlib`, `statsmodels`, `scikit-learn`, `lightgbm`,`scikit-learn`)
2. `sanity_check.py`를 먼저 실행해 데이터를 검증한다.
   ```
   python 01_midterm/2-2_eda_baseline/sanity_check.py
   ```
3. `eda_baseline.ipynb`를 셀 순서대로 실행해 EDA 결과를 확인한다.
4. `baseline_model.ipynb`를 셀 순서대로 실행해 베이스라인 모델을 학습·평가한다.

## 1. Sanity Check 결과

`product_month_labeled.parquet`의 신뢰성을 검증한다.

| 항목 | 결과 |
|---|---|
| 행 수 | 68,087행 (기대값과 일치) |
| 상품 수 | 6,613개 (기대값과 일치) |
| `parent_asin` + `year_month` 유일 키 | 중복 없음 |
| 필수 컬럼 | 전체 존재 |
| `next_*` 미래 정보 컬럼 | 5개 전부 존재 (모델 입력에서는 제외) |
| 라벨 비율 | 9.47% (양성 6,448건) |
| 결측치 | 50% 초과 컬럼 없음 |

전 항목 통과했으며, 데이터를 신뢰하고 다음 단계로 진행한다.

## 2. EDA 주요 발견

`eda_baseline.ipynb`에서 `eda.py`의 함수 8개를 순서대로 실행한 결과다.

- `review_count`는 롱테일 분포를 가진다 (중앙값 8, 최대 686). `avg_rating`은 4점대에 몰려 있고, `low_rating_ratio` 중앙값은 9.3%다.
- 평균 평점이 4.3~4.5로 비슷한 두 상품(`B002DUCEO0` 저평점 0% vs `B078BL25P3` 저평점 20%)을 실제로 확인해, 평균 평점만으로는 위험도를 구분할 수 없다는 계획서의 가설을 검증했다.
- 월별 라벨 비율은 4.5%~14.6% 사이에서 변동하며, 시간에 따라 점차 증가하는 추세를 보인다.
- `past_3m_low_rating_ratio`와 `next_low_rating_ratio`의 Pearson 상관계수는 0.575로, 과거 저평점 추세가 미래 위험을 예측하는 데 유의미한 신호임을 확인했다.
- 입력 피처 10개 중 다수가 서로 강하게 겹친다. `avg_rating`↔`low_rating_ratio`(corr -0.947), `review_count`↔`past_3m_review_count`(corr 0.940), `low_rating_count`↔`past_3m_low_rating_count`(corr 0.927)이며, VIF 기준으로도 `text_available_ratio`(435), `avg_rating`(303), `verified_purchase_ratio`(93)가 5를 크게 초과한다.
- Wilson score interval로 `reliability_ci_width`(데이터 신뢰도 피처)를 새로 만들었다. 리뷰 수가 적을수록 이 값이 커지며(리뷰 2개인 상품이 최대값 0.811에 몰려 있음), `review_count` 구간별 라벨 비율도 0-9구간 11.4% → 100+구간 1.2%로 뚜렷하게 감소해, 표본이 적은 상품의 "위험" 라벨은 통계적 근거가 얕다는 점을 확인했다.
- `text_available_ratio`는 평균 0.999로 사실상 상수에 가깝다. 이 데이터셋이 애초에 본문 있는 리뷰만 필터링된 결과이기 때문이며, 모델 입력 피처에서는 제외한다.
- `avg_rating == 5.0`인 행이 전체의 19.5%(13,280건)를 차지한다. `review_count≥100`이면서 `low_rating_ratio==0`인 14건은 전부 상품 `B0B3DB5HTC` 하나가 여러 달에 걸쳐 반복된 것으로, 별도 상품이 아니다.

## 3. 최종 피처셋

VIF·상관관계 검토 결과에 따라 아래 6개만 모델 입력으로 사용한다.

| 포함 | 제외 (사유) |
|---|---|
| `review_count` | `past_3m_review_count` (corr 0.940, 중복) |
| `low_rating_ratio` | `avg_rating`(corr -0.947), `low_rating_count`(정의상 중복) |
| `past_3m_low_rating_ratio` | `past_3m_low_rating_count` (corr 0.927, 중복) |
| `mean_helpful_vote` | `text_available_ratio` (분산 없음, 변별력 없음) |
| `verified_purchase_ratio` | |
| `reliability_ci_width` | |

## 4. 베이스라인 모델 결과

시간순으로 분할한다 (`train` < 2022-01, `valid` 2022-01~2022-08, `test` 2022-09~2023-02). 랜덤 split은 사용하지 않는다 — 미래 데이터가 학습에 섞이는 것을 방지하기 위함이다.

| 데이터 | 행 수 | 라벨 비율 |
|---|---|---|
| train | 54,813 | 8.76% |
| valid | 7,228 | 12.92% |
| test | 6,046 | 11.79% |

### 검증셋(valid) 성능

| 모델 | ROC-AUC | PR-AUC |
|---|---|---|
| 로지스틱회귀 (`class_weight="balanced"`, 표준화) | 0.601 | 0.154 |
| LightGBM (`scale_pos_weight` 적용) | 0.624 | 0.173 |

무작위 예측 시 PR-AUC는 양성 비율(약 0.13)에 수렴하므로, 두 모델 모두 무작위보다는 낫지만 절대적인 예측력은 약한 수준이다.

### 로지스틱회귀 계수 (표준화 후)

| 피처 | 계수 |
|---|---|
| `past_3m_low_rating_ratio` | 0.315 |
| `reliability_ci_width` | 0.311 |
| `mean_helpful_vote` | 0.019 |
| `low_rating_ratio` | 0.005 |
| `verified_purchase_ratio` | -0.031 |
| `review_count` | -0.275 |

`past_3m_low_rating_ratio`와 `reliability_ci_width`가 가장 큰 영향을 미치며, `low_rating_ratio`는 `past_3m_low_rating_ratio`와의 정보 중복으로 계수가 거의 0에 가깝다.

### 최종 테스트셋(test) 성능 (로지스틱회귀)

| ROC-AUC | PR-AUC |
|---|---|
| 0.616 | 0.145 |

### Walk-forward 반복검증

최종 test(2022-09~)는 재사용하지 않고, 2018년부터 6개월 단위로 검증 구간을 이동하며 9회 반복 검증했다.

| | 평균 | 표준편차 |
|---|---|---|
| ROC-AUC | 0.654 | 0.029 |
| PR-AUC | 0.136 | 0.019 |

9개 구간 모두 ROC-AUC 0.60~0.69 범위에 있어 특정 시기에만 우연히 성능이 나온 것은 아님을 확인했다. 다만 2018년(0.68~0.69)에서 2022년(0.60)으로 갈수록 성능이 점진적으로 하락하는 경향이 있으며, 같은 기간 라벨 비율도 7.9%에서 13.3%로 증가했다. 실제 서비스에 적용할 경우 주기적인 재학습을 고려해야 한다.

## 5. 결론

정형 피처만으로는 무작위보다는 나은 수준의 예측(PR-AUC 0.14~0.17)이 가능하지만, 절대적인 예측력은 제한적이다. 이는 리뷰 텍스트 기반 NLP 피처(2-3)가 왜 필요한지를 정량적으로 뒷받침하는 근거가 된다. 2-4에서 NLP 피처를 결합한 모델을 만들 때는, 본 문서의 시간순 split 기준과 최종 피처셋을 동일하게 적용해 성능 향상 여부를 공정하게 비교해야 한다.
