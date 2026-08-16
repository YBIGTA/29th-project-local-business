from pathlib import Path
import pandas as pd

base = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")

valid = pd.read_csv(base / "multiwindow_validation_metrics.csv")
rolling = pd.read_csv(base / "rolling_long_candidate_summary.csv")
importance = pd.read_csv(base / "multiwindow_feature_importance.csv")

baseline_valid = valid.query("model_name == 'baseline_existing_3m'").iloc[0]
long_valid = valid.query("model_name == 'candidate_long_1_3_6_12m'").iloc[0]

baseline_roll = rolling.query("model_name == 'baseline_existing_3m'").iloc[0]
long_roll = rolling.query("model_name == 'candidate_long_1_3_6_12m'").iloc[0]

valid_gain = (long_valid.pr_auc / baseline_valid.pr_auc - 1) * 100
rolling_gain = (long_roll.pr_auc_mean / baseline_roll.pr_auc_mean - 1) * 100

top_features = (
    importance.query("model_name == 'candidate_long_1_3_6_12m'")
    .sort_values("importance_gain", ascending=False)
    .head(10)["feature_name"]
    .tolist()
)
top_feature_text = "\n".join(f"- `{feature}`" for feature in top_features)

readme = f"""# A1 정형 시계열 모델링 기반

## 목차

1. [목적](#1-목적)
2. [데이터와 누수 방지](#2-데이터와-누수-방지)
3. [피처 구성](#3-피처-구성)
4. [시간 분할과 평가 기준](#4-시간-분할과-평가-기준)
5. [후보 모델 비교](#5-후보-모델-비교)
6. [최종 정형 기준 모델](#6-최종-정형-기준-모델)
7. [해석과 주의사항](#7-해석과-주의사항)
8. [산출물](#8-산출물)

## 1. 목적

상품×월 시점의 정형 정보만 이용해 다음 달 저평점 급증
`is_low_rating_surge`를 예측하는 기준 모델을 구축한다.

이 결과는 이후 텍스트 피처의 추가 가치와 정형+텍스트 결합 모델을 비교하는 공통 기준이다.

## 2. 데이터와 누수 방지

- 분석 데이터: 68,087 상품×월 행, 6,613개 상품
- 공통 키: `parent_asin`, `year_month`
- 라벨: `is_low_rating_surge`
- 전체 양성 비율: 9.47%
- 최종 Test 기간: 2022-09~2023-02

모든 A1 학습·검증에서 Test 기간은 제외했다. 다음은 모델 입력에서 제외했다.

- 라벨 및 미래 결과: `is_low_rating_surge`, `next_*`
- 텍스트 피처 105개: B 트랙에서 별도 검증
- 제목 기반 자동 라벨 위험 피처: `auto_title_*`
- 식별자·시간 식별값·상수 컬럼

## 3. 피처 구성

| 구분 | 피처 수 | 내용 |
|---|---:|---|
| 현재 상태 | 7 | 현재 평점, 저평점 비율·수, 리뷰 수, 인증구매 비율 등 |
| 기존 시계열 기준 | 3 | 직전 3개월 저평점 비율·수, 리뷰 수 |
| 다중 기간 이력 | 38 | 직전 1·3·6·12개월 품질·리뷰 집계, 이력 충족도, 장단기 차이 |
| 최종 정형 후보 | 45 | 현재 상태 7개 + 다중 기간 이력 38개 |

다중 기간 이력은 예측 시점 t월을 제외한 과거 달만 사용했다.

## 4. 시간 분할과 평가 기준

| 구간 | 기간 | 행 수 |
|---|---|---:|
| Train | 2016-01~2021-12 | 54,813 |
| Valid | 2022-01~2022-08 | 7,228 |
| Test | 2022-09~2023-02 | 6,046 |

- 기본 알고리즘: LightGBM
- `scale_pos_weight`: Train 음성 수 ÷ 양성 수 = 10.417
- 평가 지표: PR-AUC, ROC-AUC, Recall@Top 5%, Recall@Top 10%
- 동률 확률은 `parent_asin`, `year_month` 오름차순으로 순위를 고정한다.

## 5. 후보 모델 비교

### 공식 Valid 결과

| 모델 | 피처 수 | PR-AUC | ROC-AUC | Recall@Top 5% | Recall@Top 10% |
|---|---:|---:|---:|---:|---:|
| 기존 3개월 기준 | {int(baseline_valid.feature_count)} | {baseline_valid.pr_auc:.4f} | {baseline_valid.roc_auc:.4f} | {baseline_valid.recall_at_5pct:.2%} | {baseline_valid.recall_at_10pct:.2%} |
| 장기 1·3·6·12개월 후보 | {int(long_valid.feature_count)} | **{long_valid.pr_auc:.4f}** | **{long_valid.roc_auc:.4f}** | **{long_valid.recall_at_5pct:.2%}** | **{long_valid.recall_at_10pct:.2%}** |

장기 후보는 기존 3개월 기준 대비 Valid PR-AUC가 **{valid_gain:.1f}%** 높았다.

### 롤링 시간 검증

2020~2022년의 서로 다른 네 검증 시점에서, 각 시점 이전 데이터만 학습에 사용했다.

| 모델 | 평균 PR-AUC | PR-AUC 표준편차 | 평균 ROC-AUC | 평균 Recall@Top 10% | PR-AUC 승리 |
|---|---:|---:|---:|---:|---:|
| 기존 3개월 기준 | {baseline_roll.pr_auc_mean:.4f} | {baseline_roll.pr_auc_std:.4f} | {baseline_roll.roc_auc_mean:.4f} | {baseline_roll.recall_at_10pct_mean:.2%} | {int(baseline_roll.pr_auc_wins)}/4 |
| 장기 1·3·6·12개월 후보 | **{long_roll.pr_auc_mean:.4f}** | {long_roll.pr_auc_std:.4f} | **{long_roll.roc_auc_mean:.4f}** | **{long_roll.recall_at_10pct_mean:.2%}** | **{int(long_roll.pr_auc_wins)}/4** |

장기 후보는 네 시점 모두에서 이겼고, 평균 PR-AUC가 **{rolling_gain:.1f}%** 높았다.

## 6. 최종 정형 기준 모델

A1의 정형 기준 모델은 아래로 확정한다.

> `candidate_long_1_3_6_12m`  
> LightGBM + 현재 상태 7개 + 직전 1·3·6·12개월 이력 38개 = 총 45개 피처

이 모델은 A2에서 텍스트 피처 결합 성능을 비교할 기준선으로 사용한다.
최종 Test 평가는 아직 수행하지 않았으며, 3일 차에 모델·피처를 고정한 뒤 한 번만 수행한다.

## 7. 해석과 주의사항

### 주요 피처

{top_feature_text}

- 최근 저평점 변화뿐 아니라 장기 평균 평점·저평점 비율이 유효했다.
- 장기적 품질 수준과 최근 리뷰 흐름을 함께 볼 때 다음 달 위험을 가장 안정적으로 예측했다.
- 피처 중요도는 예측 기여도이며 인과관계를 뜻하지 않는다.
- Train 양성 비율은 8.76%, Valid 양성 비율은 12.92%로 시간에 따른 위험 분포 변화가 있다.
- 따라서 단일 Valid 결과뿐 아니라 롤링 시간 검증 결과를 함께 사용했다.

## 8. 산출물

- `structured_modeling_panel.parquet`: 기존 정형 기준 패널
- `structured_feature_spec.csv`: 기존 정형 피처 정의
- `multiwindow_history_features_pretest.parquet`: 1·3·6·12개월 이력 피처
- `multiwindow_history_feature_spec.csv`: 다중 기간 피처 정의
- `validation_metrics.csv`, `validation_predictions.csv`: 초기 정형 기준 결과
- `rolling_validation_metrics.csv`: 기존 3개월 모델 롤링 검증
- `multiwindow_validation_metrics.csv`, `multiwindow_validation_predictions.csv`: 기간별 후보 비교
- `rolling_long_candidate_metrics.csv`: 장기 후보 롤링 검증
- `models/`: A1 후보 모델 파일
"""

handoff = f"""# A1 인수인계 — A2/B2용

## 확정 기준 모델

- 모델명: `candidate_long_1_3_6_12m`
- 알고리즘: LightGBM
- 입력: 정형 피처 45개
- 공식 Valid PR-AUC: `{long_valid.pr_auc:.6f}`
- 롤링 평균 PR-AUC: `{long_roll.pr_auc_mean:.6f}`
- Test 데이터 사용 여부: 사용하지 않음

## A2에서의 비교 기준

A2의 모든 텍스트 결합 후보는 아래 정형 모델과 동일한 시간 분할에서 비교한다.

- 기준: `candidate_long_1_3_6_12m`
- Train: 2016-01~2021-12
- Valid: 2022-01~2022-08
- Test: 2022-09~2023-02, 최종 단계 전까지 사용 금지
- 주지표: PR-AUC
- 보조지표: ROC-AUC, Recall@Top 5%, Recall@Top 10%

## 결합 우선순위

1. 기준 정형 45개 + B의 최우선 텍스트 후보
2. 기준 정형 45개 + 텍스트 후보별 비교
3. 내부 시간 검증을 통과한 최대 2개 후보만 유지
4. 최종 Test 전에는 피처·설정을 고정한다.

## 주요 파일

- 기준 모델 Valid 예측값: `multiwindow_validation_predictions.csv`
- 기준 모델 성능: `multiwindow_validation_metrics.csv`
- 롤링 검증: `rolling_long_candidate_metrics.csv`
- 정형 피처 정의: `multiwindow_history_feature_spec.csv`
"""

(base / "README.md").write_text(readme, encoding="utf-8")
(base / "A1_HANDOFF.md").write_text(handoff, encoding="utf-8")

print("README.md 작성 완료")
print("A1_HANDOFF.md 작성 완료")
