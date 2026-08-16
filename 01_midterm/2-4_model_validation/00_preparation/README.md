# 00. 공통 실험 환경 및 기준

## 목차

1. 목적과 핵심 질문
2. 공통 입력 데이터와 분석 단위
3. 예측 대상과 해석 범위
4. 시간 분할과 테스트 사용 원칙
5. 정형 피처 기준
6. 텍스트 피처 기준
7. 사용 금지 피처와 누수 방지
8. 공통 모델 설정
9. 공통 평가 지표
10. 결과 전달 형식과 합류 기준
11. A와 B의 작업 경계
12. 자동 검증 방법

## 1. 목적과 핵심 질문

2-4의 목적은 다음 질문에 시간순 검증으로 답하는 것이다.

> 상품의 현재 별점·리뷰 수·과거 저평점 추세만 사용하는 모델보다 리뷰 텍스트에서 얻은 불만 토픽과 변화량을 함께 사용하는 모델이 다음 달 저평점 리뷰 급증을 더 정확하게 예측하는가?

A와 B는 같은 행, 같은 라벨, 같은 시간 분할, 같은 기본 알고리즘과 같은 평가 지표를 사용한다. 서로 다른 작업 결과를 비교할 때 데이터와 평가 조건의 차이가 성능 차이로 오인되지 않도록 하기 위함이다.

## 2. 공통 입력 데이터와 분석 단위

| 항목 | 고정 기준 |
|---|---|
| 정형·라벨 패널 | `data/processed/product_month_labeled.parquet` |
| 텍스트 패널 | `data/processed/product_month_text_features.parquet` |
| 텍스트 피처 사전 | `01_midterm/2-3_nlp_features/text_feature_dictionary.csv` |
| 분석 단위 | `parent_asin × year_month` |
| 공통 키 | `parent_asin`, `year_month` |
| 전체 행 수 | 68,087행 |
| 고유 상품 수 | 6,613개 |
| 양성 수 | 6,448행 |
| 전체 양성 비율 | 약 9.47% |
| 분석 기간 | 2016-01부터 2023-02까지 |

`parent_asin`은 색상·용량 등 세부 변형을 대표 상품 단위로 묶는 식별자다. 하나의 행은 특정 상품의 특정 t월 말 상태를 의미한다.

정형 패널과 텍스트 패널은 공통 키의 중복이 각각 0건이어야 한다. 두 패널의 키 집합과 `is_low_rating_surge` 값도 완전히 같아야 한다.

## 3. 예측 대상과 해석 범위

예측 시점은 t월 말이다. 모델 입력에는 t월 말까지 작성된 리뷰와 그 이전 이력만 사용할 수 있다.

`is_low_rating_surge = 1`은 다음 조건을 모두 만족하는 경우다.

1. t+1월의 1~2점 리뷰 비율이 30% 이상이다.
2. t+1월 저평점 비율이 t-2, t-1, t월 전체의 저평점 비율보다 15%p 이상 높다.
3. t+1월 리뷰 수가 최소 5개다.
4. t-2~t월 리뷰 수가 최소 5개다.

현재 라벨 패널은 `next_review_count >= 5`인 행만 포함한다. 따라서 이번 결과는 다음 달에도 최소 5개 리뷰가 관측된 상품·월을 조건으로 한 예측 성능이다. 모든 신규 상품과 저활성 상품에 바로 적용 가능한 서비스 성능으로 해석하면 안 된다.

## 4. 시간 분할과 테스트 사용 원칙

| 구간 | 기간 | 행 수 | 용도 |
|---|---|---:|---|
| Train | 2016-01~2021-12 | 54,813 | 모델 학습과 클래스 가중치 계산 |
| Valid | 2022-01~2022-08 | 7,228 | 피처 조합·모델 후보·설정 선택 |
| Test | 2022-09~2023-02 | 6,046 | 최종 모델 고정 후 단 한 번 평가 |

랜덤 분할은 사용하지 않는다. 미래 리뷰가 과거 학습에 섞이면 실제 조기경보 상황보다 성능이 과대평가되기 때문이다.

2-3 텍스트 패널의 기존 `split` 열은 2020/2021 기준으로 만들어져 있으므로 2-4에서 사용하지 않는다. 모든 담당자는 `experiment_config.py`의 `split_2_4()`로 분할을 다시 계산한다.

A와 B는 1일 차와 2일 차에 Test 라벨이나 Test 성능을 확인하지 않는다. 2일 차 종료 시 최종 후보, 피처 목록, 모델 파라미터를 문서로 고정한 뒤 A의 3일 차 작업에서만 Test를 평가한다. Test 결과를 확인한 뒤 피처·파라미터·임계값을 변경하지 않는다.

## 5. 정형 피처 기준

### 5.1 현재 상태 기반 모델

현재 t월의 상태만 사용하는 정형 모델은 다음 5개를 사용한다.

- `review_count`: t월 리뷰 수
- `low_rating_ratio`: t월 1~2점 리뷰 비율
- `mean_helpful_vote`: t월 리뷰당 평균 도움 수
- `verified_purchase_ratio`: t월 구매 인증 리뷰 비율
- `reliability_ci_width`: 리뷰 수를 반영한 저평점 비율 신뢰구간 폭

### 5.2 시간 변화 기반 모델

현재 상태 피처 5개에 다음 피처를 추가한다.

- `past_3m_low_rating_ratio`: t-2, t-1, t월 전체의 저평점 리뷰 비율

현재 상태 모델과 시간 변화 모델을 같은 Train/Valid 행에서 비교하여 과거 기준선의 추가 가치를 측정한다.

### 5.3 B가 독립적으로 사용하는 최소 정형 기준

B는 A의 1일 차 결과를 기다리지 않고 다음 3개를 최소 정형 기준으로 사용할 수 있다.

- `review_count`
- `low_rating_ratio`
- `past_3m_low_rating_ratio`

이 최소 기준은 B의 텍스트 후보 선별을 위한 도구다. A가 확정한 정형 기준 모델을 대신하는 최종 모델이 아니다.

## 6. 텍스트 피처 기준

| 그룹 | 안전 피처 수 | 의미 |
|---|---:|---|
| `T_basic` | 4 | 리뷰 길이·문장 수·대문자·느낌표 |
| `T_reliability` | 6 | 텍스트 리뷰 수·저표본 여부·신뢰도 |
| `T_cause` | 33 | 내구성·제작 품질·호환·성능·누수·필터·배송 문제 |
| `T_consequence` | 25 | 반품·환불·보증·설명 불일치·구매 경고 |
| `T_narrative` | 21 | 고장 시점·감정·확신·비교 서사 |
| `T_cooc` | 16 | 원인 토픽과 결과 토픽의 동시 출현 |
| 합계 | 105 | 모델 결합 후보 |

현재 상태 피처는 주로 `_rate_t`, `_rate_t_shrunk`, `_mean`으로 끝난다. 변화 정보는 주로 `_rate_p3`, `_delta`로 끝난다.

텍스트의 `_rate_p3`는 t-3, t-2, t-1월 기준이다. 라벨의 `past_3m_low_rating_ratio`는 t-2, t-1, t월 기준이다. 두 값은 목적과 기간이 다르므로 같은 의미로 설명하면 안 된다.

정확한 안전 피처 목록은 자동 생성되는 `safe_text_features.csv`를 사용한다. 담당자가 임의로 열을 추가하지 않는다.

## 7. 사용 금지 피처와 누수 방지

다음 열은 어떤 모델 입력에도 사용할 수 없다.

- `is_low_rating_surge`
- `next_review_count`
- `next_avg_rating`
- `next_low_rating_count`
- `next_low_rating_ratio`
- `next_vs_past_3m_low_rating_change`
- `next_`로 시작하는 모든 열
- `family == X_duplicate_of_structured`인 텍스트 열
- `auto_title_flag_*` 열
- 2-3에서 생성된 기존 `split` 열

`X_duplicate_of_structured` 피처는 별점 기반 정형 정보를 텍스트 패널에 복제한 값이므로 텍스트의 추가 가치를 측정할 때 제외한다.

`auto_title_flag_*`는 `One Star`, `Five Stars` 같은 아마존 자동 생성 제목을 이용한다. 이는 별점의 문자열 사본이며 연도별 생성 방식도 달라 누수와 시기 편향을 만들 수 있으므로 제외한다.

## 8. 공통 모델 설정

기본 알고리즘은 LightGBM이며 후보 간 1차 비교에서는 다음 설정을 공통으로 사용한다.

| 설정 | 값 |
|---|---:|
| objective | binary |
| n_estimators | 500 |
| learning_rate | 0.03 |
| num_leaves | 15 |
| min_child_samples | 50 |
| subsample | 0.80 |
| `subsample_freq` | 1 |
| colsample_bytree | 0.80 |
| reg_lambda | 1.0 |
| random_state | 42 |

클래스 불균형 보정값 `scale_pos_weight`는 Train 구간의 음성 행 수를 양성 행 수로 나누어 계산한다. Valid나 Test의 라벨 비율을 사용하면 안 된다.

후보마다 별도의 대규모 튜닝을 수행하지 않는다. 동일한 기본 설정으로 피처 조합을 먼저 비교한 뒤 A의 2일 차에서 정형 기준 모델과 유력 결합 모델에만 제한된 최적화를 수행한다.

## 9. 공통 평가 지표

| 지표 | 계산 기준 |
|---|---|
| PR-AUC | `average_precision_score` |
| ROC-AUC | `roc_auc_score` |
| Recall@Top 5% | 예측 확률 상위 `ceil(N×0.05)`행에 포함된 전체 양성의 비율 |
| Recall@Top 10% | 예측 확률 상위 `ceil(N×0.10)`행에 포함된 전체 양성의 비율 |

확률이 같으면 `parent_asin`, `year_month` 오름차순으로 순위를 고정한다. 각 결과에는 평가 행 수, 양성 비율, 피처 수, random seed를 함께 기록한다.

PR-AUC를 주지표로 사용한다. 양성 비율이 약 9.47%인 불균형 문제이므로 ROC-AUC만으로는 실제 상위 위험 상품 탐지력을 충분히 설명하기 어렵기 때문이다.

## 10. 결과 전달 형식과 합류 기준

모든 성능 결과 CSV는 다음 열을 동일한 순서로 사용한다.

`model_name, split, pr_auc, roc_auc, recall_at_5pct, recall_at_10pct, n_rows, positive_rate, feature_count, random_state`

A와 B는 다음 값이 모두 일치할 때만 결과를 비교하고 다음 날 작업에 사용한다.

- 전체 68,087행
- 6,613개 상품
- 양성 6,448행, 약 9.47%
- 키 중복 0건
- Train 54,813행
- Valid 7,228행
- Test 6,046행
- 안전 텍스트 피처 105개

하나라도 다르면 모델 성능을 비교하지 않고 데이터 파일 버전, 키, 설정 파일, 실행 로그를 먼저 확인한다.

## 11. A와 B의 작업 경계

A의 결과는 `01_A_structured_timeseries`와 `03_A_combined_model_optimization`에 저장한다. B의 결과는 `02_B_text_feature_validation`과 `04_B_text_stability_analysis`에 저장한다. 최종 모델·Test 결과·통합 결과는 `05_A_final_test_integration`에 저장한다.

공통 기준은 `00_preparation`에서만 관리한다. 각 담당자가 자기 폴더에서 시간 분할·금지 피처·지표 정의를 별도로 변경하지 않는다.

## 12. 자동 검증 방법

다음 두 명령으로 설정 경계와 공통 입력을 검증한다.

`python 01_midterm/2-4_model_validation/00_preparation/test_experiment_config.py`

`python 01_midterm/2-4_model_validation/00_preparation/validate_common_input.py`

검증에 성공하면 `common_input_validation.json`과 `safe_text_features.csv`가 생성된다. 두 파일은 A와 B가 같은 데이터와 피처 기준을 사용한다는 인수인계 자료다.
