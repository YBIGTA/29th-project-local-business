# A1 인수인계 — A2/B2용

## 확정 기준 모델

- 모델명: `candidate_long_1_3_6_12m`
- 알고리즘: LightGBM
- 입력: 정형 피처 45개
- 공식 Valid PR-AUC: `0.231104`
- 롤링 평균 PR-AUC: `0.218684`
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
