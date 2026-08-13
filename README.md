# Amazon 상품 리뷰 AI 분석 시스템

## 프로젝트 목표

Amazon Reviews 2023 데이터셋의 Appliances 카테고리를 이용해,
상품별 리뷰 텍스트와 별점 변화 신호로 다음 달 저평점 리뷰 급증 위험을 예측한다.

- 분석 단위: 상품(`parent_asin`) × 달력월(`year_month`)
- 예측 시점: t월 말
- 입력 정보: t월 말까지 작성된 리뷰와 이전 이력
- 예측 대상: t+1월의 저평점(1~2점) 리뷰 비율 및 급증 여부

핵심 질문은 다음과 같다.

> 별점·리뷰 수 등 정형 정보만 사용한 모델보다 리뷰 텍스트의 변화량을 함께 사용한 모델이
> 미래 상품 품질 악화를 더 정확히 예측할 수 있는가?

## 폴더 구조

```text
data/                  # raw, interim, processed 데이터 (원본 데이터는 GitHub에 올리지 않음)
docs/                  # 분석 기준, 데이터 사전, 발표 자료
src/                   # 재사용 가능한 전처리·학습 코드
tests/                 # 데이터·코드 검증

01_midterm/            # 중간발표 산출물
  2-1_data_label/      # 데이터 수집·정제 및 위험 라벨 정의
  2-2_eda_baseline/    # EDA 및 정형 피처 기반 베이스라인
  2-3_nlp_features/    # 리뷰 텍스트 기반 NLP 피처
  2-4_model_validation/# 모델 비교 및 시간순 검증

02_final/              # 최종발표 확장 산출물
  3-1_feature_expansion/
  3-2_model_improvement/
  3-3_interpretation/
  3-4_service_presentation/
