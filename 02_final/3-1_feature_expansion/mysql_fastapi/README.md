# 3-1. MySQL 스키마 설계 및 FastAPI 백엔드

## 배경 및 목표

중간발표(2-1~2-4)에서 생성된 상품·월 단위 분석 결과(정형 피처, NLP 토픽 피처, MIL 예측값, 위험도·신뢰도)는
CSV/parquet 여러 개로 흩어져 있어, 상품 검색·기간 비교·위험 상품 필터링·RAG 근거 추출 시마다
파일을 다시 읽고 결합해야 하는 문제가 있었다. 이를 해결하기 위해 서비스용 데이터 구조로 정리하여
MySQL에 적재하고, 대시보드(3-3)와 RAG(3-2)가 공통으로 사용할 FastAPI를 구축했다.

공통 식별자는 `parent_asin`(상품번호) + `year_month`(연-월)이다.

## 데이터 소스

아래 파일들은 용량 문제로 Git에 포함되지 않아 별도로 전달받았다. (`product_month_risk_confidence.parquet`만 Git에 존재)

| 파일 | 용도 |
|---|---|
| `products_meta_clean.parquet` | 상품 기본 정보 (94,327행) |
| `reviews_clean.parquet` | 정제된 리뷰 원문 (1,814,360행) |
| `product_month_text_features.parquet` | 상품×월 텍스트 피처, 토픽 12개 (68,087행 × 117열) |
| `product_month_review_volume_2m_labeled.parquet` | 상품×월 정형 피처, 1/3/6/12개월 이력 포함 (24,942행 × 56열) |
| `product_month_risk_confidence.parquet` | 상품×월 위험도·신뢰도 (3,563행, Test 기간 샘플) |
| `text_feature_dictionary.csv` (2-3) | 피처 컬럼 설명 (참고용) |
| `complaint_dictionary.json` (2-3 원본 재사용) | 불만 토픽 12개(원인 7 + 결과 5)별 매칭 키워드 사전 |
| `matching_rules.py` (2-3 원본 재사용) | 부정문·다의어 오탐을 거르는 매칭 규칙 |
| `standard_candidate_importance.csv` (2-4) | S45 모델의 전역 피처 중요도 (risk_explanations 임시 채움용) |

## 파이프라인 (`load_data_v2.py`)

1. **products** 적재 — 상품 메타데이터 그대로 적재 (categories는 JSON 문자열로 저장)
2. **product_month_metrics** 적재 — volume 파일에서 핵심 정형 지표 + 1/3/6/12개월 이력 델타 선별
3. **product_month_risk** 적재 — risk_confidence 파일 그대로 적재, `model_version` 컬럼 추가(`s45_mil_v1` 고정)
4. **product_month_topics** 적재 — 토픽 12개가 컬럼으로 펼쳐진 wide format을 상품×월×토픽 단위의 long format으로 melt.
   원본에 저장되지 않은 토픽별 언급 수(`mention_count`)는 `rate_t × 해당월 리뷰수`로 역산하여 복원
5. **evidence_reviews** 적재 — 전체 리뷰 중 1~2점 저평점만 선별(284,775행). 리뷰 텍스트에 `matching_rules.py`(부정문/다의어 처리)로
   `complaint_dictionary.json`의 12개 토픽 키워드를 매칭해 `related_topics` 컬럼에 태깅
6. **risk_explanations** 적재 — 상품별 SHAP 결과가 아직 없어(3-3 진행 전), S45 모델의 전역 피처 중요도 Top 10을
   모든 상품×월에 동일하게 채움 (임시 대체, 아래 한계 참고)

전 과정에서 `products`에 존재하지 않는 `parent_asin`은 FK 제약 위반을 막기 위해 사전에 필터링한다.

## 데이터베이스 스키마 (`schema_v2.sql`)

| 테이블 | 주요 내용 | 기본 키 |
|---|---|---|
| `products` | 상품명, 브랜드, 카테고리 | `parent_asin` |
| `product_month_metrics` | 리뷰 수, 평균 별점, 저평점 비율, 1/3/6/12개월 이력 및 변화량 | `parent_asin`, `year_month` |
| `product_month_risk` | 위험도, 기준 모델 점수, MIL 결합 점수, 운영/증거/일관성 신뢰도, 모델 버전 | `parent_asin`, `year_month`, `model_version` |
| `product_month_topics` | 토픽별 언급 수·비율(t/p3)·변화량·축소추정값 | `parent_asin`, `year_month`, `topic_name` |
| `evidence_reviews` | 리뷰 원문, 별점, 작성일, 관련 토픽, 도움됨 수 | `review_id` |
| `risk_explanations` | 위험 판단 근거 변수와 기여도 | `parent_asin`, `year_month`, `feature_name` |

> `year_month`는 MySQL 예약어(INTERVAL 구문)와 충돌하여 전 테이블에서 백틱(`` ` ``)으로 감쌌다.

### 적재 결과

| 테이블 | 행 수 |
|---|---|
| products | 94,327 |
| product_month_metrics | 24,942 |
| product_month_risk | 3,563 |
| product_month_topics | 817,044 |
| evidence_reviews | 284,775 |
| risk_explanations | 35,630 |

## FastAPI (`main.py`)

| 엔드포인트 | 역할 |
|---|---|
| `GET /health` | 서버·MySQL 연결 상태 확인 |
| `GET /products/search?q=` | 상품명 또는 parent_asin으로 검색 |
| `GET /products/{asin}/summary` | 기본정보 + 최신(또는 지정) 월 위험도·신뢰도·정형지표 통합 조회 |
| `GET /products/{asin}/trend` | 월별 위험도·리뷰수·평점·저평점비율 추이 |
| `GET /products/risky?sort_by=` | 위험 상품 목록. `risk`(위험도순) / `confidence_weighted`(위험도×신뢰도순) 지원 |
| `GET /products/{asin}/topics` | 불만 토픽별 언급수·비율·변화량 (변화량 내림차순 정렬) |
| `GET /products/{asin}/evidence-reviews` | 위험 근거 리뷰. 별점(`max_rating`), 기간(`year_month`), 토픽(`topic`) 필터 지원 |
| `GET /products/{asin}/risk-explanation` | 위험 판단 근거 변수·기여도 (현재는 전역 중요도, 아래 한계 참고) |

로컬 실행: `uvicorn main:app --reload` 이후 `localhost:8000/docs`에서 Swagger UI로 전체 엔드포인트 테스트 가능.

전 엔드포인트는 실제 요청/응답으로 검증을 완료했다 (예: `B0737B6HGR` 기준 위험도 86.7%, 12개월 평균 평점 대비 저평점 비율 상승 확인, 근거 리뷰 정렬·토픽 매칭 정상 동작).

## 한계 및 후속 작업

- **가격 정보 없음**: `products_meta_clean.parquet`에 가격 컬럼 자체가 없어 `products` 테이블에 미포함. 별도 소스 확인 필요.
- **`product_month_risk`는 전체 상품이 아닌 Test 기간 샘플(3,563건)만 존재**: 명세에서도 "DB·API 연결 및 시연 검증용"으로 명시된 부분으로, 전체 9만여 상품에 대한 운영 스코어링은 별도 작업 필요.
- **`risk_explanations`는 상품별 SHAP이 아닌 S45 모델 전역 피처 중요도**: 모든 상품×월에 동일한 Top 10 변수·기여도가 들어있다. 3-3에서 실제 SHAP 결과가 나오면 해당 로직으로 교체해야 하며, API 응답에는 이를 알리는 `note` 필드를 포함해두었다.
- **`related_topics` 매칭은 사전 등록된 12개 토픽 키워드 범위 내에서만 동작**: 사전에 없는 표현(예: "flimsy", "obscure")을 쓴 불만 리뷰는 `related_topics`가 `null`로 남는다. 필요 시 `complaint_dictionary.json`(2-3 산출물) 자체의 키워드 확장이 필요하며, 이는 2-3 담당자와 협의가 필요하다.
