# 2-3 NLP 텍스트 피처

Amazon Reviews 2023 Appliances 카테고리의 리뷰 텍스트를 `parent_asin × year_month` 단위 수치 피처로 바꾼다.
2-1이 만든 라벨 패널과 같은 키로 결합해, 정형 피처만 쓴 모델 대비 텍스트의 기여를 2-4가 측정할 수 있게 하는 것이 목표다.

> 파트 번호: 2-1 데이터 라벨링 / 2-2 EDA·정형 피처 / 2-3 NLP 피처(이 폴더) / 2-4 모델링·설명가능성

---

## 실행

2-1 파이프라인이 먼저 실행되어 `data/processed/`에 `reviews_clean.parquet`와 `product_month_labeled.parquet`이 있어야 한다.

```bash
python3 src/nlp/run_all.py            # 자동 단계 전체
python3 src/nlp/run_all.py --list     # 단계 목록
python3 src/nlp/run_all.py --from L1  # 중간부터
```

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| L0 | `build_clean_reviews.py` | 정규화, 축약형 전개, 오타 교정, 자동 생성 제목 판별 |
| L1 | `build_nlp_features.py` | 리뷰 단위 토픽·서사 피처 + 상품×월 집계 |
| L2 | `build_surge_summary.py` | 토픽별 급증 신호, 근거 리뷰 추출 |
| L3 | `build_quality_sample.py` | 사전 정밀도 검수용 표본 300건 |
| L4 | `build_quality_report.py` | 검수 결과 집계 (사람이 라벨 후 실행) |

보조 모듈은 두 개다. `config.py`는 경로와 2-1 기준값을 담고, `matching_rules.py`는 사전 매칭의 오탐을 걸러낸다. 불만 표현 목록은 `complaint_dictionary.json`에 있다.

## 산출물

| 파일 | 위치 |
|---|---|
| `clean_reviews.parquet` | `data/interim/` |
| `review_nlp_features.parquet` | `data/interim/` |
| **`product_month_text_features.parquet`** | `data/processed/` |
| `text_feature_dictionary.csv` | `01_midterm/2-3_nlp_features/` |
| `monthly_complaint_surge.csv`, `surge_topic_summary.csv` | 〃 |
| `evidence_reviews.csv` | 〃 |
| `topic_quality_sample.csv`, `topic_quality_report.csv` | 〃 |

`data/` 이하는 `.gitignore` 대상이며 위 코드로 재생성한다.

---

## 설계 결정

**조인 키는 `parent_asin`이다.** 계획서에는 `asin`으로 되어 있으나 2-1 구현이 `parent_asin`을 쓴다. 색상·용량 변형이 하나의 상품으로 묶인다.

**관측 창은 달력월이다.** 계획서의 30/60/90일 롤링 윈도우 대신 t월과 직전 3개월을 쓴다. 2-1 패널이 달력월 단위라 다른 창을 쓰면 결합 시점이 어긋난다.

**기준값은 `config.py`에 상수로 두고 실행 시 assert로 확인한다.** 결과가 우연히 맞는 것이 아니라 코드가 기준을 지키도록 했다.

**정답 정보 차단.** `next_`로 시작하는 열은 결합 단계에서 제외하고, 남아 있지 않은지 assert로 확인한다. t월 피처에는 t월 말까지의 리뷰만 들어간다.

**토픽 체계는 원인 7 + 결과 5다.** 계획서는 발열·파손 중심이었으나 train 구간 lift 상위는 반품·CS·기만 어휘가 압도적이었다. 원인은 무엇이 잘못됐는지, 결과는 그래서 무슨 일이 벌어졌는지를 나눈다.

- 원인: 내구성 고장, 제작 품질, 호환, 성능, 누수, 물맛·필터, 배송·포장
- 결과: 반품 의사, 반품 차단, 환불·보증·CS, 설명 불일치, 구매 경고

**순수 감정어는 사전에서 뺐다.** lift는 최상위지만 불만의 종류를 알려주지 않아, 넣으면 12개 토픽이 전부 감정 강도의 사본이 된다. 서사 피처로 따로 다룬다.

**시간 표현은 서사 피처다.** 월 이름이 lift 상위 100위에 11개나 올라왔는데 이는 노이즈가 아니라 "6월엔 됐는데 9월엔 죽었다" 같은 고장 시점 서사다. `time_to_failure_days`, `early_failure_flag`로 분리했다.

**피처마다 `family` 태그를 붙였다.** 2-4가 정형만 → +기본텍스트 → +원인토픽 → +결과토픽 → +서사 순으로 단계별 비교를 할 수 있게 하기 위함이다. `X_duplicate_of_structured`는 2-1 정형 피처와 값이 같아 모델 입력에서 빼야 하는 열이다.

**작은 표본 보정.** 리뷰가 적은 달의 비율이 튀지 않도록 `_rate_t_shrunk`(전체 평균 쪽으로 당긴 값)와 `text_reliability`를 함께 제공한다. 대시보드에서는 리뷰 수를 반드시 병기해야 한다.

---

## 분석 중 확인된 사항

**아마존 자동 생성 제목 오염.** "One Star", "Five Stars" 같은 제목이 2016~2017년 리뷰의 약 38%를 차지하다가 2018년 중반 이후 사실상 0%로 사라진다. 별점의 문자열 사본이라 텍스트 피처로 쓰면 정답을 미리 보는 것이고, 시기별 분포가 달라 시간순 검증도 왜곡한다. 본문에서 제외하되 `auto_title_flag`로 남겨 2-4가 처리 방식을 정하게 했다.

**동시출현이 단일 토픽보다 강하다.** 내구성 고장과 반품 차단이 한 리뷰에 함께 나오는 경우, 즉 "반품 기간이 지난 뒤 고장났다"는 서사가 단순 고장 언급보다 강한 위험 신호였다.

**lift만 보고 사전을 만들면 안 된다.** 확장 후보 1위가 특정 제조사 브랜드명이었다. 채택했다면 그 브랜드의 모든 리뷰가 CS 불만으로 잡혔을 것이다.

**사전 검수에서 오탐 유형 두 가지를 찾아 규칙으로 처리했다.** (`matching_rules.py`)

1. 부정 문맥 — `no leaks`가 누수 불만으로 잡히던 문제. 다만 부정어를 일률적으로 무시하면 안 된다. `not fit`, `not work`는 부정될 때 비로소 불만이 되기 때문이다. 그래서 부정되면 취소되는 표현만 따로 나열했다.
2. 다의어 — `window`가 반품 기간과 창문 두 뜻으로 쓰이는 문제. 단어를 빼면 진짜 신호도 잃으므로, 근처에 반품 문맥어가 있을 때만 인정한다.

**재측정 시 주의.** 사전을 고친 뒤 정밀도를 다시 잴 때는 `build_quality_sample.py`의 `SEED`를 바꿔 새 표본을 뽑아야 한다. 사전을 고치는 데 쓴 표본으로 다시 재면 정밀도가 부풀려진다.

---

## 다음 파트 안내

- 예측 대상은 `is_low_rating_surge` 하나다.
- `family == "X_duplicate_of_structured"`인 열은 2-1 정형 피처와 중복이므로 입력에서 제외한다.
- `split` 열로 시간순 분할(train/valid/test)이 되어 있다. 임계값 탐색은 train 구간에서만 한다.
