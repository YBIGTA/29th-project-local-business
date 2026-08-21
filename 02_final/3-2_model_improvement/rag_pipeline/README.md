# 3-2. RAG 기반 리포트 생성

## 배경 및 목표

3-1에서 구축한 MySQL·FastAPI를 데이터 소스로 삼아, 사용자의 질문에 대해 관련 리뷰와
분석 지표를 함께 찾아 근거가 있는 답변을 생성하는 RAG(Retrieval-Augmented Generation)
파이프라인을 구축한다.

원본 데이터(위험도, 불만 토픽, 리뷰)를 LLM에게 그대로 던지지 않고, 아래 원칙에 따라
가공한 뒤 전달한다.

- **판단은 코드에서, 서술은 LLM에서**: "이 상품이 위험한지", "표본이 충분한지",
  "어느 상품을 먼저 봐야 하는지" 같은 판단은 파이썬 코드가 먼저 계산해서 확정하고,
  LLM은 그 결과를 자연어로 설명하는 역할만 담당한다. LLM이 직접 임계값을 판단하거나
  순위를 재계산하게 두면 일관성이 떨어지고 때로는 명백히 틀린 결론을 낸다는 것을
  개발 과정에서 반복적으로 확인했다 (`threshold_analysis/` 참고).
- **임계값은 감이 아니라 데이터 분포로 정한다**: 표본부족 기준, 급증 판단 기준,
  위험도/신뢰도 등급 기준 등 모든 임계값은 `product_month_risk`,
  `product_month_topics` 실제 데이터의 백분위수를 확인해 정했다.
- **신뢰도가 낮으면 숫자를 숨기지 않고, 그대로 보여주되 경고를 함께 단다**: 표본이
  적어 판단이 불확실한 경우에도 원본 수치는 그대로 노출하고, 그 옆에 "표본 부족" 같은
  주석을 붙이는 방식을 택했다.

## 아키텍처 (검색 → 프롬프트 → 생성)

```
3-1 FastAPI (localhost:8000)
        ↓ requests
retrieval.py            : 데이터 검색 + 신뢰도/표본 판단 로직
        ↓
prompt_templates.py     : 검색 결과 → LLM 프롬프트 텍스트 변환 (기본 리포트용)
        ↓
generate_report.py      : OpenAI 호출 + 기본 위험 리포트 생성
        ↓
topic_query.py / risky_products_query.py / trend_query.py / compare_products.py
                        : 위 세 파일의 함수를 재사용하는 4가지 추가 질문 유형
```

## 파일별 역할

### `config.py`
`.env`에서 `OPENAI_API_KEY`와 MySQL 접속 정보를 읽어와, 다른 모든 파일이 공통으로
사용하는 `engine`(SQLAlchemy 커넥션)을 만든다.

`.env` 파일은 저장소에 포함되지 않는다 (`.gitignore`에 등록됨). 아래 형식으로 직접
만들어야 한다.
```
OPENAI_API_KEY=본인의_OpenAI_API_키
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=본인의_MySQL_비밀번호
DB_NAME=amazon_risk_service
```

### `retrieval.py`
3-1의 FastAPI(`/products/{asin}/summary`, `/topics`, `/evidence-reviews`,
`/risk-explanation`)를 호출해 상품 하나의 근거 자료를 모으고, 아래 판단을 코드에서
확정한다.

| 함수/상수 | 역할 |
|---|---|
| `assess_confidence()` | 위험도(`risk_probability`)와 신뢰도(`operational_confidence`, `review_evidence_confidence`)를 종합해 `risk_level`/`confidence_level`(각 high/moderate/low)과 규칙 기반 해석 문구(`interpretation`)를 만든다. `model_consistency_confidence`는 신뢰도 계산에서 제외하고 참고용으로만 남긴다 (모델이 일관되게 예측하는 것과 판단이 실제로 믿을 만한 것은 다른 질문이기 때문). `weighted_risk`(위험도×신뢰도)는 정렬·비교용 참고 지표로 별도 유지한다. |
| `annotate_topics()` | 토픽별로 `is_low_sample`(표본부족 여부)을 표시하고, 상품 전체에 급증 토픽이 있는지(`has_surging_topic`) 판단한다. |
| `annotate_evidence_reviews()` | 근거 리뷰가 실제로 존재하는지(`has_evidence`) 표시한다. |
| `get_topic_related_reviews()` | 특정 토픽에 해당하는 리뷰만 조회한다 (`topic_query.py`에서 사용). |
| `LOW_REVIEW_COUNT_THRESHOLD = 9` | `current_review_count`의 하위 10% 지점 (표본부족 판단 기준) |
| `LOW_MENTION_COUNT_THRESHOLD = 2` | 토픽 언급이 1건뿐인 경우만 표본부족으로 판단. 실제로는 토픽이 언급된 경우의 62%가 1건뿐이라, 이 경고가 자주 뜨는 것은 데이터 자체의 특성이다. |
| `SURGE_DELTA_THRESHOLD = 0.04` | `product_month_risk`가 존재하는 상품 범위로 좁힌 토픽 delta(양수)의 중앙값 |
| `RISK_HIGH/LOW_THRESHOLD = 0.60 / 0.32` | `risk_probability`의 상위/하위 25% |
| `CONFIDENCE_HIGH/LOW_THRESHOLD = 0.72 / 0.55` | `combined_confidence`의 상위/하위 25% |

모든 API 호출에 타임아웃(10초)과 재시도(최대 3회)를 적용했다.

### `prompt_templates.py`
`retrieval.py`의 결과를 사람이 읽는 텍스트로 변환하고, LLM에게 역할과 답변 형식을
지시하는 시스템 프롬프트를 만든다 (`generate_report.py`가 사용).

- 리뷰 원문이 영어이므로, 토픽 이름(`incompatibility` 등)은 한글로 번역하지 않고
  원어를 그대로 사용해 언어를 일관되게 유지한다.
- 급증 토픽을 표본 충분(`Reliable surge`)/표본 부족(`Low-sample`) 두 그룹으로 나눠
  보여주되, 표본부족 사유는 그룹 전체에 한 번만 설명한다 (토픽마다 반복하지 않음).
- `weighted_risk`(곱셈 점수)는 프롬프트에서 제외한다. 위험도와 신뢰도는 서로 다른
  질문에 대한 답이라 곱해서 하나의 숫자로 뭉개면 오해를 살 수 있기 때문이다
  (예: 위험도 99% × 신뢰도 10% = 9.9%는 "9.9%만 위험하다"는 뜻이 아니다).
  대신 `risk_level`/`confidence_level`/`interpretation`을 직접 노출한다.

### `generate_report.py`
`retrieval.py` + `prompt_templates.py`를 연결해 실제로 OpenAI API(`gpt-4o-mini`,
`temperature=0.3`)를 호출하고 리포트를 생성하는 메인 파일. 다른 질문 유형 파일들이
`call_llm()`을 공통으로 가져다 쓴다.

리포트 형식(고정): `[위험 요약] → [핵심 원인] → [근거 리뷰] → [우선 조치 제안]`

### `topic_query.py` (질문 유형 1)
"이 상품의 특정 불만 토픽 관련 리뷰만 보여줘" 유형. `get_topic_related_reviews()`로
검색하고, 각 리뷰에 매칭된 모든 토픽(`[tags: ...]`)을 프롬프트에 함께 표시해 다른
토픽과 겹치는 리뷰를 LLM이 인지하고 서술하게 한다 (완전한 해결책은 아니며, 근본적으로는
2-3의 토픽 매칭 사전(`complaint_dictionary.json`)의 정밀도에 달린 한계다. 아래 한계
참고).

### `risky_products_query.py` (질문 유형 2)
"지금 가장 위험한 상품 목록을 보여줘" 유형. 3-1의 `/products/risky`를 호출한 뒤:

- 각 상품의 `is_low_sample`을 `LOW_REVIEW_COUNT_THRESHOLD` 기준으로 코드에서 계산
- `risk_probability × operational_confidence`로 재정렬한 순위(`reprioritized_rank`)를
  코드에서 미리 계산해, LLM이 순서를 스스로 정하지 않고 그 순서를 그대로 따르도록 한다
- 순위 변동 방향(올라옴/내려감/유지)도 `original_rank`와 `reprioritized_rank`를
  비교해 코드에서 확정한다

### `trend_query.py` (질문 유형 3)
"이 상품의 위험도가 시간에 따라 어떻게 변해왔는지" 유형. 3-1의
`/products/{asin}/trend`를 호출한 뒤:

- 전월 대비 급변 시점(`significant_jumps`, 기준 29%p — 전체 상품의 전월 대비 변화량
  절댓값 분포 90분위)을 계산하고, 변화폭 상위 3개(`top_jumps`)만 추려 나머지는
  "N건 생략"으로 요약한다
- 급변 시점이 표본부족 구간과 겹치는지(`overlaps_low_sample`)도 함께 표시한다
- 단발성 급변이 없어도, 관찰 기간 전체의 월평균 변화 속도(`avg_monthly_rate`, 기준
  18.2%p/월 — 2개월 이상 데이터가 있는 상품 222개의 월평균 변화율 절댓값 분포
  90분위)가 크면 "완만하지만 지속적인 추세"로 별도 표시한다
- 월별 변화 방향의 일관성(`direction_consistency_ratio`: 총 구간 중 몇 개가
  전체 방향과 같은 방향으로 움직였는지)도 함께 계산해, "몇 개월간 꾸준히 상승/하락"
  같은 패턴을 정량적으로 보여준다

### `compare_products.py` (질문 유형 4, 관리자용)
"상품 A와 B 중 어느 쪽을 먼저 점검해야 하는지" 유형. 두 상품 각각에 대해
`get_product_context()`를 호출한 뒤, `risk_assessment`(이미 검증된 로직)의
`risk_probability`와 `weighted_risk`를 그대로 재사용해 두 가지 승자를 코드에서 확정한다.

- `raw_winner`: 위험 확률만 비교했을 때의 승자
- `weighted_winner`: 신뢰도까지 반영했을 때의 승자
- `winner_changed`: 두 결과가 다른지 (다르면 "표본이 적어 위험도가 과장된 상품"과
  "표본이 충분해 판단이 확실한 상품"이 뒤바뀌는 실제 사례를 보여줄 수 있다)
- Test 샘플 데이터 특성상 두 상품의 최신 시점이 서로 다를 수 있어, `same_period`로
  이를 감지해 다르면 리포트에 명시한다 (아래 한계 참고)

## `threshold_analysis/` — 임계값 근거 확인용 스크립트

`retrieval.py`/`trend_query.py`의 임계값을 감이 아니라 실제 데이터 분포로 정하기
위해 사용한 확인용 스크립트 모음. 서비스 로직에는 포함되지 않으며, 임계값을 다시
검증하거나 조정할 때 참고용으로 남겨둔다.

| 파일 | 확인한 것 |
|---|---|
| `check_distributions.py` | `product_month_risk`/`product_month_topics` 전체 기준 각종 컬럼 분포 (1차) |
| `check_distributions_v2.py` | 위와 동일하되 `product_month_risk`가 존재하는 상품 범위로 좁힌 분포 (1차가 범위 불일치 문제가 있어 재검증) |
| `check_topic_frequency.py` | 12개 토픽이 실제로 얼마나 자주 언급되는지 (특정 토픽 편중이 데이터 특성인지 확인) |
| `check_monthly_change_distribution.py` | 전월 대비 위험도 변화량 절댓값 분포 (`SIGNIFICANT_MONTHLY_CHANGE` 근거) |
| `check_avg_monthly_rate_distribution.py` | 상품별 월평균 위험도 변화 속도 분포 (`SUSTAINED_TREND_RATE_THRESHOLD` 근거) |
| `find_reversal_pair.py`, `find_true_reversal_pair.py` | `compare_products.py`의 `winner_changed=True` 케이스를 실제로 검증하기 위한 상품 쌍 탐색 |

## 실행 순서

1. 3-1 폴더에서 `uvicorn main:app --reload`로 서버 실행 (계속 켜둔 채로 둔다)
2. `.env` 파일 준비 (위 `config.py` 항목 참고)
3. 라이브러리 설치: `pip install openai requests sqlalchemy pymysql python-dotenv pandas`
4. 각 파일을 직접 실행해 테스트 가능:
   ```
   python generate_report.py
   python topic_query.py
   python risky_products_query.py
   python trend_query.py
   python compare_products.py
   ```
5. 다른 상품/시점으로 테스트하려면 각 파일 맨 아래 `if __name__ == "__main__":` 블록의
   `test_asin`, `year_month` 등을 수정한다.

## 검증 현황

5개 질문 유형 모두 서로 다른 특성(위험도 높음/낮음, 리뷰 많음/적음, 급증 토픽
있음/없음, 순위 역전 발생/미발생, 같은 시점/다른 시점)의 실제 상품 데이터로 반복
검증했다. 개발 과정에서 발견해 수정한 대표적인 문제:

- LLM이 규칙 설명 문구("Reliable surge", "참고 수준의 신호로만 언급합니다" 등) 자체를
  리포트 문장에 그대로 베끼는 문제 → 예시와 실제 데이터를 명확히 구분하도록 지시 강화
- 표본부족 caveat이 없는 상품에서도 "그러나"로 억지로 반전 접속사를 붙이는 문제 →
  caveat이 없으면 아예 언급하지 않도록 조건부 처리
- [위험 요약]과 [핵심 원인]에서 같은 수치를 중복 서술하는 문제 → 섹션별 책임 범위를
  명확히 분리
- 우선 조치 제안이 상품 원인과 무관하게 "지속적인 리뷰 모니터링" 같은 일반론만
  반복하는 문제 → 핵심 원인에서 언급한 구체적 토픽/리뷰를 최소 1개 이상 직접
  인용하도록 지시
- 순위 재정렬 시 LLM이 순서를 다시 판단하며 뒤섞는 문제 → 파이썬에서 이미
  정렬한 목록 자체를 프롬프트에 별도로 제공해 LLM은 그대로 나열만 하도록 변경
- 지시문 예시 문장 자체가 반말이라 그대로 복사되는 문제 → 모든 예시 문구를
  존댓말로 통일

## 한계 및 후속 작업

- **`risk_explanation`은 상품별 SHAP이 아닌 S45 모델 전역 피처 중요도**: 3-1에서
  이미 명시한 한계가 3-2에도 그대로 이어진다. 모든 리포트에 "이 상품만의 개별
  이유가 아니라 모델의 일반적 경향"이라는 단서를 반드시 동반하도록 프롬프트에
  강제해 두었다. 3-3에서 실제 SHAP 결과가 나오면 `retrieval.py`의 해당 부분만
  교체하면 된다.
- **토픽 매칭의 근본적 한계**: `performance`, `durability_failure`처럼 의미가 넓은
  토픽은 실제 리뷰 언어에서 다른 토픽(특히 `incompatibility`)과 자주 겹친다.
  각 리뷰의 매칭된 전체 토픽을 프롬프트에 노출해 LLM이 이를 인지하고 서술하게
  했으나, 이는 완화책일 뿐 근본 해결책은 아니다. 근본적으로는 2-3의
  `complaint_dictionary.json` 키워드 사전을 정교화해야 하며, 이는 2-3 담당자와의
  협의가 필요하다.
- **`compare_products.py`의 시점 불일치**: `product_month_risk`가 전체 상품×월이
  아닌 Test 샘플(3,563개)만 담고 있어, 두 상품을 비교할 때 각자의 "최신 시점"이
  서로 다를 수 있다. `same_period` 플래그로 이를 감지해 리포트에 명시하도록
  했으나, 근본적으로는 전체 상품에 대한 운영 스코어링(전 기간)이 갖춰져야
  완전히 해결되는 문제다.
- **질문 의도 분류(자유 텍스트 → 질문 유형 자동 판별)는 구현하지 않음**: 지금은
  사용자가 아니라 호출하는 쪽(3-3 등)이 어떤 함수를 부를지 직접 정하는 구조다.
  "사용자가 자유롭게 타이핑한 질문을 보고 5가지 유형 중 무엇인지, 어떤 상품/토픽을
  의미하는지 자동으로 판단"하는 것은 별도의 의도 분류 단계가 필요한, 상당히 큰
  추가 작업이라 이번 범위에서는 제외했다. 3-3에서 UI(버튼/드롭다운 등)로 질문
  유형과 대상을 선택하게 하는 방식을 권장한다.
- **`product_month_risk`가 Test 샘플만 포함**: 3-1에서부터 이어지는 한계로, 전체
  9만여 상품이 아니라 3,563개 상품×월에 대해서만 위 5개 질문 유형이 작동한다.
