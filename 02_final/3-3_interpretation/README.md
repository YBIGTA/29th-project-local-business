# 3-3. 위험 상품 분석 대시보드

Streamlit 기반 관리자 대시보드. 3-1(MySQL·FastAPI)이 제공하는 API와 3-2(RAG)의 리포트 생성 함수를 화면으로 연결한다.

## 화면 구성

| 파일 | 화면 | 역할 |
|---|---|---|
| `app.py` | 홈 | 백엔드(`/health`) 연결 상태 확인 |
| `pages/1_Overview.py` | 전체 현황 | 카테고리 전체 위험 현황 — KPI, 위험 등급별 분포, 전월 대비 급상승 상품, 필터가 적용된 상품 목록 |
| `pages/2_Product_Detail.py` | 상품 상세 | 상품 하나의 위험도·추이·불만 토픽·위험 판단 근거·근거 리뷰 |
| `pages/3_RAG_Report.py` | AI 리포트 | 3-2 RAG 파이프라인의 5가지 질문 유형(기본 리포트/토픽별 리뷰/위험 상품 요약/추이 분석/상품 비교)을 UI로 노출 |
| `utils/api_client.py` | — | 3-1 API를 호출하는 함수 모음. **대시보드는 이 파일을 통해서만 데이터를 가져온다** |
| `.streamlit/config.toml` | — | streamlit 형식 지정 |



## 구현된 기능 상세
 
### 1_Overview.py — 전체 현황
 
- **KPI 카드 4개**: 기준 월, 조회된 상품 수, 고위험 상품 수(전월 대비 증감 표시), 평균 위험도
- **사이드바 필터**: 카테고리(현재 Appliances 고정), 연/월, 정렬 기준(위험도순 / 위험도×신뢰도순), 표시 개수, 고위험 판단 기준(%, 슬라이더), 신뢰도 등급, 브랜드 검색, 최소 리뷰 수
- **위험 등급별 상품 분포**: 고/중/저위험 구간별 상품 수를 막대그래프로 표시 (Plotly)
- **전월 대비 위험도 급상승 상품**: 상승폭 상위 3개를 카드로 표시, 이전 달 대비 몇 %p 올랐는지 화살표·색상으로 강조 (사이드바 필터와 무관하게 카테고리 전체 기준)
- **상품 목록 표**: 위험도를 진행 바(progress bar)로 표시, 행을 클릭하면 해당 상품의 상세 화면으로 자동 이동 (구버전 Streamlit 대비 선택 드롭다운 폴백 포함)


### 2_Product_Detail.py — 상품 상세
 
- **사이드바 연/월 선택**: 페이지 진입 시 이 상품의 실제 최신 월을 자동으로 찾아 기본값으로 설정. 선택한 월이 아래 KPI·불만 토픽·근거 리뷰 조회에 그대로 반영됨
- **KPI 카드 7개**: 기준 월, 평균 평점, 저평점 비율, 위험도, 운영 신뢰도, 신뢰도 등급, 현재 리뷰 수
- **월별 추이 그래프**: 위험도·평점·저평점 비율 등 원하는 지표를 선택해서 선 그래프로 확인 (전체 기간 기준, 사이드바 월 선택과 무관)
- **최근 증가한 불만 토픽**: 12개 토픽별 이번 달/직전 3개월 비율과 변화량을 표로, 변화량은 막대그래프로 표시
- **위험 판단 근거**: 변수별 기여도 표, 통계 기반 근사 설명
- **근거 리뷰**: 별점(1점/2점), 연-월, 토픽(12개 드롭다운) 조건으로 저평점 리뷰 원문을 필터링해서 조회
- 하단에 `summary` API 원본 JSON을 확인할 수 있는 디버그용 expander


### 3_AI_리포트.py — AI 리포트
 
- 드롭다운으로 5가지 질문 유형 중 선택해서 조회
  1. **기본 위험 리포트**: 상품 하나의 위험도 배지 + `[위험 요약]·[핵심 원인]·[근거 리뷰]·[우선 조치 제안]` 4단 구조를 하나의 카드에 구분선으로 나눠 표시
  2. **특정 불만 토픽 리뷰**: 상품×토픽 조합으로 관련 리뷰 요약 조회 (최대 5건/15건 선택)
  3. **위험 상품 목록 요약**: 정렬 기준·개수·연월을 지정해 위험 상품 목록을 LLM이 요약
  4. **위험도 추이 분석**: 상품 하나의 시간에 따른 위험도 변화를 서술형으로 분석
  5. **상품 비교 (관리자용)**: 두 상품의 위험도·신뢰도를 비교해 우선순위와 그 근거를 제시
- 각 유형의 마지막 조회 결과는 `st.session_state`에 저장되어, 페이지를 새로고침해도 유지됨 (단, Streamlit 서버 재시작 시에는 초기화)
- 근거 데이터(JSON) 원본을 확인할 수 있는 expander 포함


## 실행 순서

이 대시보드는 단독으로 실행되지 않는다. 아래 순서를 반드시 지켜야 한다.

1. MySQL 실행 + 데이터 적재 완료 상태 (`02_final/3-1_feature_expansion/mysql_fastapi/load_data_v2.py` 실행 완료)
2. 3-1 백엔드 실행 (다른 터미널에서 계속 켜둔 채로)
   ```bash
   cd 02_final/3-1_feature_expansion/mysql_fastapi
   uvicorn main:app --reload
   ```
3. AI 리포트 페이지를 쓰려면 3-2 쪽 `.env`가 준비되어 있어야 함 (`02_final/3-2_model_improvement/rag_pipeline/.env`, `OPENAI_API_KEY` 포함)
4. 대시보드 실행
   ```bash
   cd 02_final/3-3_interpretation/dashboard
   pip install -r requirements.txt
   streamlit run app.py
   ```

기본 접속 주소: 백엔드 `http://localhost:8000`, 대시보드 `http://localhost:8501`

## 3-1 API와의 연결 방식

`utils/api_client.py`가 3-1의 8개 엔드포인트 + 대시보드 작업 중 추가한 `/products/risk-changes`(전월 대비 비교, `main.py`에 반영됨)를 감싸고 있다. `@st.cache_data(ttl=300)`로 5분 캐싱되어 있어, 같은 조건 재조회 시 API를 다시 부르지 않는다.

`api_client.py`의 `API_BASE_URL = "http://localhost:8000"`이 하드코딩되어 있다. **Docker로 통합할 때 이 값을 컨테이너 간 통신 주소(예: `http://backend:8000`)로 바꿔야 한다.**

## 3-2 RAG 파이프라인과의 연결 방식

API 호출이 아니라, `sys.path.insert()`로 `rag_pipeline` 폴더를 직접 import해서 함수를 호출하는 구조다.

```python
RAG_PIPELINE_DIR = os.path.normpath(
    os.path.join(THIS_DIR, "..", "..", "..", "3-2_model_improvement", "rag_pipeline")
)
sys.path.insert(0, RAG_PIPELINE_DIR)
```

`3-3_interpretation/dashboard`와 `3-2_model_improvement/rag_pipeline`이 `02_final` 아래 같은 깊이에 있다는 전제로 상대경로가 계산된다. **폴더를 옮기면 이 경로도 같이 조정해야 한다.**


## 3-3 작업에서 추가·수정한 파일
 
대시보드 자체(`app.py`, `pages/`, `utils/`) 외에, 3-1·2-4 산출물을 보강하기 위해 추가로 만들거나 고친 파일들이다.
 
| 파일 | 위치 | 역할 |
|---|---|---|
| `compute_stat_explanations.py` | `02_final/3-3_interpretation/` | `risk_explanations`용 근사 설명을 계산하는 스크립트. 아래 "위험 판단 근거 근사 계산" 참고 |
| `stat_explanations.csv` | `02_final/3-3_interpretation/` | 위 스크립트의 결과물. `parent_asin`, `year_month`, `feature_name`, `contribution` 4개 컬럼, 349,188행 |
| `risk_confidence_output/` | `02_final/3-3_interpretation/` | `product_month_risk_confidence.parquet` 등 위험도·신뢰도 원본 데이터가 담긴 폴더. `load_data_v2.py`의 `RISK_CONFIDENCE_PATH`가 이 경로를 가리킴 |
| `load_data_v2.py` (3-1 원본 수정) | `02_final/3-1_feature_expansion/mysql_fastapi/` | 경로를 절대경로에서 프로젝트 표준 상대경로(data/interim/, data/processed/ 등)로 변경. 6번 단계(`risk_explanations` 적재)의 데이터 소스를 `standard_candidate_importance.csv`(전역 중요도)에서 `stat_explanations.csv`(상품·월별 근사 기여도)로 교체. 나머지 1~5단계는 3-1 원본 그대로 |
| `main.py` (3-1 원본 수정) | `02_final/3-1_feature_expansion/mysql_fastapi/` | `/products/risk-changes` 엔드포인트 신규 추가(전월 대비 위험도·고위험 상품 수 변화 계산).|
| `requirements.txt` (수정) | 프로젝트 루트 | 대시보드 실행에 필요한 패키지(`streamlit`, `plotly`, `requests` 등) 추가 |
 
### 위험 판단 근거 근사 계산 — 배경과 방법
 
원래 목표는 상품별 SHAP이었다. 다만 최종 채택된 예측 파이프라인(`01_midterm/2-4_model_validation/2-4-03_Randomness_Correction/run_true_bag_mil_final_test.py`)은 S45·raw context·MIL 세 모델을 실행 시점마다 함께 학습해 최종 점수를 만드는 구조라, 재사용 가능한 단일 모델 파일로 저장되어 있지 않다. 이 구조에서 SHAP을 계산하려면 파이프라인 자체를 다시 실행하며 별도로 값을 뽑아내는 작업이 필요해, 상품 위험 판단 근거를 근사 계산하는 방식을 채택했다.

`compute_stat_explanations.py`가 다음을 계산한다.
 
1. `data/processed/product_month_review_volume_2m_labeled.parquet`에서 정형 변수 14개(리뷰 수, 평점, 저평점 비율 및 1/3/6/12개월 이력)를 가져온다.
2. 각 변수마다 위험 방향(값이 클수록 위험한지 작을수록 위험한지)을 미리 정의해둔다.
3. 상품×월별로 그 변수 값이 **전체 상품 분포에서 몇 백분위(0~100)에 있는지** 계산한다. 값이 클수록(=위험할수록) 백분위 점수가 높아지도록 방향을 맞춘다.
4. 결과를 `risk_explanations` 테이블과 동일한 형식(`parent_asin`, `year_month`, `feature_name`, `contribution`)으로 저장한다.
**주의**: 이 값은 SHAP이 아니라 통계적 근사치다. "모델이 이 변수를 실제로 어떻게 판단했는지"가 아니라 "이 상품이 이 변수 기준으로 전체 대비 얼마나 튀는지"를 보여줄 뿐이다. 

 
## 제한사항
 
- **`risk_explanations`(위험 판단 근거)는 실제 SHAP이 아니라 통계 기반 근사치다.** 위 "위험 판단 근거 근사 계산" 참고.
- **`product_month_risk`는 전체 9만여 상품이 아니라 Test 기간 샘플(3,563건)만 존재한다.** 특정 월 조회 시 데이터가 없을 수 있다.
- **가격 정보가 없다.** 원본 데이터(`products_meta_clean.parquet`)에 가격 컬럼 자체가 없어 대시보드에서 표시 불가능.
- **"전월 대비 급상승 상품" 카드는 사이드바 필터와 무관하게 카테고리 전체 기준으로 계산된다.**