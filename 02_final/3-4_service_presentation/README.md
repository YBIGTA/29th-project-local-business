# 3-4. Docker 배포 및 서비스 통합

## 배경 및 목표

3-1(MySQL·FastAPI), 3-2(RAG 파이프라인), 3-3(Streamlit 대시보드)은 각자 개인 환경에서
개별 실행되는 상태였다. 실행 순서(MySQL 기동 → 데이터 적재 → 백엔드 실행 → 대시보드 실행)를
사람이 직접 지켜야 했고, DB 접속 정보와 API 주소가 코드에 하드코딩되어 있어 다른 사람의
환경에서는 그대로 동작하지 않았다.

이를 Docker Compose 기반의 하나의 실행 가능한 서비스로 통합하여, 저장소와 데이터 파일만
있으면 명령어 두 줄로 전체 서비스가 뜨도록 구성했다.

## 목차

1. [컨테이너 구성](#컨테이너-구성)
2. [실행 방법](#실행-방법)
3. [통합 과정에서 수정한 사항](#통합-과정에서-수정한-사항)
4. [설계 결정과 근거](#설계-결정과-근거)
5. [트러블슈팅](#트러블슈팅)
6. [한계 및 후속 작업](#한계-및-후속-작업)

## 컨테이너 구성

| 서비스 | 이미지/베이스 | 역할 | 포트(호스트→컨테이너) |
|---|---|---|---|
| `db` | `mysql:8.4` | 서비스 데이터베이스 | 3307 → 3306 |
| `loader` | `python:3.12-slim` | 최초 1회 데이터 적재 (일회성) | — |
| `backend` | `python:3.12-slim` | 3-1 FastAPI | 8000 → 8000 |
| `dashboard` | `python:3.12-slim` | 3-3 Streamlit + 3-2 RAG | 8501 → 8501 |

```
사용자
  │
  ▼
dashboard (8501)  ── HTTP ──▶  backend (8000)  ── SQL ──▶  db (3306)
  └─ rag_pipeline                   ▲                        ▲
       ├─ HTTP ─────────────────────┘                        │
       ├─ SQL ───────────────────────────────────────────────┘
       └─ OpenAI API (외부)
```

**3-2 RAG는 별도 컨테이너로 분리하지 않았다.** 3-3 대시보드가 `sys.path.insert()`로
`rag_pipeline` 폴더를 직접 import하는 구조이기 때문에, 독립 서비스가 아니라 대시보드
이미지 안에 함께 포함된다. 이때 `02_final/3-2_.../rag_pipeline`과
`02_final/3-3_.../dashboard`의 상대 경로 깊이가 유지되어야 import가 성립하므로,
컨테이너 내부에서도 저장소와 동일한 폴더 구조를 그대로 복사한다.

## 실행 방법

### 사전 준비

**1) 데이터 파일 배치**

용량 문제로 Git에 포함되지 않은 아래 파일들을 저장소 루트 기준 경로에 배치한다.
(3-1 담당자가 전달한 `amazon_3_1_data_01~04.zip` 압축 해제 후 복사)

```
data/interim/products_meta_clean.parquet
data/processed/reviews_clean.parquet
data/processed/product_month_text_features.parquet
data/processed/product_month_review_volume_2m_labeled.parquet
02_final/3-3_interpretation/risk_confidence_output/product_month_risk_confidence.parquet
02_final/3-3_interpretation/stat_explanations.csv
```


리뷰 원문은 세 조각으로 분할 전달되므로 합쳐야 한다.

```bash
cat data/processed/reviews_clean.parquet.part_* > data/processed/reviews_clean.parquet
rm data/processed/reviews_clean.parquet.part_*
```

**2) 환경 변수 설정**

```bash
cd 02_final/3-4_service_presentation
cp .env.example .env
```

`.env`를 열어 아래 값을 채운다. `MYSQL_ROOT_PASSWORD`와 `DB_PASSWORD`는 **반드시 같은 값**이어야 하며,
URL 파싱 문제를 피하기 위해 영문·숫자만 사용한다.

| 변수 | 설명 |
|---|---|
| `MYSQL_ROOT_PASSWORD` / `DB_PASSWORD` | MySQL root 비밀번호 (동일 값) |
| `DB_HOST` | `db` (compose 서비스명, 수정 불필요) |
| `API_BASE_URL` | `http://backend:8000` (수정 불필요) |
| `OPENAI_API_KEY` | RAG 리포트 생성용. 없으면 AI 리포트 페이지만 동작하지 않음 |

`.env`는 `.gitignore`에 등록되어 저장소에 포함되지 않는다.

### 최초 실행

```bash
cd 02_final/3-4_service_presentation

# 1. DB 기동 (스키마 자동 생성)
docker compose up -d db

# 2. 데이터 적재 (최초 1회, 10~20분 소요)
docker compose --profile init run --rm loader

# 3. 백엔드 + 대시보드 기동
docker compose up -d
```

접속: 대시보드 `http://localhost:8501`, API 문서 `http://localhost:8000/docs`

### 이후 실행

데이터는 Docker 볼륨(`mysql_data`)에 남으므로 적재를 반복할 필요가 없다.

```bash
docker compose up -d
```

### 상태 확인 및 종료

```bash
docker compose ps
docker compose logs -f backend
curl http://localhost:8000/health          # {"status":"ok","db_connected":true}

docker compose down                        # 컨테이너만 중지 (데이터 유지)
docker compose down -v                     # 볼륨까지 삭제 (재적재 필요)
```

## 통합 과정에서 수정한 사항

컨테이너 환경에서는 각 서비스가 서로 다른 호스트에 있으므로, 하드코딩된 접속 정보를
환경 변수로 전환했다. 기본값을 기존 로컬 값(`localhost`)으로 두어 **개인 환경에서
도커 없이 실행하던 방식도 그대로 동작한다.**

| 파일 | 수정 내용 |
|---|---|
| `3-1/mysql_fastapi/main.py` | DB 접속 정보 5개를 `os.getenv()`로 전환 |
| `3-1/mysql_fastapi/load_data_v2.py` | 위와 동일 |
| `3-3/dashboard/utils/api_client.py` | `API_BASE_URL`을 `os.getenv()`로 전환 |
| `3-2/rag_pipeline/retrieval.py` | 위와 동일 |
| `3-2/rag_pipeline/trend_query.py` | 위와 동일 |
| `3-2/rag_pipeline/risky_products_query.py` | 위와 동일 |
| `requirements.txt` (루트) | 문법 오류 2건 수정 (`SQLAlchemy=` → `==`, `PyMySQL`/`requests` 줄바꿈 누락) |

`3-2/rag_pipeline/config.py`는 이미 `os.getenv()` 패턴이었으므로 수정하지 않았다.

> `load_data_v2.py`와 `main.py`에 MySQL 비밀번호가 평문으로 커밋되어 있던 것도 이 과정에서
> 함께 제거되었다. 다만 커밋 히스토리에는 남아 있으므로, 해당 비밀번호를 다른 곳에서
> 사용 중이라면 변경을 권장한다.

## 설계 결정과 근거

**DB 스키마는 `docker-entrypoint-initdb.d`로 자동 적용한다**

`load_data_v2.py`는 INSERT만 수행하고 테이블을 생성하지 않는다. MySQL 공식 이미지는
최초 기동 시 `/docker-entrypoint-initdb.d/` 안의 `.sql`을 자동 실행하므로,
3-1의 `schema_v2.sql`을 이 경로에 읽기 전용으로 마운트했다. 스키마 파일을 복사하지 않고
마운트하는 방식이라, 3-1에서 스키마가 수정되면 컨테이너 재생성만으로 반영된다.

**데이터 적재는 별도의 일회성 컨테이너로 분리한다**

적재는 리뷰 원문 471MB를 읽어 6개 테이블에 약 157만 행을 넣는 작업으로 10~20분이 걸린다.
서비스 컨테이너의 시작 과정에 포함하면 매 기동마다 반복되므로, `profiles: ["init"]`을 지정한
`loader` 서비스로 분리해 평상시 `docker compose up`에는 뜨지 않도록 했다.
적재 결과는 `mysql_data` 볼륨에 영속되므로 최초 1회만 실행하면 된다.

**데이터 파일은 이미지에 굽지 않고 볼륨으로 마운트한다**

471MB를 `COPY`하면 이미지가 1GB를 넘고 빌드도 느려진다. 반면 데이터 파일은 어차피
Git에 없어서 실행자가 별도로 받아야 하므로, 이미지에 포함해도 배포 편의성이 개선되지 않는다.
따라서 `loader`에는 데이터 폴더를 읽기 전용으로 마운트하는 방식을 택했다.

**MySQL 포트를 3307로 노출한다**

로컬에 MySQL이 이미 설치된 환경에서 3306이 점유되어 있을 수 있어 충돌을 피했다.
컨테이너 사이의 통신은 내부 네트워크의 3306을 그대로 쓰므로 이 설정은 호스트에서
직접 DB에 접속할 때만 영향을 준다.

**서비스 기동 순서는 healthcheck로 보장한다**

`depends_on`만으로는 컨테이너가 "시작"된 시점만 보장하고 MySQL이 접속을 받을 준비가
되었는지는 알 수 없다. `mysqladmin ping` 기반 healthcheck와 `condition: service_healthy`를
함께 지정해, `loader`와 `backend`가 DB 준비 완료 후에 실행되도록 했다.

## 동작 확인

세 컨테이너를 모두 기동한 상태에서 아래를 확인했다.

| 확인 항목 | 결과 |
|---|---|
| `GET /health` | `{"status":"ok","db_connected":true}` |
| 대시보드 홈 | 백엔드 연결 상태 정상 (`http://backend:8000`) |
| Overview | KPI·위험 등급 분포·급상승 상품·상품 목록 정상 렌더링 |
| Product Detail | KPI·월별 추이·불만 토픽·근거 리뷰 정상 조회 |
| RAG Report | 5개 질문 유형 리포트 생성 정상 |

적재된 행 수는 3-1 README의 명세와 일치한다.

| 테이블 | 행 수 |
|---|---|
| products | 94,327 |
| product_month_metrics | 24,942 |
| product_month_risk | 3,563 |
| product_month_topics | 817,044 |
| evidence_reviews | 284,775 |
| risk_explanations | 349,188 |

`risk_explanations`는 3-1 README 기준(35,630)과 다른데, 3-3에서 전역 피처 중요도를
상품·월별 근사 기여도(`stat_explanations.csv`)로 교체하면서 늘어난 값이다.

## 트러블슈팅

**1. WSL2에서 `docker: command not found`**

Docker Desktop이 Resource Saver 모드로 절전되면 WSL 통합이 끊긴다. Docker Desktop 좌하단의
재생 버튼으로 엔진을 깨우면 해결된다. 엔진이 켜져 있는데도 발생하면
Settings → Resources → WSL Integration에서 해당 배포판 토글을 확인한다.

**2. `requirements.txt` 설치 실패**

`SQLAlchemy=2.0.52`(등호 1개), `PyMySQL==1.2.0requests==2.34.2`(줄바꿈 누락) 두 건의
문법 오류로 `pip install -r`이 전체 실패했다. 이미지 빌드 단계에서 바로 드러나는 문제라
수정 후 재빌드했다.

**3. 리뷰 원문 파일 무결성 확인**

분할 전달된 조각을 합친 뒤에는 행 수로 검증하는 것이 확실하다.

```bash
python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('data/processed/reviews_clean.parquet').metadata.num_rows)"
# 1814360 이어야 함
```

## 한계 및 후속 작업

- **외부 배포는 하지 않았다.** 현재 구성은 로컬 환경에서 `localhost` 접속을 전제로 한다.
  외부 접속이 필요하면 클라우드 인스턴스에 배포하고 방화벽·보안 그룹에서 8501 포트를
  열어야 하며, `API_BASE_URL`은 컨테이너 내부 통신용이므로 그대로 두면 된다.
- **데이터 파일 배포가 수동이다.** 약 490MB의 parquet 파일을 별도로 전달받아 배치해야 하므로,
  저장소만으로는 서비스를 띄울 수 없다. 근본적으로는 오브젝트 스토리지(S3 등)에 올리고
  `loader`가 실행 시점에 내려받는 구조가 적절하다.
- **HTTPS와 인증이 없다.** 대시보드가 인증 없이 열려 있어, 외부에 노출할 경우
  리버스 프록시(nginx 등)를 통한 TLS 종료와 접근 제어가 필요하다.
- **`mysql:8.4` 이미지는 x86_64 기준이다.** Apple Silicon 등 ARM 환경에서는
  플랫폼 지정 또는 대체 이미지가 필요할 수 있다.