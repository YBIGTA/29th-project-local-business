"""
2-3 NLP 파이프라인 공통 설정.

경로와 기준값을 한 곳에 모아 둔다. 기준값은 2-1(데이터 라벨링)에서 정한 것을
그대로 가져온 것이며, 값이 어긋나면 상품x월 결합이 깨지므로 실행 시 assert로
확인한다.

파트 번호는 다음과 같다.
  2-1 데이터 수집·정제 및 위험 라벨 정의
  2-2 EDA 및 정형 피처
  2-3 NLP 텍스트 피처 (이 폴더)
  2-4 모델링 및 설명가능성
"""
from pathlib import Path

# ---------------------------------------------------------------
# 경로
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "01_midterm" / "2-3_nlp_features"

for directory in (INTERIM_DIR, PROCESSED_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# 2-1 산출물. 파일명이 다를 수 있어 후보를 순서대로 찾는다.
def _find(candidates, description):
    for name in candidates:
        for directory in (PROCESSED_DIR, INTERIM_DIR, RAW_DIR):
            path = directory / name
            if path.exists():
                return path
    return PROCESSED_DIR / candidates[0]


REVIEWS_CLEAN = _find(
    ["reviews_clean.parquet", "clean_reviews.parquet", "reviews_clean.csv"],
    "2-1 정제 리뷰",
)
LABELED_PANEL = _find(
    ["product_month_labeled.parquet", "product_month_labeled.csv"],
    "2-1 라벨 패널",
)

# 2-3 산출물
CLEAN_REVIEWS = INTERIM_DIR / "clean_reviews.parquet"
REVIEW_FEATURES = INTERIM_DIR / "review_nlp_features.parquet"
MONTH_FEATURES = PROCESSED_DIR / "product_month_text_features.parquet"

DICTIONARY_PATH = Path(__file__).resolve().parent / "complaint_dictionary.json"

# ---------------------------------------------------------------
# 2-1에서 정한 기준값
# ---------------------------------------------------------------
# 모델 행이 되는 t월의 범위
ANALYSIS_START = "2016-01"
ANALYSIS_END = "2023-02"
# 직전 3개월 기준선을 계산하기 위해 필요한 가장 이른 월
BASELINE_START = "2015-11"
# 직전 3개월의 정의
PAST_WINDOW_OFFSETS = (1, 2, 3)

# 저평점과 고평점의 경계
LOW_RATING_MAX = 2
HIGH_RATING_MIN = 4

# 한 달 최소 리뷰 수. 이보다 적으면 비율이 불안정해 분석 대상에서 제외한다.
MIN_MONTH_REVIEWS = 5
# 위험 라벨 기준
SURGE_RATE_THRESHOLD = 0.30
SURGE_DELTA_THRESHOLD = 0.15

# 시간순 분할. 사전 구축과 임계값 탐색은 train 구간만 보고 해야 한다.
TRAIN_END = "2020-12"
VALID_END = "2021-12"

# 2-1 산출물의 정답 정보. 피처로 절대 쓰지 않는다.
FUTURE_COLUMNS = (
    "next_review_count",
    "next_low_rating_count",
    "next_low_rating_ratio",
    "next_avg_rating",
    "next_vs_past_3m_low_rating_change",
)
LABEL_COLUMN = "is_low_rating_surge"
KEY_COLUMNS = ["parent_asin", "year_month"]

# 2-1 검증 보고서에 기록된 기대값
EXPECTED_ROWS = 68087
EXPECTED_PRODUCTS = 6613

assert ANALYSIS_START > BASELINE_START, "기준선 시작이 분석 시작보다 늦다"
assert LOW_RATING_MAX < HIGH_RATING_MIN, "저평점과 고평점 경계가 겹친다"
assert TRAIN_END < ANALYSIS_END, "train 종료가 분석 종료보다 늦다"


def month_index(year_month):
    """'2019-06' 같은 문자열을 정수 월 번호로 바꾼다. 월 차이 계산에 쓴다."""
    year, month = str(year_month).split("-")
    return int(year) * 12 + int(month)


def split_of(year_month):
    """시간순 분할 이름을 돌려준다."""
    if year_month <= TRAIN_END:
        return "train"
    if year_month <= VALID_END:
        return "valid"
    return "test"
