"""2-4 모델 검증에서 A와 B가 함께 사용하는 고정 실험 설정."""

KEY_COLUMNS = ["parent_asin", "year_month"]
TARGET_COLUMN = "is_low_rating_surge"

LABELED_DATA_PATH = "data/processed/product_month_labeled.parquet"
TEXT_DATA_PATH = "data/processed/product_month_text_features.parquet"
TEXT_DICTIONARY_PATH = "01_midterm/2-3_nlp_features/text_feature_dictionary.csv"

ANALYSIS_START = "2016-01"
ANALYSIS_END = "2023-02"

TRAIN_START = "2016-01"
TRAIN_END = "2021-12"
VALID_START = "2022-01"
VALID_END = "2022-08"
TEST_START = "2022-09"
TEST_END = "2023-02"

EXPECTED_ROWS = 68_087
EXPECTED_PRODUCTS = 6_613
EXPECTED_POSITIVES = 6_448
EXPECTED_SPLIT_COUNTS = {
    "train": 54_813,
    "valid": 7_228,
    "test": 6_046,
}

# 현재 t월의 상태만 사용하는 A의 첫 번째 정형 모델.
STRUCTURED_CURRENT_FEATURES = [
    "review_count",
    "low_rating_ratio",
    "mean_helpful_vote",
    "verified_purchase_ratio",
    "reliability_ci_width",
]

# 현재 상태에 t-2~t월 저평점 기준선을 추가하는 A의 시계열 정형 모델.
STRUCTURED_TEMPORAL_FEATURES = STRUCTURED_CURRENT_FEATURES + [
    "past_3m_low_rating_ratio",
]

# B가 A의 결과를 기다리지 않고 사용할 최소 정형 기준.
MINIMUM_STRUCTURED_FEATURES_FOR_B = [
    "review_count",
    "low_rating_ratio",
    "past_3m_low_rating_ratio",
]

FORBIDDEN_EXACT_COLUMNS = {
    "is_low_rating_surge",
    "next_review_count",
    "next_avg_rating",
    "next_low_rating_count",
    "next_low_rating_ratio",
    "next_vs_past_3m_low_rating_change",
}
FORBIDDEN_PREFIXES = ("next_",)
FORBIDDEN_TEXT_FAMILIES = {"key", "X_duplicate_of_structured"}
FORBIDDEN_AUTO_TITLE_COLUMNS = {
    "auto_title_flag_rate_t",
    "auto_title_flag_rate_p3",
    "auto_title_flag_delta",
    "auto_title_flag_rate_t_shrunk",
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

METRICS = [
    "pr_auc",
    "roc_auc",
    "recall_at_5pct",
    "recall_at_10pct",
]

RESULT_COLUMNS = [
    "model_name",
    "split",
    "pr_auc",
    "roc_auc",
    "recall_at_5pct",
    "recall_at_10pct",
    "n_rows",
    "positive_rate",
    "feature_count",
    "random_state",
]


def split_2_4(year_month: str) -> str:
    """2-2 베이스라인과 동일한 2-4 공통 시간순 분할."""
    month = str(year_month)
    if TRAIN_START <= month <= TRAIN_END:
        return "train"
    if VALID_START <= month <= VALID_END:
        return "valid"
    if TEST_START <= month <= TEST_END:
        return "test"
    raise ValueError(f"2-4 분석 기간 밖의 year_month: {month}")
