from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/interim/product_month_with_next_month.parquet"
OUTPUT_PATH = ROOT / "data/processed/product_month_labeled.parquet"
REPORT_DIR = ROOT / "01_midterm/2-1_data_label"

ANALYSIS_START = pd.Period("2016-01", freq="M")
ANALYSIS_END = pd.Period("2023-02", freq="M")

data = pd.read_parquet(INPUT_PATH)
data["year_month"] = pd.PeriodIndex(data["year_month"], freq="M")

# t월을 포함한 직전 3개월(t-2, t-1, t)을 모아
# 상품별 기존 저평점 수준을 계산한다.
current = data[
    ["parent_asin", "year_month", "review_count", "low_rating_count"]
].copy()

history_parts = []
for month_offset in [0, 1, 2]:
    part = current.copy()
    part["target_month"] = part["year_month"] + month_offset
    part = part.rename(
        columns={
            "review_count": f"review_count_lag_{month_offset}",
            "low_rating_count": f"low_rating_count_lag_{month_offset}",
        }
    )
    history_parts.append(
        part[
            [
                "parent_asin",
                "target_month",
                f"review_count_lag_{month_offset}",
                f"low_rating_count_lag_{month_offset}",
            ]
        ]
    )

history = history_parts[0]
for part in history_parts[1:]:
    history = history.merge(
        part,
        on=["parent_asin", "target_month"],
        how="outer",
    )

count_cols = [
    "review_count_lag_0",
    "review_count_lag_1",
    "review_count_lag_2",
]
low_count_cols = [
    "low_rating_count_lag_0",
    "low_rating_count_lag_1",
    "low_rating_count_lag_2",
]

history[count_cols + low_count_cols] = history[
    count_cols + low_count_cols
].fillna(0)

history["past_3m_review_count"] = history[count_cols].sum(axis=1)
history["past_3m_low_rating_count"] = history[low_count_cols].sum(axis=1)
history["past_3m_low_rating_ratio"] = (
    history["past_3m_low_rating_count"]
    / history["past_3m_review_count"].where(
        history["past_3m_review_count"] > 0
    )
)

dataset = data.merge(
    history[
        [
            "parent_asin",
            "target_month",
            "past_3m_review_count",
            "past_3m_low_rating_count",
            "past_3m_low_rating_ratio",
        ]
    ],
    left_on=["parent_asin", "year_month"],
    right_on=["parent_asin", "target_month"],
    how="left",
).drop(columns="target_month")

# 라벨 신뢰도를 확보할 수 있는 행과 확정한 분석 기간만 남긴다.
dataset = dataset[
    (dataset["year_month"] >= ANALYSIS_START)
    & (dataset["year_month"] <= ANALYSIS_END)
    & (dataset["next_review_count"] >= 5)
    & (dataset["past_3m_review_count"] >= 5)
].copy()

dataset["next_vs_past_3m_low_rating_change"] = (
    dataset["next_low_rating_ratio"]
    - dataset["past_3m_low_rating_ratio"]
)

dataset["is_low_rating_surge"] = (
    (dataset["next_low_rating_ratio"] >= 0.30)
    & (dataset["next_vs_past_3m_low_rating_change"] >= 0.15)
).astype("int8")

dataset["year_month"] = dataset["year_month"].astype(str)
dataset = dataset.sort_values(["year_month", "parent_asin"]).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset.to_parquet(OUTPUT_PATH, index=False)

data_dictionary = pd.DataFrame(
    [
        ["parent_asin", "상품 대표 식별자", "입력 키", "사용"],
        ["year_month", "기준 시점 t월", "입력 키", "사용"],
        ["review_count", "t월 전체 리뷰 수", "t월 집계", "사용"],
        ["avg_rating", "t월 평균 별점", "t월 집계", "사용"],
        ["low_rating_count", "t월 1~2점 리뷰 수", "t월 집계", "사용"],
        ["low_rating_ratio", "t월 1~2점 리뷰 비율", "t월 집계", "사용"],
        ["verified_purchase_ratio", "t월 실제 구매 인증 리뷰 비율", "t월 집계", "사용"],
        ["mean_helpful_vote", "t월 리뷰당 평균 도움 수", "t월 집계", "사용"],
        ["text_available_ratio", "t월 본문이 있는 리뷰 비율", "t월 집계", "사용"],
        ["past_3m_review_count", "t-2~t월 전체 리뷰 수", "과거 집계", "사용"],
        ["past_3m_low_rating_count", "t-2~t월 1~2점 리뷰 수", "과거 집계", "사용"],
        ["past_3m_low_rating_ratio", "t-2~t월 1~2점 리뷰 비율", "과거 집계", "사용"],
        ["next_review_count", "t+1월 전체 리뷰 수", "미래값", "라벨 검증만"],
        ["next_avg_rating", "t+1월 평균 별점", "미래값", "라벨 검증만"],
        ["next_low_rating_count", "t+1월 1~2점 리뷰 수", "미래값", "라벨 검증만"],
        ["next_low_rating_ratio", "t+1월 1~2점 리뷰 비율", "미래값", "라벨 검증만"],
        ["next_vs_past_3m_low_rating_change", "다음 달과 직전 3개월 저평점 비율 차이", "미래값 포함", "라벨 검증만"],
        ["is_low_rating_surge", "확정 위험 라벨(0/1)", "미래값으로 생성", "예측 대상"],
    ],
    columns=["column_name", "description", "source", "model_usage"],
)
data_dictionary.to_csv(
    REPORT_DIR / "data_dictionary.csv",
    index=False,
    encoding="utf-8-sig",
)

summary = pd.DataFrame(
    {
        "metric": [
            "analysis_start",
            "analysis_end",
            "final_row_count",
            "positive_count",
            "positive_ratio",
            "unique_product_count",
        ],
        "value": [
            str(ANALYSIS_START),
            str(ANALYSIS_END),
            len(dataset),
            int(dataset["is_low_rating_surge"].sum()),
            round(float(dataset["is_low_rating_surge"].mean()), 4),
            dataset["parent_asin"].nunique(),
        ],
    }
)
summary.to_csv(
    REPORT_DIR / "labeled_dataset_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=== 최종 상품×월 학습 테이블 생성 완료 ===")
print(summary.to_string(index=False))
print("\n라벨 분포")
print(dataset["is_low_rating_surge"].value_counts().sort_index().to_string())

print("\n생성 파일")
print(OUTPUT_PATH)
print(REPORT_DIR / "data_dictionary.csv")
print(REPORT_DIR / "labeled_dataset_summary.csv")
