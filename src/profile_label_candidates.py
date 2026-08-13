from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_PATH = ROOT / "data/interim/product_month_activity.parquet"
REPORT_DIR = ROOT / "01_midterm/2-1_data_label"
OUTPUT_PATH = ROOT / "data/interim/product_month_with_next_month.parquet"

activity = pd.read_parquet(ACTIVITY_PATH)
activity["year_month"] = pd.PeriodIndex(
    activity["year_month"], freq="M"
)

# 같은 상품의 다음 달 집계값을 t월 행 옆에 붙인다.
next_month = activity[
    [
        "parent_asin",
        "year_month",
        "review_count",
        "avg_rating",
        "low_rating_count",
        "low_rating_ratio",
    ]
].copy()

next_month["year_month"] = next_month["year_month"] - 1
next_month = next_month.rename(
    columns={
        "review_count": "next_review_count",
        "avg_rating": "next_avg_rating",
        "low_rating_count": "next_low_rating_count",
        "low_rating_ratio": "next_low_rating_ratio",
    }
)

dataset = activity.merge(
    next_month,
    on=["parent_asin", "year_month"],
    how="left",
)

dataset["year_month"] = dataset["year_month"].astype(str)
dataset.to_parquet(OUTPUT_PATH, index=False)

eligible = dataset[dataset["next_review_count"] >= 5].copy()

threshold_rows = []
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
    threshold_rows.append(
        {
            "next_low_rating_ratio_threshold": threshold,
            "eligible_row_count": len(eligible),
            "positive_count": int(
                (eligible["next_low_rating_ratio"] >= threshold).sum()
            ),
            "positive_ratio": round(
                (eligible["next_low_rating_ratio"] >= threshold).mean(),
                4,
            ),
        }
    )

threshold_summary = pd.DataFrame(threshold_rows)
threshold_summary.to_csv(
    REPORT_DIR / "label_threshold_candidates.csv",
    index=False,
)

distribution = (
    eligible["next_low_rating_ratio"]
    .describe(percentiles=[0.25, 0.5, 0.75, 0.8, 0.9, 0.95])
    .rename("next_low_rating_ratio")
)
distribution.to_csv(
    REPORT_DIR / "next_low_rating_ratio_distribution.csv",
    header=True,
)

print("=== 다음 달 라벨 후보 점검 완료 ===")
print(f"전체 상품×월 행: {len(dataset):,}")
print(f"다음 달 리뷰 5개 이상 예측 가능 행: {len(eligible):,}")

print("\n=== 다음 달 저평점 비율 분포 ===")
print(distribution.to_string())

print("\n=== 임계값별 양성 비율 ===")
print(threshold_summary.to_string(index=False))

print("\n생성 파일")
print(OUTPUT_PATH)
print(REPORT_DIR / "label_threshold_candidates.csv")
print(REPORT_DIR / "next_low_rating_ratio_distribution.csv")
