from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/interim/product_month_with_next_month.parquet"

data = pd.read_parquet(INPUT_PATH)
data["year_month"] = pd.PeriodIndex(data["year_month"], freq="M")

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
history["past_3m_low_rating_ratio"] = (
    history[low_count_cols].sum(axis=1)
    / history["past_3m_review_count"].where(
        history["past_3m_review_count"] > 0
    )
)

data = data.merge(
    history[
        [
            "parent_asin",
            "target_month",
            "past_3m_review_count",
            "past_3m_low_rating_ratio",
        ]
    ],
    left_on=["parent_asin", "year_month"],
    right_on=["parent_asin", "target_month"],
    how="left",
).drop(columns="target_month")

eligible = data[
    (data["next_review_count"] >= 5)
    & (data["past_3m_review_count"] >= 5)
].copy()

eligible["is_low_rating_surge"] = (
    (eligible["next_low_rating_ratio"] >= 0.30)
    & (
        eligible["next_low_rating_ratio"]
        - eligible["past_3m_low_rating_ratio"]
        >= 0.15
    )
).astype(int)

eligible["year"] = eligible["year_month"].astype(str).str[:4]

summary = (
    eligible.groupby("year", as_index=False)
    .agg(
        eligible_product_month_count=("parent_asin", "size"),
        surge_count=("is_low_rating_surge", "sum"),
    )
)
summary["surge_ratio"] = (
    summary["surge_count"] / summary["eligible_product_month_count"]
).round(4)

print("=== 연도별 최종 라벨 후보 분포 ===")
print(summary.to_string(index=False))
