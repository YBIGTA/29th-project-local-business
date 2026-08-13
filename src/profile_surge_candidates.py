from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/interim/product_month_with_next_month.parquet"
REPORT_DIR = ROOT / "01_midterm/2-1_data_label"

data = pd.read_parquet(INPUT_PATH)
data["year_month"] = pd.PeriodIndex(data["year_month"], freq="M")

# t월을 포함한 직전 3개월(t-2, t-1, t)의
# 리뷰 수와 저평점 수를 상품별·달력월 기준으로 합산한다.
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

count_columns = [
    "review_count_lag_0",
    "review_count_lag_1",
    "review_count_lag_2",
]
low_count_columns = [
    "low_rating_count_lag_0",
    "low_rating_count_lag_1",
    "low_rating_count_lag_2",
]

history[count_columns + low_count_columns] = history[
    count_columns + low_count_columns
].fillna(0)

history["past_3m_review_count"] = history[count_columns].sum(axis=1)
history["past_3m_low_rating_count"] = history[low_count_columns].sum(axis=1)
history["past_3m_low_rating_ratio"] = (
    history["past_3m_low_rating_count"]
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
            "past_3m_low_rating_count",
            "past_3m_low_rating_ratio",
        ]
    ],
    left_on=["parent_asin", "year_month"],
    right_on=["parent_asin", "target_month"],
    how="left",
).drop(columns="target_month")

# 다음 달 라벨이 안정적이고, 비교할 과거 이력도 있는 행만 남긴다.
eligible = data[
    (data["next_review_count"] >= 5)
    & (data["past_3m_review_count"] >= 5)
].copy()

eligible["next_vs_past_3m_low_rating_change"] = (
    eligible["next_low_rating_ratio"]
    - eligible["past_3m_low_rating_ratio"]
)

rows = []
for increase_threshold in [0.10, 0.15, 0.20]:
    is_risk = (
        (eligible["next_low_rating_ratio"] >= 0.30)
        & (
            eligible["next_vs_past_3m_low_rating_change"]
            >= increase_threshold
        )
    )
    rows.append(
        {
            "absolute_next_low_rating_ratio": 0.30,
            "minimum_increase_vs_past_3m": increase_threshold,
            "eligible_row_count": len(eligible),
            "risk_count": int(is_risk.sum()),
            "risk_ratio": round(float(is_risk.mean()), 4),
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(
    REPORT_DIR / "surge_label_candidates.csv",
    index=False,
)

print("=== 저평점 급증 라벨 후보 점검 완료 ===")
print(f"다음 달·직전 3개월 모두 리뷰 5개 이상인 행: {len(eligible):,}")
print("\n=== 절대 30% + 직전 3개월 대비 상승 기준 ===")
print(summary.to_string(index=False))

print("\n생성 파일")
print(REPORT_DIR / "surge_label_candidates.csv")
