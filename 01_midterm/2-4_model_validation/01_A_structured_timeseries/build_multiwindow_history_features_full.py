from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")
LABELED_PATH = Path("data/processed/product_month_labeled.parquet")
ACTIVITY_PATH = Path("data/interim/product_month_activity.parquet")
TEST_START = "2023-03"
WINDOWS = [1, 3, 6, 12]

KEYS = ["parent_asin", "year_month"]
COUNT_COLUMNS = ["review_count", "low_rating_count"]
WEIGHTED_MEAN_COLUMNS = [
    "avg_rating",
    "verified_purchase_ratio",
    "mean_helpful_vote",
    "text_available_ratio",
]

labeled = pd.read_parquet(LABELED_PATH)[KEYS].copy()
activity = pd.read_parquet(ACTIVITY_PATH).copy()

required = set(KEYS + COUNT_COLUMNS + WEIGHTED_MEAN_COLUMNS)
missing = required - set(activity.columns)
if missing:
    raise ValueError(f"활동 데이터에 필요한 열이 없습니다: {sorted(missing)}")

if activity.duplicated(KEYS).any():
    raise ValueError("활동 데이터의 상품×월 키가 중복됩니다.")

# 최종 평가용: 라벨 없이 모든 대상 월의 과거 이력 피처를 생성한다.
targets = labeled.loc[labeled["year_month"].astype(str) < TEST_START].copy()
targets = targets.reset_index(drop=True)
targets["month_code"] = pd.PeriodIndex(
    targets["year_month"].astype(str), freq="M"
).asi8

activity = activity.copy()
activity["month_code"] = pd.PeriodIndex(
    activity["year_month"].astype(str), freq="M"
).asi8
activity = activity.sort_values(["parent_asin", "month_code"]).reset_index(drop=True)

n_targets = len(targets)
output = targets[KEYS].copy()
feature_values: dict[str, np.ndarray] = {}

for window in WINDOWS:
    for suffix in [
        "review_count",
        "low_rating_count",
        "low_rating_ratio",
        "avg_rating",
        "verified_purchase_ratio",
        "mean_helpful_vote",
        "text_available_ratio",
        "active_month_count",
        "coverage_ratio",
    ]:
        feature_values[f"history_{window}m_{suffix}"] = np.full(n_targets, np.nan)

target_groups = targets.groupby("parent_asin", sort=False).groups
activity_groups = {
    asin: group for asin, group in activity.groupby("parent_asin", sort=False)
}

for asin, target_index in target_groups.items():
    group = activity_groups.get(asin)
    if group is None:
        continue

    target_index = np.asarray(list(target_index))
    target_months = targets.loc[target_index, "month_code"].to_numpy()
    month_codes = group["month_code"].to_numpy()

    reviews = pd.to_numeric(group["review_count"], errors="coerce").fillna(0).to_numpy(float)
    low_counts = pd.to_numeric(
        group["low_rating_count"], errors="coerce"
    ).fillna(0).to_numpy(float)

    weighted_arrays = {}
    for col in WEIGHTED_MEAN_COLUMNS:
        values = pd.to_numeric(group[col], errors="coerce").fillna(0).to_numpy(float)
        weighted_arrays[col] = values * reviews

    prefix_reviews = np.r_[0.0, np.cumsum(reviews)]
    prefix_low = np.r_[0.0, np.cumsum(low_counts)]
    prefix_weighted = {
        col: np.r_[0.0, np.cumsum(values)]
        for col, values in weighted_arrays.items()
    }

    for window in WINDOWS:
        # [t-window, t-1]의 달만 사용한다. t월과 미래 정보는 포함하지 않는다.
        left = np.searchsorted(month_codes, target_months - window, side="left")
        right = np.searchsorted(month_codes, target_months, side="left")

        review_sum = prefix_reviews[right] - prefix_reviews[left]
        low_sum = prefix_low[right] - prefix_low[left]
        active_month_count = right - left

        feature_values[f"history_{window}m_review_count"][target_index] = review_sum
        feature_values[f"history_{window}m_low_rating_count"][target_index] = low_sum
        feature_values[f"history_{window}m_active_month_count"][target_index] = active_month_count
        feature_values[f"history_{window}m_coverage_ratio"][target_index] = (
            active_month_count / window
        )

        valid_review = review_sum > 0
        low_ratio = np.full(len(target_index), np.nan)
        low_ratio[valid_review] = low_sum[valid_review] / review_sum[valid_review]
        feature_values[f"history_{window}m_low_rating_ratio"][target_index] = low_ratio

        for col in WEIGHTED_MEAN_COLUMNS:
            weighted_sum = prefix_weighted[col][right] - prefix_weighted[col][left]
            mean_value = np.full(len(target_index), np.nan)
            mean_value[valid_review] = weighted_sum[valid_review] / review_sum[valid_review]
            feature_values[f"history_{window}m_{col}"][target_index] = mean_value

for name, values in feature_values.items():
    output[name] = values

# 최근 3개월과 장기 평균의 차이: 장기 평균보다 최근 품질이 악화됐는지 나타낸다.
for long_window in [6, 12]:
    output[f"history_3m_vs_{long_window}m_low_rating_ratio_delta"] = (
        output["history_3m_low_rating_ratio"]
        - output[f"history_{long_window}m_low_rating_ratio"]
    )

feature_columns = [c for c in output.columns if c not in KEYS]
spec_rows = []
for col in feature_columns:
    window = int(col.split("_")[1].replace("m", "")) if col.startswith("history_") else None
    if "vs_" in col:
        definition = "직전 3개월 저평점 비율 - 장기 저평점 비율"
    elif col.endswith("coverage_ratio"):
        definition = "해당 기간 중 리뷰가 있었던 월의 비율"
    elif col.endswith("active_month_count"):
        definition = "해당 기간 중 리뷰가 있었던 월 수"
    elif col.endswith("low_rating_ratio"):
        definition = "해당 기간 저평점 리뷰 수 / 전체 리뷰 수"
    elif col.endswith("review_count"):
        definition = "해당 기간 전체 리뷰 수"
    elif col.endswith("low_rating_count"):
        definition = "해당 기간 저평점 리뷰 수"
    else:
        definition = "리뷰 수 가중 기간 평균"

    spec_rows.append(
        {
            "feature_name": col,
            "window_months": window,
            "definition": definition,
            "missing_rate": round(float(output[col].isna().mean()), 6),
            "uses_current_month": False,
            "uses_future_data": False,
        }
    )

spec = pd.DataFrame(spec_rows)
summary = {
    "test_data_used": False,
    "feature_period": "2016-01~2023-02",
    "rows": len(output),
    "unique_products": int(output["parent_asin"].nunique()),
    "key_duplicate_count": int(output.duplicated(KEYS).sum()),
    "multiwindow_feature_count": len(feature_columns),
    "windows": WINDOWS,
    "uses_current_month": False,
    "uses_future_data": False,
}

if summary["key_duplicate_count"] != 0:
    raise ValueError("생성 결과에 상품×월 키 중복이 있습니다.")

output.to_parquet(BASE / "multiwindow_history_features_full.parquet", index=False)
spec.to_csv(BASE / "multiwindow_history_feature_spec_full.csv", index=False, encoding="utf-8-sig")
(BASE / "multiwindow_history_validation_full.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A1 다중 기간 이력 피처 생성 완료 =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n[기간별 저평점 비율 피처 결측률]")
print(
    spec.loc[
        spec["feature_name"].str.endswith("low_rating_ratio"),
        ["feature_name", "missing_rate"],
    ].to_string(index=False)
)
