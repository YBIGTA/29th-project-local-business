from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")
LABELED_PATH = Path("data/processed/product_month_labeled.parquet")
ACTIVITY_PATH = Path("data/interim/product_month_activity.parquet")
TEST_START = "2022-09"
WINDOWS = [1, 3, 6, 12]

labeled = pd.read_parquet(LABELED_PATH)[["parent_asin", "year_month"]].copy()
activity = pd.read_parquet(ACTIVITY_PATH).copy()

required = {"parent_asin", "year_month"}
missing = required - set(activity.columns)
if missing:
    raise ValueError(f"활동 데이터 필수 열 누락: {sorted(missing)}")

# 최종 Test 시작 전의 상품×월만 이력 확보율에 사용한다.
targets = labeled.loc[labeled["year_month"].astype(str) < TEST_START].copy()
targets["month_code"] = pd.PeriodIndex(
    targets["year_month"].astype(str), freq="M"
).asi8

activity = activity.copy()
activity["month_code"] = pd.PeriodIndex(
    activity["year_month"].astype(str), freq="M"
).asi8

if activity.duplicated(["parent_asin", "year_month"]).any():
    raise ValueError("product_month_activity에 상품×월 키 중복이 있습니다.")

target_keys = set(zip(targets["parent_asin"], targets["year_month"].astype(str)))
activity_keys = set(zip(activity["parent_asin"], activity["year_month"].astype(str)))
current_key_match_rate = len(target_keys & activity_keys) / len(target_keys)

activity_months = {
    asin: np.sort(group["month_code"].unique())
    for asin, group in activity.groupby("parent_asin", sort=False)
}

coverage_rows = []
for window in WINDOWS:
    counts = np.zeros(len(targets), dtype=np.int16)

    for asin, idx in targets.groupby("parent_asin", sort=False).groups.items():
        months = activity_months.get(asin)
        if months is None:
            continue

        target_months = targets.loc[idx, "month_code"].to_numpy()
        left = np.searchsorted(months, target_months - window, side="left")
        right = np.searchsorted(months, target_months, side="left")
        counts[np.asarray(idx)] = right - left

    coverage_rows.append(
        {
            "window_months": window,
            "target_rows": len(targets),
            "mean_active_months_in_window": round(float(counts.mean()), 4),
            "rows_with_any_history": int((counts >= 1).sum()),
            "rate_with_any_history": round(float((counts >= 1).mean()), 6),
            "rows_with_half_window_history": int(
                (counts >= int(np.ceil(window / 2))).sum()
            ),
            "rate_with_half_window_history": round(
                float((counts >= int(np.ceil(window / 2))).mean()), 6
            ),
            "rows_with_full_window_history": int((counts >= window).sum()),
            "rate_with_full_window_history": round(
                float((counts >= window).mean()), 6
            ),
            "rows_with_no_history": int((counts == 0).sum()),
            "rate_with_no_history": round(float((counts == 0).mean()), 6),
        }
    )

coverage = pd.DataFrame(coverage_rows)
schema = pd.DataFrame(
    {
        "column": activity.columns.astype(str),
        "dtype": activity.dtypes.astype(str).to_numpy(),
    }
)

summary = {
    "test_data_used": False,
    "target_period": f"2016-01~2022-08",
    "target_rows": len(targets),
    "target_unique_products": int(targets["parent_asin"].nunique()),
    "activity_rows": len(activity),
    "activity_unique_products": int(activity["parent_asin"].nunique()),
    "current_target_activity_key_match_rate": round(current_key_match_rate, 6),
    "activity_columns": activity.columns.astype(str).tolist(),
}

coverage.to_csv(BASE / "history_window_coverage.csv", index=False, encoding="utf-8-sig")
schema.to_csv(BASE / "activity_data_schema.csv", index=False, encoding="utf-8-sig")
(BASE / "history_window_audit.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A1 기간별 이력 확보율 점검 완료 =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n[1·3·6·12개월 이력 확보율]")
print(coverage.to_string(index=False))
print("\n[활동 데이터 컬럼]")
print(", ".join(activity.columns.astype(str).tolist()))
