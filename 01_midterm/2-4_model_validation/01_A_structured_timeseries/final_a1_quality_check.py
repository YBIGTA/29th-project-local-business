from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")
KEYS = ["parent_asin", "year_month"]
TEST_START = "2022-09"
WINDOWS = [1, 3, 6, 12]

issues = []
checks = {}

panel = pd.read_parquet(BASE / "structured_modeling_panel.parquet")
history = pd.read_parquet(BASE / "multiwindow_history_features_pretest.parquet")
activity = pd.read_parquet("data/interim/product_month_activity.parquet")
spec = pd.read_csv(BASE / "structured_feature_spec.csv")

current_features = spec.loc[
    (spec["decision"] == "use")
    & (spec["feature_group"] == "current_state"),
    "feature_name",
].tolist()

history_features = [c for c in history.columns if c not in KEYS]
final_features = current_features + history_features

# 1. 구조·키·금지 피처 확인
checks["panel_rows"] = len(panel)
checks["history_rows"] = len(history)
checks["panel_key_duplicates"] = int(panel.duplicated(KEYS).sum())
checks["history_key_duplicates"] = int(history.duplicated(KEYS).sum())
checks["current_feature_count"] = len(current_features)
checks["history_feature_count"] = len(history_features)
checks["final_feature_count"] = len(final_features)

if len(panel) != 68087:
    issues.append(f"정형 패널 행 수 불일치: {len(panel)}")
if len(history) != 62041:
    issues.append(f"다중 기간 패널 행 수 불일치: {len(history)}")
if checks["panel_key_duplicates"] or checks["history_key_duplicates"]:
    issues.append("상품×월 공통 키 중복 발견")
if len(final_features) != 45:
    issues.append(f"최종 후보 피처 수 불일치: {len(final_features)}")

forbidden = [
    c for c in final_features
    if c == "is_low_rating_surge"
    or c.startswith("next_")
    or c.startswith("auto_title_")
    or c.startswith("T_")
]
checks["forbidden_features_in_final_candidate"] = forbidden
if forbidden:
    issues.append(f"금지 피처 포함: {forbidden}")

history_max_month = history["year_month"].astype(str).max()
checks["history_max_year_month"] = history_max_month
if history["year_month"].astype(str).ge(TEST_START).any():
    issues.append("다중 기간 이력 파일에 Test 기간이 포함됨")

# 2. 현재 상태 7개가 해당 t월 활동 데이터와 일치하는지 확인
current_merged = panel[KEYS + current_features].merge(
    activity[KEYS + current_features],
    on=KEYS,
    how="left",
    suffixes=("_panel", "_activity"),
    validate="one_to_one",
)

if len(current_merged) != len(panel):
    issues.append("정형 패널과 활동 데이터의 현재 시점 키 결합 행 수 불일치")

current_mismatch = {}
for col in current_features:
    left = pd.to_numeric(current_merged[f"{col}_panel"], errors="coerce").to_numpy(float)
    right = pd.to_numeric(current_merged[f"{col}_activity"], errors="coerce").to_numpy(float)
    mismatch = int((~np.isclose(left, right, equal_nan=True, atol=1e-9)).sum())
    current_mismatch[col] = mismatch
    if mismatch:
        issues.append(f"현재 상태 피처 불일치: {col} {mismatch}건")

checks["current_feature_mismatch_counts"] = current_mismatch

# 3. 1·3·6·12개월 이력 피처가 [t-window, t-1]만 사용했는지 전수 재계산
history = history.reset_index(drop=True).copy()
history["month_code"] = pd.PeriodIndex(
    history["year_month"].astype(str), freq="M"
).asi8

activity = activity.copy()
activity["month_code"] = pd.PeriodIndex(
    activity["year_month"].astype(str), freq="M"
).asi8
activity = activity.sort_values(["parent_asin", "month_code"]).reset_index(drop=True)

activity_groups = {
    asin: group for asin, group in activity.groupby("parent_asin", sort=False)
}

history_mismatch = {
    f"{window}m_review_count": 0
    for window in WINDOWS
}
history_mismatch.update({
    f"{window}m_low_rating_count": 0
    for window in WINDOWS
})
history_mismatch.update({
    f"{window}m_low_rating_ratio": 0
    for window in WINDOWS
})
history_mismatch.update({
    f"{window}m_avg_rating": 0
    for window in WINDOWS
})

for asin, index in history.groupby("parent_asin", sort=False).groups.items():
    group = activity_groups.get(asin)
    if group is None:
        issues.append(f"활동 데이터에 없는 상품: {asin}")
        continue

    index = np.asarray(list(index))
    target_months = history.loc[index, "month_code"].to_numpy()
    month_codes = group["month_code"].to_numpy()

    reviews = pd.to_numeric(group["review_count"], errors="coerce").fillna(0).to_numpy(float)
    low_counts = pd.to_numeric(group["low_rating_count"], errors="coerce").fillna(0).to_numpy(float)
    avg_rating = pd.to_numeric(group["avg_rating"], errors="coerce").fillna(0).to_numpy(float)

    review_prefix = np.r_[0.0, np.cumsum(reviews)]
    low_prefix = np.r_[0.0, np.cumsum(low_counts)]
    rating_weighted_prefix = np.r_[0.0, np.cumsum(avg_rating * reviews)]

    for window in WINDOWS:
        left = np.searchsorted(month_codes, target_months - window, side="left")
        right = np.searchsorted(month_codes, target_months, side="left")

        expected_reviews = review_prefix[right] - review_prefix[left]
        expected_low = low_prefix[right] - low_prefix[left]
        expected_ratio = np.full(len(index), np.nan)
        expected_avg = np.full(len(index), np.nan)
        valid = expected_reviews > 0
        expected_ratio[valid] = expected_low[valid] / expected_reviews[valid]
        expected_avg[valid] = (
            rating_weighted_prefix[right][valid]
            - rating_weighted_prefix[left][valid]
        ) / expected_reviews[valid]

        actuals = {
            "review_count": history.loc[index, f"history_{window}m_review_count"].to_numpy(float),
            "low_rating_count": history.loc[index, f"history_{window}m_low_rating_count"].to_numpy(float),
            "low_rating_ratio": history.loc[index, f"history_{window}m_low_rating_ratio"].to_numpy(float),
            "avg_rating": history.loc[index, f"history_{window}m_avg_rating"].to_numpy(float),
        }
        expecteds = {
            "review_count": expected_reviews,
            "low_rating_count": expected_low,
            "low_rating_ratio": expected_ratio,
            "avg_rating": expected_avg,
        }

        for name in actuals:
            mismatch = int(
                (~np.isclose(
                    actuals[name],
                    expecteds[name],
                    equal_nan=True,
                    atol=1e-8,
                )).sum()
            )
            history_mismatch[f"{window}m_{name}"] += mismatch

checks["history_feature_full_recalculation_mismatch_counts"] = history_mismatch
bad_history = {k: v for k, v in history_mismatch.items() if v}
if bad_history:
    issues.append(f"과거 이력 피처 전수 재계산 불일치: {bad_history}")

# 4. Valid 예측값에 Test 기간이 없는지 확인
prediction_checks = {}
for filename in [
    "validation_predictions.csv",
    "multiwindow_validation_predictions.csv",
]:
    pred = pd.read_csv(BASE / filename)
    max_month = pred["year_month"].astype(str).max()
    has_test = bool(pred["year_month"].astype(str).ge(TEST_START).any())
    prediction_checks[filename] = {
        "rows": len(pred),
        "max_year_month": max_month,
        "contains_test_period": has_test,
    }
    if has_test:
        issues.append(f"{filename}에 Test 기간 예측값 포함")

checks["validation_prediction_checks"] = prediction_checks
checks["status"] = "PASSED" if not issues else "FAILED"
checks["issues"] = issues

(BASE / "final_a1_quality_check.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A1 최종 누수·정합성 점검 =====")
print(f"상태: {checks['status']}")
print(f"최종 후보 피처 수: {checks['final_feature_count']} (현재 {checks['current_feature_count']} + 이력 {checks['history_feature_count']})")
print(f"금지 피처 포함: {checks['forbidden_features_in_final_candidate']}")
print(f"Test 포함 다중 기간 이력: {history_max_month >= TEST_START}")
print(f"현재 상태 피처 불일치 합계: {sum(current_mismatch.values())}")
print(f"과거 이력 전수 재계산 불일치 합계: {sum(history_mismatch.values())}")
print(f"Valid 예측에 Test 포함: {any(v['contains_test_period'] for v in prediction_checks.values())}")
print(f"상세 저장: {BASE / 'final_a1_quality_check.json'}")

if issues:
    print("\n[발견 항목]")
    for issue in issues:
        print(f"- {issue}")
    raise SystemExit(1)

print("\nA1의 입력 피처·시간 분할·과거 이력 계산에서 누수와 정합성 문제를 발견하지 못했습니다.")
