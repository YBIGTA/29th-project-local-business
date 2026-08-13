import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/product_month_labeled.parquet"
DICTIONARY_PATH = ROOT / "01_midterm/2-1_data_label/data_dictionary.csv"
REPORT_PATH = ROOT / "01_midterm/2-1_data_label/validation_report.json"

data = pd.read_parquet(DATA_PATH)
dictionary = pd.read_csv(DICTIONARY_PATH)

required_columns = {
    "parent_asin",
    "year_month",
    "past_3m_review_count",
    "past_3m_low_rating_ratio",
    "next_review_count",
    "next_low_rating_ratio",
    "next_vs_past_3m_low_rating_change",
    "is_low_rating_surge",
}

missing_columns = sorted(required_columns - set(data.columns))
assert not missing_columns, f"필수 컬럼 누락: {missing_columns}"

duplicate_row_count = int(
    data.duplicated(["parent_asin", "year_month"]).sum()
)
assert duplicate_row_count == 0, "상품×월 중복 행이 존재합니다."

year_month = pd.PeriodIndex(data["year_month"], freq="M")
assert year_month.min() == pd.Period("2016-01", freq="M")
assert year_month.max() == pd.Period("2023-02", freq="M")

assert (data["next_review_count"] >= 5).all()
assert (data["past_3m_review_count"] >= 5).all()

expected_label = (
    (data["next_low_rating_ratio"] >= 0.30)
    & (data["next_vs_past_3m_low_rating_change"] >= 0.15)
).astype("int8")

label_mismatch_count = int(
    (data["is_low_rating_surge"] != expected_label).sum()
)
assert label_mismatch_count == 0, "라벨 계산 규칙이 일치하지 않습니다."

future_columns = sorted(
    column
    for column in data.columns
    if column.startswith("next_")
)

dictionary_usage = dictionary.set_index("column_name")["model_usage"].to_dict()
future_columns_not_blocked = [
    column
    for column in future_columns
    if dictionary_usage.get(column) != "라벨 검증만"
]
assert not future_columns_not_blocked, (
    "미래 정보 열의 사용 제한 표기가 누락되었습니다: "
    f"{future_columns_not_blocked}"
)

report = {
    "status": "PASS",
    "row_count": int(len(data)),
    "unique_product_count": int(data["parent_asin"].nunique()),
    "analysis_start": str(year_month.min()),
    "analysis_end": str(year_month.max()),
    "positive_count": int(data["is_low_rating_surge"].sum()),
    "positive_ratio": round(float(data["is_low_rating_surge"].mean()), 6),
    "duplicate_product_month_row_count": duplicate_row_count,
    "label_mismatch_count": label_mismatch_count,
    "future_columns": future_columns,
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

print("=== 최종 학습 데이터 검증 통과 ===")
for key, value in report.items():
    print(f"{key}: {value}")
print(f"\n검증 보고서: {REPORT_PATH}")
