from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("01_midterm/2-3_nlp_features")

CLEAN_REVIEWS_PATH = Path("data/interim/clean_reviews.parquet")
BASE_TEXT_PATH = Path("data/processed/product_month_text_features.parquet")
OUTPUT_PATH = Path("data/processed/product_month_text_features.parquet")
SUMMARY_PATH = OUT / "intensity_feature_summary.json"

KEYS = ["parent_asin", "year_month"]
LABEL = "is_low_rating_surge"
TRAIN_END = "2021-12"

# 약한 불만: 단순히 'little'이 아니라 문제 단어와 함께 나타난 경우만 잡는다.
MILD_ISSUE = re.compile(
    r"\b(?:slightly|somewhat|minor|mild|a little|a bit|little bit)\b"
    r".{0,40}\b(?:hot|warm|noise|noisy|leak|loose|weak|slow|broken|crack|"
    r"scratch|smell|odor|damage|defect|issue|problem)\b|"
    r"\b(?:hot|warm|noise|noisy|leak|loose|weak|slow|broken|crack|scratch|"
    r"smell|odor|damage|defect|issue|problem)\b.{0,40}"
    r"\b(?:slightly|somewhat|minor|mild|a little|a bit|little bit)\b",
    flags=re.IGNORECASE,
)

# 강한 정도 표현과 결함이 같이 등장한 경우.
SEVERE_ISSUE = re.compile(
    r"\b(?:extremely|incredibly|unbearably|dangerously|severely|completely|"
    r"totally|absolutely|barely)\b.{0,45}"
    r"\b(?:hot|warm|noise|noisy|leak|loose|weak|slow|broken|crack|smell|"
    r"odor|damage|defect|issue|problem|useless)\b|"
    r"\b(?:hot|warm|noise|noisy|leak|loose|weak|slow|broken|crack|smell|"
    r"odor|damage|defect|issue|problem|useless)\b.{0,45}"
    r"\b(?:extremely|incredibly|unbearably|dangerously|severely|completely|"
    r"totally|absolutely|barely)\b",
    flags=re.IGNORECASE,
)

# 사람의 안전과 직접 연결되는 표현은 가장 높은 강도로 분리한다.
SAFETY_RISK = re.compile(
    r"\b(?:burn(?:ed|ing)?|scald(?:ed|ing)?|electric shock|electrocut|"
    r"fire|smoke|sparks?|explod(?:e|ed|ing)|catch(?:es|ing)? fire|"
    r"dangerous|unsafe|hazard(?:ous)?|toxic|chemical burn|overheat(?:ed|ing)?)\b",
    flags=re.IGNORECASE,
)

# 심각한 결과·사용 불능. 단순 반품 의도와 달리 손해/작동 불능을 포착한다.
SEVERE_CONSEQUENCE = re.compile(
    r"\b(?:stopped working|quit working|does not work|would not work|"
    r"dead on arrival|dead after|broke(?:n)? (?:after|within|in)|"
    r"ruined|destroyed|damaged my|injur(?:y|ed)|hurt my|cut my|"
    r"melted|leaked everywhere|flooded|wasted (?:my )?money)\b",
    flags=re.IGNORECASE,
)


def prior_three_month_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    """동일 상품의 직전 최대 3개월 평균. 현재 월은 포함하지 않는다."""
    return (
        frame.groupby("parent_asin", group_keys=False)[column]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


for path in [CLEAN_REVIEWS_PATH, BASE_TEXT_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"필수 입력 파일 없음: {path}")

reviews = pd.read_parquet(
    CLEAN_REVIEWS_PATH,
    columns=KEYS + ["text_norm", "auto_title_flag"],
)
base = pd.read_parquet(BASE_TEXT_PATH)

if base.duplicated(KEYS).any():
    raise ValueError("기존 상품×월 텍스트 패널 키가 중복됩니다.")
if LABEL not in base.columns:
    raise ValueError(f"기존 텍스트 패널에 라벨이 없습니다: {LABEL}")

reviews["year_month"] = reviews["year_month"].astype(str)
base["year_month"] = base["year_month"].astype(str)

# 자동 생성 제목·빈 텍스트를 제외해 실제 작성 문장만 신호로 사용한다.
reviews = reviews.loc[
    reviews["auto_title_flag"].eq(0)
    & reviews["text_norm"].notna()
    & reviews["text_norm"].str.strip().ne("")
].copy()
text = reviews["text_norm"].astype(str)

reviews["intensity_mild_issue_flag"] = text.str.contains(MILD_ISSUE, na=False).astype("int8")
reviews["intensity_severe_issue_flag"] = text.str.contains(SEVERE_ISSUE, na=False).astype("int8")
reviews["intensity_safety_risk_flag"] = text.str.contains(SAFETY_RISK, na=False).astype("int8")
reviews["intensity_severe_consequence_flag"] = text.str.contains(
    SEVERE_CONSEQUENCE, na=False
).astype("int8")

# 안전 위험(3점) > 심각 결과·강한 결함(각 2점) > 약한 불만(1점).
reviews["intensity_score"] = (
    reviews["intensity_mild_issue_flag"]
    + 2 * reviews["intensity_severe_issue_flag"]
    + 3 * reviews["intensity_safety_risk_flag"]
    + 2 * reviews["intensity_severe_consequence_flag"]
).astype("int8")

flag_columns = [
    "intensity_mild_issue_flag",
    "intensity_severe_issue_flag",
    "intensity_safety_risk_flag",
    "intensity_severe_consequence_flag",
]

monthly = (
    reviews.groupby(KEYS, as_index=False)
    .agg(
        intensity_review_count=("text_norm", "size"),
        intensity_score_mean=("intensity_score", "mean"),
        intensity_score_max=("intensity_score", "max"),
        **{column: (column, "mean") for column in flag_columns},
    )
    .sort_values(KEYS)
    .reset_index(drop=True)
)

monthly = monthly.rename(
    columns={column: column.replace("_flag", "_rate_t") for column in flag_columns}
)
monthly["intensity_score_mean_t"] = monthly.pop("intensity_score_mean")
monthly["intensity_score_max_t"] = monthly.pop("intensity_score_max")

rate_columns = [
    "intensity_mild_issue_rate_t",
    "intensity_severe_issue_rate_t",
    "intensity_safety_risk_rate_t",
    "intensity_severe_consequence_rate_t",
    "intensity_score_mean_t",
]
for current in rate_columns:
    stem = current.removesuffix("_t")
    prior = f"{stem}_p3"
    delta = f"{stem}_delta"
    monthly[prior] = prior_three_month_mean(monthly, current).fillna(0.0)
    monthly[delta] = monthly[current] - monthly[prior]

feature_columns = [
    "intensity_review_count",
    "intensity_score_mean_t",
    "intensity_score_max_t",
] + rate_columns[:4] + [
    column
    for current in rate_columns
    for column in (
        f"{current.removesuffix('_t')}_p3",
        f"{current.removesuffix('_t')}_delta",
    )
]

result = base.merge(
    monthly[KEYS + feature_columns],
    on=KEYS,
    how="left",
    validate="one_to_one",
)
result[feature_columns] = result[feature_columns].fillna(0.0)

if len(result) != len(base):
    raise ValueError("강도 피처 결합 뒤 행 수가 달라졌습니다.")
if result.duplicated(KEYS).any():
    raise ValueError("강도 피처 결합 뒤 키가 중복됩니다.")

train = result.loc[result["year_month"] <= TRAIN_END].copy()
summary = {
    "input_review_rows_after_text_filter": int(len(reviews)),
    "product_month_rows": int(len(result)),
    "intensity_feature_count": int(len(feature_columns)),
    "feature_columns": feature_columns,
    "review_level_match_rates": {
        column: float(reviews[column].mean()) for column in flag_columns
    },
    "train_label_association": {
        column: {
            "negative_mean": float(train.loc[train[LABEL].eq(0), column].mean()),
            "positive_mean": float(train.loc[train[LABEL].eq(1), column].mean()),
        }
        for column in feature_columns
        if column != "intensity_review_count"
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT_PATH, index=False)
SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"saved: {OUTPUT_PATH}")
print(f"rows={len(result):,}, columns={len(result.columns):,}")
print(f"new_intensity_features={len(feature_columns)}")
print("\n[review-level match rates]")
for column, rate in summary["review_level_match_rates"].items():
    print(f"{column}: {rate:.4%}")
print("\n[train: positive vs negative mean]")
for column, values in summary["train_label_association"].items():
    print(
        f"{column}: neg={values['negative_mean']:.6f}, "
        f"pos={values['positive_mean']:.6f}"
    )
