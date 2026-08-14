"""
L1. 리뷰 단위 NLP 피처와 상품x월 텍스트 피처 생성.

두 산출물을 한 스크립트에서 만든다. 리뷰 단위 결과를 만든 직후 같은 메모리에서
집계하면 130만 건짜리 중간 파일을 다시 읽지 않아도 되기 때문이다.

(1) 리뷰 단위
    12개 불만 토픽의 등장 여부와 횟수, 원인과 결과의 동시 언급, 그리고 서사
    피처를 만든다. 서사 피처란 불만의 종류가 아니라 상황을 나타내는 것으로,
    고장까지 걸린 시간, 감정 강도, 확실성, 비교 표현이 여기 속한다.
    train 구간 분석에서 시간 표현이 가장 강한 신호였기 때문에 별도로 둔다.

(2) 상품x월 집계
    t월(달력월)과 직전 3개월을 각각 집계하고 그 차이를 델타로 만든다.
    예측 대상이 수준이 아니라 급증이므로 델타가 핵심 피처가 된다.
    관측 창을 달력월로 잡은 이유는 2-1 패널이 달력월 단위여서 다른 창을 쓰면
    결합 시점이 어긋나기 때문이다.

    t월 피처에는 t월 말까지 작성된 리뷰만 들어간다. t+1월 리뷰는 어떤 경우에도
    쓰지 않는다.

    python3 src/nlp/build_nlp_features.py

출력: data/interim/review_nlp_features.parquet
      data/processed/product_month_text_features.parquet
      01_midterm/2-3_nlp_features/text_feature_dictionary.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matching_rules as rules  # noqa: E402
from config import (  # noqa: E402
    ANALYSIS_END, ANALYSIS_START, CLEAN_REVIEWS, DICTIONARY_PATH,
    FUTURE_COLUMNS, KEY_COLUMNS, LABELED_PANEL, LABEL_COLUMN,
    MIN_MONTH_REVIEWS, MONTH_FEATURES, PAST_WINDOW_OFFSETS, REPORT_DIR,
    REVIEW_FEATURES, month_index, split_of,
)

# ---------------------------------------------------------------
# 서사 표현
# ---------------------------------------------------------------
MONTH_NAMES = (
    "january february march april may june july august september october "
    "november december jan feb mar apr jun jul aug sep sept oct nov dec"
).split()

DURATION = re.compile(r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b")
DURATION_UNIT_DAYS = {
    "day": 1, "days": 1, "week": 7, "weeks": 7,
    "month": 30, "months": 30, "year": 365, "years": 365,
}
# 기간 표현 뒤에 고장 동사가 오면 고장까지 걸린 시간으로 본다
FAILURE_AFTER_DURATION = re.compile(
    r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b[^.]{0,40}?"
    r"\b(broke|broken|died|failed|stopped|quit|cracked|leaked)\b"
)
FAILURE_BEFORE_DURATION = re.compile(
    r"\b(broke|broken|died|failed|stopped|quit|cracked|leaked)\b[^.]{0,25}?"
    r"\b(?:after|within|in)\s+(\d+)\s*"
    r"(day|days|week|weeks|month|months|year|years)\b"
)

EMOTIONAL = {
    "horrible", "terrible", "awful", "worst", "pathetic", "garbage",
    "junk", "trash", "furious", "angry", "disgusted", "ridiculous",
    "unacceptable", "outrageous", "disappointed", "disappointing",
}
CERTAINTY = {
    "definitely", "absolutely", "never", "always", "completely",
    "totally", "certainly", "clearly", "obviously",
}
COMPARATIVE = {
    "original", "oem", "previous", "before", "used to", "compared",
    "instead", "unlike", "worse than", "better than",
}

EARLY_FAILURE_DAYS = 90


def load_dictionary():
    with open(DICTIONARY_PATH, encoding="utf-8") as file:
        dictionary = json.load(file)
    topics = []
    for layer in ("cause", "consequence"):
        for key, topic in dictionary[layer].items():
            topics.append((layer, key, topic["terms"]))
    return topics


# ---------------------------------------------------------------
# 1. 리뷰 단위 피처
# ---------------------------------------------------------------
print("=== 리뷰 단위 피처 생성 ===")
if not CLEAN_REVIEWS.exists():
    raise SystemExit(f"입력이 없다: {CLEAN_REVIEWS}\nbuild_clean_reviews.py를 먼저 실행한다.")

data = pd.read_parquet(CLEAN_REVIEWS)
print(f"리뷰 {len(data):,}건")

topics = load_dictionary()
cause_keys = [key for layer, key, _ in topics if layer == "cause"]
consequence_keys = [key for layer, key, _ in topics if layer == "consequence"]
print(f"토픽 {len(topics)}개 (원인 {len(cause_keys)}, 결과 {len(consequence_keys)})")

token_lists = [text.split() if text else [] for text in data["text_norm"]]

for index, (layer, key, terms) in enumerate(topics, start=1):
    counts = [rules.count_terms(tokens, terms) for tokens in token_lists]
    data[f"topic_{key}_count"] = counts
    data[f"topic_{key}"] = (np.array(counts) > 0).astype(int)
    print(f"  [{index}/{len(topics)}] {key}: {data[f'topic_{key}'].mean():.2%}")

data["cause_topic_count"] = data[[f"topic_{k}" for k in cause_keys]].sum(axis=1)
data["consequence_topic_count"] = data[[f"topic_{k}" for k in consequence_keys]].sum(axis=1)
data["any_cause_topic"] = (data["cause_topic_count"] > 0).astype(int)
data["any_consequence_topic"] = (data["consequence_topic_count"] > 0).astype(int)

# 원인과 결과가 한 리뷰에 함께 나오면 신호가 더 강하다.
# 특히 반품 기간이 지난 뒤 고장났다는 조합이 train 구간에서 lift가 가장 높았다.
data["cooc_durability_return"] = (
    data["topic_durability_failure"] & data["topic_return_intent"]
).astype(int)
data["cooc_durability_blocked"] = (
    data["topic_durability_failure"] & data["topic_return_blocked"]
).astype(int)
data["cooc_shipping_return"] = (
    data["topic_shipping_packaging"] & data["topic_return_intent"]
).astype(int)
data["cooc_incompat_return"] = (
    data["topic_incompatibility"] & data["topic_return_intent"]
).astype(int)

print("서사 피처 생성 중...")


def time_to_failure(text):
    """고장까지 걸린 시간을 일수로 추정한다. 없으면 결측."""
    match = FAILURE_BEFORE_DURATION.search(text)
    if match:
        return int(match.group(2)) * DURATION_UNIT_DAYS[match.group(3)]
    match = FAILURE_AFTER_DURATION.search(text)
    if match:
        return int(match.group(1)) * DURATION_UNIT_DAYS[match.group(2)]
    return np.nan


texts = data["text_norm"].fillna("")
data["month_name_count"] = [
    sum(1 for token in tokens if token in MONTH_NAMES) for tokens in token_lists
]
data["duration_mention_flag"] = texts.str.contains(DURATION.pattern, regex=True).astype(int)
data["time_narrative_flag"] = (
    (data["month_name_count"] > 0) | (data["duration_mention_flag"] == 1)
).astype(int)
data["time_to_failure_days"] = [time_to_failure(text) for text in texts]
data["has_time_to_failure"] = data["time_to_failure_days"].notna().astype(int)
data["early_failure_flag"] = (
    data["time_to_failure_days"] <= EARLY_FAILURE_DAYS
).fillna(False).astype(int)

data["emotional_count"] = [
    sum(1 for token in tokens if token in EMOTIONAL) for tokens in token_lists
]
data["emotional_flag"] = (data["emotional_count"] > 0).astype(int)
data["certainty_count"] = [
    sum(1 for token in tokens if token in CERTAINTY) for tokens in token_lists
]
data["comparative_flag"] = texts.apply(
    lambda t: int(any(word in t for word in COMPARATIVE))
).astype(int)

review_columns = (
    ["review_id", "parent_asin", "year_month", "review_datetime_utc", "rating",
     "is_low_rating", "verified_purchase", "helpful_vote", "auto_title_flag",
     "is_likely_spanish", "word_count", "char_count", "sentence_count",
     "exclaim_count", "question_count", "caps_ratio"]
    + [f"topic_{k}" for _, k, _ in topics]
    + [f"topic_{k}_count" for _, k, _ in topics]
    + ["cause_topic_count", "consequence_topic_count", "any_cause_topic",
       "any_consequence_topic", "cooc_durability_return",
       "cooc_durability_blocked", "cooc_shipping_return",
       "cooc_incompat_return", "month_name_count", "duration_mention_flag",
       "time_narrative_flag", "time_to_failure_days", "has_time_to_failure",
       "early_failure_flag", "emotional_count", "emotional_flag",
       "certainty_count", "comparative_flag"]
)
data[review_columns].to_parquet(REVIEW_FEATURES, index=False)
print(f"저장: {REVIEW_FEATURES}")

# ---------------------------------------------------------------
# 2. 상품x월 집계
# ---------------------------------------------------------------
print("\n=== 상품x월 집계 ===")

topic_keys = [key for _, key, _ in topics]
FLAG_COLUMNS = (
    [f"topic_{k}" for k in topic_keys]
    + ["any_cause_topic", "any_consequence_topic",
       "cooc_durability_return", "cooc_durability_blocked",
       "cooc_shipping_return", "cooc_incompat_return",
       "time_narrative_flag", "early_failure_flag", "emotional_flag",
       "comparative_flag", "auto_title_flag", "is_low_rating"]
)
MEAN_COLUMNS = [
    "word_count", "sentence_count", "caps_ratio", "exclaim_count",
    "cause_topic_count", "consequence_topic_count", "emotional_count",
    "certainty_count", "month_name_count",
]

aggregation = {column: (column, "sum") for column in FLAG_COLUMNS}
aggregation.update({f"{column}_mean": (column, "mean") for column in MEAN_COLUMNS})
aggregation["n_reviews"] = ("review_id", "size")
aggregation["ttf_median_days"] = ("time_to_failure_days", "median")
aggregation["n_ttf_reviews"] = ("has_time_to_failure", "sum")

monthly = data.groupby(KEY_COLUMNS).agg(**aggregation).reset_index()
monthly["ym_index"] = monthly["year_month"].map(month_index)
print(f"상품x월 {len(monthly):,}행")

# 직전 3개월은 각 월의 원시 개수를 더한 뒤 비율을 계산한다.
# 월별 비율을 평균 내면 리뷰가 적은 달이 과대 반영되기 때문이다.
past_columns = FLAG_COLUMNS + ["n_reviews"]
past = None
for offset in PAST_WINDOW_OFFSETS:
    shifted = monthly[KEY_COLUMNS[:1] + ["ym_index"] + past_columns].copy()
    shifted["ym_index"] = shifted["ym_index"] + offset
    past = shifted if past is None else pd.concat([past, shifted], ignore_index=True)

past = past.groupby(["parent_asin", "ym_index"], as_index=False)[past_columns].sum()
past = past.rename(columns={column: f"{column}_p3sum" for column in past_columns})

result = monthly.merge(past, on=["parent_asin", "ym_index"], how="left")
for column in past_columns:
    result[f"{column}_p3sum"] = result[f"{column}_p3sum"].fillna(0)

result = result.rename(columns={"n_reviews": "n_reviews_t"})
result["n_reviews_p3"] = result["n_reviews_p3sum"]

# 비율과 델타
prior_weight = MIN_MONTH_REVIEWS  # 축소 보정 강도. 리뷰 5건이면 사전분포와 반반이 된다.
for column in FLAG_COLUMNS:
    rate_t = result[column] / result["n_reviews_t"]
    rate_p3 = result[f"{column}_p3sum"] / result["n_reviews_p3"].replace(0, np.nan)
    prior = rate_t.mean()
    result[f"{column}_rate_t"] = rate_t.round(5)
    result[f"{column}_rate_p3"] = rate_p3.round(5)
    result[f"{column}_delta"] = (rate_t - rate_p3).round(5)
    # 리뷰가 적은 달의 비율이 튀지 않도록 전체 평균 쪽으로 당긴다
    result[f"{column}_rate_t_shrunk"] = (
        (result[column] + prior_weight * prior)
        / (result["n_reviews_t"] + prior_weight)
    ).round(5)

# 리뷰 수가 적을수록 텍스트 지표를 덜 믿어야 한다
result["text_reliability"] = (
    result["n_reviews_t"] / (result["n_reviews_t"] + MIN_MONTH_REVIEWS)
).round(4)
result["low_volume_t"] = (result["n_reviews_t"] < MIN_MONTH_REVIEWS).astype(int)

# ---------------------------------------------------------------
# 3. 2-1 패널과 결합
# ---------------------------------------------------------------
print(f"\n2-1 패널 결합: {LABELED_PANEL}")
if not LABELED_PANEL.exists():
    raise SystemExit(
        f"2-1 패널이 없다: {LABELED_PANEL}\n2-1 파이프라인을 먼저 실행한다."
    )

panel = (
    pd.read_csv(LABELED_PANEL) if LABELED_PANEL.suffix == ".csv"
    else pd.read_parquet(LABELED_PANEL)
)
panel["year_month"] = panel["year_month"].astype(str)
print(f"패널 {len(panel):,}행")

assert panel["year_month"].min() >= ANALYSIS_START, "패널 시작이 기준과 다르다"
assert panel["year_month"].max() <= ANALYSIS_END, "패널 종료가 기준과 다르다"

keep_suffixes = ("_rate_t", "_rate_p3", "_rate_t_shrunk", "_delta", "_mean")
text_columns = [
    column for column in result.columns if column.endswith(keep_suffixes)
]
text_columns += ["n_reviews_t", "n_reviews_p3", "text_reliability",
                 "low_volume_t", "ttf_median_days", "n_ttf_reviews"]

output = panel[KEY_COLUMNS + [LABEL_COLUMN]].merge(
    result[KEY_COLUMNS + text_columns], on=KEY_COLUMNS, how="left"
)
output["has_text_t"] = output["n_reviews_t"].notna().astype(int)
output["has_text_p3"] = (output["n_reviews_p3"].fillna(0) > 0).astype(int)
output[text_columns] = output[text_columns].fillna(0)
output["split"] = output["year_month"].map(split_of)

# 정답 정보가 섞여 들어가지 않았는지 확인한다
leaked = [c for c in output.columns if c in FUTURE_COLUMNS or c.startswith("next_")]
assert not leaked, f"미래 정보가 포함됐다: {leaked}"

output.to_parquet(MONTH_FEATURES, index=False)
print(f"저장: {MONTH_FEATURES}  ({len(output):,}행 x {len(output.columns)}열)")

# ---------------------------------------------------------------
# 4. 피처 사전
# ---------------------------------------------------------------
def family_of(column):
    """
    2-4가 단계별로 피처를 늘려가며 비교할 수 있도록 성격을 표시한다.
    정형 피처와 값이 같은 열은 모델 입력에서 빼야 하므로 따로 표시한다.
    """
    if column in KEY_COLUMNS + [LABEL_COLUMN, "split"]:
        return "key"
    if column.startswith("is_low_rating"):
        return "X_duplicate_of_structured"
    if column in ("n_reviews_t", "n_reviews_p3", "text_reliability",
                  "low_volume_t", "has_text_t", "has_text_p3"):
        return "T_reliability"
    if column.startswith("cooc_"):
        return "T_cooc"
    for key in cause_keys:
        if column.startswith(f"topic_{key}") or column == "cause_topic_count_mean":
            return "T_cause"
    for key in consequence_keys:
        if column.startswith(f"topic_{key}") or column == "consequence_topic_count_mean":
            return "T_consequence"
    if column.startswith(("any_cause", "any_consequence")):
        return "T_cause" if "cause" in column else "T_consequence"
    if column.startswith(("time_narrative", "early_failure", "emotional",
                          "certainty", "comparative", "month_name", "ttf_",
                          "n_ttf")):
        return "T_narrative"
    if column.startswith("auto_title"):
        return "T_basic"
    return "T_basic"


dictionary_rows = [
    {"column": column, "family": family_of(column), "dtype": str(output[column].dtype)}
    for column in output.columns
]
feature_dictionary = pd.DataFrame(dictionary_rows)
feature_dictionary.to_csv(
    REPORT_DIR / "text_feature_dictionary.csv", index=False, encoding="utf-8-sig"
)

print("\n=== family 분포 ===")
print(feature_dictionary["family"].value_counts().to_string())
print(f"\n텍스트가 있는 행: {output['has_text_t'].mean():.1%}")
print(f"생성 파일: {REPORT_DIR / 'text_feature_dictionary.csv'}")
