"""
L2. 월별 불만 급증 신호와 근거 리뷰 추출.

(1) 급증 신호
    상품x월마다 토픽별로 t월 비율이 직전 3개월보다 얼마나 올랐는지를 세로로
    펼쳐 순위를 매긴다. 새로 계산하는 값은 없고 L1이 만든 델타를 정리하는
    단계다.

    급증 기준을 2-1의 위험 라벨과 같은 15%p로 맞췄다. 그래야 텍스트 급증과
    저평점 급증을 같은 잣대로 비교할 수 있다.

    t월 리뷰가 최소 기준에 못 미치는 행은 제외한다. 리뷰 2건 중 1건이 불만이면
    델타가 50%p로 튀지만 이는 급증이 아니라 표본이 작은 것이다.

(2) 근거 리뷰
    급증한 상품x월마다 그 불만을 실제로 언급한 리뷰를 토픽별로 최대 5건 뽑는다.
    대시보드 상세 화면과 리포트에서 위험 판단의 근거로 쓰인다.
    저평점, 검증 구매, helpful vote가 높은 순으로 고른다.

    급증 건수가 만 단위이므로 매번 전체 리뷰를 훑으면 매우 느리다.
    대상 상품x월의 리뷰만 먼저 추린 뒤 키별로 묶어두고 조회한다.

    python3 src/nlp/build_surge_summary.py

출력: 01_midterm/2-3_nlp_features/monthly_complaint_surge.csv
      01_midterm/2-3_nlp_features/surge_topic_summary.csv
      01_midterm/2-3_nlp_features/evidence_reviews.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CLEAN_REVIEWS, DICTIONARY_PATH, KEY_COLUMNS, MIN_MONTH_REVIEWS,
    MONTH_FEATURES, REPORT_DIR, REVIEW_FEATURES, SURGE_DELTA_THRESHOLD,
)

TOP_EXAMPLES_PER_TOPIC = 5
EVIDENCE_TEXT_LIMIT = 400

with open(DICTIONARY_PATH, encoding="utf-8") as file:
    dictionary = json.load(file)
TOPIC_KEYS = [
    key for layer in ("cause", "consequence") for key in dictionary[layer]
]
TOPIC_LABELS = {
    key: dictionary[layer][key]["label"]
    for layer in ("cause", "consequence") for key in dictionary[layer]
}

print("=== 급증 신호 정리 ===")
features = pd.read_parquet(MONTH_FEATURES)
print(f"입력 {len(features):,}행")

usable = features[features["n_reviews_t"] >= MIN_MONTH_REVIEWS].copy()
print(f"t월 리뷰 {MIN_MONTH_REVIEWS}건 이상: {len(usable):,}행 "
      f"({len(usable) / len(features):.1%})")

records = []
for key in TOPIC_KEYS:
    subset = usable[KEY_COLUMNS + ["n_reviews_t", "n_reviews_p3",
                                   "text_reliability"]].copy()
    subset["topic"] = key
    subset["topic_label"] = TOPIC_LABELS[key]
    subset["rate_t"] = usable[f"topic_{key}_rate_t"]
    subset["rate_p3"] = usable[f"topic_{key}_rate_p3"]
    subset["rate_t_shrunk"] = usable[f"topic_{key}_rate_t_shrunk"]
    subset["delta"] = usable[f"topic_{key}_delta"]
    records.append(subset)

surge = pd.concat(records, ignore_index=True)
surge["is_surge"] = (surge["delta"] >= SURGE_DELTA_THRESHOLD).astype(int)
surge = surge.sort_values("delta", ascending=False).reset_index(drop=True)
surge.to_csv(REPORT_DIR / "monthly_complaint_surge.csv",
             index=False, encoding="utf-8-sig")
print(f"급증 신호 표: {len(surge):,}행 (토픽 {len(TOPIC_KEYS)}개 x 상품x월)")

summary = (
    surge.groupby(["topic", "topic_label"])
    .agg(
        n_product_months=("delta", "size"),
        mean_delta=("delta", "mean"),
        p90_delta=("delta", lambda s: s.quantile(0.90)),
        max_delta=("delta", "max"),
        n_surge=("is_surge", "sum"),
        surge_ratio=("is_surge", "mean"),
    )
    .round(4)
    .sort_values("n_surge", ascending=False)
)
summary.to_csv(REPORT_DIR / "surge_topic_summary.csv", encoding="utf-8-sig")

print(f"\n=== 토픽별 급증 요약 (델타 {SURGE_DELTA_THRESHOLD:.0%}p 이상) ===")
print(summary.to_string())

# ---------------------------------------------------------------
# 근거 리뷰
# ---------------------------------------------------------------
print("\n=== 근거 리뷰 추출 ===")
surging = surge[surge["is_surge"] == 1].copy()
print(f"급증 상품x월-토픽 {len(surging):,}건")

if surging.empty:
    pd.DataFrame().to_csv(REPORT_DIR / "evidence_reviews.csv", index=False)
    raise SystemExit("급증 사례가 없어 근거 리뷰를 만들지 않는다.")

reviews = pd.read_parquet(REVIEW_FEATURES)
raw = pd.read_parquet(CLEAN_REVIEWS, columns=["review_id", "text_norm"])
reviews = reviews.merge(raw, on="review_id", how="left")

# 대상 상품x월의 리뷰만 남긴다
targets = surging[KEY_COLUMNS].drop_duplicates()
reviews = reviews.merge(targets, on=KEY_COLUMNS, how="inner")
print(f"대상 상품x월 {len(targets):,}개, 해당 리뷰 {len(reviews):,}건")

# 우선순위대로 미리 정렬해 두면 그룹 안에서 앞부분만 잘라 쓰면 된다
reviews = reviews.sort_values(
    ["is_low_rating", "verified_purchase", "helpful_vote"], ascending=False
)
groups = {key: group for key, group in reviews.groupby(KEY_COLUMNS, sort=False)}

evidence = []
for row in surging.itertuples(index=False):
    group = groups.get((row.parent_asin, row.year_month))
    if group is None:
        continue
    matched = group[group[f"topic_{row.topic}"] == 1].head(TOP_EXAMPLES_PER_TOPIC)
    for review in matched.itertuples(index=False):
        evidence.append({
            "parent_asin": row.parent_asin,
            "year_month": row.year_month,
            "topic": row.topic,
            "topic_label": row.topic_label,
            "delta": row.delta,
            "n_reviews_t": row.n_reviews_t,
            "rating": review.rating,
            "verified_purchase": review.verified_purchase,
            "helpful_vote": review.helpful_vote,
            "review_text": str(review.text_norm)[:EVIDENCE_TEXT_LIMIT],
        })

evidence = pd.DataFrame(evidence)
evidence.to_csv(REPORT_DIR / "evidence_reviews.csv",
                index=False, encoding="utf-8-sig")
print(f"근거 리뷰 {len(evidence):,}건")
if len(evidence):
    print(f"급증 사례당 평균 {len(evidence) / len(surging):.1f}건")

print("\n생성 파일")
print(REPORT_DIR / "monthly_complaint_surge.csv")
print(REPORT_DIR / "surge_topic_summary.csv")
print(REPORT_DIR / "evidence_reviews.csv")
