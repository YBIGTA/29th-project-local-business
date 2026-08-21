"""
raw_winner와 weighted_winner가 실제로 다르게 나오는 상품 쌍(순위 역전 사례)을 찾는다.
compare_products.py의 winner_changed=True 케이스를 테스트하기 위함.

계산 방식은 retrieval.py의 assess_confidence와 동일하게 맞춘다:
    combined_confidence = min(operational_confidence, review_evidence_confidence)
    weighted_risk = risk_probability * combined_confidence

실행 방법:
    python find_reversal_pair.py
"""

import pandas as pd
from config import engine

df = pd.read_sql(
    "SELECT parent_asin, `year_month`, risk_probability, "
    "operational_confidence, review_evidence_confidence, current_review_count "
    "FROM product_month_risk",
    con=engine
)

df["combined_confidence"] = df[["operational_confidence", "review_evidence_confidence"]].min(axis=1)
df["weighted_risk"] = df["risk_probability"] * df["combined_confidence"]

# risk_probability 상위 15개 안에서, weighted_risk 순위가 크게 밀리는 쌍을 찾는다
top15 = df.sort_values("risk_probability", ascending=False).head(15).reset_index(drop=True)
top15["raw_rank"] = top15.index + 1
top15_by_weighted = top15.sort_values("weighted_risk", ascending=False).reset_index(drop=True)
top15_by_weighted["weighted_rank"] = top15_by_weighted.index + 1

merged = top15.merge(
    top15_by_weighted[["parent_asin", "year_month", "weighted_rank"]],
    on=["parent_asin", "year_month"]
)
merged["rank_diff"] = merged["weighted_rank"] - merged["raw_rank"]

print("--- risk_probability 상위 15개 (raw_rank) vs weighted_risk 순위(weighted_rank) ---")
print(merged[["parent_asin", "year_month", "risk_probability", "combined_confidence",
              "weighted_risk", "current_review_count", "raw_rank", "weighted_rank", "rank_diff"]]
      .sort_values("rank_diff").to_string(index=False))
