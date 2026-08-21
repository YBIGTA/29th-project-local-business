"""
1:1로 비교했을 때 실제로 순위가 뒤집히는 쌍(risk_probability는 A가 높은데
weighted_risk는 B가 더 높은 경우)을 전체 데이터에서 직접 검색한다.

실행 방법:
    python find_true_reversal_pair.py
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

# risk_probability가 높은 편(상위 30개)인 후보들을 A로 놓고,
# 그보다 risk_probability는 낮지만 weighted_risk는 더 높은 B가 있는지 전체에서 검색
candidates_a = df.sort_values("risk_probability", ascending=False).head(30)

found_pairs = []
for _, row_a in candidates_a.iterrows():
    # A보다 risk_probability는 낮지만(최소 5%p 이상 차이), weighted_risk는 더 높은 B 찾기
    mask = (
        (df["risk_probability"] < row_a["risk_probability"] - 0.05) &
        (df["weighted_risk"] > row_a["weighted_risk"]) &
        (df["parent_asin"] != row_a["parent_asin"])
    )
    matches = df[mask]
    if len(matches) > 0:
        # weighted_risk가 가장 높은 B를 선택
        row_b = matches.sort_values("weighted_risk", ascending=False).iloc[0]
        found_pairs.append({
            "asin_a": row_a["parent_asin"], "year_month_a": row_a["year_month"],
            "risk_a": row_a["risk_probability"], "weighted_a": row_a["weighted_risk"],
            "asin_b": row_b["parent_asin"], "year_month_b": row_b["year_month"],
            "risk_b": row_b["risk_probability"], "weighted_b": row_b["weighted_risk"],
        })

if not found_pairs:
    print("역전 쌍을 찾지 못했습니다. 조건을 완화해서 다시 시도해보세요.")
else:
    result_df = pd.DataFrame(found_pairs).drop_duplicates(subset=["asin_a", "asin_b"])
    print(f"역전 쌍 {len(result_df)}건 발견:\n")
    print(result_df.to_string(index=False))
