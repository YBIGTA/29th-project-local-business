"""
다양한 특성의 상품 asin을 뽑아, generate_report.py 테스트용으로 쓴다.

카테고리:
    1. 위험도 매우 높음 (risk_probability 상위)
    2. 위험도 매우 낮음 (risk_probability 하위)
    3. 리뷰 수 많음 (신뢰도 높은 케이스)
    4. 리뷰 수 매우 적음 (표본부족 케이스)
    5. 급증한 토픽이 없는 경우 (delta가 낮은 상품)

실행 방법:
    python pick_test_products.py
"""

import pandas as pd
from config import engine

risk_df = pd.read_sql(
    "SELECT parent_asin, `year_month`, risk_probability, current_review_count, "
    "operational_confidence, review_evidence_confidence "
    "FROM product_month_risk",
    con=engine
)

print("=" * 60)
print("1. 위험도 매우 높음 (상위 5개)")
print("=" * 60)
print(risk_df.sort_values("risk_probability", ascending=False).head(5)[
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

print("\n" + "=" * 60)
print("2. 위험도 매우 낮음 (하위 5개)")
print("=" * 60)
print(risk_df.sort_values("risk_probability", ascending=True).head(5)[
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

print("\n" + "=" * 60)
print("3. 리뷰 수 많음 (상위 5개, 신뢰도 높은 케이스)")
print("=" * 60)
print(risk_df.sort_values("current_review_count", ascending=False).head(5)[
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

print("\n" + "=" * 60)
print("4. 리뷰 수 매우 적음 (하위 5개, 표본부족 케이스)")
print("=" * 60)
print(risk_df.sort_values("current_review_count", ascending=True).head(5)[
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

# 5. 급증한 토픽이 없는 상품: product_month_topics에서 delta의 최댓값이 낮은 상품
topics_df = pd.read_sql(
    "SELECT t.parent_asin, t.year_month, MAX(t.delta) as max_delta "
    "FROM product_month_topics t "
    "INNER JOIN product_month_risk r ON t.parent_asin = r.parent_asin AND t.year_month = r.year_month "
    "GROUP BY t.parent_asin, t.year_month",
    con=engine
)
print("\n" + "=" * 60)
print("5. 급증한 토픽이 없는 경우 (max delta 하위 5개)")
print("=" * 60)
print(topics_df.sort_values("max_delta", ascending=True).head(5).to_string(index=False))
