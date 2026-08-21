"""
다양한 특성의 상품을 아까와 겹치지 않게 새로 뽑는다 (각 카테고리 6~10위).

카테고리:
    1. 위험도 매우 높음
    2. 위험도 매우 낮음
    3. 리뷰 수 많음
    4. 리뷰 수 매우 적음 (동률이 많아 무작위로 5개 선택)
    5. 급증한 토픽이 없는 경우

실행 방법:
    python pick_test_products_v2.py
"""

import pandas as pd
from config import engine

risk_df = pd.read_sql(
    "SELECT parent_asin, `year_month`, risk_probability, current_review_count "
    "FROM product_month_risk",
    con=engine
)

print("=" * 60)
print("1. 위험도 매우 높음 (6~10위)")
print("=" * 60)
print(risk_df.sort_values("risk_probability", ascending=False).iloc[5:10][
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

print("\n" + "=" * 60)
print("2. 위험도 매우 낮음 (6~10위)")
print("=" * 60)
print(risk_df.sort_values("risk_probability", ascending=True).iloc[5:10][
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

print("\n" + "=" * 60)
print("3. 리뷰 수 많음 (6~10위)")
print("=" * 60)
print(risk_df.sort_values("current_review_count", ascending=False).iloc[5:10][
    ["parent_asin", "year_month", "risk_probability", "current_review_count"]
].to_string(index=False))

print("\n" + "=" * 60)
print("4. 리뷰 수 매우 적음 (최소값 동률 중 다른 5개)")
print("=" * 60)
min_count = risk_df["current_review_count"].min()
print(risk_df[risk_df["current_review_count"] == min_count].sample(
    n=min(5, len(risk_df[risk_df["current_review_count"] == min_count])), random_state=42
)[["parent_asin", "year_month", "risk_probability", "current_review_count"]].to_string(index=False))

topics_df = pd.read_sql(
    "SELECT t.parent_asin, t.year_month, MAX(t.delta) as max_delta "
    "FROM product_month_topics t "
    "INNER JOIN product_month_risk r ON t.parent_asin = r.parent_asin AND t.year_month = r.year_month "
    "GROUP BY t.parent_asin, t.year_month",
    con=engine
)
print("\n" + "=" * 60)
print("5. 급증한 토픽이 없는 경우 (6~10위)")
print("=" * 60)
print(topics_df.sort_values("max_delta", ascending=True).iloc[5:10].to_string(index=False))
