"""
"완만하지만 지속적인 추세"를 판단할 기준을 데이터 기반으로 잡기 위해,
상품별로 (전체 관찰기간 동안의 위험도 변화량) / (관찰 개월 수 - 1)를 계산해서
그 분포를 확인한다.

이는 전월 대비 변화량 분포(check_monthly_change_distribution.py, 급변 시점용)와는
다른 지표다. 여기서는 "꾸준한 평균 변화 속도"를 본다.

실행 방법:
    python check_avg_monthly_rate_distribution.py
"""

import pandas as pd
from config import engine

df = pd.read_sql(
    "SELECT parent_asin, `year_month`, risk_probability FROM product_month_risk "
    "ORDER BY parent_asin, `year_month`",
    con=engine
)

# 상품별로 관찰 개월 수, 시작/종료 시점의 risk_probability 계산
grouped = df.groupby("parent_asin").agg(
    n_months=("year_month", "count"),
    first_risk=("risk_probability", "first"),
    last_risk=("risk_probability", "last"),
)

# 관찰 기간이 2개월 이상인 상품만 (변화율 계산 가능한 경우)
grouped = grouped[grouped["n_months"] >= 2].copy()
grouped["overall_change"] = grouped["last_risk"] - grouped["first_risk"]
grouped["avg_monthly_rate"] = grouped["overall_change"] / (grouped["n_months"] - 1)

print(f"관찰 개월 2개월 이상인 상품 수: {len(grouped)}")
print("\n--- 상품별 평균 월간 변화율(avg_monthly_rate) 절댓값 분포 ---")
print(grouped["avg_monthly_rate"].abs().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
