"""
SIGNIFICANT_MONTHLY_CHANGE(월간 위험도 급변 기준) 임계값을 데이터 기반으로 잡기 위해,
전월 대비 risk_probability 변화량의 분포를 확인한다.

주의: product_month_risk는 상품별로 연속된 달이 다 있는 게 아니라 Test 샘플만
있어서, "전월"이 진짜 바로 전달인지는 보장 안 됨. 그래도 대략적인 분포 참고용으로 확인.

실행 방법:
    python check_monthly_change_distribution.py
"""

import pandas as pd
from config import engine

df = pd.read_sql(
    "SELECT parent_asin, `year_month`, risk_probability FROM product_month_risk "
    "ORDER BY parent_asin, `year_month`",
    con=engine
)

# 같은 상품 안에서 전월 대비 변화량 계산
df["prev_risk"] = df.groupby("parent_asin")["risk_probability"].shift(1)
df["delta"] = df["risk_probability"] - df["prev_risk"]
df = df.dropna(subset=["delta"])

print(f"전월 대비 비교 가능한 행 수: {len(df)}")
print("\n--- 변화량(delta) 절댓값 분포 ---")
print(df["delta"].abs().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
