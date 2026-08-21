"""
임계값 재검증: product_month_risk(Test 샘플 3,563건)로 범위를 좁혀서
mention_count, delta 분포를 다시 확인한다.

아까 check_distributions.py에서는 product_month_topics 전체(68,087 상품x월,
817,044 상품x월x토픽)를 기준으로 봤는데, 이는 실제로 RAG가 다루게 될
product_month_risk의 3,563개 범위보다 훨씬 넓은 데이터였다.
이 스크립트는 두 테이블을 parent_asin + year_month로 JOIN해서,
"실제로 위험도 예측이 있는 상품x월"로 한정한 뒤 다시 분포를 본다.

실행 방법:
    python check_distributions_v2.py
"""

import pandas as pd
from config import engine

# product_month_risk에 있는 3,563개 상품x월로 범위를 좁혀서 topics 조회
query = """
SELECT t.mention_count, t.delta
FROM product_month_topics t
INNER JOIN product_month_risk r
    ON t.parent_asin = r.parent_asin AND t.year_month = r.year_month
"""
scoped_df = pd.read_sql(query, con=engine)

n_total = len(scoped_df)
print(f"Test 샘플 범위로 좁힌 topics 행 수: {n_total} (3,563개 상품x월 x 12토픽 예상치: {3563*12})")

# --- mention_count 분포 (0 포함 / 0 제외) ---
print("\n--- mention_count (Test 샘플 범위, 0 포함) ---")
print(scoped_df["mention_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

nonzero = scoped_df[scoped_df["mention_count"] > 0]["mention_count"]
print(f"\n--- mention_count (Test 샘플 범위, 0 제외, n={len(nonzero)}) ---")
print(nonzero.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

# 정확한 비율: mention_count == 1인 것이 (0 제외 기준으로) 몇 %인지
count_eq_1 = (nonzero == 1).sum()
count_eq_2_or_less = (nonzero <= 2).sum()
print(f"\nmention_count == 1 인 비율 (0 제외 기준): {count_eq_1/len(nonzero)*100:.1f}%")
print(f"mention_count <= 2 인 비율 (0 제외 기준): {count_eq_2_or_less/len(nonzero)*100:.1f}%")

# 임계값을 2로 잡았을 때, 즉 "mention_count < 2" (=1)만 표본부족으로 볼 때
# 전체 언급(0 제외) 대비 표본부족 비율
print(f"\n[임계값=2일 때] '표본부족(mention_count < 2)'으로 분류되는 비율: {count_eq_1/len(nonzero)*100:.1f}%")
print(f"[임계값=3일 때] '표본부족(mention_count < 3)'으로 분류되는 비율: {count_eq_2_or_less/len(nonzero)*100:.1f}%")

# --- delta 분포 (양수만) ---
positive_delta = scoped_df[scoped_df["delta"] > 0]["delta"]
print(f"\n--- delta (Test 샘플 범위, 양수만, n={len(positive_delta)}) ---")
print(positive_delta.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
