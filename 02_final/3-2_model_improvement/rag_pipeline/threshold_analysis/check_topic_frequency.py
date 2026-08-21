"""
전체 데이터셋에서 각 토픽이 얼마나 자주 "실제로 언급"되는지 확인한다.
incompatibility/purchase_warning 같은 토픽이 유독 자주 나오는 게
리포트 생성 로직의 편향인지, 데이터 자체의 특성인지 구분하기 위함이다.

실행 방법:
    python check_topic_frequency.py
"""

import pandas as pd
from config import engine

df = pd.read_sql(
    "SELECT t.topic_name, COUNT(*) as total_rows, "
    "SUM(CASE WHEN t.mention_count > 0 THEN 1 ELSE 0 END) as mentioned_count "
    "FROM product_month_topics t "
    "INNER JOIN product_month_risk r ON t.parent_asin = r.parent_asin AND t.year_month = r.year_month "
    "GROUP BY t.topic_name "
    "ORDER BY mentioned_count DESC",
    con=engine
)
df["mentioned_ratio"] = (df["mentioned_count"] / df["total_rows"] * 100).round(1)
print(df.to_string(index=False))
