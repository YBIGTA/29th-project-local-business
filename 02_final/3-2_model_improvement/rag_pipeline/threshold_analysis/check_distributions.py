"""
임계값(threshold) 설정 근거 확인용 스크립트

retrieval.py에 있는 5개 임계값(LOW_REVIEW_COUNT_THRESHOLD, LOW_MENTION_COUNT_THRESHOLD,
SURGE_DELTA_THRESHOLD, RISK_HIGH/LOW_THRESHOLD, CONFIDENCE_HIGH/LOW_THRESHOLD)이
실제 데이터 분포상 타당한지 확인하기 위해, 관련 컬럼들의 백분위수를 조회한다.

실행 방법:
    python check_distributions.py
"""

import pandas as pd
from config import engine


def print_percentiles(series: pd.Series, name: str):
    print(f"\n--- {name} (n={len(series)}) ---")
    print(series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))


# 1. current_review_count 분포 (product_month_risk 테이블, LOW_REVIEW_COUNT_THRESHOLD 근거용)
review_count_df = pd.read_sql(
    "SELECT current_review_count FROM product_month_risk", con=engine
)
print_percentiles(review_count_df["current_review_count"], "current_review_count")

# 2. mention_count 분포 (product_month_topics 테이블, LOW_MENTION_COUNT_THRESHOLD 근거용)
# 0은 "언급 없음"이라 분포 왜곡을 피하기 위해 1 이상인 것만 별도로도 확인
mention_df = pd.read_sql(
    "SELECT mention_count FROM product_month_topics", con=engine
)
print_percentiles(mention_df["mention_count"], "mention_count (전체, 0 포함)")
print_percentiles(
    mention_df[mention_df["mention_count"] > 0]["mention_count"],
    "mention_count (0 제외, 실제 언급된 것만)"
)

# 3. delta 분포 (product_month_topics 테이블, SURGE_DELTA_THRESHOLD 근거용)
# delta는 음수/양수 다 있을 수 있어서, 양수(=증가한 경우)만 따로 확인
delta_df = pd.read_sql(
    "SELECT delta FROM product_month_topics WHERE delta IS NOT NULL", con=engine
)
print_percentiles(delta_df["delta"], "delta (전체)")
print_percentiles(
    delta_df[delta_df["delta"] > 0]["delta"],
    "delta (양수만, 즉 증가한 경우만)"
)

# 4. risk_probability 분포 (product_month_risk 테이블, RISK_HIGH/LOW_THRESHOLD 근거용)
risk_df = pd.read_sql(
    "SELECT risk_probability FROM product_month_risk", con=engine
)
print_percentiles(risk_df["risk_probability"], "risk_probability")

# 5. combined_confidence 분포 (operational_confidence, review_evidence_confidence의 최솟값)
# CONFIDENCE_HIGH/LOW_THRESHOLD 근거용
conf_df = pd.read_sql(
    "SELECT operational_confidence, review_evidence_confidence FROM product_month_risk", con=engine
)
conf_df["combined_confidence"] = conf_df[["operational_confidence", "review_evidence_confidence"]].min(axis=1)
print_percentiles(conf_df["combined_confidence"], "combined_confidence (min of operational, review_evidence)")

print("\n\n=== 확인 끝. 이 결과를 근거로 임계값을 다시 잡습니다. ===")
