"""
'진짜 SHAP'이 아니라, 통계 기반 근사 설명을 계산하는 스크립트.

원리: 각 상품의 변수 값이 전체 상품 분포에서 몇 퍼센타일에 있는지를 계산해서,
      그 백분위를 "기여도"처럼 보여준다.

주의: 이건 SHAP이 아니다. 모델이 실제로 그 변수를 어떻게 썼는지가 아니라,
      "이 상품이 이 변수에서 얼마나 튀는지"를 보여줄 뿐이다.
"""
import pandas as pd

# ============================================
# 1. 원본 데이터 로드
# ============================================
df = pd.read_parquet("data/processed/product_month_review_volume_2m_labeled.parquet")

# 위험 방향을 아는 변수들 (값이 클수록 위험한지, 작을수록 위험한지)
# True = 값이 클수록 위험, False = 값이 작을수록 위험
FEATURE_DIRECTION = {
    "history_12m_low_rating_ratio": True,
    "history_3m_low_rating_ratio": True,
    "history_6m_low_rating_ratio": True,
    "low_rating_ratio": True,
    "history_3m_vs_6m_low_rating_ratio_delta": True,
    "history_3m_vs_12m_low_rating_ratio_delta": True,
    "avg_rating": False,
    "history_12m_avg_rating": False,
    "history_6m_avg_rating": False,
    "history_3m_avg_rating": False,
    "review_count": False,             # 리뷰 수 급감이 위험 신호이므로, 적을수록 위험
    "history_3m_review_count": False,
    "history_12m_review_count": False,
    "history_1m_review_count": False,
}

FEATURES = list(FEATURE_DIRECTION.keys())
FEATURES = [f for f in FEATURES if f in df.columns]

# ============================================
# 2. 각 변수의 전체 분포 대비 백분위 계산
# ============================================
print("백분위 계산 중...")
for feat in FEATURES:
    # rank(pct=True): 0~1 사이 백분위. 값이 클수록 그 변수 기준 순위가 높음
    pct = df[feat].rank(pct=True)
    if not FEATURE_DIRECTION[feat]:
        # "작을수록 위험"인 변수는 백분위를 뒤집어서, 위험할수록 점수가 높아지게 함
        pct = 1 - pct
    df[f"{feat}__risk_pct"] = (pct * 100).round(1)

# ============================================
# 3. 상품×월×변수 형태(risk_explanations 테이블 형식)로 변환
# ============================================
print("결과 정리 중...")
rows = []
for _, row in df.iterrows():
    for feat in FEATURES:
        rows.append({
            "parent_asin": row["parent_asin"],
            "year_month": row["year_month"],
            "feature_name": feat,
            "contribution": row[f"{feat}__risk_pct"],  # 0~100, 높을수록 위험 기여가 큼(근사)
        })

result_df = pd.DataFrame(rows)
result_df.to_csv("stat_explanations.csv", index=False)
print(f"완료: {len(result_df)}행 생성, {df['parent_asin'].nunique()}개 상품")
print(result_df.head(10))