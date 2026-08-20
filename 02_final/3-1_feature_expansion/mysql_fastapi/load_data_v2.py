"""
3-1 데이터 적재 스크립트 v2 (명세 파일 기준 6개 테이블)

실행 전 준비:
    pip install sqlalchemy pymysql pandas pyarrow

실행 방법:
    python load_data_v2.py
"""

import pandas as pd
from sqlalchemy import create_engine

# ============================================
# 0. DB 연결 정보 
# ============================================
DB_USER = "root"
DB_PASSWORD = "305040"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "amazon_risk_service"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

# ============================================
# 파일 경로 
# ============================================
PRODUCTS_META_PATH = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\amazon_3_1_data_04\data\interim\products_meta_clean.parquet"
REVIEWS_PATH = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\reviews_clean.parquet"
TEXT_FEATURES_PATH = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\amazon_3_1_data_04\data\processed\product_month_text_features.parquet"
VOLUME_PATH = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\amazon_3_1_data_04\data\processed\product_month_review_volume_2m_labeled.parquet"
RISK_CONFIDENCE_PATH = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\amazon_3_1_data_04\02_final\3-3_interpretation\risk_confidence_output\product_month_risk_confidence.parquet"

# ============================================
# 1. products 적재
# ============================================
print("1. products 적재 중...")
products = pd.read_parquet(PRODUCTS_META_PATH)
products["categories"] = products["categories"].astype(str)
products.to_sql("products", con=engine, if_exists="append", index=False)
print(f"   -> {len(products)}행 적재 완료")


# ============================================
# 2. product_month_metrics 적재 (volume 파일 기반)
# ============================================
print("2. product_month_metrics 적재 중...")
volume = pd.read_parquet(VOLUME_PATH)

metrics_cols = [
    "parent_asin", "year_month",
    "review_count", "avg_rating", "low_rating_count", "low_rating_ratio",
    "verified_purchase_ratio", "mean_helpful_vote", "text_available_ratio",
    "past_3m_review_count",
    "history_3m_low_rating_ratio", "history_6m_low_rating_ratio",
    "history_3m_vs_6m_low_rating_ratio_delta",
    "history_12m_review_count", "history_12m_avg_rating", "history_12m_low_rating_ratio",
    "history_3m_vs_12m_low_rating_ratio_delta",
    "is_review_volume_drop_2m",
]
metrics_for_db = volume[metrics_cols].copy()

# products에 존재하는 parent_asin만 남기기 (FK 제약 위반 방지)
valid_asins = set(products["parent_asin"])
metrics_for_db = metrics_for_db[metrics_for_db["parent_asin"].isin(valid_asins)]

metrics_for_db.to_sql(
    "product_month_metrics", con=engine, if_exists="append", index=False, chunksize=5000
)
print(f"   -> {len(metrics_for_db)}행 적재 완료")


# ============================================
# 3. product_month_risk 적재 (risk_confidence 파일 기반)
# ============================================
print("3. product_month_risk 적재 중...")
risk = pd.read_parquet(RISK_CONFIDENCE_PATH)

risk_for_db = risk[[
    "parent_asin", "year_month", "risk_probability", "current_review_count",
    "s45_raw_score", "s45_raw_mil_score", "mil_score_change",
    "review_evidence_confidence", "model_consistency_confidence",
    "operational_confidence", "confidence_level"
]].copy()
risk_for_db["model_version"] = "s45_mil_v1"

risk_for_db = risk_for_db[risk_for_db["parent_asin"].isin(valid_asins)]

risk_for_db.to_sql(
    "product_month_risk", con=engine, if_exists="append", index=False, chunksize=5000
)
print(f"   -> {len(risk_for_db)}행 적재 완료")


# ============================================
# 4. product_month_topics 적재 (text_features 파일 melt)
# ============================================
print("4. product_month_topics 적재 중...")
text_feat = pd.read_parquet(TEXT_FEATURES_PATH)

topic_names = [
    "durability_failure", "build_quality", "incompatibility", "performance",
    "leak_damage", "water_taste_filter", "shipping_packaging",
    "return_intent", "return_blocked", "refund_warranty_cs",
    "misrepresentation", "purchase_warning",
]

topic_rows = []
for topic in topic_names:
    rate_t = text_feat.get(f"topic_{topic}_rate_t")
    # 언급 수 = 비율 x 해당 월 리뷰 수 (반올림, 원본에서 카운트 자체가 저장 안 됐어서 역산)
    mention_count = (rate_t * text_feat["n_reviews_t"]).round().astype("Int64") if rate_t is not None else None

    cols = {
        "parent_asin": text_feat["parent_asin"],
        "year_month": text_feat["year_month"],
        "topic_name": topic,
        "mention_count": mention_count,
        "rate_t": rate_t,
        "rate_p3": text_feat.get(f"topic_{topic}_rate_p3"),
        "delta": text_feat.get(f"topic_{topic}_delta"),
        "rate_t_shrunk": text_feat.get(f"topic_{topic}_rate_t_shrunk"),
    }
    topic_rows.append(pd.DataFrame(cols))

topics_long = pd.concat(topic_rows, ignore_index=True)
topics_long = topics_long[topics_long["parent_asin"].isin(valid_asins)]

topics_long.to_sql(
    "product_month_topics", con=engine, if_exists="append", index=False, chunksize=5000
)
print(f"   -> {len(topics_long)}행 적재 완료")


# ============================================
# 5. evidence_reviews 적재 (저평점 1~2점 리뷰만 선별)
# ============================================
print("5. evidence_reviews 적재 중...")
reviews = pd.read_parquet(REVIEWS_PATH)

import json
import sys

# matching_rules.py 파일 위치를 파이썬이 찾을 수 있게 경로 추가
MATCHING_RULES_DIR = rMATCHING_RULES_DIR = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\3_1_files"
sys.path.insert(0, MATCHING_RULES_DIR)
from matching_rules import count_terms   # 부정문·다의어까지 처리하는 원본 매칭 함수

COMPLAINT_DICT_PATH = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\3_1_files\complaint_dictionary.json"
with open(COMPLAINT_DICT_PATH, encoding="utf-8") as f:
    complaint_dict = json.load(f)

# 12개 토픽의 키워드를 {토픽명: [키워드...]} 형태로 정리
topic_terms = {}
for group in ["cause", "consequence"]:
    for topic, info in complaint_dict[group].items():
        topic_terms[topic] = info["terms"]

def find_related_topics(text_value):
    if not isinstance(text_value, str) or not text_value:
        return None
    tokens = text_value.lower().split()
    matched = [topic for topic, terms in topic_terms.items()
               if count_terms(tokens, terms) > 0]
    return ",".join(matched) if matched else None

# 위험 근거로 쓸 리뷰만 선별: 1~2점 저평점 리뷰
evidence = reviews[reviews["rating"] <= 2].copy()

print("   토픽 매칭 계산 중... (부정문/다의어 규칙 반영, 시간이 좀 걸릴 수 있어요)")
evidence["related_topics"] = evidence["review_text"].apply(find_related_topics)

evidence_for_db = evidence[[
    "parent_asin", "year_month", "rating", "review_text", "related_topics",
    "review_datetime_utc", "verified_purchase", "helpful_vote"
]].copy()
evidence_for_db = evidence_for_db[evidence_for_db["parent_asin"].isin(valid_asins)]

evidence_for_db.to_sql(
    "evidence_reviews", con=engine, if_exists="append", index=False, chunksize=5000
)
print(f"   -> {len(evidence_for_db)}행 적재 완료 (전체 리뷰 중 1~2점만 선별)")


# ============================================
# 6. risk_explanations : 구조만 생성 (SHAP 결과 나오면 추후 적재)
# ============================================
print("6. risk_explanations 적재 중...")

# ⚠️ 주의: 아직 상품별 SHAP 결과가 없어서, 대신 S45 모델의
# "전역 피처 중요도"(이 모델 전체에서 어떤 변수가 중요한지)를 임시로 채워 넣는다.
# 상품마다 다른 진짜 SHAP 기여도가 아니라, 모든 상품×월에 동일한 값이 들어간다.
# 3-3 단계에서 진짜 SHAP 결과가 나오면 이 값들을 교체해야 한다.

importance_path = r"C:\Users\lucy0\Documents\YONSEI\YBIGTA\신기플\amazon_3_1_data\3_1_files\standard_candidate_importance.csv"
importance = pd.read_csv(importance_path)

s45_importance = importance[importance["model_name"] == "S45_structured_only"].copy()
top_features = s45_importance.sort_values("importance_gain", ascending=False).head(10)

# product_month_risk에 있는 상품x월 조합마다, 동일한 top10 피처 중요도를 매칭
risk_keys = risk_for_db[["parent_asin", "year_month"]].drop_duplicates()

explanations_rows = []
for _, feat_row in top_features.iterrows():
    temp = risk_keys.copy()
    temp["feature_name"] = feat_row["feature_name"]
    temp["contribution"] = feat_row["importance_gain"]
    explanations_rows.append(temp)

explanations_for_db = pd.concat(explanations_rows, ignore_index=True)

explanations_for_db.to_sql(
    "risk_explanations", con=engine, if_exists="append", index=False, chunksize=5000
)
print(f"   -> {len(explanations_for_db)}행 적재 완료 (전역 피처 중요도 기준, 상품별 SHAP 아님)")

print("\n전체 적재 완료!")
