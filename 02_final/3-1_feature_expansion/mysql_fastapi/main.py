"""
3-1 FastAPI 백엔드

실행 전 준비:
    pip install fastapi uvicorn sqlalchemy pymysql

실행 방법:
    uvicorn main:app --reload

실행 후 확인:
    브라우저에서 http://localhost:8000/docs 접속
"""

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text
from typing import Optional

# ============================================
# 0. DB 연결 설정
# ============================================
DB_USER = "root"
DB_PASSWORD = "305040"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "amazon_risk_service"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

app = FastAPI(title="Amazon 품질 위험 분석 서비스 API")


# ============================================
# 1. GET /health : 서버·DB 연결 상태 확인
# ============================================
@app.get("/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_connected": True}
    except Exception as e:
        return {"status": "error", "db_connected": False, "detail": str(e)}


# ============================================
# 2. GET /products/search : 상품명 또는 parent_asin 검색
# ============================================
@app.get("/products/search")
def search_products(q: str = Query(..., description="상품명 또는 parent_asin"), limit: int = 20):
    query = text("""
        SELECT parent_asin, product_title, store, main_category
        FROM products
        WHERE parent_asin LIKE :q OR product_title LIKE :q
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"q": f"%{q}%", "limit": limit}).mappings().all()
    return {"count": len(rows), "results": [dict(r) for r in rows]}


# ============================================
# 3. GET /products/{asin}/summary : 상품 종합 정보
# ============================================
@app.get("/products/{asin}/summary")
def get_product_summary(asin: str, year_month: Optional[str] = None):
    with engine.connect() as conn:
        # 기본 상품 정보
        product = conn.execute(
            text("SELECT * FROM products WHERE parent_asin = :asin"),
            {"asin": asin}
        ).mappings().first()

        if not product:
            raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다")

        # year_month 지정 없으면 가장 최근 월 사용
        if year_month is None:
            latest = conn.execute(
                text("SELECT MAX(`year_month`) as ym FROM product_month_risk WHERE parent_asin = :asin"),
                {"asin": asin}
            ).mappings().first()
            year_month = latest["ym"] if latest else None

        risk = None
        metrics = None
        if year_month:
            risk = conn.execute(
                text("""SELECT risk_probability, operational_confidence, review_evidence_confidence,
                                model_consistency_confidence, confidence_level, current_review_count
                         FROM product_month_risk
                         WHERE parent_asin = :asin AND `year_month` = :ym"""),
                {"asin": asin, "ym": year_month}
            ).mappings().first()

            metrics = conn.execute(
                text("""SELECT review_count, avg_rating, low_rating_ratio,
                                history_3m_vs_6m_low_rating_ratio_delta,
                                history_12m_avg_rating, history_12m_low_rating_ratio,
                                history_3m_vs_12m_low_rating_ratio_delta
                         FROM product_month_metrics
                         WHERE parent_asin = :asin AND `year_month` = :ym"""),
                {"asin": asin, "ym": year_month}
            ).mappings().first()

    return {
        "product": dict(product),
        "year_month": year_month,
        "risk": dict(risk) if risk else None,
        "metrics": dict(metrics) if metrics else None,
    }


# ============================================
# 4. GET /products/{asin}/trend : 월별 추이
# ============================================
@app.get("/products/{asin}/trend")
def get_product_trend(asin: str):
    query = text("""
        SELECT m.`year_month`, m.review_count, m.avg_rating, m.low_rating_ratio,
               m.history_12m_avg_rating, m.history_12m_low_rating_ratio,
               m.history_3m_vs_12m_low_rating_ratio_delta,
               r.risk_probability, r.confidence_level
        FROM product_month_metrics m
        LEFT JOIN product_month_risk r
          ON m.parent_asin = r.parent_asin AND m.`year_month` = r.`year_month`
        WHERE m.parent_asin = :asin
        ORDER BY m.`year_month` ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"asin": asin}).mappings().all()
    return {"parent_asin": asin, "trend": [dict(r) for r in rows]}


# ============================================
# 5. GET /products/risky : 위험 상품 목록
# ============================================
@app.get("/products/risky")
def get_risky_products(
    sort_by: str = Query("risk", enum=["risk", "confidence_weighted"]),
    limit: int = 20,
    year_month: Optional[str] = None,
):
    with engine.connect() as conn:
        if year_month is None:
            latest = conn.execute(text("SELECT MAX(`year_month`) as ym FROM product_month_risk")).mappings().first()
            year_month = latest["ym"]

        if sort_by == "risk":
            order_clause = "r.risk_probability DESC"
        else:  # confidence_weighted
            order_clause = "(r.risk_probability * r.operational_confidence) DESC"

        query = text(f"""
            SELECT p.parent_asin, p.product_title, p.store,
                   r.risk_probability, r.operational_confidence, r.confidence_level,
                   r.current_review_count
            FROM product_month_risk r
            JOIN products p ON r.parent_asin = p.parent_asin
            WHERE r.`year_month` = :ym
            ORDER BY {order_clause}
            LIMIT :limit
        """)
        rows = conn.execute(query, {"ym": year_month, "limit": limit}).mappings().all()

    return {"year_month": year_month, "sort_by": sort_by, "count": len(rows), "results": [dict(r) for r in rows]}


# ============================================
# 6. GET /products/{asin}/topics : 불만 토픽별 비율/변화량
# ============================================
@app.get("/products/{asin}/topics")
def get_product_topics(asin: str, year_month: Optional[str] = None):
    with engine.connect() as conn:
        if year_month is None:
            latest = conn.execute(
                text("SELECT MAX(`year_month`) as ym FROM product_month_topics WHERE parent_asin = :asin"),
                {"asin": asin}
            ).mappings().first()
            year_month = latest["ym"] if latest else None

        if year_month is None:
            return {"parent_asin": asin, "year_month": None, "topics": []}

        query = text("""
            SELECT topic_name, mention_count, rate_t, rate_p3, delta, rate_t_shrunk
            FROM product_month_topics
            WHERE parent_asin = :asin AND `year_month` = :ym
            ORDER BY delta DESC
        """)
        rows = conn.execute(query, {"asin": asin, "ym": year_month}).mappings().all()

    return {"parent_asin": asin, "year_month": year_month, "topics": [dict(r) for r in rows]}


# ============================================
# 7. GET /products/{asin}/evidence-reviews : 위험 근거 리뷰
# ============================================
@app.get("/products/{asin}/evidence-reviews")
def get_evidence_reviews(
    asin: str,
    year_month: Optional[str] = None,
    max_rating: int = Query(2, description="이 별점 이하만 조회"),
    topic: Optional[str] = Query(None, description="이 토픽이 언급된 리뷰만 조회 (예: durability_failure)"),
    limit: int = 10,
):
    query = """
        SELECT review_id, `year_month`, rating, review_text, related_topics,
               review_datetime_utc, helpful_vote
        FROM evidence_reviews
        WHERE parent_asin = :asin AND rating <= :max_rating
    """
    params = {"asin": asin, "max_rating": max_rating, "limit": limit}

    if year_month:
        query += " AND `year_month` = :ym"
        params["ym"] = year_month

    if topic:
        # related_topics는 "durability_failure,return_blocked" 처럼 콤마로 이어붙인 문자열이라
        # 콤마를 포함해서 찾아야 "durability" 같은 부분 문자열 오매칭을 피할 수 있음
        query += " AND (related_topics LIKE :topic_exact OR related_topics LIKE :topic_start OR related_topics LIKE :topic_mid OR related_topics = :topic_only)"
        params["topic_only"] = topic
        params["topic_exact"] = f"%,{topic}"
        params["topic_start"] = f"{topic},%"
        params["topic_mid"] = f"%,{topic},%"

    query += " ORDER BY helpful_vote DESC, review_datetime_utc DESC LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    return {"parent_asin": asin, "count": len(rows), "reviews": [dict(r) for r in rows]}


# ============================================
# 8. GET /products/{asin}/risk-explanation : 위험 판단 근거 (SHAP 변수/기여도)
# ============================================
@app.get("/products/{asin}/risk-explanation")
def get_risk_explanation(asin: str, year_month: Optional[str] = None):
    with engine.connect() as conn:
        if year_month is None:
            latest = conn.execute(
                text("SELECT MAX(`year_month`) as ym FROM risk_explanations WHERE parent_asin = :asin"),
                {"asin": asin}
            ).mappings().first()
            year_month = latest["ym"] if latest else None

        if year_month is None:
            return {"parent_asin": asin, "year_month": None, "explanations": [],
                    "note": "아직 SHAP 결과가 없습니다"}

        query = text("""
            SELECT feature_name, contribution
            FROM risk_explanations
            WHERE parent_asin = :asin AND `year_month` = :ym
            ORDER BY contribution DESC
        """)
        rows = conn.execute(query, {"asin": asin, "ym": year_month}).mappings().all()

    return {
        "parent_asin": asin,
        "year_month": year_month,
        "explanations": [dict(r) for r in rows],
        "note": "현재는 모델 전역 피처 중요도이며, 상품별 SHAP 값이 아닙니다. 3-3 완료 후 교체 예정입니다.",
    }
