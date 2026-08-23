"""
FastAPI 백엔드(3-1)를 호출하는 함수 모음.

원칙: Streamlit은 MySQL을 직접 만지지 않는다.
      항상 이 파일을 통해서만 데이터를 가져온다.

"""
import requests
import streamlit as st

# Docker Compose로 묶을 때는 "http://backend:8000" 처럼
# 서비스 이름으로 바뀔 수 있음. 지금은 로컬 개발 기준.
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _get(path: str, params: dict | None = None):
    """공통 GET 요청 wrapper. 실패해도 앱 전체가 죽지 않도록 None 반환."""
    try:
        res = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 호출 실패: `{path}` — {e}")
        return None


@st.cache_data(ttl=300)  
def get_health():
    return _get("/health")


@st.cache_data(ttl=300)
def search_products(query: str):
    """GET /products/search?q= -> {"count":..., "results":[...]} 에서 results 리스트만 반환"""
    if not query:
        return None
    data = _get("/products/search", params={"q": query})
    return data["results"] if data else None


@st.cache_data(ttl=300)
def get_risky_products(sort_by: str = "risk", limit: int = 50, year_month: str | None = None):
    """GET /products/risky?sort_by=risk|confidence_weighted
    응답: {"year_month":..., "sort_by":..., "count":..., "results":[...]}
    -> (results 리스트, 실제 조회에 쓰인 year_month) 튜플로 반환
    """
    params = {"sort_by": sort_by, "limit": limit}
    if year_month:
        params["year_month"] = year_month
    data = _get("/products/risky", params=params)
    if not data:
        return None, None
    return data["results"], data.get("year_month")


@st.cache_data(ttl=300)
def get_product_summary(asin: str, year_month: str | None = None):
    """GET /products/{asin}/summary
    응답: {"product": {...}, "year_month":..., "risk": {...}|None, "metrics": {...}|None}
    """
    params = {"year_month": year_month} if year_month else None
    return _get(f"/products/{asin}/summary", params=params)


@st.cache_data(ttl=300)
def get_product_trend(asin: str):
    """GET /products/{asin}/trend -> {"parent_asin":..., "trend":[...]} 에서 trend 리스트만 반환"""
    data = _get(f"/products/{asin}/trend")
    return data["trend"] if data else None


@st.cache_data(ttl=300)
def get_product_topics(asin: str, year_month: str | None = None):
    """GET /products/{asin}/topics -> {"parent_asin":..., "year_month":..., "topics":[...]}"""
    params = {"year_month": year_month} if year_month else None
    data = _get(f"/products/{asin}/topics", params=params)
    return (data["topics"], data.get("year_month")) if data else (None, None)


@st.cache_data(ttl=300)
def get_evidence_reviews(
    asin: str,
    max_rating: int = 2,
    year_month: str | None = None,
    topic: str | None = None,
    limit: int = 10,
):
    """GET /products/{asin}/evidence-reviews -> {"parent_asin":..., "count":..., "reviews":[...]}"""
    params = {"max_rating": max_rating, "limit": limit}
    if year_month:
        params["year_month"] = year_month
    if topic:
        params["topic"] = topic
    data = _get(f"/products/{asin}/evidence-reviews", params=params)
    return data["reviews"] if data else None


@st.cache_data(ttl=300)
def get_risk_explanation(asin: str, year_month: str | None = None):
    """GET /products/{asin}/risk-explanation
    응답: {"parent_asin":..., "year_month":..., "explanations":[{feature_name, contribution}], "note": "..."}
    """
    params = {"year_month": year_month} if year_month else None
    return _get(f"/products/{asin}/risk-explanation", params=params)

@st.cache_data(ttl=300)
def get_risk_changes(year_month: str | None = None, limit: int = 20, high_risk_threshold: float = 0.7):
    """GET /products/risk-changes"""
    params = {"limit": limit, "high_risk_threshold": high_risk_threshold}
    if year_month:
        params["year_month"] = year_month
    return _get("/products/risk-changes", params=params)