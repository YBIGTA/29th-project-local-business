"""
3-2 검색(Retrieval) 로직 (v3)

3-1에서 만든 FastAPI(main.py)를 통해 데이터를 가져온다.
RAG가 LLM에 넘길 "근거 자료 묶음"을 이 파일이 만들어준다.

v3에서 추가된 것:
    - assess_confidence()에 위험도x신뢰도 조합에 따른 규칙 기반 해석 문구(interpretation) 추가
      (숫자 하나로 뭉뚱그리지 않고, 숫자와 해석 문구를 함께 제공)

실행 전 준비:
    1. 3-1 폴더에서 uvicorn main:app --reload 로 서버를 먼저 켜둘 것
    2. pip install requests
"""

import time
import requests

import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

REQUEST_TIMEOUT = 10        # 초. 서버가 이 시간 안에 응답 안 하면 실패 처리
MAX_RETRIES = 3             # 실패 시 최대 재시도 횟수
RETRY_DELAY_SEC = 1.5       # 재시도 사이 대기 시간

LOW_REVIEW_COUNT_THRESHOLD = 9        # product_month_risk의 current_review_count 하위 10% 기준
LOW_MENTION_COUNT_THRESHOLD = 2       # mention_count==1 인 경우만 표본부족으로 판단 (A안 확정)
                                        # 주의: 실제 데이터에서 언급된 토픽의 약 62%가 1건뿐이라,
                                        # 이 경고가 자주 뜨는 게 정상이다 (표본이 적은 데이터셋 특성)
SURGE_DELTA_THRESHOLD = 0.04           # product_month_risk 범위로 좁힌 delta(양수) 중앙값(0.042) 기준

# 위험도/신뢰도 각각을 몇 단계로 나눌지 기준값
# product_month_risk(3,563건) 실제 분포의 25%/75% 백분위수 기준
RISK_HIGH_THRESHOLD = 0.60      # risk_probability 75% 지점
RISK_LOW_THRESHOLD = 0.32       # risk_probability 25% 지점
CONFIDENCE_HIGH_THRESHOLD = 0.72   # combined_confidence 75% 지점
CONFIDENCE_LOW_THRESHOLD = 0.55    # combined_confidence 25% 지점


def _request_with_retry(url: str, params: dict | None = None) -> requests.Response | None:
    """
    타임아웃과 재시도를 적용해 GET 요청을 보낸다.
    최종적으로 실패하면 None을 반환한다 (404는 재시도하지 않고 바로 반환).
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if res.status_code == 404:
                return res  # 데이터가 원래 없는 경우이므로 재시도 의미 없음
            res.raise_for_status()
            return res
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    print(f"[retrieval] 요청 실패 (재시도 {MAX_RETRIES}회 소진): {url} - {last_error}")
    return None


def _risk_level(risk_probability: float) -> str:
    if risk_probability >= RISK_HIGH_THRESHOLD:
        return "high"
    if risk_probability >= RISK_LOW_THRESHOLD:
        return "moderate"
    return "low"


def _confidence_level(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if confidence >= CONFIDENCE_LOW_THRESHOLD:
        return "moderate"
    return "low"


# 위험도 단계 x 신뢰도 단계 조합별 해석 문구
_INTERPRETATION_MATRIX = {
    ("high", "high"): "위험 신호가 뚜렷하고 판단 근거도 충분하여, 위험 가능성을 신뢰할 수 있습니다.",
    ("high", "moderate"): "위험 신호가 뚜렷하나 판단 근거가 다소 제한적이어서, 주의 깊게 관찰할 필요가 있습니다.",
    ("high", "low"): "위험 신호는 높게 나타나지만 판단 근거(리뷰 표본 등)가 부족하여, 확정적으로 보기는 어렵습니다.",
    ("moderate", "high"): "중간 수준의 위험 신호가 확인되며, 판단 근거는 충분한 편입니다.",
    ("moderate", "moderate"): "중간 수준의 위험 신호가 확인되나, 판단 근거도 다소 제한적입니다.",
    ("moderate", "low"): "중간 수준의 위험 신호가 있으나, 판단 근거가 부족해 참고 수준으로만 활용해야 합니다.",
    ("low", "high"): "위험 신호가 낮고 판단 근거도 충분하여, 현재로서는 위험 가능성이 낮다고 볼 수 있습니다.",
    ("low", "moderate"): "위험 신호는 낮으나 판단 근거가 다소 제한적입니다.",
    ("low", "low"): "위험 신호는 낮게 나타나지만 판단 근거 자체가 부족하여, 판단을 유보하는 것이 안전합니다.",
}


def assess_confidence(risk_probability: float, operational_confidence: float,
                       review_evidence_confidence: float, model_consistency_confidence: float,
                       current_review_count: int) -> dict:
    """
    위험도와 신뢰도를 종합해 LLM에 넘길 판단 정보를 만든다.

    - combined_confidence = min(operational_confidence, review_evidence_confidence)
        model_consistency_confidence는 "모델이 일관되게 예측하는가"를 나타낼 뿐,
        이 판단이 실제로 믿을 만한지와는 결이 달라 계산에서 제외하고 참고값으로만 남긴다.
    - interpretation: 위험도x신뢰도를 곱해서 숫자 하나로 뭉개는 대신,
        두 축을 각각 3단계(high/moderate/low)로 나눠 조합한 규칙 기반 해석 문구를 추가한다.
        weighted_risk(곱셈 점수)는 정렬/비교용 참고 지표로 별도 유지한다.
    """
    combined_confidence = min(operational_confidence, review_evidence_confidence)
    weighted_risk = risk_probability * combined_confidence  # 정렬/비교용 참고 지표 (원래 값 형태 유지)

    risk_lvl = _risk_level(risk_probability)
    confidence_lvl = _confidence_level(combined_confidence)
    interpretation = _INTERPRETATION_MATRIX[(risk_lvl, confidence_lvl)]

    caveats = []
    if current_review_count < LOW_REVIEW_COUNT_THRESHOLD:
        caveats.append(
            f"이번 달 리뷰 수가 {current_review_count}개로 적어, 표본 부족으로 추가 관찰이 필요합니다."
        )

    return {
        "risk_probability": risk_probability,           # 원래 위험도 숫자는 그대로 보존 (숨기지 않음)
        "combined_confidence": round(combined_confidence, 4),
        "risk_level": risk_lvl,                          # high / moderate / low
        "confidence_level": confidence_lvl,               # high / moderate / low
        "interpretation": interpretation,                 # 규칙 기반 해석 문구
        "weighted_risk": round(weighted_risk, 4),          # 참고용 정량 지표 (정렬/비교 시 활용, % 아님)
        "model_consistency_confidence_reference": model_consistency_confidence,  # 참고용, 계산엔 미반영
        "caveats": caveats,
    }


def annotate_topics(topics: list[dict]) -> dict:
    """
    토픽 리스트를 받아, '급증한 토픽이 있는지'와 '표본이 부족한 토픽'을 표시한다.
    """
    if not topics:
        return {"topics": [], "has_surging_topic": False, "note": "토픽 데이터가 없습니다."}

    annotated = []
    for t in topics:
        entry = dict(t)
        entry["is_low_sample"] = entry.get("mention_count", 0) < LOW_MENTION_COUNT_THRESHOLD
        annotated.append(entry)

    top_delta = max((t.get("delta") or 0) for t in topics)
    has_surging_topic = top_delta >= SURGE_DELTA_THRESHOLD

    return {
        "topics": annotated,
        "has_surging_topic": has_surging_topic,
        "note": None if has_surging_topic else "특별히 급증한 불만 토픽은 확인되지 않았습니다.",
    }


def annotate_evidence_reviews(reviews: list[dict]) -> dict:
    """근거 리뷰가 충분한지 표시한다."""
    if not reviews:
        return {"reviews": [], "has_evidence": False, "note": "이 판단을 뒷받침할 저평점 근거 리뷰가 확인되지 않았습니다."}
    return {"reviews": reviews, "has_evidence": True, "note": None}


def get_product_context(asin: str, year_month: str | None = None) -> dict:
    """
    상품 하나에 대해 LLM 리포트 작성에 필요한 모든 근거 자료를 모아온다.
    각 항목은 원본 데이터와 함께, 신뢰도/표본 부족 여부를 보완한 형태로 반환된다.
    """
    params = {"year_month": year_month} if year_month else {}

    # 1. 상품 종합 정보
    summary_res = _request_with_retry(f"{API_BASE_URL}/products/{asin}/summary", params)
    if summary_res is None:
        return {"error": f"summary 조회 실패 (서버 응답 없음): {asin}"}
    if summary_res.status_code == 404:
        return {"error": f"상품을 찾을 수 없습니다: {asin}"}
    summary = summary_res.json()

    actual_year_month = summary.get("year_month")
    shared_params = {"year_month": actual_year_month} if actual_year_month else {}

    # 위험도 + 신뢰도 종합 판단 (summary.risk가 없을 수도 있으니 방어적으로 처리)
    risk_assessment = None
    if summary.get("risk"):
        r = summary["risk"]
        risk_assessment = assess_confidence(
            risk_probability=r["risk_probability"],
            operational_confidence=r["operational_confidence"],
            review_evidence_confidence=r["review_evidence_confidence"],
            model_consistency_confidence=r["model_consistency_confidence"],
            current_review_count=r["current_review_count"],
        )

    # 2. 불만 토픽
    topics_res = _request_with_retry(f"{API_BASE_URL}/products/{asin}/topics", shared_params)
    topics_raw = topics_res.json().get("topics", []) if topics_res else []
    topics = annotate_topics(topics_raw)

    # 3. 위험 근거 리뷰
    evidence_params = dict(shared_params)
    evidence_params["limit"] = 10
    evidence_res = _request_with_retry(f"{API_BASE_URL}/products/{asin}/evidence-reviews", evidence_params)
    evidence_raw = evidence_res.json().get("reviews", []) if evidence_res else []
    evidence_reviews = annotate_evidence_reviews(evidence_raw)

    # 4. 위험 판단 근거 변수 (통계 기반 근사치 — 상품별 백분위 기여도, 진짜 SHAP은 아님. main.py 참고)
    explanation_res = _request_with_retry(f"{API_BASE_URL}/products/{asin}/risk-explanation", shared_params)
    risk_explanation = explanation_res.json() if explanation_res else {"explanations": [], "note": "조회 실패"}

    return {
        "summary": summary,
        "risk_assessment": risk_assessment,
        "topics": topics,
        "evidence_reviews": evidence_reviews,
        "risk_explanation": risk_explanation,
    }


def get_topic_related_reviews(asin: str, topic: str, year_month: str | None = None, limit: int = 5) -> dict:
    """특정 불만 토픽에 관련된 리뷰만 따로 가져온다."""
    params = {"topic": topic, "limit": limit}
    if year_month:
        params["year_month"] = year_month

    res = _request_with_retry(f"{API_BASE_URL}/products/{asin}/evidence-reviews", params)
    if res is None:
        return {"error": "요청 실패"}
    return res.json()


if __name__ == "__main__":
    test_asin = "B0737B6HGR"
    context = get_product_context(test_asin)

    if "error" in context:
        print(context["error"])
    else:
        print("=== summary (상품 기본정보) ===")
        print(context["summary"])
        print("\n=== risk_assessment (해석 문구 포함) ===")
        print(context["risk_assessment"])
        print("\n=== topics (급증 여부 포함) ===")
        print("급증 토픽 있음:", context["topics"]["has_surging_topic"])
        print("상위 3개:", context["topics"]["topics"][:3])
        print("\n=== evidence_reviews (근거 충분 여부 포함) ===")
        print("근거 있음:", context["evidence_reviews"]["has_evidence"])
        print("상위 2개:", context["evidence_reviews"]["reviews"][:2])
