"""
질문 유형 3: 현재 위험 상품 목록을 조회하고 자연어로 요약한다.

예시 질문: "지금 가장 위험한 상품 Top 5 알려줘"

기존 파일과의 관계:
    - 3-1의 GET /products/risky API를 그대로 호출 (검색 로직 재사용)
    - generate_report.py의 call_llm()을 재사용 (OpenAI 호출 로직 재사용)

실행 전 준비:
    1. 3-1 폴더에서 uvicorn main:app --reload 로 서버를 먼저 켜둘 것
    2. .env에 OPENAI_API_KEY가 채워져 있을 것
"""

import requests

from generate_report import call_llm
from retrieval import LOW_REVIEW_COUNT_THRESHOLD

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 10


def get_risky_products(sort_by: str = "risk", limit: int = 5, year_month: str | None = None) -> dict:
    """3-1의 /products/risky API를 호출해 위험 상품 목록을 가져온다."""
    params = {"sort_by": sort_by, "limit": limit}
    if year_month:
        params["year_month"] = year_month

    try:
        res = requests.get(f"{API_BASE_URL}/products/risky", params=params, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"위험 상품 목록 조회 실패: {e}"}


def annotate_and_reprioritize(results: list[dict]) -> list[dict]:
    """
    각 상품에 표본 충분 여부를 명시적으로 표시하고,
    "위험확률 x 운영신뢰도"로 재계산한 점수 기준 우선순위를 미리 계산해둔다.
    (LLM이 직접 판단/재정렬하지 않고, 이미 계산된 결과만 서술하도록 하기 위함)

    주의: /products/risky API는 review_evidence_confidence를 제공하지 않아,
    여기서는 operational_confidence만으로 가중 점수를 계산한다 (retrieval.py의
    combined_confidence보다 단순화된 버전).
    """
    annotated = []
    for p in results:
        entry = dict(p)
        entry["is_low_sample"] = p["current_review_count"] < LOW_REVIEW_COUNT_THRESHOLD
        entry["weighted_score"] = p["risk_probability"] * p["operational_confidence"]
        annotated.append(entry)

    # 원래 순위(입력 순서)를 보존해두고, 재계산된 점수 기준 순위도 매김
    for i, entry in enumerate(annotated, 1):
        entry["original_rank"] = i

    reprioritized = sorted(annotated, key=lambda e: e["weighted_score"], reverse=True)
    for i, entry in enumerate(reprioritized, 1):
        entry["reprioritized_rank"] = i

    return annotated, reprioritized


def build_risky_products_messages(data: dict, sort_by: str) -> list[dict]:
    """위험 상품 목록 요약을 위한 프롬프트를 만든다."""
    sort_desc = {
        "risk": "위험 확률이 높은 순",
        "confidence_weighted": "위험 확률에 신뢰도를 반영한 순",
    }.get(sort_by, sort_by)

    system_prompt = f"""당신은 아마존 가전용품 카테고리의 품질 위험 분석 애널리스트입니다.
주어진 위험 상품 목록만을 근거로 답변하며, 목록에 없는 내용은 지어내지 않습니다.
리포트 전체는 정중한 존댓말로 작성합니다.
아래 데이터의 is_low_sample과 reprioritized_rank는 이미 코드로 계산되어 주어진 것이므로,
당신이 직접 재판단하거나 재계산하지 않고 그 값을 그대로 서술에 반영합니다.

다음 형식을 따릅니다:

[전체 요약]
목록에 나타난 상품들의 위험 수준과 신뢰도를 종합해 2~3문장으로 요약합니다.
목록에 특정 카테고리(예: 제빙기, 필터류)가 반복해서 나타난다면 그 경향도 언급합니다.

[상품별 요약]
목록의 각 상품에 대해 아래 형식으로 한 줄씩 작성합니다:
"- 상품명 (브랜드): 위험확률 X%, 신뢰도 <confidence_level 값 그대로>, 리뷰 N건 — [해석]"
[해석]은 주어진 is_low_sample 값을 그대로 따릅니다:
- is_low_sample이 true인 상품: "리뷰 N건으로 표본이 적어 추가 관찰 필요" (리뷰 수를 반드시 언급)
- is_low_sample이 false인 상품: "리뷰 N건으로 표본이 충분하여 판단을 신뢰할 수 있음" (리뷰 수를 반드시 언급)
목록에 없는 세부 원인(구체적 불만 토픽 등)은 언급하지 않습니다.

[우선 확인 순서]
**반드시 reprioritized_rank가 1, 2, 3, 4, 5 순서(오름차순)가 되도록 상품을 나열합니다.**
다른 기준(원래 순위, 변동 방향 등)으로 재배열하지 않습니다. 즉 이 섹션의 첫 줄은 항상
reprioritized_rank=1인 상품, 둘째 줄은 reprioritized_rank=2인 상품이어야 합니다.
각 상품에 대해 반드시 아래 규칙에 따라 순위 변동 방향을 정확히 서술합니다:
- reprioritized_rank가 original_rank보다 작은 숫자면(예: 3위였는데 1위): "위험확률만으로는
  N위였으나 신뢰도를 반영하면 M위로 올라옵니다" (숫자가 작아지는 것 = 올라오는 것)
- reprioritized_rank가 original_rank보다 큰 숫자면(예: 2위였는데 3위): "위험확률만으로는
  N위였으나 신뢰도를 반영하면 M위로 내려갑니다" (숫자가 커지는 것 = 내려가는 것)
- 두 순위가 같으면: "위험확률과 신뢰도 모두 고려해도 순위가 그대로 유지됩니다"
어느 경우든, "올라온다"와 "내려간다"를 실제 숫자 비교 결과와 반대로 쓰지 않도록
반드시 두 순위를 다시 비교한 뒤에 씁니다.

목록이 비어 있다면 "조회된 위험 상품이 없습니다"라고만 답합니다."""

    original_order, reprioritized_order = annotate_and_reprioritize(data.get("results", []))

    if not original_order:
        product_lines = "(목록 없음)"
        reprioritized_lines = "(목록 없음)"
    else:
        product_lines = "\n".join(
            f"{p['original_rank']}. {p['product_title'][:80]} ({p['store']}): "
            f"risk_probability={p['risk_probability']*100:.1f}%, "
            f"operational_confidence={p['operational_confidence']*100:.1f}%, "
            f"confidence_level={p['confidence_level']}, "
            f"current_review_count={p['current_review_count']}, "
            f"is_low_sample={p['is_low_sample']} (기준: 리뷰 {LOW_REVIEW_COUNT_THRESHOLD}건 미만)"
            for p in original_order
        )
        # reprioritized_rank 오름차순으로 이미 정렬된 리스트를 그대로 텍스트로 만들어,
        # LLM이 순서를 다시 정할 필요 없이 이 순서를 그대로 따르기만 하면 되게 한다.
        reprioritized_lines = "\n".join(
            f"reprioritized_rank={p['reprioritized_rank']} (원래 순위 original_rank={p['original_rank']}): "
            f"{p['product_title'][:80]} ({p['store']})"
            for p in reprioritized_order
        )

    user_prompt = f"""다음은 {sort_desc}으로 정렬한 위험 상품 목록입니다. (기준 시점: {data.get('year_month', '알 수 없음')}, 정렬: {sort_by})

{product_lines}

---
아래는 [우선 확인 순서] 섹션 작성을 위해 이미 reprioritized_rank 오름차순으로 정렬해 둔
목록입니다. 이 순서를 그대로 따라 나열하고, 각 상품마다 original_rank와의 차이를 설명하세요.

{reprioritized_lines}

---
위 데이터를 바탕으로 요약해주세요."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_risky_products_query(
    sort_by: str = "risk", limit: int = 5, year_month: str | None = None
) -> dict:
    """
    위험 상품 목록을 조회하고 요약을 생성한다.

    Args:
        sort_by: "risk"(위험도순) 또는 "confidence_weighted"(신뢰도 반영 위험도순)
        limit: 조회할 상품 수
        year_month: 특정 월 지정 (없으면 최신 월 자동 사용)

    Returns:
        {"answer", "product_count"} 또는 실패 시 {"error"}
    """
    data = get_risky_products(sort_by=sort_by, limit=limit, year_month=year_month)

    if "error" in data:
        return {"error": data["error"]}

    messages = build_risky_products_messages(data, sort_by)
    answer = call_llm(messages)

    if answer is None:
        return {"error": "LLM 호출 실패"}

    return {
        "answer": answer,
        "product_count": data.get("count", 0),
        "year_month": data.get("year_month"),
    }



if __name__ == "__main__":
    print("위험 상품 Top 5 조회 중... (2020-06 기준)\n")
    result = answer_risky_products_query(sort_by="risk", limit=5, year_month="2020-06")

    if result.get("error"):
        print("실패:", result["error"])
    else:
        print(f"(기준 시점: {result['year_month']}, {result['product_count']}건)\n")
        print(result["answer"])