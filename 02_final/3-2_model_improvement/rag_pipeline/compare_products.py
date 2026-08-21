"""
질문 유형 5: 두 상품의 위험도를 비교한다. (관리자용)

예시 질문: "상품 A와 B 중 어느 쪽을 먼저 확인해야 해?"

설계 원칙 (지금까지의 교훈 반영):
    - "어느 쪽이 더 우선인지" 판단은 LLM이 하지 않는다. retrieval.py의 risk_assessment가
      이미 계산해둔 weighted_risk(위험도x신뢰도, 데이터 기반 임계값으로 검증된 로직)를
      그대로 재사용해서 코드에서 승자를 확정하고, LLM은 그 결과를 설명만 한다.
    - 새로운 임계값을 감으로 만들지 않고, 기존에 검증된 risk_assessment 로직만 재사용한다.

기존 파일과의 관계:
    - retrieval.py의 get_product_context()를 상품 두 개에 대해 각각 호출 (재사용)
    - generate_report.py의 call_llm()을 재사용

실행 전 준비:
    1. 3-1 폴더에서 uvicorn main:app --reload 로 서버를 먼저 켜둘 것
    2. .env에 OPENAI_API_KEY가 채워져 있을 것
"""

from retrieval import get_product_context
from generate_report import call_llm


def compare_risk(context_a: dict, context_b: dict) -> dict:
    """
    두 상품의 risk_assessment를 비교해, 어느 쪽을 우선 확인해야 하는지 코드에서 확정한다.
    (risk_assessment의 risk_probability, weighted_risk는 이미 retrieval.py에서
     데이터 기반으로 검증된 로직으로 계산된 값이므로, 여기서는 그대로 재사용한다.)
    """
    ra_a = context_a.get("risk_assessment")
    ra_b = context_b.get("risk_assessment")

    if ra_a is None or ra_b is None:
        return {"comparable": False, "reason": "한 쪽 이상 위험도 데이터가 없어 비교할 수 없습니다."}

    # 원본 위험확률 기준 승자
    raw_winner = "A" if ra_a["risk_probability"] > ra_b["risk_probability"] else "B"
    # 신뢰도 반영(weighted_risk) 기준 승자 - risk_assessment에 이미 계산되어 있음
    weighted_winner = "A" if ra_a["weighted_risk"] > ra_b["weighted_risk"] else "B"

    winner_changed = raw_winner != weighted_winner

    # 데이터셋(Test 샘플) 한계: 전체 상품x월이 다 있는 게 아니라서,
    # 두 상품의 "최신 시점"이 서로 다를 수 있다. year_month가 다르면 그 사실을
    # 명시적으로 표시해서, "동일 시점 비교"인 것처럼 오해하지 않게 한다.
    year_month_a = context_a["summary"].get("year_month")
    year_month_b = context_b["summary"].get("year_month")
    same_period = year_month_a == year_month_b

    return {
        "comparable": True,
        "raw_winner": raw_winner,
        "weighted_winner": weighted_winner,
        "winner_changed": winner_changed,
        "year_month_a": year_month_a,
        "year_month_b": year_month_b,
        "same_period": same_period,
        "risk_assessment_a": ra_a,
        "risk_assessment_b": ra_b,
    }


def build_compare_messages(asin_a: str, asin_b: str, context_a: dict, context_b: dict, comparison: dict) -> list[dict]:
    """두 상품 비교를 위한 프롬프트를 만든다."""
    system_prompt = """당신은 아마존 가전용품 카테고리의 품질 위험 분석 애널리스트입니다.
이 리포트는 관리자가 여러 상품 중 어디를 먼저 점검할지 결정하는 데 사용됩니다.
주어진 데이터만을 근거로 답변하며, 데이터에 없는 내용은 추측하지 않습니다.
리포트 전체는 정중한 존댓말로 작성합니다.

아래 raw_winner와 weighted_winner는 이미 코드로 계산되어 확정된 값이므로,
당신이 직접 재계산하거나 다른 결론을 내지 않고 그 값을 그대로 서술에 반영합니다.

주의: same_period가 false라면, 두 상품의 데이터가 서로 다른 시점(year_month_a, year_month_b)을
기준으로 한다는 뜻입니다. 이 경우 [비교 요약] 시작 부분에 반드시 "두 상품은 서로 다른 시점
(year_month_a 대 year_month_b) 데이터를 기준으로 비교되었습니다"라고 존댓말로 먼저 명시한 뒤
비교를 시작합니다. same_period가 true라면 이 언급을 생략합니다.

다음 형식을 따릅니다:

[비교 요약]
두 상품의 위험 확률, 신뢰도 수준(risk_level, confidence_level), interpretation을
각각 한 문장씩 요약합니다.

[우선순위 판단]
raw_winner를 근거로 "위험 확률만 보면 {{raw_winner}} 상품이 더 위험하다"고 먼저 밝힙니다.
그 다음 weighted_winner를 근거로 서술합니다:
- winner_changed가 false이면: "신뢰도를 반영해도 우선순위는 바뀌지 않는다"고 명시합니다.
- winner_changed가 true이면: 반드시 "다만 신뢰도를 반영하면 우선순위가 {{weighted_winner}}
  상품으로 바뀐다"고 명시하고, 그 이유(표본 크기나 신뢰도 차이)를 데이터에서 찾아 설명합니다.

[핵심 차이]
두 상품의 주요 불만 토픽 차이, 근거 리뷰의 성격 차이를 대조하여 설명합니다.
한쪽에 급증 토픽이나 근거 리뷰가 없다면 그 사실도 명시합니다.

[관리자를 위한 제안]
weighted_winner를 최종 결론으로 삼아, 관리자가 지금 어느 상품을 우선 점검해야 하는지
1~2문장으로 제시합니다. [우선순위 판단]에서 이미 설명한 이유를 반복하지 말고,
"따라서 관리자는 weighted_winner 상품을 우선 점검하시기 바랍니다"처럼 결론만 간결하게
제시합니다. 표본이 부족한 쪽의 데이터가 있다면 그 상품명을 특정하여 "참고용으로만
고려해야 한다"고 짧게 덧붙입니다.

데이터가 비교 불가능하다면 그 이유만 명시하고 다른 섹션은 생략합니다."""

    if not comparison["comparable"]:
        user_prompt = f"상품 {asin_a}와 {asin_b}를 비교하려 했으나: {comparison['reason']}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _product_block(label: str, asin: str, context: dict) -> str:
        product = context["summary"]["product"]
        ra = context["risk_assessment"]
        topics_block = context["topics"]
        evidence_block = context["evidence_reviews"]

        lines = [f"[상품 {label}: {asin}]"]
        lines.append(f"- 상품명: {product['product_title'][:80]} ({product['store']})")
        lines.append(f"- risk_probability: {ra['risk_probability']*100:.1f}%")
        lines.append(f"- risk_level: {ra['risk_level']}, confidence_level: {ra['confidence_level']}")
        lines.append(f"- interpretation: {ra['interpretation']}")
        lines.append(f"- weighted_risk(신뢰도 반영 점수): {ra['weighted_risk']:.4f}")
        if ra["caveats"]:
            lines.append(f"- caveat: {'; '.join(ra['caveats'])}")

        if topics_block["has_surging_topic"]:
            surging = sorted(
                [t for t in topics_block["topics"] if (t.get("delta") or 0) > 0],
                key=lambda t: t["delta"], reverse=True
            )[:3]
            topic_str = ", ".join(f"{t['topic_name']}(+{t['delta']*100:.1f}%p)" for t in surging)
            lines.append(f"- 주요 급증 토픽: {topic_str}")
        else:
            lines.append(f"- 주요 급증 토픽: 없음 ({topics_block['note']})")

        if evidence_block["has_evidence"]:
            lines.append(f"- 근거 리뷰 예시: \"{evidence_block['reviews'][0]['review_text'][:150]}\"")
        else:
            lines.append("- 근거 리뷰: 없음")

        return "\n".join(lines)

    block_a = _product_block("A", asin_a, context_a)
    block_b = _product_block("B", asin_b, context_b)

    user_prompt = f"""다음은 비교할 두 상품의 데이터입니다.

{block_a}

{block_b}

[미리 계산된 비교 결과 - 이 값을 그대로 사용하세요]
- raw_winner (위험확률만 기준): {comparison['raw_winner']}
- weighted_winner (신뢰도 반영 기준): {comparison['weighted_winner']}
- winner_changed (우선순위가 바뀌었는지): {comparison['winner_changed']}
- year_month_a: {comparison['year_month_a']}, year_month_b: {comparison['year_month_b']}
- same_period (두 상품이 같은 시점 데이터인지): {comparison['same_period']}

---
위 데이터를 바탕으로 두 상품을 비교해주세요."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_compare_query(
    asin_a: str, asin_b: str,
    year_month_a: str | None = None, year_month_b: str | None = None,
) -> dict:
    """
    두 상품의 위험도를 비교하고 설명을 생성한다.

    Args:
        asin_a, asin_b: 비교할 두 상품
        year_month_a, year_month_b: 각 상품의 시점을 따로 지정 (없으면 각자 최신 월 자동 사용)

    Returns:
        {"asin_a", "asin_b", "answer", "comparison"} 또는 실패 시 {"error"}
    """
    context_a = get_product_context(asin_a, year_month=year_month_a)
    context_b = get_product_context(asin_b, year_month=year_month_b)

    if "error" in context_a:
        return {"error": f"상품 A 조회 실패: {context_a['error']}"}
    if "error" in context_b:
        return {"error": f"상품 B 조회 실패: {context_b['error']}"}

    comparison = compare_risk(context_a, context_b)
    messages = build_compare_messages(asin_a, asin_b, context_a, context_b, comparison)
    answer = call_llm(messages)

    if answer is None:
        return {"error": "LLM 호출 실패"}

    return {"asin_a": asin_a, "asin_b": asin_b, "answer": answer, "comparison": comparison}


if __name__ == "__main__":
    test_asin_a = "B0C58TH22G"
    test_year_month_a = "2021-09"
    test_asin_b = "B0C7CC35KH"
    test_year_month_b = "2017-04" 

    print(f"상품 {test_asin_a}({test_year_month_a}) vs {test_asin_b}({test_year_month_b}) 비교 중...\n")
    result = answer_compare_query(
        test_asin_a, test_asin_b,
        year_month_a=test_year_month_a, year_month_b=test_year_month_b,
    )

    if result.get("error"):
        print("실패:", result["error"])
    elif not result["comparison"]["comparable"]:
        print("비교 불가:", result["comparison"]["reason"])
        print("\n=== LLM 응답 ===")
        print(result["answer"])
    else:
        print(f"(raw_winner={result['comparison']['raw_winner']}, "
              f"weighted_winner={result['comparison']['weighted_winner']}, "
              f"winner_changed={result['comparison']['winner_changed']})\n")
        print(result["answer"])
