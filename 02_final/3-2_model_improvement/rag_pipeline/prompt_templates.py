"""
3-2 프롬프트 템플릿 (v2)

retrieval.py의 get_product_context()가 만든 딕셔너리 데이터를
LLM(OpenAI)이 읽을 수 있는 자연어 프롬프트로 변환한다.
"""

from retrieval import SURGE_DELTA_THRESHOLD, LOW_REVIEW_COUNT_THRESHOLD


def format_context_for_prompt(context: dict) -> str:
    """
    retrieval.py의 get_product_context() 결과를 사람이 읽는 텍스트로 변환한다.
    """
    if "error" in context:
        return f"[조회 실패] {context['error']}"

    lines = []

    # --- 상품 기본 정보 ---
    product = context["summary"]["product"]
    lines.append("## Product Info")
    lines.append(f"- Title: {product['product_title']}")
    lines.append(f"- Brand: {product['store']}")
    lines.append(f"- Reference month: {context['summary'].get('year_month', 'unknown')}")
    lines.append("")

    # --- 위험도/신뢰도 종합 판단 ---
    ra = context.get("risk_assessment")
    lines.append("## Risk & Confidence")
    if ra is None:
        lines.append("- No risk prediction available for this period.")
    else:
        lines.append(f"- Risk probability (raw model output): {ra['risk_probability']*100:.1f}%")
        lines.append(f"- Risk level: {ra['risk_level']} / Confidence level: {ra['confidence_level']}")
        lines.append(f"- Interpretation: {ra['interpretation']}")
        lines.append(
            f"- (Reference only, not used in confidence calc) Model consistency confidence: "
            f"{ra['model_consistency_confidence_reference']*100:.1f}%"
        )
        if ra["caveats"]:
            for c in ra["caveats"]:
                lines.append(f"- Caveat: {c}")
        else:
            lines.append("- No sample-size caveat for this product (review count is sufficient).")
    lines.append("")

    # --- 정형 지표 ---
    metrics = context["summary"].get("metrics")
    if metrics:
        lines.append("## Structured Metrics")
        review_count = metrics['review_count']
        lines.append(f"- This month TOTAL review count: {review_count} "
                      f"(this is the count of ALL reviews, not just low-rating ones)")
        lines.append(f"- This month avg rating: {metrics['avg_rating']:.2f}")
        lines.append(f"- This month low-rating(1-2 star) ratio: {metrics['low_rating_ratio']*100:.1f}%")
        lines.append(f"- 12-month avg rating: {metrics['history_12m_avg_rating']:.2f}, "
                      f"12-month low-rating ratio: {metrics['history_12m_low_rating_ratio']*100:.1f}%")
        lines.append(f"- Recent 3-month vs 12-month low-rating ratio change: "
                      f"{metrics['history_3m_vs_12m_low_rating_ratio_delta']*100:+.1f}%p")
        if review_count < LOW_REVIEW_COUNT_THRESHOLD:
            lines.append(
                f"- Note: total review count ({review_count}) is low, so the ratios above "
                f"(low-rating ratio, avg rating) are also based on a small sample and should "
                f"be treated as reference, not confirmed figures."
            )
    lines.append("")

    # --- 불만 토픽: 표본 충분 / 표본 부족 그룹으로 분리 ---
    topics_block = context["topics"]
    lines.append("## Complaint Topic Changes")
    if not topics_block["has_surging_topic"]:
        lines.append(f"- {topics_block['note']}")
        lines.append("- (There is no low-sample surging topic to report either, since no topic surged at all.)")
    else:
        surging = [
            t for t in topics_block["topics"]
            if (t.get("delta") or 0) >= SURGE_DELTA_THRESHOLD
        ]
        surging.sort(key=lambda t: t["delta"], reverse=True)

        reliable = [t for t in surging if not t["is_low_sample"]]
        low_sample = [t for t in surging if t["is_low_sample"]]

        if reliable:
            lines.append("[Reliable surge - sufficient mentions]")
            for t in reliable[:5]:
                lines.append(
                    f"- {t['topic_name']}: this month {t['rate_t']*100:.1f}%, "
                    f"change vs 3 months ago {t['delta']*100:+.1f}%p, {t['mention_count']} mentions"
                )
        if low_sample:
            lines.append("[Low-sample - reference only, may be coincidental]")
            for t in low_sample[:5]:
                lines.append(
                    f"- {t['topic_name']}: this month {t['rate_t']*100:.1f}%, "
                    f"change vs 3 months ago {t['delta']*100:+.1f}%p, only {t['mention_count']} mention(s)"
                )
            lines.append(
                "  (Note: this product has few reviews overall, so individual topic mentions "
                "are often based on just 1 review. Treat these as reference, not confirmed signals.)"
            )
        if not reliable and not low_sample:
            lines.append(f"- No topic reached the surge threshold ({SURGE_DELTA_THRESHOLD*100:.0f}%p).")
    lines.append("")

    # --- 근거 리뷰 ---
    evidence_block = context["evidence_reviews"]
    lines.append("## Evidence Reviews")
    if not evidence_block["has_evidence"]:
        lines.append(f"- {evidence_block['note']}")
    else:
        for i, r in enumerate(evidence_block["reviews"][:5], 1):
            topic_str = r["related_topics"] if r.get("related_topics") else "unclassified"
            lines.append(f"{i}. [{r['rating']} star, {topic_str}] \"{r['review_text'][:200]}\"")
    lines.append("")

    # --- 위험 판단 근거 변수 ---
    explanation = context.get("risk_explanation", {})
    lines.append("## Variables Influencing Risk Prediction (approximate, based on percentile rank across products)")
    for e in explanation.get("explanations", [])[:5]:
        lines.append(f"- {e['feature_name']}")
    if explanation.get("note"):
        lines.append(f"- Note: {explanation['note']}")

    return "\n".join(lines)


def build_system_prompt() -> str:
    """LLM에게 역할과 답변 형식을 지시하는 고정 프롬프트."""
    return """당신은 아마존 가전용품 카테고리의 품질 위험 분석 애널리스트입니다.
주어진 데이터만을 근거로 리포트를 작성하며, 데이터에 없는 내용은 추측하거나 지어내지 않습니다.
리포트 전체는 반드시 정중한 존댓말("~습니다")로 작성합니다. 아래 지시문에 포함된 예시 문구도
내용 전달용일 뿐이므로, 그대로 베끼지 말고 항상 존댓말로 바꿔서 씁니다.

**중요**: 아래는 당신이 "지켜야 할 규칙"이지, 리포트에 그대로 옮겨 적을 "데이터 내용"이 아닙니다.
"Reliable surge", "Low-sample", "참고 수준의 신호로만 언급합니다" 같은 규칙 설명 문구 자체를
리포트 문장에 그대로 베끼지 말고, 반드시 실제 수치와 토픽명을 사용해 자연스러운 문장으로 다시 씁니다.
해당하는 토픽이나 caveat이 실제로 없다면, 그 규칙 자체를 언급하지 말고 그냥 생략합니다.

리포트는 아래 형식을 정확히, 순서와 소제목을 바꾸지 않고 따릅니다.
맨 첫 줄은 반드시 "상품명: {실제 상품명}", 둘째 줄은 "브랜드: {실제 브랜드}" 형식으로,
다른 문구나 마크다운 기호 없이 작성합니다.

[위험 요약]
아래 내용만 포함한 두세 문장으로 작성합니다. 정형 지표의 구체적 수치(예: 몇 % 증가)는
[핵심 원인]에서만 다루므로 여기서는 반복하지 않습니다.
1. interpretation 문구의 핵심 내용을 거의 그대로 반영합니다 (임의로 축약하지 않습니다).
2. Risk probability(원본 확률 수치)를 인용할 경우, 반드시 confidence_level과 함께 언급하여
   "수치가 높다고 곧 확실하다"는 인상을 주지 않도록 합니다.
3. Caveat이 실제로 존재할 때만, 그 안의 구체적 수치(예: 리뷰 개수)를 포함합니다.
   "No sample-size caveat" 이라고 되어 있다면 표본 관련 경고를 아예 언급하지 않습니다.
   앞 문장과 논리적으로 자연스럽게 연결되지 않는 접속사("그러나" 등)를 억지로 쓰지 않습니다.

[핵심 원인]
다음 세 가지를, 반드시 아래 소제목 그대로("- 불만 토픽 변화:", "- 정형 지표 변화:",
"- 위험 판단 근거 변수:") 각각 별도의 줄에 작성합니다. 소제목을 바꾸거나 생략하지 않습니다.

- 불만 토픽 변화: 실제로 급증한 토픽이 있다면, 표본이 충분한 토픽과 표본이 적은(1건 근처) 토픽을
  구분해서 구체적 수치(비율, 변화량, 언급 건수)와 함께 설명합니다. 급증한 토픽이 아예 없다면
  "특별히 급증한 불만 토픽은 없다"고만 명시하고, 존재하지 않는 표본부족 토픽을 언급하지 않습니다.
- 정형 지표 변화: "This month TOTAL review count"는 전체 리뷰 수이며 저평점 리뷰 수가 아닙니다.
  이 둘을 혼동해서 서술하지 않습니다. 저평점 비율을 서술할 때는 반드시 "이번 달 X% (12개월 평균
  Y%, 변화 Z%p)" 형식으로, 이번 달 수치와 12개월 평균 수치를 한 문장 안에 나란히 명시합니다.
  두 값을 서로 다른 문장에 떨어뜨려 쓰지 않습니다. 델타(%p) 부호가 "이번달 - 12개월평균"과
  일치하는지 다시 계산해서 확인한 뒤 씁니다. 데이터에 "reference only" 노트가 있다면
  ("This month TOTAL review count" 바로 다음 줄에 Note로 표시됨) 반드시 이 문장 안에서
  "표본이 적어 참고용"이라는 취지를 언급합니다. 그런 노트가 없다면 참고용이라는 말을 쓰지
  않습니다. 이 문장에는 모델/변수 관련 단서를 붙이지 않습니다.
- 위험 판단 근거 변수: "Variables Influencing Risk Prediction" 목록에 있는 변수명을
  그대로 나열하고, 반드시 "이 값은 각 변수를 상품 간 상대적 위치(백분위)로 환산한 근사 지표이며, 
  모델의 실제 판단 근거는 아니다"라는 단서를 이 항목 안에서만 붙입니다. 이 단서를 위 두 항목에 붙이지 않습니다.

[근거 리뷰]
실제 리뷰 원문을 짧게 인용하여 핵심 원인을 뒷받침합니다. 리뷰는 영어 원문을 그대로 인용합니다.
근거 리뷰가 없다면 "확인된 저평점 근거 리뷰가 없습니다"라고 존댓말로 명시합니다
(전체 리뷰가 없다는 뜻으로 서술하지 않습니다).

[우선 조치 제안]
반드시 "1." "2." "3." 형식의 번호 목록으로 1~3개를 작성합니다. 산문이나 문단 형식으로 쓰지 않습니다.
각 제안은 반드시 [핵심 원인]에서 실제로 언급한 구체적 토픽명, 수치, 또는 근거 리뷰의 내용을
직접 지목하여 작성합니다. "지속적인 리뷰 모니터링을 통해 피드백을 살펴보시기 바랍니다"처럼
어떤 상품에나 붙일 수 있는 일반론적 문구만으로는 안 되며, 최소 1개 이상의 제안은 이 상품의
핵심 원인(예: 특정 불만 토픽, 특정 리뷰에서 지적된 문제)과 명시적으로 연결되어야 합니다.
근거 리뷰나 급증 토픽이 확인되지 않았다면, 그 상황에서도 번호 목록 형식은 유지하되
"지속적인 리뷰 모니터링"류의 조치를 제안하고 확정적 문제가 있는 것처럼 서술하지 않습니다.

주의사항:
- Model consistency confidence는 참고 정보일 뿐 신뢰도 판단에 직접 반영되지 않았다는 점을 인지합니다.
- 주어지지 않은 데이터를 임의로 추측하지 않습니다."""


def build_messages(asin: str, context: dict, user_question: str | None = None) -> list[dict]:
    """
    OpenAI API에 넘길 최종 messages 리스트를 만든다.

    Args:
        asin: 상품 parent_asin
        context: retrieval.py의 get_product_context() 결과
        user_question: 사용자가 직접 입력한 질문 (없으면 기본 리포트 요청으로 대체)
    """
    context_text = format_context_for_prompt(context)
    question = user_question or f"상품({asin})의 품질 위험 리포트를 작성해줘."

    user_prompt = f"""다음은 상품 {asin}에 대한 실제 데이터입니다.

{context_text}

---
질문: {question}
위 데이터만 근거로 답변하세요."""

    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


if __name__ == "__main__":
    from retrieval import get_product_context

    test_asin = "B0737B6HGR"
    context = get_product_context(test_asin)
    messages = build_messages(test_asin, context)

    print("=== SYSTEM ===")
    print(messages[0]["content"])
    print("\n=== USER (LLM에게 전달될 최종 프롬프트) ===")
    print(messages[1]["content"])
