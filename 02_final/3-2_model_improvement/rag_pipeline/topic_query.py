"""
질문 유형 1: 특정 상품의 특정 불만 토픽에 대한 리뷰만 조회하고 요약한다.

예시 질문: "B0737B6HGR 상품에서 incompatibility(호환성) 관련 리뷰만 보여줘"

기존 파일과의 관계:
    - retrieval.py의 get_topic_related_reviews()를 그대로 재사용 (검색 로직 재사용)
    - generate_report.py의 call_llm()을 그대로 재사용 (OpenAI 호출 로직 재사용)
    - 새로 만든 건 이 파일의 프롬프트 구성과 실행 흐름뿐

실행 전 준비:
    1. 3-1 폴더에서 uvicorn main:app --reload 로 서버를 먼저 켜둘 것
    2. .env에 OPENAI_API_KEY가 채워져 있을 것
"""

from retrieval import get_topic_related_reviews
from generate_report import call_llm

# 12개 토픽 이름 (evidence_reviews의 related_topics와 동일한 값을 써야 함)
VALID_TOPICS = [
    "durability_failure", "build_quality", "incompatibility", "performance",
    "leak_damage", "water_taste_filter", "shipping_packaging",
    "return_intent", "return_blocked", "refund_warranty_cs",
    "misrepresentation", "purchase_warning",
]


def build_topic_query_messages(asin: str, topic: str, reviews_data: dict, show_all: bool = False) -> list[dict]:
    """토픽 관련 리뷰 요약을 위한 프롬프트를 만든다.

    Args:
        show_all: True면 조회된 리뷰 전체를 [대표 리뷰]에 인용하도록 지시.
                  False면(기본값) 대표적인 2~3개만 짧게 인용.
    """
    if show_all:
        review_instruction = (
            "[대표 리뷰]\n"
            "아래 리뷰 목록에 있는 리뷰를 하나도 빠짐없이 전부, 원문 그대로 번호를 매겨 인용합니다. "
            "요약하거나 일부만 골라내지 않습니다."
        )
    else:
        review_instruction = (
            "[대표 리뷰]\n"
            "가장 대표적인 리뷰 2~3개를 영어 원문 그대로 짧게 인용합니다."
        )

    system_prompt = f"""당신은 아마존 가전용품 리뷰 분석가입니다.
주어진 리뷰 목록만을 근거로 답변하며, 목록에 없는 내용은 지어내지 않습니다.
리포트 전체는 정중한 존댓말로 작성합니다.

각 리뷰에는 [tags: ...]로 이 리뷰가 매칭된 모든 토픽이 표시되어 있습니다. 여러 토픽이 함께
표시된 리뷰는, 지금 질문한 토픽 외의 다른 문제도 같이 언급하고 있다는 뜻입니다. 이런 리뷰를
요약에 포함할 때는 "이 리뷰는 {{다른 토픽}} 문제도 함께 지적하고 있습니다"처럼 자연스럽게
언급하여, 지금 토픽만의 문제인 것처럼 단정하지 않습니다.

다음 형식을 따릅니다:

[요약]
리뷰들이 공통적으로 지적하는 문제를 2~3문장으로 요약합니다. 다른 토픽과 겹치는 리뷰가
섞여 있다면 그 사실도 요약에 반영합니다.

{review_instruction}

리뷰가 없다면, "해당 토픽에 대한 리뷰가 확인되지 않았습니다"라고만 답하고 다른 섹션은 생략합니다."""

    reviews = reviews_data.get("reviews", [])
    if not reviews:
        review_text = "(해당 토픽에 대한 리뷰 없음)"
    else:
        review_lines = []
        for i, r in enumerate(reviews, 1):
            tags = r.get("related_topics") or topic
            review_lines.append(
                f"{i}. [{r['rating']} star] [tags: {tags}] \"{r['review_text'][:200]}\""
            )
        review_text = "\n".join(review_lines)

    # 이 목록이 상한선(limit)에 도달했는지 여부에 따라 문구를 다르게 함
    count = len(reviews)
    limit_hit_note = (
        f"(참고: 조회 상한({count}건)에 도달했을 수 있어, 실제로는 더 있을 수 있습니다.)"
        if count >= 5 else ""
    )

    user_prompt = f"""상품 {asin}의 "{topic}" 토픽으로 매칭된 리뷰 목록입니다. (이번 조회에서 {count}건 확인) {limit_hit_note}

{review_text}

---
위 리뷰들을 요약해주세요."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_topic_query(asin: str, topic: str, year_month: str | None = None, show_all: bool = False) -> dict:
    """
    특정 상품의 특정 토픽에 대한 리뷰를 조회하고 요약을 생성한다.

    Args:
        asin: 상품 parent_asin
        topic: 12개 토픽 이름 중 하나 (예: "incompatibility")
        year_month: 특정 월 지정 (없으면 전체 기간)
        show_all: True면 조회된 리뷰 전체를 인용, False면 대표 2~3개만 (기본값)

    Returns:
        {"asin", "topic", "answer", "review_count"} 또는 실패 시 {"error"}
    """
    if topic not in VALID_TOPICS:
        return {"error": f"알 수 없는 토픽입니다: {topic}. 사용 가능한 토픽: {VALID_TOPICS}"}

    # limit도 show_all이면 넉넉하게, 아니면 기본값(5) 사용
    reviews_data = get_topic_related_reviews(
        asin, topic, year_month=year_month, limit=50 if show_all else 5
    )

    if "error" in reviews_data:
        return {"error": reviews_data["error"]}

    messages = build_topic_query_messages(asin, topic, reviews_data, show_all=show_all)
    answer = call_llm(messages)

    if answer is None:
        return {"error": "LLM 호출 실패"}

    count = reviews_data.get("count", 0)
    count_limit = 15 if show_all else 5
    is_limited = count >= count_limit

    return {
        "asin": asin,
        "topic": topic,
        "answer": answer,
        "review_count": count,
        "review_count_note": (
            f"조회 상한({count_limit}건)에 도달하여, 실제로는 더 있을 수 있습니다."
            if is_limited else "확인된 전체 건수입니다."
        ),
    }


if __name__ == "__main__":
    test_asin = "B0737B6HGR"
    test_topic = "performance"

    print(f"상품 {test_asin}의 '{test_topic}' 토픽 리뷰 조회 중... (전체 보기)\n")
    result = answer_topic_query(test_asin, test_topic)

    if result.get("error"):
        print("실패:", result["error"])
    else:
        print(f"(관련 리뷰 {result['review_count']}건 - {result['review_count_note']})\n")
        print(result["answer"])
