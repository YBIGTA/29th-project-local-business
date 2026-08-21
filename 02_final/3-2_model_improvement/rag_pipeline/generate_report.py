"""
3-2 리포트 생성 (최종 실행 파일)

retrieval.py(검색) + prompt_templates.py(프롬프트 조립)를 연결하여
실제 OpenAI API를 호출하고, 상품 품질 위험 리포트를 생성한다.
"""

import time
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

from config import OPENAI_API_KEY
from retrieval import get_product_context
from prompt_templates import build_messages

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4o-mini"   # 저렴하고 빠른 모델로 시작. 품질 부족하면 상위 모델로 교체
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2
REQUEST_TIMEOUT = 30


def call_llm(messages: list[dict]) -> str | None:
    """
    OpenAI API를 호출해 리포트 텍스트를 받아온다.
    실패 시 재시도하고, 최종 실패하면 None을 반환한다.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,   # 리포트는 일관성이 중요하니 낮은 온도로 설정
                timeout=REQUEST_TIMEOUT,
            )
            return response.choices[0].message.content
        except (APITimeoutError, APIConnectionError, APIError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    print(f"[generate_report] OpenAI 호출 실패 (재시도 {MAX_RETRIES}회 소진): {last_error}")
    return None


def generate_report(asin: str, year_month: str | None = None, user_question: str | None = None) -> dict:
    """
    상품 하나에 대한 품질 위험 리포트를 생성한다.

    Args:
        asin: 상품 parent_asin
        year_month: 특정 월 지정 (없으면 최신 월 자동 사용)
        user_question: 사용자가 직접 입력한 질문 (없으면 기본 리포트 요청)

    Returns:
        {
            "asin": ...,
            "report": "생성된 리포트 텍스트" 또는 None(실패 시),
            "context": retrieval.py가 가져온 원본 데이터 (근거 확인/디버깅용),
            "error": 실패 시에만 존재
        }
    """
    context = get_product_context(asin, year_month=year_month)

    if "error" in context:
        return {"asin": asin, "report": None, "context": context, "error": context["error"]}

    messages = build_messages(asin, context, user_question=user_question)
    report_text = call_llm(messages)

    if report_text is None:
        return {"asin": asin, "report": None, "context": context, "error": "LLM 호출 실패"}

    return {"asin": asin, "report": report_text, "context": context}


if __name__ == "__main__":
    test_asin = "B092T5Y46X"
    print(f"상품 {test_asin} 리포트 생성 중...\n")

    result = generate_report(test_asin, year_month="2017-03")
    
    if result.get("error"):
        print("실패:", result["error"])
    else:
        print("=== 생성된 리포트 ===\n")
        print(result["report"])
