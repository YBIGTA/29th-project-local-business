"""
질문 유형 4: 상품의 시간에 따른 위험도 추이를 설명한다.

예시 질문: "이 상품 위험도가 최근 어떻게 변해왔어?"

설계 원칙 (어제 배운 교훈 반영):
    - "언제부터 나빠졌는지", "가장 위험했던 시점" 등은 LLM이 원본 리스트를 보고 스스로
      계산하게 하지 않는다. 파이썬에서 미리 계산(피크/저점/월별 변화량/전체 방향)해서
      숫자로 확정한 뒤, LLM은 그 결과를 자연어로 설명하는 역할만 한다.

기존 파일과의 관계:
    - 3-1의 GET /products/{asin}/trend API를 그대로 호출 (검색 로직 재사용)
    - generate_report.py의 call_llm()을 재사용 (OpenAI 호출 로직 재사용)
    - retrieval.py의 LOW_REVIEW_COUNT_THRESHOLD를 재사용 (표본부족 판단 기준 일관성 유지)

실행 전 준비:
    1. 3-1 폴더에서 uvicorn main:app --reload 로 서버를 먼저 켜둘 것
    2. .env에 OPENAI_API_KEY가 채워져 있을 것
"""

import requests

from generate_report import call_llm
from retrieval import LOW_REVIEW_COUNT_THRESHOLD

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 10

# 전월 대비 risk_probability가 이 값(%p) 이상 변하면 "급변 시점"으로 표시.
# product_month_risk 전체(3,310개 상품x월 쌍)의 전월 대비 변화량 절댓값 분포를 확인한 결과:
#   중앙값 10.1%p, 75% 19.8%p, 90% 29.2%p, 95% 35.8%p
# "흔한 변동"이 아니라 "정말 두드러진 변화"만 잡기 위해 상위 10%(90분위) 기준을 사용한다.
SIGNIFICANT_MONTHLY_CHANGE = 0.29

# "완만하지만 지속적인 악화/개선" 추세를 판단하는 기준 (월평균 변화율, %p/월).
# 2개월 이상 데이터가 있는 상품 222개를 대상으로 (전체 변화량 / 관찰개월수-1)의
# 절댓값 분포를 확인한 결과: 중앙값 2.1%p, 75% 7.3%p, 90% 18.2%p, 95% 29.4%p
# 단발성 급변(SIGNIFICANT_MONTHLY_CHANGE)보다는 낮은 값으로도 "꾸준한 추세"로 볼 수
# 있으므로 상위 10%(90분위) 기준을 사용한다.
SUSTAINED_TREND_RATE_THRESHOLD = 0.182


def get_trend(asin: str) -> dict:
    """3-1의 /products/{asin}/trend API를 호출해 월별 추이를 가져온다."""
    try:
        res = requests.get(f"{API_BASE_URL}/products/{asin}/trend", timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"추이 조회 실패: {e}"}


def analyze_trend(trend: list[dict]) -> dict:
    """
    월별 추이 리스트를 받아, LLM에게 넘길 핵심 통계를 미리 계산한다.
    (피크/저점/전체 방향/급변 시점/표본부족 구간 등을 파이썬에서 확정)
    """
    if not trend:
        return {"has_data": False}

    start = trend[0]
    end = trend[-1]
    peak = max(trend, key=lambda t: t["risk_probability"])
    trough = min(trend, key=lambda t: t["risk_probability"])

    overall_change = end["risk_probability"] - start["risk_probability"]
    overall_direction = (
        "악화" if overall_change > 0.05 else "개선" if overall_change < -0.05 else "큰 변화 없음"
    )

    # 전월 대비 급변 시점 계산 (파이썬에서 확정, LLM이 다시 계산하지 않게 함)
    significant_jumps = []
    for i in range(1, len(trend)):
        prev, curr = trend[i - 1], trend[i]
        delta = curr["risk_probability"] - prev["risk_probability"]
        if abs(delta) >= SIGNIFICANT_MONTHLY_CHANGE:
            significant_jumps.append({
                "from_month": prev["year_month"],
                "to_month": curr["year_month"],
                "delta": round(delta, 4),
                "direction": "급등" if delta > 0 else "급락",
            })

    # 표본부족(리뷰 수 적음) 구간이 전체의 몇 %인지
    low_sample_months = [t for t in trend if t.get("review_count", 0) < LOW_REVIEW_COUNT_THRESHOLD]
    low_sample_ratio = len(low_sample_months) / len(trend)
    low_sample_month_set = {t["year_month"] for t in low_sample_months}

    # 급변 시점 중 표본부족 구간과 겹치는 것이 있는지 미리 확인
    # (LLM이 "리뷰 수 감소와 위험도 급등이 겹친다"를 스스로 대조하지 않도록,
    #  겹침 여부를 코드에서 확정해서 알려준다)
    for jump in significant_jumps:
        jump["overlaps_low_sample"] = (
            jump["from_month"] in low_sample_month_set or jump["to_month"] in low_sample_month_set
        )

    # "완만하지만 지속적인 추세" 판단: 단발성 급변 여부와 무관하게 항상 계산한다.
    n_months = len(trend)
    avg_monthly_rate = overall_change / (n_months - 1) if n_months > 1 else 0.0
    is_sustained_trend = abs(avg_monthly_rate) >= SUSTAINED_TREND_RATE_THRESHOLD

    # 급변 시점이 많을 경우, 절댓값 기준 상위 3개만 "핵심"으로 표시
    top_jumps = sorted(significant_jumps, key=lambda j: abs(j["delta"]), reverse=True)[:3]

    # 월간 변화 방향의 일관성: 전체 방향(overall_direction)과 같은 방향으로 움직인
    # 구간이 몇 번인지 (단순 카운트라 별도 임계값 불필요)
    step_deltas = [
        trend[i]["risk_probability"] - trend[i - 1]["risk_probability"]
        for i in range(1, len(trend))
    ]
    total_steps = len(step_deltas)
    if overall_change > 0:
        consistent_steps = sum(1 for d in step_deltas if d > 0)
    elif overall_change < 0:
        consistent_steps = sum(1 for d in step_deltas if d < 0)
    else:
        consistent_steps = 0
    direction_consistency_ratio = (consistent_steps / total_steps) if total_steps > 0 else 0.0

    return {
        "has_data": True,
        "total_months": len(trend),
        "start": start,
        "end": end,
        "peak": peak,
        "trough": trough,
        "overall_change": round(overall_change, 4),
        "overall_direction": overall_direction,
        "significant_jumps": significant_jumps,
        "top_jumps": top_jumps,
        "avg_monthly_rate": round(avg_monthly_rate, 4),
        "is_sustained_trend": is_sustained_trend,
        "consistent_steps": consistent_steps,
        "total_steps": total_steps,
        "direction_consistency_ratio": round(direction_consistency_ratio, 4),
        "low_sample_month_count": len(low_sample_months),
        "low_sample_ratio": round(low_sample_ratio, 4),
    }


def build_trend_messages(asin: str, trend: list[dict], analysis: dict) -> list[dict]:
    """추이 설명을 위한 프롬프트를 만든다."""
    system_prompt = f"""당신은 아마존 가전용품 카테고리의 품질 위험 분석 애널리스트입니다.
주어진 월별 데이터와, 이미 계산되어 주어진 통계(피크/저점/전체방향/급변시점 등)만을
근거로 답변합니다. 피크가 언제인지, 얼마나 변했는지 등은 이미 계산되어 주어졌으므로
당신이 원본 리스트를 보고 다시 계산하거나 다른 값을 추정하지 않습니다.
리포트 전체는 정중한 존댓말로 작성합니다.

다음 형식을 따릅니다:

[전체 흐름]
overall_direction과 overall_change, 전체 관찰 기간(total_months개월)을 반영해 2~3문장으로
요약합니다. start와 end 시점의 risk_probability를 반드시 함께 언급합니다.

[주요 시점]
peak(가장 위험했던 시점)과 trough(가장 안전했던 시점)를 각각 연월과 수치로 명시합니다.
top_jumps(급변 시점 중 변화폭이 가장 큰 상위 항목)만 골라 하나씩 설명합니다. 각 항목에 대해
"YYYY-MM에서 YYYY-MM 사이 위험도가 X%p 급등/급락" 형식으로 쓰고, overlaps_low_sample이
true라면 "이 시기는 리뷰 수가 적어(표본부족) 변화폭이 실제보다 과장되었을 수 있다"는 점을
반드시 함께 언급합니다. top_jumps가 비어 있다면 "뚜렷한 단발성 급변 시점은 확인되지 않았다"고
명시합니다. 전체 significant_jumps 개수가 top_jumps보다 많다면, "이 외에도 N건의 변동이
더 있었으나 변화폭이 상대적으로 작아 생략한다"고 짧게 언급합니다.
이 섹션에서는 단발성 급변만 다루고, 지속적인 추세는 다음 섹션에서 다루므로 여기서
avg_monthly_rate나 direction_consistency_ratio는 언급하지 않습니다.

[지속 추세]
이 섹션은 top_jumps 유무와 무관하게 항상 작성합니다.
avg_monthly_rate와 is_sustained_trend를 근거로, 반드시 아래 두 문장 중 하나를 골라
(수치만 채워서) 그대로 사용합니다. 다른 표현으로 바꾸거나 확정적 어조를 섞지 않습니다.
- is_sustained_trend가 true인 경우: "관찰 기간 동안 특정 달의 급변이 아니라, 월평균
  {{avg_monthly_rate}}%p만큼 완만하지만 꾸준하게 [악화/개선]되는 추세를 보입니다."
- is_sustained_trend가 false인 경우: "월평균 변화율은 {{avg_monthly_rate}}%p로, 위험도가
  [악화/개선]되는 방향으로 조금씩 움직였으나, 뚜렷한 지속 추세로 분류할 만큼 크지는
  않았습니다." (숫자와 방향은 반드시 포함하되, "완만하지만 꾸준하게 [악화/개선]되는
  추세를 보인다"처럼 확정적으로 단정하지는 않습니다.)
추가로 direction_consistency_ratio와 consistent_steps, total_steps를 근거로
"총 {{total_steps}}개 구간 중 {{consistent_steps}}개 구간에서 같은 방향으로 움직였다"는
식으로 방향의 일관성도 함께 언급합니다.

[신뢰도 관련 참고]
low_sample_month_count와 low_sample_ratio를 근거로, 전체 관찰 기간 중 표본이 부족했던
달이 어느 정도 비중인지 언급합니다. [주요 시점]에서 이미 표본부족과 겹치는 급변을
설명했다면 여기서 반복하지 말고, 그 외 전반적인 표본 수준만 짧게 언급합니다.

목록이 비어 있다면 "조회된 추이 데이터가 없습니다"라고만 답합니다."""

    if not analysis["has_data"]:
        month_lines = "(데이터 없음)"
        stats_text = "(계산할 데이터 없음)"
    else:
        month_lines = "\n".join(
            f"{t['year_month']}: risk_probability={t['risk_probability']*100:.1f}%, "
            f"review_count={t.get('review_count', '?')}"
            for t in trend
        )
        jumps_text = "\n".join(
            f"- {j['from_month']} → {j['to_month']}: {j['direction']} ({j['delta']*100:+.1f}%p), "
            f"overlaps_low_sample={j['overlaps_low_sample']}"
            for j in analysis["top_jumps"]
        ) or "(급변 시점 없음)"
        extra_jumps_count = len(analysis["significant_jumps"]) - len(analysis["top_jumps"])

        stats_text = f"""
- 관찰 기간: {analysis['total_months']}개월
- 시작 시점: {analysis['start']['year_month']}, risk_probability={analysis['start']['risk_probability']*100:.1f}%
- 종료 시점: {analysis['end']['year_month']}, risk_probability={analysis['end']['risk_probability']*100:.1f}%
- 전체 변화(overall_change): {analysis['overall_change']*100:+.1f}%p ({analysis['overall_direction']})
- 최고 위험 시점(peak): {analysis['peak']['year_month']}, {analysis['peak']['risk_probability']*100:.1f}%
- 최저 위험 시점(trough): {analysis['trough']['year_month']}, {analysis['trough']['risk_probability']*100:.1f}%
- 핵심 급변 시점(top_jumps, 변화폭 상위 {len(analysis['top_jumps'])}개, 기준 {SIGNIFICANT_MONTHLY_CHANGE*100:.0f}%p 이상):
{jumps_text}
- 생략된 추가 급변 건수: {max(extra_jumps_count, 0)}건
- 월평균 변화율(avg_monthly_rate): {analysis['avg_monthly_rate']*100:+.1f}%p/월
- 완만한 지속 추세 여부(is_sustained_trend): {analysis['is_sustained_trend']} (기준: 월평균 {SUSTAINED_TREND_RATE_THRESHOLD*100:.1f}%p 이상, 상품 222개 분포의 상위 10%)
- 방향 일관성(direction_consistency): 총 {analysis['total_steps']}개 구간 중 {analysis['consistent_steps']}개 구간이 전체 방향과 같은 방향으로 움직임 (비율 {analysis['direction_consistency_ratio']*100:.1f}%)
- 표본부족(리뷰 {LOW_REVIEW_COUNT_THRESHOLD}건 미만) 달 수: {analysis['low_sample_month_count']}개월 중 (전체의 {analysis['low_sample_ratio']*100:.1f}%)
"""

    user_prompt = f"""상품 {asin}의 월별 위험도 추이 데이터와, 미리 계산된 통계입니다.

[월별 원본 데이터]
{month_lines}

[미리 계산된 통계 - 이 값을 그대로 사용하세요]
{stats_text}

---
위 데이터를 바탕으로 추이를 설명해주세요."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_trend_query(asin: str) -> dict:
    """
    상품의 위험도 추이를 조회하고 설명을 생성한다.

    Returns:
        {"asin", "answer", "analysis"} 또는 실패 시 {"error"}
    """
    trend_data = get_trend(asin)

    if "error" in trend_data:
        return {"error": trend_data["error"]}

    trend = trend_data.get("trend", [])
    if not trend:
        return {"error": f"상품 {asin}의 추이 데이터가 없습니다."}

    analysis = analyze_trend(trend)
    messages = build_trend_messages(asin, trend, analysis)
    answer = call_llm(messages)

    if answer is None:
        return {"error": "LLM 호출 실패"}

    return {"asin": asin, "answer": answer, "analysis": analysis}


if __name__ == "__main__":
    test_asin = "B0C7CC35KH"

    print(f"상품 {test_asin}의 위험도 추이 조회 중...\n")

    trend_data = get_trend(test_asin)
    trend = trend_data.get("trend", [])

    print("=== 원본 월별 데이터 (전체) ===")
    for t in trend:
        print(f"  {t['year_month']}: risk={t['risk_probability']*100:.1f}%, "
              f"review_count={t.get('review_count', '?')}, "
              f"avg_rating={t.get('avg_rating', '?')}")

    analysis = analyze_trend(trend)
    print("\n=== 계산된 통계(analysis) ===")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    print("\n=== LLM 리포트 ===")
    result = answer_trend_query(test_asin)

    if result.get("error"):
        print("실패:", result["error"])
    else:
        print(f"(총 {result['analysis']['total_months']}개월 데이터)\n")
        print(result["answer"])
