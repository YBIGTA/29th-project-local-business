"""
불만 표현 매칭 보정 규칙.

사전의 표현이 리뷰에 등장했다고 해서 항상 그 불만이 있는 것은 아니다.
수동 검수 300건에서 확인된 두 가지 오탐 유형을 여기서 걸러낸다.

유형 1. 부정 문맥
  "no leaks", "never leaked"는 누수가 없었다는 뜻인데 leak이 매칭된다.
  다만 부정어를 일률적으로 무시하면 안 된다. "not work", "not fit"처럼
  부정될 때 비로소 불만이 되는 표현이 있기 때문이다.

    leak, damage, crack 같은 문제 명사   -> 부정되면 불만 아님 (세지 않는다)
    work, fit, last 같은 기능 동사        -> 부정되면 불만 (세야 한다)

  그래서 "부정되면 취소되는 표현"만 NEGATABLE에 나열한다.

유형 2. 다의어
  window는 반품 기간과 창문 두 뜻으로 쓰인다. 검수에서 return_blocked의
  고평점 오탐이 대부분 이 경우였다.
  ("돈을 창밖에 버린다", "모델번호 입력 팝업창")
  단어를 사전에서 빼면 진짜 반품 차단 신호까지 잃으므로, 근처에 반품
  문맥어가 있을 때만 인정한다.

리뷰 본문은 정규화 과정에서 문장부호가 제거되므로 문장 단위 구분이
불가능하다. 따라서 문장이 아니라 토큰 거리(window size)로 범위를 정한다.
"""

# ---------------------------------------------------------------
# 유형 1. 부정 문맥
# ---------------------------------------------------------------
NEGATION_CUES = {
    "no", "not", "never", "without", "zero", "nor", "none", "hardly",
    "barely", "havent", "hasnt", "didnt", "doesnt", "dont", "wasnt",
    "werent", "isnt", "arent", "wont", "cant", "couldnt",
}

# 앞에 부정어가 오면 불만으로 세지 않는 표현.
# 문제 상태를 가리키는 명사와 그 활용형만 넣는다.
NEGATABLE = {
    "leak", "leaks", "leaked", "leaking", "leakage",
    "damage", "damaged", "damages",
    "crack", "cracks", "cracked", "cracking",
    "break", "breaks", "broke", "broken", "breaking",
    "rust", "rusted", "rusty",
    "smell", "smells", "smelled", "odor",
    "problem", "problems", "issue", "issues",
    "defect", "defects", "defective",
    "dent", "dents", "dented",
    "spill", "spills", "spilled", "spilling",
    "clog", "clogs", "clogged", "clogging",
    "noise", "noisy",
    "complaint", "complaints",
    "trouble", "troubles",
    "return", "returns", "returned", "returning",
    "refund", "refunds", "refunded",
    "aftertaste", "residue", "sediment",
}

# 부정어와 대상 표현 사이에 허용할 최대 토큰 수.
# "no leaks", "no water leaks", "never had any leaks"까지 잡되
# 문장을 건너뛰지 않을 정도로 좁게 둔다.
NEGATION_SPAN = 3


def is_negated(tokens, position):
    """tokens[position]이 앞쪽 부정어의 영향을 받는지 판정한다."""
    if tokens[position] not in NEGATABLE:
        return False
    start = max(0, position - NEGATION_SPAN)
    return any(token in NEGATION_CUES for token in tokens[start:position])


# ---------------------------------------------------------------
# 유형 2. 다의어
# ---------------------------------------------------------------
# 이 표현들은 근처에 문맥어가 함께 있을 때만 불만으로 센다.
RETURN_CONTEXT = {
    "return", "returns", "returned", "returning", "returnable",
    "refund", "refunded", "exchange", "exchanged", "restocking",
    "warranty", "policy", "sent", "send", "ship", "shipped",
}

AMBIGUOUS_TERMS = {
    "window": RETURN_CONTEXT,
    "windows": RETURN_CONTEXT,
    "too late": RETURN_CONTEXT,
    "stuck with": RETURN_CONTEXT,
    "eligible": RETURN_CONTEXT,
    "policy": RETURN_CONTEXT | {"seller", "vendor", "company", "amazon"},
    "expired": RETURN_CONTEXT | {"filter", "warranty"},
}

# 문맥어를 찾을 범위. 부정어보다 넓게 둔다.
CONTEXT_SPAN = 12


def has_context(tokens, position, length, required):
    """tokens[position:position+length] 주변에 문맥어가 있는지 확인한다."""
    start = max(0, position - CONTEXT_SPAN)
    end = min(len(tokens), position + length + CONTEXT_SPAN)
    scope = tokens[start:position] + tokens[position + length:end]
    return any(token in required for token in scope)


# ---------------------------------------------------------------
# 매칭
# ---------------------------------------------------------------
def count_term(tokens, term):
    """
    정규화된 토큰 목록에서 term이 불만으로 등장한 횟수를 센다.

    tokens: text_norm.split() 결과
    term:   사전 표현. 공백이 있으면 다중어로 처리한다.

    위 두 규칙에 걸린 등장은 세지 않는다.
    """
    parts = term.split()
    length = len(parts)
    required = AMBIGUOUS_TERMS.get(term)

    count = 0
    for i in range(len(tokens) - length + 1):
        if tokens[i:i + length] != parts:
            continue
        if length == 1 and is_negated(tokens, i):
            continue
        if required and not has_context(tokens, i, length, required):
            continue
        count += 1
    return count


def count_terms(tokens, terms):
    """사전 표현 목록 전체의 등장 횟수 합계를 돌려준다."""
    return sum(count_term(tokens, term) for term in terms)


# ---------------------------------------------------------------
# 규칙이 의도대로 동작하는지 확인하는 자체 점검
# ---------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # (문장, 표현, 기대값, 설명)
        ("no leaks and water tastes fine", "leaks", 0, "부정된 문제 명사"),
        ("it leaks everywhere", "leaks", 1, "실제 누수"),
        ("never had any leaks", "leaks", 0, "부정어와 거리 3"),
        ("the filter does not work at all", "not work", 1, "부정이 불만인 표현"),
        ("this does not fit my dryer", "not fit", 1, "부정이 불만인 표현"),
        ("i might as well have thrown the cash out the window",
         "window", 0, "창문 비유"),
        ("it was outside of the return window", "window", 1, "반품 기간"),
        ("you will get a window that asks for your model number",
         "window", 0, "팝업창"),
        ("stuck with an expensive item i cannot return",
         "stuck with", 1, "반품 문맥 있음"),
        ("until i get my new stove top i am stuck with the old one",
         "stuck with", 0, "반품 문맥 없음"),
        ("glad i found it before it was too late to cause a fire",
         "too late", 0, "화재 문맥"),
        ("it was too late to return it", "too late", 1, "반품 문맥"),
    ]

    failed = 0
    for text, term, expected, note in cases:
        actual = count_term(text.split(), term)
        status = "OK " if actual == expected else "FAIL"
        if actual != expected:
            failed += 1
        print(f"{status} {term:12s} 기대 {expected} 실제 {actual}  {note}")
        if actual != expected:
            print(f"       {text}")

    print(f"\n{len(cases)}건 중 {len(cases) - failed}건 통과")
