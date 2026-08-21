"""
여러 상품의 리포트를 한 번에 순서대로 생성해서 보여준다.
매번 generate_report.py를 수동으로 고쳐서 실행하는 번거로움을 없애기 위함.

실행 방법:
    python run_test_batch.py
"""

from generate_report import generate_report

# 테스트할 상품 목록: (asin, year_month, 케이스 설명)
TEST_CASES = [
    ("B097F1B375", "2021-10", "1. 위험도 매우 높음"),
    ("B01FF4ASTS", "2021-04", "2. 위험도 매우 낮음"),
    ("B09LM1H5N9", "2020-09", "3. 리뷰 수 많음"),
    ("B08224KN82", "2020-10", "4. 리뷰 수 매우 적음"),
    ("B07RZ4MGXR", "2019-10", "5. 급증 토픽 없음"),
]

for asin, year_month, description in TEST_CASES:
    print("\n" + "=" * 70)
    print(f"[{description}] asin={asin}, year_month={year_month}")
    print("=" * 70)

    result = generate_report(asin, year_month=year_month)

    if result.get("error"):
        print("실패:", result["error"])
    else:
        print(result["report"])
