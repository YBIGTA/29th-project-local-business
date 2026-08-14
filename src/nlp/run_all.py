"""
2-3 NLP 파이프라인 일괄 실행.

2-1 산출물(reviews_clean, product_month_labeled)이 준비된 상태에서 아래 순서로
최종 산출물까지 재현한다.

    python3 src/nlp/run_all.py           자동 단계 전체
    python3 src/nlp/run_all.py --all     수동 단계까지 포함
    python3 src/nlp/run_all.py --from L1 중간부터 이어서
    python3 src/nlp/run_all.py --only L2 한 단계만
    python3 src/nlp/run_all.py --list    단계 목록만

L4는 사람이 topic_quality_sample.csv의 is_correct를 채운 뒤에 의미가 있으므로
기본 실행에서 제외한다.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

# (코드, 스크립트, 설명, 사람 개입 여부)
STEPS = [
    ("L0", "build_clean_reviews.py",
     "리뷰 정제와 정규화 -> clean_reviews.parquet", False),
    ("L1", "build_nlp_features.py",
     "리뷰 단위 피처와 상품x월 집계 -> product_month_text_features.parquet", False),
    ("L2", "build_surge_summary.py",
     "급증 신호와 근거 리뷰 -> monthly_complaint_surge.csv", False),
    ("L3", "build_quality_sample.py",
     "사전 검수용 표본 -> topic_quality_sample.csv", False),
    ("L4", "build_quality_report.py",
     "검수 결과 집계 -> topic_quality_report.csv", True),
]

CODES = [code for code, _, _, _ in STEPS]


def run(code, script, description):
    path = SRC_DIR / script
    if not path.exists():
        print(f"[{code}] 스크립트를 찾을 수 없다: {path}")
        return False

    print("=" * 70)
    print(f"[{code}] {script}")
    print(f"      {description}")
    print("=" * 70)

    started = time.time()
    result = subprocess.run([sys.executable, str(path)])
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"\n[{code}] 실패 (exit {result.returncode}, {elapsed:.1f}초)")
        return False
    print(f"\n[{code}] 완료 ({elapsed:.1f}초)\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="2-3 NLP 파이프라인 실행")
    parser.add_argument("--all", action="store_true", help="수동 단계까지 포함")
    parser.add_argument("--from", dest="start", choices=CODES, help="이 단계부터")
    parser.add_argument("--only", dest="only", choices=CODES, help="이 단계만")
    parser.add_argument("--list", action="store_true", help="목록만 출력")
    args = parser.parse_args()

    if args.list:
        for code, script, description, manual in STEPS:
            tag = "수동" if manual else "자동"
            print(f"{code} [{tag}] {script:30s} {description}")
        return

    if args.only:
        targets = [step for step in STEPS if step[0] == args.only]
    else:
        targets = STEPS
        if args.start:
            targets = STEPS[CODES.index(args.start):]
        if not args.all:
            targets = [step for step in targets if not step[3]]

    print(f"실행 대상: {', '.join(step[0] for step in targets)}\n")

    started = time.time()
    for code, script, description, _ in targets:
        if not run(code, script, description):
            print("중단한다. 위 오류를 먼저 해결할 것.")
            sys.exit(1)

    print("=" * 70)
    print(f"전체 완료 ({time.time() - started:.1f}초)")
    print("=" * 70)


if __name__ == "__main__":
    main()
