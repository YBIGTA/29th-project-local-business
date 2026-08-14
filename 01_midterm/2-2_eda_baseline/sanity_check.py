from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# data_dictionary.csv / 2-1 인수인계 문서 기준 스키마 정의
# ---------------------------------------------------------------------------
KEY_COLUMNS = ["parent_asin", "year_month"]

FEATURE_COLUMNS = [
    "review_count",
    "avg_rating",
    "low_rating_count",
    "low_rating_ratio",
    "verified_purchase_ratio",
    "mean_helpful_vote",
    "text_available_ratio",
    "past_3m_review_count",
    "past_3m_low_rating_count",
    "past_3m_low_rating_ratio",
]

LEAK_COLUMNS = [
    "next_review_count",
    "next_avg_rating",
    "next_low_rating_count",
    "next_low_rating_ratio",
    "next_vs_past_3m_low_rating_change",
]

TARGET = "is_low_rating_surge"

REQUIRED_COLUMNS = KEY_COLUMNS + FEATURE_COLUMNS + LEAK_COLUMNS + [TARGET]

# 기준값 
EXPECTED_ROW_COUNT = 68_087
EXPECTED_PRODUCT_COUNT = 6_613
EXPECTED_LABEL_RATIO = 0.0947  # 9.47%
RATIO_TOLERANCE = 0.01  # ±1%p까지는 정상 범위로 간주


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class SanityReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.results.append(CheckResult(name, passed, detail))

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def print_summary(self) -> None:
        print("=" * 70)
        print("SANITY CHECK 결과")
        print("=" * 70)
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {r.name}\n       -> {r.detail}")
        print("-" * 70)
        if self.all_passed:
            print("전체 통과. EDA(분포/시계열/산점도/상관관계) 단계로 진행해도 됩니다.")
        else:
            print("FAIL 항목이 있습니다. 2-1 담당자에게 원인을 먼저 확인하세요.")
        print("=" * 70)


def run_sanity_checks(df: pd.DataFrame) -> SanityReport:
    report = SanityReport()

    # 1) 행 수 / 상품 수
    n_rows = len(df)
    n_products = df["parent_asin"].nunique() if "parent_asin" in df.columns else None
    report.add(
        "행 수 일치 (기대값 68,087)",
        n_rows == EXPECTED_ROW_COUNT,
        f"실제 {n_rows:,}행 (기대 {EXPECTED_ROW_COUNT:,}행)",
    )
    if n_products is not None:
        report.add(
            "상품 수 일치 (기대값 6,613)",
            n_products == EXPECTED_PRODUCT_COUNT,
            f"실제 {n_products:,}개 상품 (기대 {EXPECTED_PRODUCT_COUNT:,}개)",
        )

    # 2) 유일 키 검증
    if set(KEY_COLUMNS).issubset(df.columns):
        n_dup = df.duplicated(subset=KEY_COLUMNS).sum()
        report.add(
            "parent_asin + year_month 유일 키",
            n_dup == 0,
            f"중복 행 {n_dup:,}건" if n_dup else "중복 행 없음",
        )
    else:
        report.add("parent_asin + year_month 유일 키", False, "키 컬럼 자체가 없음")

    # 3) 필수 컬럼 존재 확인
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.add(
        "필수 컬럼 전체 존재",
        len(missing_cols) == 0,
        "모든 컬럼 존재" if not missing_cols else f"누락 컬럼: {missing_cols}",
    )

    # 4) next_* 컬럼 존재 확인 (분리는 모델링 코드에서 별도 강제)
    present_leak_cols = [c for c in LEAK_COLUMNS if c in df.columns]
    report.add(
        "next_* 미래 정보 컬럼 존재 확인",
        len(present_leak_cols) == len(LEAK_COLUMNS),
        f"{len(present_leak_cols)}/{len(LEAK_COLUMNS)}개 존재 "
        "(EDA/모델 입력 X에는 절대 포함 금지)",
    )

    # 5) 타겟 클래스 비율
    if TARGET in df.columns:
        label_ratio = df[TARGET].mean()
        within_tol = abs(label_ratio - EXPECTED_LABEL_RATIO) <= RATIO_TOLERANCE
        report.add(
            f"라벨 비율 (기대값 {EXPECTED_LABEL_RATIO:.2%} ±{RATIO_TOLERANCE:.0%}p)",
            within_tol,
            f"실제 {label_ratio:.2%} (양성 {int(df[TARGET].sum()):,}건)",
        )
    else:
        report.add("라벨 비율", False, f"{TARGET} 컬럼이 없음")

    # 6) 결측치 비율이 비정상적으로 높은 컬럼 확인 (50% 초과 시 경고)
    na_ratio = df.isna().mean()
    high_na_cols = na_ratio[na_ratio > 0.5].to_dict()
    report.add(
        "결측치 비율 50% 초과 컬럼 없음",
        len(high_na_cols) == 0,
        "이상 없음" if not high_na_cols else f"주의 필요: { {k: f'{v:.1%}' for k, v in high_na_cols.items()} }",
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="product_month_labeled.parquet sanity check")
    parser.add_argument(
        "--path",
        default="data/processed/product_month_labeled.parquet",
        help="parquet 파일 경로 (기본값: data/processed/product_month_labeled.parquet)",
    )
    args = parser.parse_args()

    try:
        df = pd.read_parquet(args.path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {args.path}")
        return 1

    report = run_sanity_checks(df)
    report.print_summary()
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())