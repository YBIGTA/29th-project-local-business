"""
B-2일차 4단계. 중복도와 라벨 신호의 관계 검증.

3단계에서 텍스트 91개의 중앙 R2가 0.0467로 나왔다.
정형이 텍스트를 담고 있어서 기여가 없다는 설명은 성립하지 않는다.

표를 보면 다른 구조가 보인다.
  정형이 잘 재현하는 텍스트 피처는 라벨 상관도 높았고
  정형이 재현하지 못하는 피처는 라벨 상관이 거의 0이었다.

가설
  텍스트의 예측 가능한 부분은 정형과 겹치고
  정형과 겹치지 않는 부분은 라벨과 무관하다.
  그래서 텍스트를 더해도 얻을 것이 없고 피처 수만 늘어난다.

feature_audit.csv(1일차)와 text_redundancy.csv(3단계)를 결합해
학습 없이 확인한다.

    python3 analyze_signal_vs_redundancy.py

출력: signal_vs_redundancy.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR

HERE = Path(__file__).resolve().parent
AUDIT_PATH = B1_DIR / "feature_audit.csv"
REDUNDANCY_PATH = HERE / "text_redundancy.csv"
OUTPUT_PATH = HERE / "signal_vs_redundancy.csv"


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def main() -> None:
    audit = pd.read_csv(AUDIT_PATH, encoding="utf-8-sig")
    red = pd.read_csv(REDUNDANCY_PATH, encoding="utf-8-sig")

    df = red.merge(
        audit[["column", "corr_label", "nonzero_ratio"]],
        on="column", how="left",
    )
    df["abs_corr_label"] = df["corr_label"].abs()
    df = df.dropna(subset=["abs_corr_label", "r2_valid"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    rho = spearman(df["r2_valid"], df["abs_corr_label"])

    print("=" * 84)
    print("가설 검증: 정형이 재현할 수 있는 텍스트일수록 라벨 신호가 강한가")
    print("=" * 84)
    print(f"  대상 피처            {len(df)}개")
    print(f"  R2 x |라벨상관| 순위상관   {rho:+.4f}")
    if rho > 0.5:
        print("  강한 양의 관계. 가설과 일치한다.")
    elif rho > 0.25:
        print("  뚜렷한 양의 관계. 가설과 대체로 일치한다.")
    else:
        print("  관계가 약하다. 다른 설명이 필요하다.")

    # ---------------- 중복도 구간별 라벨 신호 ----------------
    bins = [-np.inf, 0.02, 0.05, 0.10, 0.20, np.inf]
    labels = ["R2<0.02", "0.02~0.05", "0.05~0.10", "0.10~0.20", "R2>=0.20"]
    df["r2_bin"] = pd.cut(df["r2_valid"], bins=bins, labels=labels)

    table = (df.groupby("r2_bin", observed=True)
             .agg(n=("column", "count"),
                  median_abs_corr=("abs_corr_label", "median"),
                  max_abs_corr=("abs_corr_label", "max"),
                  median_nonzero=("nonzero_ratio", "median"))
             .round(4))

    print("\n" + "=" * 84)
    print("정형 재현 가능성 구간별 라벨 신호 크기")
    print("=" * 84)
    print(table.to_string())
    print("\n  왼쪽으로 갈수록 정형이 못 담는 정보이고, 라벨 신호도 함께 작아지면")
    print("  텍스트의 독립적인 부분에는 예측에 쓸 신호가 없다는 뜻이다.")

    # ---------------- 사분면 ----------------
    r2_med = df["r2_valid"].median()
    sig_med = df["abs_corr_label"].median()

    def quad(r):
        hi_r2 = r["r2_valid"] >= r2_med
        hi_sig = r["abs_corr_label"] >= sig_med
        if hi_r2 and hi_sig:
            return "1_신호O_정형이재현O"
        if not hi_r2 and hi_sig:
            return "2_신호O_정형이재현X"
        if hi_r2 and not hi_sig:
            return "3_신호X_정형이재현O"
        return "4_신호X_정형이재현X"

    df["quadrant"] = df.apply(quad, axis=1)

    print("\n" + "=" * 84)
    print(f"사분면 분류 (R2 중앙값 {r2_med:.4f}, |라벨상관| 중앙값 {sig_med:.4f})")
    print("=" * 84)
    counts = df["quadrant"].value_counts().sort_index()
    for name, n in counts.items():
        print(f"  {name:<24} {n:>3}개 ({n / len(df):>5.1%})")

    print("\n  2번 사분면이 텍스트가 단독으로 기여할 수 있는 유일한 영역이다.")
    q2 = df[df["quadrant"] == "2_신호O_정형이재현X"].sort_values(
        "abs_corr_label", ascending=False)
    if len(q2):
        print(f"\n  2번 사분면 {len(q2)}개 (라벨 신호 순)")
        print(q2[["column", "family", "variant", "r2_valid",
                  "abs_corr_label", "nonzero_ratio"]].head(15).to_string(index=False))

    print("\n" + "=" * 84)
    print("family별 위치")
    print("=" * 84)
    fam = (df.groupby("family")
           .agg(n=("column", "count"),
                median_r2=("r2_valid", "median"),
                median_signal=("abs_corr_label", "median"))
           .round(4)
           .sort_values("median_signal", ascending=False))
    print(fam.to_string())
    print("\n  T_cooc은 정형과 가장 직교하면서 라벨 신호도 가장 약한 그룹이다.")
    print("  독립적인 정보라는 것과 예측에 쓸모 있다는 것은 다르다.")

    print(f"\n저장: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
