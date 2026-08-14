"""
eda.py
======
sanity_check.py 통과 후에 실행한다.

포함된 것
--------
1. check_feature_distributions          : review_count / avg_rating / low_rating_ratio 분포
2. check_monthly_label_trend             : year_month별 is_low_rating_surge 비율 시계열
3. check_label_validity_scatter          : past_3m_low_rating_ratio vs next_low_rating_ratio
                                            산점도 + 상관계수로 라벨 타당성 확인, next_* 누수 재점검
4. check_correlation_multicollinearity   : 입력 피처 10개 상관관계 heatmap + VIF
5. check_reliability_features            : Wilson score interval 기반 데이터 신뢰도 피처 추가
6. check_remaining_feature_distributions : verified_purchase_ratio / mean_helpful_vote / text_available_ratio 분포
7. check_label_ratio_by_review_count_bucket : review_count 구간별 라벨 비율
8. check_outliers                        : 이상치 / 극단값 점검

그림은 plt.show()로 화면(또는 노트북 셀)에 띄우는 동시에, eda.py 위치 기준
figures/ 폴더에도 PNG로 저장한다. 

5번(check_reliability_features)만 df에 컬럼을 추가하고 반환하므로
`df = check_reliability_features(df)`처럼 반드시 반환값을 다시 받아야 한다.
나머지 함수는 df를 읽기만 하고 수정하지 않으므로 반환값이 없다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# 그래프의 한글 라벨이 깨지지 않도록 OS별로 설치된 한글 폰트를 찾아서 적용한다.
for _font_name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if _font_name in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

# eda.py 파일 위치 기준 경로라 터미널/노트북 어디서 실행해도 항상 같은 곳에 저장된다.
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def save_fig(fig, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")


try:
    from sanity_check import FEATURE_COLUMNS, LEAK_COLUMNS, TARGET
except ImportError:
    # sanity_check.py와 같은 폴더에 없을 경우를 대비한 fallback
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

DIST_COLUMNS = ["review_count", "avg_rating", "low_rating_ratio"]


# ---------------------------------------------------------------------------
# 1. 정형 피처 분포 확인
# ---------------------------------------------------------------------------
def check_feature_distributions(df: pd.DataFrame) -> None:
    print("\n[1/8] 정형 피처 분포 확인")
    print(df[DIST_COLUMNS].describe().to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, DIST_COLUMNS):
        # review_count는 롱테일이 강하므로 상위 1%는 클리핑해서 같이 확인
        if col == "review_count":
            ax.hist(df[col].clip(upper=df[col].quantile(0.99)), bins=50)
            ax.set_title(f"{col} (99th pct clip)")
        else:
            ax.hist(df[col], bins=50)
            ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    fig.tight_layout()
    save_fig(fig, "01_feature_distributions.png")
    plt.show()

    # 평균 평점은 비슷한데 저평점 비율이 다른 상품 쌍 찾기
    latest = df.sort_values("year_month").groupby("parent_asin").tail(1)
    similar_rating = latest[(latest["avg_rating"] >= 4.3) & (latest["avg_rating"] <= 4.5)]
    if len(similar_rating) >= 2:
        stable = similar_rating.sort_values("low_rating_ratio").iloc[0]
        volatile = similar_rating.sort_values("low_rating_ratio").iloc[-1]
        print(
            "  예시: 평균평점 4.3~4.5 구간에서 low_rating_ratio가 가장 낮은 상품 "
            f"{stable['parent_asin']}({stable['low_rating_ratio']:.1%}) vs "
            f"가장 높은 상품 {volatile['parent_asin']}({volatile['low_rating_ratio']:.1%})"
        )


# ---------------------------------------------------------------------------
# 2. 월별 라벨 비율 시계열
# ---------------------------------------------------------------------------
def check_monthly_label_trend(df: pd.DataFrame) -> None:
    print("\n[2/8] 월별 라벨 비율 시계열")
    monthly = (
        df.groupby("year_month")[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "label_ratio", "count": "n_rows"})
        .sort_index()
    )
    print(monthly.describe().to_string())

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(monthly.index.astype(str), monthly["label_ratio"], color="tab:red", marker="o", markersize=2)
    ax1.set_ylabel("is_low_rating_surge ratio", color="tab:red")
    ax1.tick_params(axis="x", rotation=90)
    ax1.axhline(df[TARGET].mean(), linestyle="--", color="gray", label="전체 평균 9.47%")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.bar(monthly.index.astype(str), monthly["n_rows"], alpha=0.15, color="tab:blue")
    ax2.set_ylabel("행 수", color="tab:blue")

    fig.tight_layout()
    save_fig(fig, "02_monthly_label_trend.png")
    plt.show()


# ---------------------------------------------------------------------------
# 3. 라벨 타당성 산점도 + 데이터 누수 재점검
# ---------------------------------------------------------------------------
def check_label_validity_scatter(df: pd.DataFrame) -> None:
    print("\n[3/8] 라벨 타당성 산점도 + 데이터 누수 재점검")

    x = df["past_3m_low_rating_ratio"]
    y = df["next_low_rating_ratio"]
    corr = x.corr(y)
    print(f"  past_3m_low_rating_ratio vs next_low_rating_ratio Pearson corr = {corr:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(df[TARGET] == 1, "tab:red", "tab:blue")
    ax.scatter(x, y, s=6, alpha=0.3, c=colors)
    ax.set_xlabel("past_3m_low_rating_ratio (t월 기준 과거)")
    ax.set_ylabel("next_low_rating_ratio (t+1월 실제)")
    ax.set_title(f"라벨 타당성 확인 (corr={corr:.3f}, 빨강=surge)")
    fig.tight_layout()
    save_fig(fig, "03_label_validity_scatter.png")
    plt.show()

    # 데이터 누수 재점검: next_* 컬럼이 FEATURE_COLUMNS에 섞여있지 않은지 확인
    leaked = [c for c in LEAK_COLUMNS if c in FEATURE_COLUMNS]
    if leaked:
        print(f"  [경고] FEATURE_COLUMNS에 미래 정보 컬럼이 섞여 있습니다: {leaked}")
    else:
        print("  [확인 완료] next_* 컬럼은 FEATURE_COLUMNS에 포함되어 있지 않습니다.")


# ---------------------------------------------------------------------------
# 4. 입력 피처 상관관계 / 다중공선성 (VIF)
# ---------------------------------------------------------------------------
def check_correlation_multicollinearity(df: pd.DataFrame) -> None:
    print("\n[4/8] 입력 피처 상관관계 / 다중공선성 확인")
    X = df[FEATURE_COLUMNS].dropna()

    corr = X.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90)
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticklabels(FEATURE_COLUMNS)
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, "04_correlation_heatmap.png")
    plt.show()

    high_corr_pairs = [
        (FEATURE_COLUMNS[i], FEATURE_COLUMNS[j], corr.iloc[i, j])
        for i in range(len(FEATURE_COLUMNS))
        for j in range(i + 1, len(FEATURE_COLUMNS))
        if abs(corr.iloc[i, j]) >= 0.8
    ]
    if high_corr_pairs:
        print("  상관계수 0.8 이상 쌍:")
        for a, b, v in high_corr_pairs:
            print(f"    {a} <-> {b} : {v:.3f}")

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "VIF": [
                    variance_inflation_factor(X.values, i) for i in range(len(FEATURE_COLUMNS))
                ],
            }
        ).sort_values("VIF", ascending=False)
        print(vif_df.to_string(index=False))
        high_vif = vif_df[vif_df["VIF"] > 5]
        if not high_vif.empty:
            print(
                "  [주의] VIF 5 초과 피처 존재 (statsmodels 권장 기준): "
                f"{high_vif['feature'].tolist()} -> 베이스라인 모델링 때 피처 통합/제거 고려"
            )
    except ImportError:
        print("  statsmodels가 설치되어 있지 않아 VIF는 건너뜁니다. `pip install statsmodels`로 설치 후 재실행하세요.")


# ---------------------------------------------------------------------------
# 5. 데이터 신뢰도 피처 (Wilson score interval)
# ---------------------------------------------------------------------------
def add_reliability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    review_count가 적은 상품일수록 low_rating_ratio의 표본오차가 커진다.
    Wilson score interval로 이항비율 신뢰구간을 구하고, 그 폭(width)을
    '데이터 신뢰도'가 낮다는 신호로 피처화한다. 폭이 넓을수록 신뢰도가 낮다.

    statsmodels.stats.proportion.proportion_confint(method="wilson")을 사용한다.
    참고: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
    """
    from statsmodels.stats.proportion import proportion_confint

    low, high = proportion_confint(
        count=df["low_rating_count"],
        nobs=df["review_count"].clip(lower=1),  # 0으로 나누기 방지
        alpha=0.05,
        method="wilson",
    )
    df = df.copy()
    df["low_rating_ratio_wilson_low"] = low
    df["low_rating_ratio_wilson_high"] = high
    df["reliability_ci_width"] = high - low  # 넓을수록 신뢰도 낮음
    return df


def check_reliability_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/8] 데이터 신뢰도 피처 (Wilson score interval)")
    df = add_reliability_features(df)

    print(df["reliability_ci_width"].describe().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(df["review_count"], df["reliability_ci_width"], s=4, alpha=0.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("review_count (log scale)")
    axes[0].set_ylabel("reliability_ci_width (Wilson 구간 폭)")
    axes[0].set_title("리뷰 수가 적을수록 신뢰구간이 넓어지는지 확인")

    axes[1].hist(df["reliability_ci_width"], bins=50)
    axes[1].set_xlabel("reliability_ci_width")
    axes[1].set_ylabel("count")
    axes[1].set_title("신뢰도 폭 분포")

    fig.tight_layout()
    save_fig(fig, "05_reliability_features.png")
    plt.show()

    # 위험도는 높지만 신뢰도는 낮은 상품 예시 (surge=1 이면서 ci_width가 넓은 상위 5개)
    risky_but_unreliable = (
        df[df[TARGET] == 1].sort_values("reliability_ci_width", ascending=False).head(5)
    )
    if len(risky_but_unreliable):
        cols = ["parent_asin", "year_month", "review_count", "low_rating_ratio", "reliability_ci_width"]
        print("\n  위험(surge=1)하지만 신뢰구간이 가장 넓은(=근거가 얕은) 상위 5건:")
        print(risky_but_unreliable[cols].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# 6. 나머지 피처 분포 확인 (verified_purchase_ratio, mean_helpful_vote, text_available_ratio)
# ---------------------------------------------------------------------------
def check_remaining_feature_distributions(df: pd.DataFrame) -> None:
    print("\n[6/8] 나머지 피처 분포 확인")
    cols = ["verified_purchase_ratio", "mean_helpful_vote", "text_available_ratio"]
    print(df[cols].describe().to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cols):
        if col == "mean_helpful_vote":
            # helpful vote도 롱테일일 가능성이 높음
            ax.hist(df[col].clip(upper=df[col].quantile(0.99)), bins=50)
            ax.set_title(f"{col} (99th pct clip)")
        else:
            ax.hist(df[col], bins=50)
            ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    fig.tight_layout()
    save_fig(fig, "06_remaining_feature_distributions.png")
    plt.show()

    # text_available_ratio: 리뷰 본문이 아예 없는 상품×월이 얼마나 되는지
    no_text_ratio = (df["text_available_ratio"] == 0).mean()
    low_text_ratio = (df["text_available_ratio"] < 0.5).mean()
    print(
        f"\n  text_available_ratio == 0 (리뷰 본문 전혀 없음): {no_text_ratio:.2%}"
        f"\n  text_available_ratio < 0.5 (본문 있는 리뷰가 절반 미만): {low_text_ratio:.2%}"
    )


# ---------------------------------------------------------------------------
# 7. review_count 구간별 라벨 비율
# ---------------------------------------------------------------------------
def check_label_ratio_by_review_count_bucket(df: pd.DataFrame) -> None:
    print("\n[7/8] review_count 구간별 라벨 비율")
    bins = [0, 10, 30, 100, np.inf]
    labels = ["0-9", "10-29", "30-99", "100+"]
    df = df.copy()
    df["review_count_bucket"] = pd.cut(df["review_count"], bins=bins, labels=labels, right=False)

    bucket_stats = (
        df.groupby("review_count_bucket", observed=True)[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "label_ratio", "count": "n_rows"})
    )
    print(bucket_stats.to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bucket_stats.index.astype(str), bucket_stats["label_ratio"])
    ax.axhline(df[TARGET].mean(), linestyle="--", color="gray", label="전체 평균")
    ax.set_xlabel("review_count 구간")
    ax.set_ylabel("is_low_rating_surge 비율")
    ax.set_title("표본이 적은 상품일수록 라벨 비율이 불안정한지 확인")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "07_label_ratio_by_review_count_bucket.png")
    plt.show()


# ---------------------------------------------------------------------------
# 8. 이상치 / 극단값 점검 (그림 없음, 콘솔 출력만)
# ---------------------------------------------------------------------------
def check_outliers(df: pd.DataFrame) -> None:
    print("\n[8/8] 이상치 / 극단값 점검")

    n_perfect_5 = (df["avg_rating"] == 5.0).sum()
    n_perfect_1 = (df["avg_rating"] == 1.0).sum()
    print(f"  avg_rating == 5.0 (만점 고정): {n_perfect_5:,}건")
    print(f"  avg_rating == 1.0 (최저점 고정): {n_perfect_1:,}건")

    # 리뷰가 많은데 저평점이 하나도 없는, 지나치게 이상적인 케이스
    suspicious = df[(df["review_count"] >= 100) & (df["low_rating_ratio"] == 0)]
    print(f"  review_count>=100 인데 low_rating_ratio==0: {len(suspicious):,}건")
    if len(suspicious):
        print(suspicious[["parent_asin", "year_month", "review_count", "low_rating_ratio"]].head(5).to_string(index=False))

    # verified_purchase_ratio가 0에 가까운 상품 (리뷰 신뢰도 자체가 의심되는 케이스)
    low_verified = (df["verified_purchase_ratio"] < 0.1).mean()
    print(f"\n  verified_purchase_ratio < 0.1 인 행 비율: {low_verified:.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="2-2 EDA 실행 스크립트")
    parser.add_argument(
        "--path",
        default="data/processed/product_month_labeled.parquet",
        help="parquet 파일 경로",
    )
    args = parser.parse_args()

    try:
        df = pd.read_parquet(args.path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {args.path}")
        return 1

    check_feature_distributions(df)
    check_monthly_label_trend(df)
    check_label_validity_scatter(df)
    check_correlation_multicollinearity(df)
    check_reliability_features(df)
    check_remaining_feature_distributions(df)
    check_label_ratio_by_review_count_bucket(df)
    check_outliers(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())