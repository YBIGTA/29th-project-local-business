"""
B-2일차 9단계. 집계 방식 개선.

7단계 X1_core4가 4개 fold 전부에서 개선됐다(평균 +0.00248).
집계는 mean / max / p75 / high_rate 네 가지였다.

남은 문제
  비율은 표본 크기를 무시한다.
  리뷰 10개 중 1개 불만과 100개 중 10개 불만이 모두 0.1이 된다.
  후자가 훨씬 확실한 신호인데 같은 값을 갖는다.

개선 1. Wilson score interval 하한
  비율과 표본 크기를 함께 반영한다.
    10개 중 1개  -> 비율 0.100, 하한 0.018
    100개 중 10개 -> 비율 0.100, 하한 0.055
  2-2가 reliability_ci_width로 이미 쓴 도구이며, 축소추정과 달리
  비율을 리뷰 수로 대체하지 않고 결합한다.

개선 2. 집계 함수 다양화
  top-k 평균  max는 이상치 하나에 흔들리고 mean은 희석된다. 그 중간
  위험 리뷰 수  비율이 아닌 절대 개수. log 변환
  표준편차     의견이 갈리는 상품과 일관되게 나쁜 상품을 구분

개선 3. 도움 투표 가중 평균
  다른 구매자가 유용하다고 인정한 리뷰에 더 큰 가중을 준다.

누수 방지는 7단계와 같다. 텍스트 모델은 fold별 train 구간으로만 학습한다.

    python3 weighted_text_score.py

출력: weighted_text_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from supervised_text_score import (
    FOLDS, VECTORIZER_PARAMS, find_reviews, run_fold,
)

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "weighted_text_metrics.csv"

HIGH_RISK_THRESHOLD = 0.5
TOP_K = 3
WILSON_Z = 1.96


def load_reviews_with_meta() -> pd.DataFrame:
    """도움 투표와 인증구매까지 함께 읽는다. 가중 집계에 사용한다."""
    path = find_reviews()
    cols = set(pd.read_parquet(path).head(0).columns)

    text_col = next((c for c in ["text_norm", "review_text", "text"] if c in cols), None)
    rating_col = next((c for c in ["rating", "star_rating", "overall"] if c in cols), None)
    if text_col is None or rating_col is None:
        raise KeyError(f"텍스트/별점 컬럼을 찾지 못했다: {sorted(cols)}")

    optional = [c for c in ["auto_title_flag", "helpful_vote", "verified_purchase"]
                if c in cols]
    df = pd.read_parquet(path, columns=cfg.KEY_COLUMNS + [text_col, rating_col] + optional)
    df = df.rename(columns={text_col: "text", rating_col: "rating"})
    df["year_month"] = df["year_month"].astype(str)

    if "auto_title_flag" in df.columns:
        df = df[df["auto_title_flag"].eq(0)]
    df = df[df["text"].notna() & df["text"].astype(str).str.strip().ne("")]
    df["is_low"] = (df["rating"] <= 2).astype("int8")

    # 도움 투표는 롱테일이라 log로 눌러서 가중치로 쓴다.
    if "helpful_vote" in df.columns:
        df["w"] = 1.0 + np.log1p(df["helpful_vote"].fillna(0).clip(lower=0))
    else:
        df["w"] = 1.0
        print("  helpful_vote 없음. 가중치는 균등으로 둔다")

    print(f"리뷰 {len(df):,}건  저평점 {df['is_low'].mean():.2%}  "
          f"가중치 중앙값 {df['w'].median():.3f}")
    return df.reset_index(drop=True)


def wilson_lower(successes: np.ndarray, n: np.ndarray, z: float = WILSON_Z) -> np.ndarray:
    """Wilson score interval의 하한.

    표본이 작으면 하한이 크게 내려가고, 커질수록 관측 비율에 수렴한다.
    비율과 표본 크기를 대체가 아니라 결합해서 표현한다.
    """
    n = np.maximum(n.astype(float), 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return np.clip((center - margin) / denom, 0.0, 1.0)


def score_reviews(reviews: pd.DataFrame, train_end: str) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    fit_mask = reviews["year_month"].le(train_end).to_numpy()
    texts = reviews["text"].astype(str)

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_fit = vec.fit_transform(texts[fit_mask])
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_fit, reviews.loc[fit_mask, "is_low"].to_numpy())

    proba = clf.predict_proba(vec.transform(texts))[:, 1]
    holdout = ~fit_mask
    if holdout.sum():
        auc = roc_auc_score(reviews.loc[holdout, "is_low"], proba[holdout])
        print(f"    리뷰 단위 ROC-AUC 이후구간 {auc:.4f}")
    return proba


def aggregate(reviews: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    work = reviews[cfg.KEY_COLUMNS].copy()
    work["p"] = proba
    work["w"] = reviews["w"].to_numpy()
    work["high"] = (proba >= HIGH_RISK_THRESHOLD).astype("int8")
    work["pw"] = work["p"] * work["w"]

    def topk_mean(s: pd.Series) -> float:
        return float(np.sort(s.to_numpy())[-TOP_K:].mean())

    g = work.groupby(cfg.KEY_COLUMNS)
    out = g.agg(
        n_reviews=("p", "size"),
        txt_mean=("p", "mean"),
        txt_max=("p", "max"),
        txt_p75=("p", lambda s: float(np.percentile(s, 75))),
        txt_std=("p", "std"),
        txt_topk=("p", topk_mean),
        n_high=("high", "sum"),
        sum_pw=("pw", "sum"),
        sum_w=("w", "sum"),
    ).reset_index()

    out["txt_std"] = out["txt_std"].fillna(0.0)

    # 표본 크기를 반영한 위험 리뷰 비율의 보수적 추정.
    out["txt_wilson"] = wilson_lower(out["n_high"].to_numpy(), out["n_reviews"].to_numpy())
    # 비율이 아닌 절대 개수. 100개 중 10개와 10개 중 1개를 구분한다.
    out["txt_high_log"] = np.log1p(out["n_high"].to_numpy())
    # 도움 투표 가중 평균.
    out["txt_wmean"] = out["sum_pw"] / out["sum_w"].replace(0, np.nan)
    out["txt_wmean"] = out["txt_wmean"].fillna(out["txt_mean"])

    return out.drop(columns=["n_reviews", "n_high", "sum_pw", "sum_w"])


BASE4 = ["txt_mean", "txt_max", "txt_p75"]
SETS = {
    "W0_base3": BASE4,
    "W1_plus_wilson": BASE4 + ["txt_wilson"],
    "W2_plus_count": BASE4 + ["txt_wilson", "txt_high_log"],
    "W3_plus_topk_std": BASE4 + ["txt_wilson", "txt_high_log", "txt_topk", "txt_std"],
    "W4_plus_weighted": BASE4 + ["txt_wilson", "txt_high_log", "txt_topk",
                                 "txt_std", "txt_wmean"],
}


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews_with_meta()

    records = []
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}] 텍스트 모델 학습 (리뷰 <= {tend})")
        agg = aggregate(reviews, score_reviews(reviews, tend))
        all_cols = [c for c in agg.columns if c.startswith("txt_")]

        df = panel.merge(agg, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
        df[all_cols] = df[all_cols].fillna(0.0)

        base = run_fold(df, a_features, tend, vs, ve)
        records.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"    기준선 {base['pr_auc']:.5f}")

        for name, feats_text in SETS.items():
            feats = list(dict.fromkeys(a_features + feats_text))
            m = run_fold(df, feats, tend, vs, ve)
            records.append({"fold": fold, "model_name": name,
                            "n_text": len(feats_text), **m})
            gain = m["pr_auc"] - base["pr_auc"]
            print(f"    {'+' if gain > 0 else '-'} {name:<20} {len(feats_text)}개  "
                  f"{m['pr_auc']:.5f}  ({gain:+.5f})")

    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    base_by_fold = (metrics[metrics["model_name"] == "A_baseline_45"]
                    .set_index("fold")["pr_auc"])
    work = metrics[metrics["model_name"] != "A_baseline_45"].copy()
    work["gain"] = work.apply(lambda r: r["pr_auc"] - base_by_fold[r["fold"]], axis=1)

    summary = (work.groupby("model_name")
               .agg(n_text=("n_text", "first"), gain_mean=("gain", "mean"),
                    gain_std=("gain", "std"), gain_min=("gain", "min"),
                    gain_max=("gain", "max"),
                    wins=("gain", lambda s: int((s > 0).sum())))
               .round(5).sort_values("gain_mean", ascending=False))

    print("\n" + "=" * 80)
    print("4개 fold 종합")
    print("=" * 80)
    print(summary.to_string())
    print("\n  비교 기준")
    print("    X1_core4 (mean/max/p75/high_rate)  +0.00248  4/4승")
    print("    강도 17개                          -0.00012  2/4승")
    print("    기존 사전 기반 91개                -0.00710  1/4승")
    print(f"\n저장: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
