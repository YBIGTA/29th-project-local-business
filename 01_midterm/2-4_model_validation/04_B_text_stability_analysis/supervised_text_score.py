"""
B-2일차 7단계. 지도학습 기반 텍스트 위험 점수.

기존 접근의 한계
  - 불만 사전을 사람이 정의해 데이터가 단어를 고르지 못했다
  - "not broken", "works without issue" 같은 부정문을 놓친다
  - 91개 피처가 정형 45개에 더해지며 분산만 키웠다

대안
  리뷰 텍스트로 저평점 여부를 예측하는 모델을 학습하고,
  그 예측 확률을 상품×월로 집계해 소수의 피처로 압축한다.
  단어 선택은 데이터가 하고, bigram이 부정 표현을 일부 흡수한다.

누수 방지
  텍스트 모델은 각 fold의 train 구간 리뷰로만 학습한다.
  fold별로 다시 학습하므로 검증 구간의 어휘를 미리 보지 않는다.
  라벨은 리뷰의 별점(1~2점)이며 다음 달 급증 라벨은 쓰지 않는다.

    python3 supervised_text_score.py

출력: supervised_text_metrics.csv
      supervised_text_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402
from evaluation import compute_scale_pos_weight, evaluate  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
METRICS_PATH = HERE / "supervised_text_metrics.csv"
FEATURES_PATH = HERE / "supervised_text_features.parquet"

FOLDS = [
    ("fold_1_2020_01_08", "2019-12", "2020-01", "2020-08"),
    ("fold_2_2020_09_2021_04", "2020-08", "2020-09", "2021-04"),
    ("fold_3_2021_05_12", "2021-04", "2021-05", "2021-12"),
    ("fold_4_2022_01_08", "2021-12", "2022-01", "2022-08"),
]

# bigram까지 사용해 "not broken", "stopped working" 같은 표현을 한 단위로 본다.
VECTORIZER_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 10,
    "max_features": 120_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}
HIGH_RISK_THRESHOLD = 0.5


def find_reviews() -> Path:
    for name in ["reviews_clean.parquet", "clean_reviews.parquet"]:
        p = REPO_ROOT / "data" / "interim" / name
        if p.exists():
            return p
    raise FileNotFoundError("data/interim 아래에서 정제 리뷰 parquet을 찾지 못했다")


def load_reviews() -> pd.DataFrame:
    path = find_reviews()
    head = pd.read_parquet(path).head(0)
    cols = set(head.columns)

    text_col = next((c for c in ["text_norm", "review_text", "text"] if c in cols), None)
    rating_col = next((c for c in ["rating", "star_rating", "overall"] if c in cols), None)
    if text_col is None or rating_col is None:
        raise KeyError(f"텍스트/별점 컬럼을 찾지 못했다. 보유 컬럼: {sorted(cols)}")

    use = cfg.KEY_COLUMNS + [text_col, rating_col]
    if "auto_title_flag" in cols:
        use.append("auto_title_flag")

    df = pd.read_parquet(path, columns=use)
    df = df.rename(columns={text_col: "text", rating_col: "rating"})
    df["year_month"] = df["year_month"].astype(str)

    # 아마존 자동 생성 제목은 별점의 문자열 사본이라 제외한다.
    if "auto_title_flag" in df.columns:
        df = df[df["auto_title_flag"].eq(0)]
    df = df[df["text"].notna() & df["text"].astype(str).str.strip().ne("")]
    df["is_low"] = (df["rating"] <= 2).astype("int8")

    print(f"리뷰 {len(df):,}건  저평점 비율 {df['is_low'].mean():.2%}  "
          f"기간 {df['year_month'].min()}~{df['year_month'].max()}")
    return df.reset_index(drop=True)


def score_reviews(reviews: pd.DataFrame, train_end: str) -> np.ndarray:
    """train 구간 리뷰로만 학습하고 전체 리뷰에 위험 점수를 매긴다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    fit_mask = reviews["year_month"].le(train_end).to_numpy()
    texts = reviews["text"].astype(str)

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_fit = vec.fit_transform(texts[fit_mask])
    y_fit = reviews.loc[fit_mask, "is_low"].to_numpy()

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(X_fit, y_fit)

    X_all = vec.transform(texts)
    proba = clf.predict_proba(X_all)[:, 1]

    holdout = ~fit_mask
    auc_in = roc_auc_score(y_fit, proba[fit_mask])
    auc_out = (roc_auc_score(reviews.loc[holdout, "is_low"], proba[holdout])
               if holdout.sum() > 0 else np.nan)
    print(f"    어휘 {len(vec.vocabulary_):,}개  "
          f"리뷰 단위 ROC-AUC  학습 {auc_in:.4f} / 이후구간 {auc_out:.4f}")
    return proba


def aggregate(reviews: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """리뷰 점수를 상품×월로 집계한다. 피처를 소수로 유지한다."""
    work = reviews[cfg.KEY_COLUMNS].copy()
    work["p"] = proba
    work["high"] = (proba >= HIGH_RISK_THRESHOLD).astype("int8")

    out = (work.groupby(cfg.KEY_COLUMNS)
           .agg(txt_risk_mean_t=("p", "mean"),
                txt_risk_max_t=("p", "max"),
                txt_risk_p75_t=("p", lambda s: float(np.percentile(s, 75))),
                txt_risk_high_rate_t=("high", "mean"))
           .reset_index())

    # 직전 3개월 평균과 그 차이. 현재 월은 포함하지 않는다.
    out = out.sort_values(cfg.KEY_COLUMNS).reset_index(drop=True)
    prior = (out.groupby("parent_asin", group_keys=False)["txt_risk_mean_t"]
             .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    out["txt_risk_mean_p3"] = prior.reset_index(level=0, drop=True).fillna(0.0)
    out["txt_risk_mean_delta"] = out["txt_risk_mean_t"] - out["txt_risk_mean_p3"]
    return out


def run_fold(df, features, train_end, valid_start, valid_end):
    import lightgbm as lgb

    train = df[df["year_month"] <= train_end]
    valid = df[(df["year_month"] >= valid_start) & (df["year_month"] <= valid_end)]
    params = dict(cfg.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(train[cfg.TARGET_COLUMN])
    params.setdefault("verbose", -1)
    model = lgb.LGBMClassifier(**params).fit(train[features], train[cfg.TARGET_COLUMN])
    return evaluate(valid, model.predict_proba(valid[features])[:, 1])


CORE = ["txt_risk_mean_t", "txt_risk_max_t", "txt_risk_p75_t", "txt_risk_high_rate_t"]
WITH_CHANGE = CORE + ["txt_risk_mean_p3", "txt_risk_mean_delta"]


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews()

    records, keep_features = [], None
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}] 텍스트 모델 학습 (리뷰 <= {tend})")
        proba = score_reviews(reviews, tend)
        agg = aggregate(reviews, proba)

        df = panel.merge(agg, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
        df[WITH_CHANGE] = df[WITH_CHANGE].fillna(0.0)
        if fold == FOLDS[-1][0]:
            keep_features = agg

        base = run_fold(df, a_features, tend, vs, ve)
        records.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"    기준선 PR-AUC {base['pr_auc']:.5f}")

        for name, feats_text in [("X1_core4", CORE), ("X2_core_plus_change", WITH_CHANGE)]:
            feats = list(dict.fromkeys(a_features + feats_text))
            m = run_fold(df, feats, tend, vs, ve)
            records.append({"fold": fold, "model_name": name,
                            "n_text": len(feats_text), **m})
            gain = m["pr_auc"] - base["pr_auc"]
            print(f"    {'+' if gain > 0 else '-'} {name:<22} 텍스트 {len(feats_text)}개  "
                  f"{m['pr_auc']:.5f}  ({gain:+.5f})")

    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    if keep_features is not None:
        keep_features.to_parquet(FEATURES_PATH, index=False)

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

    print("\n" + "=" * 82)
    print("4개 fold 종합")
    print("=" * 82)
    print(summary.to_string())
    print("\n  비교 기준")
    print("    기존 텍스트 91개   -0.00710  승수 1/4")
    print("    강도 17개          -0.00012  승수 2/4")
    print(f"\n저장: {METRICS_PATH.name}, {FEATURES_PATH.name}")


if __name__ == "__main__":
    main()
