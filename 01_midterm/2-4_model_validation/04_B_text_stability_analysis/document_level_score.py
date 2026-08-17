"""
B-2일차 10단계. 월 단위 문서 기반 급증 예측.

지금까지의 접근은 모두 리뷰 단위로 저평점 여부를 예측하고 집계했다.
그런데 예측 대상은 저평점이 아니라 다음 달 급증이다.
저평점을 타깃으로 하면 결과물이 별점의 복제에 가까워지고,
정형 피처가 이미 담고 있는 정보와 중복된다.

새 접근
  상품×월의 모든 리뷰를 이어붙여 하나의 문서로 만들고,
  그 문서에서 is_low_rating_surge를 직접 예측한다.
  급증을 예고하는 표현을 모델이 스스로 찾게 한다.

TF-IDF가 해결하는 것
  사전 방식은 리뷰마다 0/1 플래그를 찍어 길이를 무시했다.
  10줄 리뷰의 "broken"과 100줄 리뷰의 "broken"이 같은 값이었다.
  TF-IDF는 단어 빈도를 문서 길이로 정규화하므로 전자가 더 큰 값을 갖는다.

누수 방지
  1. 벡터라이저와 분류기는 각 fold의 train 구간 문서로만 학습한다.
  2. train 행의 점수는 K-겹 교차적합으로 만든다.
     자기 자신을 학습한 모델로 점수를 매기면 과적합된 피처가 되어
     LightGBM이 그 값에 과도하게 의존한다.
  3. valid 행의 점수는 train 전체로 학습한 모델로 매긴다.

    python3 document_level_score.py

출력: document_level_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from supervised_text_score import FOLDS, load_reviews, run_fold

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "document_level_metrics.csv"
DOC_CACHE = HERE / "month_documents.parquet"

DOC_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 10,
    "max_features": 80_000,
    "sublinear_tf": True,      # 긴 문서에서 반복 등장하는 단어를 눌러 준다
    "strip_accents": "unicode",
    "lowercase": True,
    "dtype": np.float32,
}
N_OOF_FOLDS = 5
MAX_DOC_CHARS = 20_000


def build_documents(reviews: pd.DataFrame) -> pd.DataFrame:
    """상품×월의 리뷰를 하나의 문서로 잇는다."""
    if DOC_CACHE.exists():
        docs = pd.read_parquet(DOC_CACHE)
        print(f"문서 캐시 사용: {len(docs):,}개")
        return docs

    print("월 단위 문서 생성 중...")
    docs = (reviews.groupby(cfg.KEY_COLUMNS)["text"]
            .apply(lambda s: " ".join(s.astype(str))[:MAX_DOC_CHARS])
            .reset_index()
            .rename(columns={"text": "doc"}))
    docs["doc_chars"] = docs["doc"].str.len()
    docs.to_parquet(DOC_CACHE, index=False)
    print(f"  문서 {len(docs):,}개  길이 중앙값 {docs['doc_chars'].median():,.0f}자  "
          f"최대 {docs['doc_chars'].max():,}자")
    return docs


def score_documents(panel: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """train 구간으로 학습해 전체 행에 급증 예측 점수를 매긴다.

    train 행은 K-겹 교차적합으로 자기 자신을 보지 않은 점수를 얻는다.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score, roc_auc_score

    is_train = panel["year_month"].le(train_end).to_numpy()
    docs = panel["doc"].fillna("").astype(str)
    y = panel[cfg.TARGET_COLUMN].to_numpy()

    vec = TfidfVectorizer(**DOC_PARAMS)
    X_train = vec.fit_transform(docs[is_train])
    X_all = vec.transform(docs)
    y_train = y[is_train]

    def make_clf():
        return LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")

    # --- train 행: K-겹 교차적합 ---
    oof = np.zeros(is_train.sum(), dtype=np.float32)
    skf = StratifiedKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        clf = make_clf().fit(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]

    # --- 나머지 행: train 전체로 학습한 모델 ---
    full = make_clf().fit(X_train, y_train)
    proba = full.predict_proba(X_all)[:, 1].astype(np.float32)
    proba[is_train] = oof

    holdout = ~is_train
    print(f"    어휘 {len(vec.vocabulary_):,}개")
    print(f"    문서 단위 성능  train OOF PR-AUC {average_precision_score(y_train, oof):.4f}"
          f" / ROC {roc_auc_score(y_train, oof):.4f}")
    if holdout.sum():
        print(f"                    이후구간 PR-AUC "
              f"{average_precision_score(y[holdout], proba[holdout]):.4f}"
              f" / ROC {roc_auc_score(y[holdout], proba[holdout]):.4f}")

    out = panel[cfg.KEY_COLUMNS].copy()
    out["doc_surge_t"] = proba

    # 직전 3개월 평균과 변화. 현재 월은 포함하지 않는다.
    out = out.sort_values(cfg.KEY_COLUMNS).reset_index(drop=True)
    prior = (out.groupby("parent_asin", group_keys=False)["doc_surge_t"]
             .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    out["doc_surge_p3"] = prior.reset_index(level=0, drop=True).fillna(0.0)
    out["doc_surge_delta"] = out["doc_surge_t"] - out["doc_surge_p3"]
    return out


SETS = {
    "M1_doc_only": ["doc_surge_t"],
    "M2_doc_p3_delta": ["doc_surge_t", "doc_surge_p3", "doc_surge_delta"],
}


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews()
    docs = build_documents(reviews)

    panel = panel.merge(docs[cfg.KEY_COLUMNS + ["doc"]],
                        on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
    n_missing = int(panel["doc"].isna().sum())
    print(f"\n결합: {len(panel):,}행  문서 없는 행 {n_missing:,}개")

    records = []
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}] 문서 모델 학습 (문서 <= {tend})")
        scores = score_documents(panel, tend)

        df = panel.merge(scores, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
        cols = ["doc_surge_t", "doc_surge_p3", "doc_surge_delta"]
        df[cols] = df[cols].fillna(0.0)

        base = run_fold(df, a_features, tend, vs, ve)
        records.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"    기준선 PR-AUC {base['pr_auc']:.5f}")

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
    w = metrics[metrics["model_name"] != "A_baseline_45"].copy()
    w["gain"] = w.apply(lambda r: r["pr_auc"] - base_by_fold[r["fold"]], axis=1)

    g = w.groupby("model_name")["gain"]
    summary = pd.DataFrame({
        "n_text": w.groupby("model_name")["n_text"].first(),
        "gain_mean": g.mean(), "gain_min": g.min(), "gain_max": g.max(),
        "wins": g.apply(lambda s: int((s > 0).sum())),
    }).round(5).sort_values("gain_mean", ascending=False)

    print("\n" + "=" * 80)
    print("4개 fold 종합")
    print("=" * 80)
    print(summary.to_string())
    print("\n[fold별]")
    print(w.pivot(index="model_name", columns="fold", values="gain").round(5).to_string())
    print("\n  비교 기준")
    print("    X1_core4 (리뷰 단위, 저평점 타깃)  +0.00248  4/4승")
    print("    W0_base3 (집계 3종)                +0.00081  3/4승")
    print("    사전 기반 91개                     -0.00710  1/4승")
    print(f"\n저장: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
