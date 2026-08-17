"""
B-2일차 8단계. 문장 단위 텍스트 위험 점수.

7단계에서 리뷰 전체를 하나의 문서로 보고 TF-IDF + 로지스틱으로
위험 점수를 만들었더니 4개 fold 전부에서 개선됐다(평균 +0.00248).

남은 한계
  리뷰를 통째로 보면 긍정 문단에 섞인 한 문장짜리 위험 신호가 희석된다.
  "Great price, arrived fast. It stopped working after three weeks."
  이 리뷰는 문서 전체로는 긍정이 우세해 점수가 낮게 나온다.

개선
  1. 리뷰를 문장으로 쪼개 각 문장에 점수를 매기고 최댓값을 취한다.
     긍정 리뷰 안의 위험 신호를 살린다.
  2. 문자 n-gram을 함께 사용해 오타와 활용형을 흡수한다.
     stoped/stopped, broke/broken/breaking 등.

누수 방지는 7단계와 같다. 텍스트 모델은 fold별 train 구간 리뷰로만 학습한다.

    python3 sentence_text_score.py

출력: sentence_text_metrics.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from supervised_text_score import FOLDS, load_reviews, run_fold

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "sentence_text_metrics.csv"

# 문장 분리. 마침표/물음표/느낌표/줄바꿈 기준이며 약어는 완벽히 처리하지 않는다.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_SENTENCE_CHARS = 15
MAX_SENTENCES_PER_REVIEW = 5

WORD_PARAMS = {
    "ngram_range": (1, 2), "min_df": 20, "max_features": 60_000,
    "sublinear_tf": True, "strip_accents": "unicode", "lowercase": True,
    "dtype": np.float32,
}
HIGH_RISK_THRESHOLD = 0.5

# 학습에 사용할 문장 수 상한. 메모리 보호용.
MAX_FIT_SENTENCES = 1_000_000
INFER_CHUNK = 500_000


def explode_sentences(reviews: pd.DataFrame) -> pd.DataFrame:
    """리뷰를 문장 단위로 펼친다. 문장의 라벨은 리뷰의 라벨을 물려받는다."""
    texts = reviews["text"].astype(str).str.slice(0, 4000)
    parts = texts.str.split(SENTENCE_SPLIT)

    out = pd.DataFrame({
        "review_id": np.arange(len(reviews)),
        "year_month": reviews["year_month"].to_numpy(),
        "is_low": reviews["is_low"].to_numpy(),
        "sent": parts.to_numpy(),
    }).explode("sent", ignore_index=True)

    out["sent"] = out["sent"].fillna("").astype(str).str.strip()
    out = out[out["sent"].str.len() >= MIN_SENTENCE_CHARS]
    out = out.groupby("review_id", group_keys=False).head(MAX_SENTENCES_PER_REVIEW)

    print(f"  문장 {len(out):,}개 (리뷰당 평균 "
          f"{len(out) / reviews['text'].notna().sum():.2f}개)")
    return out.reset_index(drop=True)


def score_sentences(sents: pd.DataFrame, train_end: str) -> np.ndarray:
    """train 구간 문장으로만 학습하고 전체 문장에 점수를 매긴다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    fit_mask = sents["year_month"].le(train_end).to_numpy()
    text = sents["sent"]

    # 메모리 보호를 위해 학습 표본을 상한선까지만 사용한다.
    # 추론은 전체 문장에 대해 수행하므로 피처 값은 영향받지 않는다.
    fit_idx = np.flatnonzero(fit_mask)
    if len(fit_idx) > MAX_FIT_SENTENCES:
        rng = np.random.default_rng(42)
        fit_idx = np.sort(rng.choice(fit_idx, MAX_FIT_SENTENCES, replace=False))
        fit_mask = np.zeros(len(sents), dtype=bool)
        fit_mask[fit_idx] = True
        print(f"    학습 문장 {MAX_FIT_SENTENCES:,}개로 표본화")

    # 문자 n-gram은 480만 문장에서 메모리를 감당할 수 없어 제외한다.
    # 문장 분해 효과만 단독으로 측정한다.
    word_vec = TfidfVectorizer(**WORD_PARAMS)
    X_fit = word_vec.fit_transform(text[fit_mask])
    y_fit = sents.loc[fit_mask, "is_low"].to_numpy()

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_fit, y_fit)

    # 전체 문장을 한 번에 변환하면 메모리가 급증하므로 청크로 처리한다.
    chunks = []
    for start in range(0, len(text), INFER_CHUNK):
        part = text.iloc[start:start + INFER_CHUNK]
        Xp = word_vec.transform(part)
        chunks.append(clf.predict_proba(Xp)[:, 1].astype(np.float32))
        del Xp
    proba = np.concatenate(chunks)
    del chunks

    holdout = ~fit_mask
    auc_out = (roc_auc_score(sents.loc[holdout, "is_low"], proba[holdout])
               if holdout.sum() > 0 else np.nan)
    print(f"    어휘 {len(word_vec.vocabulary_):,}개   "
          f"문장 단위 ROC-AUC 이후구간 {auc_out:.4f}")
    return proba


def aggregate(reviews: pd.DataFrame, sents: pd.DataFrame,
              proba: np.ndarray) -> pd.DataFrame:
    """문장 점수를 리뷰로 올리고 다시 상품×월로 집계한다."""
    work = pd.DataFrame({
        "review_id": sents["review_id"].to_numpy(),
        "p": proba,
    })
    # 리뷰 단위: 가장 위험한 문장이 그 리뷰를 대표한다.
    per_review = work.groupby("review_id")["p"].agg(["max", "mean"])
    per_review.columns = ["sent_max", "sent_mean"]

    keys = reviews[cfg.KEY_COLUMNS].reset_index(drop=True)
    keys["review_id"] = np.arange(len(reviews))
    merged = keys.merge(per_review, left_on="review_id", right_index=True, how="inner")
    merged["high"] = (merged["sent_max"] >= HIGH_RISK_THRESHOLD).astype("int8")

    return (merged.groupby(cfg.KEY_COLUMNS)
            .agg(snt_risk_mean_t=("sent_max", "mean"),
                 snt_risk_max_t=("sent_max", "max"),
                 snt_risk_p75_t=("sent_max", lambda s: float(np.percentile(s, 75))),
                 snt_risk_high_rate_t=("high", "mean"))
            .reset_index())


CORE = ["snt_risk_mean_t", "snt_risk_max_t", "snt_risk_p75_t", "snt_risk_high_rate_t"]


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews()

    print("\n문장 분해 중...")
    sents = explode_sentences(reviews)

    records = []
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}] 문장 모델 학습 (문장 <= {tend})")
        proba = score_sentences(sents, tend)
        agg = aggregate(reviews, sents, proba)

        df = panel.merge(agg, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
        df[CORE] = df[CORE].fillna(0.0)

        base = run_fold(df, a_features, tend, vs, ve)
        records.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"    기준선 PR-AUC {base['pr_auc']:.5f}")

        feats = list(dict.fromkeys(a_features + CORE))
        m = run_fold(df, feats, tend, vs, ve)
        records.append({"fold": fold, "model_name": "Y1_sentence4",
                        "n_text": len(CORE), **m})
        gain = m["pr_auc"] - base["pr_auc"]
        print(f"    {'+' if gain > 0 else '-'} Y1_sentence4  텍스트 4개  "
              f"{m['pr_auc']:.5f}  ({gain:+.5f})")

    metrics = pd.DataFrame(records)
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    base_by_fold = (metrics[metrics["model_name"] == "A_baseline_45"]
                    .set_index("fold")["pr_auc"])
    work = metrics[metrics["model_name"] != "A_baseline_45"].copy()
    work["gain"] = work.apply(lambda r: r["pr_auc"] - base_by_fold[r["fold"]], axis=1)

    print("\n" + "=" * 78)
    print("4개 fold 종합")
    print("=" * 78)
    print(f"  Y1_sentence4  평균 {work['gain'].mean():+.5f}  "
          f"최소 {work['gain'].min():+.5f}  최대 {work['gain'].max():+.5f}  "
          f"승수 {int((work['gain'] > 0).sum())}/4")
    print("\n  비교 기준")
    print("    X1_core4 (리뷰 단위)  +0.00248  4/4승")
    print("    강도 17개             -0.00012  2/4승")
    print("    기존 텍스트 91개      -0.00710  1/4승")
    print(f"\n저장: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
