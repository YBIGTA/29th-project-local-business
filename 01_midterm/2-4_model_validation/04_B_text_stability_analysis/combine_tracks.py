"""
B-2일차 11단계. 문서 단위와 리뷰 단위 결합.

두 트랙이 각각 4개 fold에서 개선을 보였다.
  X1_core4   리뷰 단위, 저평점 타깃    +0.00248  4/4
  M2_doc     월 문서 단위, 급증 타깃   +0.00128  4/4

두 접근은 타깃과 단위가 모두 다르다.
  리뷰 단위는 소수의 심각한 개별 리뷰를 잡는다.
  월 문서 단위는 한 달 전체의 분위기를 잡는다.
서로 다른 정보라면 함께 넣을 때 더 큰 개선이 나온다.

추가 시도
  직전 3개월 리뷰를 하나의 문서로 이어 붙여 점수를 낸다.
  현재 doc_surge_p3는 월별 점수의 이동평균이라 다른 값이다.
  누적된 불만 패턴을 직접 학습시킨다.

    python3 combine_tracks.py

출력: combined_text_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from supervised_text_score import (
    FOLDS, VECTORIZER_PARAMS, load_reviews, run_fold,
)
from document_level_score import (
    DOC_PARAMS, N_OOF_FOLDS, build_documents, score_documents,
)

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "combined_text_metrics.csv"

HIGH_RISK_THRESHOLD = 0.5
DOC_COLS = ["doc_surge_t", "doc_surge_p3", "doc_surge_delta"]
REVIEW_COLS = ["txt_risk_mean_t", "txt_risk_max_t",
               "txt_risk_p75_t", "txt_risk_high_rate_t"]
WINDOW_COLS = ["docw_surge_p3"]


def review_track(reviews: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """리뷰 단위 저평점 예측 점수를 상품×월로 집계한다. X1_core4와 동일."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    fit_mask = reviews["year_month"].le(train_end).to_numpy()
    texts = reviews["text"].astype(str)

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_fit = vec.fit_transform(texts[fit_mask])
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_fit, reviews.loc[fit_mask, "is_low"].to_numpy())
    proba = clf.predict_proba(vec.transform(texts))[:, 1]

    work = reviews[cfg.KEY_COLUMNS].copy()
    work["p"] = proba
    work["high"] = (proba >= HIGH_RISK_THRESHOLD).astype("int8")
    return (work.groupby(cfg.KEY_COLUMNS)
            .agg(txt_risk_mean_t=("p", "mean"),
                 txt_risk_max_t=("p", "max"),
                 txt_risk_p75_t=("p", lambda s: float(np.percentile(s, 75))),
                 txt_risk_high_rate_t=("high", "mean"))
            .reset_index())


def window_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """직전 3개월 리뷰를 하나의 문서로 잇는다. 현재 월은 포함하지 않는다."""
    d = docs.sort_values(cfg.KEY_COLUMNS).reset_index(drop=True)

    def join_prior(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return pd.Series(
            [" ".join(shifted.iloc[max(0, i - 2):i + 1].dropna().astype(str))[:30_000]
             for i in range(len(s))],
            index=s.index,
        )

    d["docw"] = (d.groupby("parent_asin", group_keys=False)["doc"]
                 .apply(join_prior))
    return d[cfg.KEY_COLUMNS + ["docw"]]


def score_window(panel: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """직전 3개월 문서에서 급증을 직접 예측한다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score

    is_train = panel["year_month"].le(train_end).to_numpy()
    text = panel["docw"].fillna("").astype(str)
    y = panel[cfg.TARGET_COLUMN].to_numpy()

    vec = TfidfVectorizer(**DOC_PARAMS)
    X_tr = vec.fit_transform(text[is_train])
    X_all = vec.transform(text)
    y_tr = y[is_train]

    def make_clf():
        return LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")

    oof = np.zeros(is_train.sum(), dtype=np.float32)
    for tr, va in StratifiedKFold(N_OOF_FOLDS, shuffle=True,
                                  random_state=42).split(X_tr, y_tr):
        oof[va] = make_clf().fit(X_tr[tr], y_tr[tr]).predict_proba(X_tr[va])[:, 1]

    proba = make_clf().fit(X_tr, y_tr).predict_proba(X_all)[:, 1].astype(np.float32)
    proba[is_train] = oof
    print(f"    3개월 창 문서  train OOF PR-AUC "
          f"{average_precision_score(y_tr, oof):.4f}")

    out = panel[cfg.KEY_COLUMNS].copy()
    out["docw_surge_p3"] = proba
    return out


SETS = {
    "R_review_only": REVIEW_COLS,
    "D_doc_only": DOC_COLS,
    "W_window_only": WINDOW_COLS,
    "RD_review_doc": REVIEW_COLS + DOC_COLS,
    "RDW_all": REVIEW_COLS + DOC_COLS + WINDOW_COLS,
    "DW_doc_window": DOC_COLS + WINDOW_COLS,
}


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews()
    docs = build_documents(reviews)

    panel = panel.merge(docs[cfg.KEY_COLUMNS + ["doc"]],
                        on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
    panel["doc"] = panel["doc"].fillna("")

    print("\n3개월 창 문서 생성 중...")
    win = window_documents(panel[cfg.KEY_COLUMNS + ["doc"]])
    panel = panel.merge(win, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
    panel["docw"] = panel["docw"].fillna("")
    print(f"  창 문서 길이 중앙값 {panel['docw'].str.len().median():,.0f}자")

    records = []
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}]")
        print("  리뷰 트랙 학습")
        rev = review_track(reviews, tend)
        print("  문서 트랙 학습")
        doc = score_documents(panel, tend)
        print("  창 문서 트랙 학습")
        wdw = score_window(panel, tend)

        df = (panel.merge(rev, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
              .merge(doc, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
              .merge(wdw, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one"))
        all_cols = REVIEW_COLS + DOC_COLS + WINDOW_COLS
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
            print(f"    {'+' if gain > 0 else '-'} {name:<18} {len(feats_text)}개  "
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
        "gain_mean": g.mean(), "gain_std": g.std(),
        "gain_min": g.min(), "gain_max": g.max(),
        "wins": g.apply(lambda s: int((s > 0).sum())),
    }).round(5).sort_values("gain_mean", ascending=False)

    print("\n" + "=" * 82)
    print("4개 fold 종합")
    print("=" * 82)
    print(summary.to_string())
    print("\n[fold별]")
    print(w.pivot(index="model_name", columns="fold", values="gain").round(5).to_string())

    best = summary[(summary["wins"] == 4)]
    print("\n  4개 fold 전부 개선한 조합")
    print(best.to_string() if len(best) else "    없음")
    print(f"\n저장: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
