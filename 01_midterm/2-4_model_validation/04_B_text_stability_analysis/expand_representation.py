"""
B-2일차 12단계. 텍스트 표현 확장.

11단계에서 RD_review_doc가 4개 fold 전부 개선했다(+0.00464).
남은 병목은 압축이다.

  8만 차원 TF-IDF -> 로지스틱 회귀 -> 확률 하나

이 과정에서 "어떤 종류의 불만인가"가 사라진다.
내구성 고장이 많은 상품과 배송 불만이 많은 상품이 같은 점수를 받으면
정형 이력과의 상호작용을 LightGBM이 찾을 수 없다.

시도 1. SVD 성분을 피처로 직접 사용
  TF-IDF를 20~50차원으로 줄여 LightGBM에 넣는다.
  밀집·직교 차원이라 사전 기반 91개처럼 노이즈가 쌓이지 않는다.
  각 성분이 대략 하나의 주제 축이므로 상호작용 학습이 가능하다.
  SVD는 비지도 변환이므로 라벨 누수가 없다. train 문서로만 적합한다.

시도 2. 리뷰 단위 모델의 타깃을 급증으로 변경
  현재 리뷰 트랙은 "이 리뷰가 1~2점인가"를 학습한다.
  상품×월의 급증 라벨을 리뷰에 전파해 학습하면 다른 신호가 나온다.
  MIL에서 묶음 라벨을 항목에 물려주는 방식이다.

    python3 expand_representation.py

출력: expanded_text_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from supervised_text_score import (
    FOLDS, VECTORIZER_PARAMS, load_reviews, run_fold,
)
from document_level_score import DOC_PARAMS, N_OOF_FOLDS, build_documents, score_documents
from combine_tracks import DOC_COLS, REVIEW_COLS, review_track

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "expanded_text_metrics.csv"

SVD_DIMS = [20, 50]
HIGH_RISK_THRESHOLD = 0.5


def svd_features(panel: pd.DataFrame, train_end: str, n_comp: int):
    """월 문서 TF-IDF를 SVD로 축소해 피처로 만든다.

    SVD는 라벨을 쓰지 않지만 어휘와 성분 방향이 train 구간에 고정되도록
    train 문서로만 적합한다.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    is_train = panel["year_month"].le(train_end).to_numpy()
    text = panel["doc"].fillna("").astype(str)

    vec = TfidfVectorizer(**DOC_PARAMS)
    X_tr = vec.fit_transform(text[is_train])
    svd = TruncatedSVD(n_components=n_comp, random_state=42, algorithm="randomized")
    svd.fit(X_tr)

    Z = svd.transform(vec.transform(text)).astype(np.float32)
    cols = [f"svd{n_comp}_{i:02d}" for i in range(n_comp)]
    print(f"    SVD {n_comp}차원  설명 분산 {svd.explained_variance_ratio_.sum():.3f}")

    out = panel[cfg.KEY_COLUMNS].copy()
    for i, c in enumerate(cols):
        out[c] = Z[:, i]
    return out, cols


def review_track_surge(reviews: pd.DataFrame, panel: pd.DataFrame,
                       train_end: str) -> pd.DataFrame:
    """리뷰 단위 모델을 급증 라벨로 학습한다.

    상품×월의 라벨을 그 달 리뷰 전체에 물려준다.
    개별 리뷰 라벨이 없으므로 약지도학습이며 노이즈가 크지만,
    저평점 타깃과 다른 표현을 학습할 수 있다.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    lab = panel[cfg.KEY_COLUMNS + [cfg.TARGET_COLUMN]]
    rv = reviews.merge(lab, on=cfg.KEY_COLUMNS, how="inner")

    fit_mask = rv["year_month"].le(train_end).to_numpy()
    texts = rv["text"].astype(str)

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_fit = vec.fit_transform(texts[fit_mask])
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_fit, rv.loc[fit_mask, cfg.TARGET_COLUMN].to_numpy())
    proba = clf.predict_proba(vec.transform(texts))[:, 1]

    work = rv[cfg.KEY_COLUMNS].copy()
    work["p"] = proba
    work["high"] = (proba >= HIGH_RISK_THRESHOLD).astype("int8")
    return (work.groupby(cfg.KEY_COLUMNS)
            .agg(srg_mean_t=("p", "mean"),
                 srg_max_t=("p", "max"),
                 srg_high_rate_t=("high", "mean"))
            .reset_index())


SURGE_COLS = ["srg_mean_t", "srg_max_t", "srg_high_rate_t"]


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews()
    docs = build_documents(reviews)

    panel = panel.merge(docs[cfg.KEY_COLUMNS + ["doc"]],
                        on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
    panel["doc"] = panel["doc"].fillna("")

    records = []
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}]")
        print("  리뷰 트랙 (저평점 타깃)")
        rev = review_track(reviews, tend)
        print("  리뷰 트랙 (급증 타깃)")
        srg = review_track_surge(reviews, panel, tend)
        print("  문서 트랙")
        doc = score_documents(panel, tend)

        df = (panel.merge(rev, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
              .merge(srg, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
              .merge(doc, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one"))

        svd_cols = {}
        for n in SVD_DIMS:
            print(f"  SVD {n}차원")
            sv, cols = svd_features(panel, tend, n)
            df = df.merge(sv, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
            svd_cols[n] = cols

        fill = REVIEW_COLS + SURGE_COLS + DOC_COLS + sum(svd_cols.values(), [])
        df[fill] = df[fill].fillna(0.0)

        RD = REVIEW_COLS + DOC_COLS
        sets = {
            "E0_RD": RD,
            "E1_RD_surge": RD + SURGE_COLS,
            "E2_RD_svd20": RD + svd_cols[20],
            "E3_RD_svd50": RD + svd_cols[50],
            "E4_svd20_only": svd_cols[20],
            "E5_all": RD + SURGE_COLS + svd_cols[20],
        }

        base = run_fold(df, a_features, tend, vs, ve)
        records.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"    기준선 {base['pr_auc']:.5f}")

        for name, feats_text in sets.items():
            feats = list(dict.fromkeys(a_features + feats_text))
            m = run_fold(df, feats, tend, vs, ve)
            records.append({"fold": fold, "model_name": name,
                            "n_text": len(feats_text), **m})
            gain = m["pr_auc"] - base["pr_auc"]
            print(f"    {'+' if gain > 0 else '-'} {name:<16} {len(feats_text):>3}개  "
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

    print("\n" + "=" * 84)
    print("4개 fold 종합")
    print("=" * 84)
    print(summary.to_string())
    print("\n[fold별]")
    print(w.pivot(index="model_name", columns="fold", values="gain").round(5).to_string())
    print("\n  직전 최고: RD_review_doc  +0.00464  4/4승  표준편차 0.00180")
    print(f"\n저장: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
