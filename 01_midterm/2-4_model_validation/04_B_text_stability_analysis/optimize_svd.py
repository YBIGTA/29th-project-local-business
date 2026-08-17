"""
B-2일차 13단계. SVD 표현 최적화.

12단계에서 SVD 20차원 단독이 최고였다(+0.00534, 4/4).
50차원은 +0.00076으로 떨어져 차원 수에 최적점이 있음을 시사한다.

세 방향으로 확장한다.

1. 차원 수 탐색
   10 / 15 / 20 / 30 을 비교해 최적점을 찾는다.

2. 문서 구성 변경
   지금은 그 달 리뷰 전부를 이어 붙인다.
   저평점 리뷰만 모은 문서를 따로 만들면 소수의 심각한 신호가
   다수의 만족 리뷰에 희석되지 않는다.

3. SVD 축의 이력 확장
   A는 정형에서 1·3·6·12개월 창을 만들어 가장 큰 이득을 얻었다.
   텍스트 축은 아직 현재 월만 본다.
   각 SVD 축의 직전 3개월 평균과 변화량을 추가해
   "어떤 불만 축이 최근 증가했는가"를 표현한다.

E1_RD_surge 계열은 제외한다. 급증 라벨을 리뷰에 전파하면 같은
상품×월의 리뷰가 라벨을 공유해 모델이 그 달을 외워버린다.
교차적합 없이는 누수가 발생하므로 사용하지 않는다.

    python3 optimize_svd.py

출력: svd_optimization_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from supervised_text_score import FOLDS, load_reviews, run_fold
from document_level_score import DOC_PARAMS, build_documents, score_documents
from combine_tracks import DOC_COLS, REVIEW_COLS, review_track

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "svd_optimization_metrics.csv"
LOWDOC_CACHE = HERE / "month_documents_low.parquet"

DIMS = [10, 15, 20, 30]
HIST_DIM = 20          # 이력 확장에 사용할 차원 수
MAX_DOC_CHARS = 20_000


def build_low_documents(reviews: pd.DataFrame) -> pd.DataFrame:
    """저평점(1~2점) 리뷰만 이어 붙인 문서.

    다수의 만족 리뷰에 소수의 심각한 불만이 희석되는 것을 막는다.
    """
    if LOWDOC_CACHE.exists():
        d = pd.read_parquet(LOWDOC_CACHE)
        print(f"저평점 문서 캐시 사용: {len(d):,}개")
        return d

    print("저평점 문서 생성 중...")
    low = reviews[reviews["is_low"].eq(1)]
    d = (low.groupby(cfg.KEY_COLUMNS)["text"]
         .apply(lambda s: " ".join(s.astype(str))[:MAX_DOC_CHARS])
         .reset_index().rename(columns={"text": "doc_low"}))
    d.to_parquet(LOWDOC_CACHE, index=False)
    print(f"  저평점 문서 {len(d):,}개")
    return d


def svd_transform(panel: pd.DataFrame, text_col: str, train_end: str,
                  n_comp: int, prefix: str):
    """문서 TF-IDF를 SVD로 축소한다. train 문서로만 적합한다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    is_train = panel["year_month"].le(train_end).to_numpy()
    text = panel[text_col].fillna("").astype(str)

    vec = TfidfVectorizer(**DOC_PARAMS)
    X_tr = vec.fit_transform(text[is_train])
    svd = TruncatedSVD(n_components=n_comp, random_state=42).fit(X_tr)
    Z = svd.transform(vec.transform(text)).astype(np.float32)

    cols = [f"{prefix}{i:02d}" for i in range(n_comp)]
    out = panel[cfg.KEY_COLUMNS].copy()
    for i, c in enumerate(cols):
        out[c] = Z[:, i]
    return out, cols


def add_history(frame: pd.DataFrame, cols: list[str]):
    """각 축의 직전 3개월 평균과 변화량을 만든다. 현재 월은 제외한다.

    LightGBM은 subsample_freq=1에서 행 순서에 따라 샘플링 결과가 달라진다.
    따라서 정렬은 계산용으로만 쓰고 원래 행 순서를 그대로 돌려준다.
    """
    ordered = frame.sort_values(cfg.KEY_COLUMNS)
    pieces = {}
    grouped = ordered.groupby("parent_asin", group_keys=False)
    for c in cols:
        p3 = grouped[c].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        p3 = p3.reset_index(level=0, drop=True) if isinstance(p3.index, pd.MultiIndex) else p3
        pieces[f"{c}_p3"] = p3.fillna(0.0)
        pieces[f"{c}_dl"] = ordered[c] - pieces[f"{c}_p3"]

    add = pd.DataFrame(pieces, index=ordered.index).reindex(frame.index)
    out = pd.concat([frame, add], axis=1)
    return out, list(pieces.keys())


def main() -> None:
    panel, a_features, _ = load_combined(verbose=False)
    reviews = load_reviews()

    docs = build_documents(reviews)
    lows = build_low_documents(reviews)

    panel = (panel.merge(docs[cfg.KEY_COLUMNS + ["doc"]], on=cfg.KEY_COLUMNS,
                         how="left", validate="one_to_one")
             .merge(lows, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one"))
    panel["doc"] = panel["doc"].fillna("")
    panel["doc_low"] = panel["doc_low"].fillna("")
    print(f"저평점 문서가 있는 행 {int(panel['doc_low'].str.len().gt(0).sum()):,} / {len(panel):,}")

    records = []
    for fold, tend, vs, ve in FOLDS:
        print(f"\n[{fold}]")
        # 모든 병합은 원래 행 순서를 유지해야 한다.
        # subsample_freq=1 환경에서 순서가 바뀌면 결과가 재현되지 않는다.
        df = panel.copy()
        base_index = df.index.copy()
        sets = {}

        # --- 차원 수 탐색 ---
        for n in DIMS:
            sv, cols = svd_transform(df, "doc", tend, n, f"sv{n}_")
            df = df.merge(sv, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
            sets[f"S{n}_all_doc"] = cols
        print(f"    전체 문서 SVD {DIMS} 완료")

        # --- 저평점 문서 SVD ---
        lo, lo_cols = svd_transform(df, "doc_low", tend, HIST_DIM, "lo_")
        df = df.merge(lo, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
        sets["L20_low_doc"] = lo_cols
        sets["B40_all_plus_low"] = sets[f"S{HIST_DIM}_all_doc"] + lo_cols
        print("    저평점 문서 SVD 완료")

        # --- 이력 축 확장 ---
        base_cols = sets[f"S{HIST_DIM}_all_doc"]
        df, hist_cols = add_history(df, base_cols)
        sets[f"H{HIST_DIM}_with_history"] = base_cols + hist_cols
        print(f"    이력 축 {len(hist_cols)}개 생성")

        # --- 기존 최고 조합 재현 ---
        rev = review_track(reviews, tend)
        doc = score_documents(df, tend)
        df = (df.merge(rev, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
              .merge(doc, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one"))
        sets["E0_RD"] = REVIEW_COLS + DOC_COLS
        sets["BEST_svd20_plus_RD_hist"] = (base_cols + hist_cols
                                           + REVIEW_COLS + DOC_COLS)

        assert df.index.equals(base_index), "행 순서가 변경되었다"
        fill = sorted({c for v in sets.values() for c in v})
        df[fill] = df[fill].fillna(0.0)
        df = df.copy()  # 조각난 프레임 정리

        base = run_fold(df, a_features, tend, vs, ve)
        records.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"    기준선 {base['pr_auc']:.5f}")

        for name, feats_text in sets.items():
            feats = list(dict.fromkeys(a_features + feats_text))
            m = run_fold(df, feats, tend, vs, ve)
            records.append({"fold": fold, "model_name": name,
                            "n_text": len(feats_text), **m})
            gain = m["pr_auc"] - base["pr_auc"]
            print(f"    {'+' if gain > 0 else '-'} {name:<26} {len(feats_text):>3}개  "
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

    print("\n" + "=" * 88)
    print("4개 fold 종합")
    print("=" * 88)
    print(summary.to_string())
    print("\n[fold별]")
    print(w.pivot(index="model_name", columns="fold", values="gain").round(5).to_string())
    print("\n  직전 최고")
    print("    E4_svd20_only   +0.00534  4/4  표준편차 0.00414")
    print("    E0_RD           +0.00464  4/4  표준편차 0.00180")
    print(f"\n저장: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
