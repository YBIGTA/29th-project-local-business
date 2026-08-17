"""
B-2일차 6단계. 세그먼트별 텍스트 기여 분해.

지금까지 68,087행 전체 평균으로만 쟀다. 텍스트가 특정 조건에서만
유용하다면 전체 평균에서는 묻힌다.

검증할 가설
  정형 모델은 이미 평점이 낮은 상품을 잘 잡는다.
  텍스트의 가치는 평점이 아직 높은데 불만이 쌓이는 상품,
  즉 정형 지표가 아직 정상인 구간의 조기 경보에 있다.

근거
  2-3 정밀도 감사에서 저평점 구간 0.889, 고평점 구간 0.423이었다.
  지금까지 이를 "고평점에서 사전이 부정확하다"로 읽었지만,
  예측 관점에서는 고평점이 텍스트가 별점을 앞설 수 있는 영역이다.

설계 주의
  세그먼트마다 따로 학습하면 train이 줄어 성능이 떨어지고,
  그것은 텍스트 효과가 아니라 표본 감소 효과다.
  따라서 학습은 전체에서 한 번만 하고 평가만 세그먼트로 나눈다.
  세그먼트 기준은 예측 시점 t월에 알 수 있는 값만 사용한다.

    python3 segment_analysis.py

출력: segment_gain.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined
from validate_intensity import attach_intensity

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402
from evaluation import compute_scale_pos_weight, evaluate  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUPS_PATH = B1_DIR / "text_feature_groups.csv"
OUTPUT_PATH = HERE / "segment_gain.csv"

FOLDS = [
    ("fold_1_2020_01_08", "2019-12", "2020-01", "2020-08"),
    ("fold_2_2020_09_2021_04", "2020-08", "2020-09", "2021-04"),
    ("fold_3_2021_05_12", "2021-04", "2021-05", "2021-12"),
    ("fold_4_2022_01_08", "2021-12", "2022-01", "2022-08"),
]

MIN_SEGMENT_ROWS = 300
MIN_SEGMENT_POSITIVES = 25


def add_segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """예측 시점에 알 수 있는 값으로만 세그먼트를 만든다."""
    seg = {}

    # 현재 평점 수준. 가설의 핵심 축이다.
    seg["avg_rating"] = pd.cut(
        df["avg_rating"], bins=[-np.inf, 3.0, 4.0, 4.5, np.inf],
        labels=["~3.0", "3.0~4.0", "4.0~4.5", "4.5~"])

    # 리뷰 수. 저표본에서 텍스트가 불안정한지 확인한다.
    seg["review_count"] = pd.cut(
        df["review_count"], bins=[-np.inf, 9, 29, 99, np.inf],
        labels=["0-9", "10-29", "30-99", "100+"])

    # 장기 품질 수준. 원래 나빴던 상품과 새로 나빠지는 상품을 가른다.
    seg["history_12m_low_rating_ratio"] = pd.cut(
        df["history_12m_low_rating_ratio"], bins=[-np.inf, 0.05, 0.15, 0.30, np.inf],
        labels=["~0.05", "0.05~0.15", "0.15~0.30", "0.30~"])

    # 이력 충족도. 신규 상품에서 텍스트가 대체재가 되는지 본다.
    seg["history_12m_coverage_ratio"] = pd.cut(
        df["history_12m_coverage_ratio"], bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["~0.25", "0.25~0.50", "0.50~0.75", "0.75~"])

    return seg


def fit_predict(df, features, train_end, valid_start, valid_end):
    import lightgbm as lgb

    train = df[df["year_month"] <= train_end]
    valid = df[(df["year_month"] >= valid_start) & (df["year_month"] <= valid_end)]
    params = dict(cfg.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(train[cfg.TARGET_COLUMN])
    params.setdefault("verbose", -1)
    model = lgb.LGBMClassifier(**params).fit(train[features], train[cfg.TARGET_COLUMN])
    return valid, model.predict_proba(valid[features])[:, 1]


def main() -> None:
    df, a_features, _ = load_combined(verbose=False)
    df, intensity = attach_intensity(df)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    kept = groups[groups["kept"]]["column"].tolist()
    shrunk = groups[(groups["kept"]) & (groups["variant"] == "shrunk")]["column"].tolist()

    segments = add_segments(df)
    for name, series in segments.items():
        df[f"seg_{name}"] = series

    # 롤링에서 손해가 가장 적었던 조합들만 남긴다.
    text_sets = {
        "intensity17": intensity,
        "raw_only": groups[(groups["kept"])
                           & (groups["variant"].isin(["raw_t", "raw_mean"]))
                           ]["column"].tolist(),
        "no_shrunk_plus_intensity": [c for c in kept if c not in shrunk] + intensity,
    }

    print(f"세그먼트 {len(segments)}종 x 조합 {len(text_sets)}개 x fold {len(FOLDS)}개\n")

    records = []
    for fold, tend, vs, ve in FOLDS:
        valid, base_proba = fit_predict(df, a_features, tend, vs, ve)
        print(f"[{fold}] valid {len(valid):,}행")

        for set_name, text_feats in text_sets.items():
            feats = list(dict.fromkeys(a_features + text_feats))
            _, proba = fit_predict(df, feats, tend, vs, ve)

            for seg_name in segments:
                col = f"seg_{seg_name}"
                for level, part in valid.groupby(col, observed=True):
                    idx = valid.index.get_indexer(part.index)
                    n_pos = int(part[cfg.TARGET_COLUMN].sum())
                    if len(part) < MIN_SEGMENT_ROWS or n_pos < MIN_SEGMENT_POSITIVES:
                        continue
                    b = evaluate(part, base_proba[idx])
                    t = evaluate(part, proba[idx])
                    records.append({
                        "fold": fold, "text_set": set_name,
                        "segment": seg_name, "level": str(level),
                        "n_rows": len(part), "n_pos": n_pos,
                        "positive_rate": round(float(part[cfg.TARGET_COLUMN].mean()), 4),
                        "pr_auc_base": round(b["pr_auc"], 5),
                        "pr_auc_text": round(t["pr_auc"], 5),
                        "gain": round(t["pr_auc"] - b["pr_auc"], 5),
                    })

    res = pd.DataFrame(records)
    res.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = (res.groupby(["text_set", "segment", "level"])
               .agg(n_rows=("n_rows", "mean"),
                    pos_rate=("positive_rate", "mean"),
                    gain_mean=("gain", "mean"),
                    gain_min=("gain", "min"),
                    wins=("gain", lambda s: int((s > 0).sum())),
                    folds=("gain", "size"))
               .round(5))

    for set_name in text_sets:
        print("\n" + "=" * 92)
        print(f"세그먼트별 텍스트 기여: {set_name}")
        print("=" * 92)
        sub = summary.loc[set_name]
        print(sub.to_string())

    print("\n" + "=" * 92)
    print("모든 fold에서 기여한 세그먼트 (wins == folds)")
    print("=" * 92)
    solid = summary[(summary["wins"] == summary["folds"])
                    & (summary["gain_mean"] > 0)]
    if len(solid):
        print(solid.sort_values("gain_mean", ascending=False).to_string())
        print("\n  전체 평균이 음수여도 이 조건에서는 텍스트가 일관되게 기여한다.")
    else:
        print("  없음. 어떤 세그먼트에서도 일관된 기여가 관찰되지 않는다.")

    print("\n" + "=" * 92)
    print("평균 기여 상위 10개 세그먼트")
    print("=" * 92)
    print(summary.sort_values("gain_mean", ascending=False).head(10).to_string())

    print("\n" + "=" * 92)
    print("평균 기여 하위 5개 (텍스트가 해로운 조건)")
    print("=" * 92)
    print(summary.sort_values("gain_mean").head(5).to_string())

    print(f"\n저장: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
