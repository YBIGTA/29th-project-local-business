"""
B-2일차. A의 정형 기준 모델 재현 및 텍스트 결합 기반.

A1이 확정한 `candidate_long_1_3_6_12m`(정형 45개)을 그대로 재현하고,
공식 Valid PR-AUC 0.231104와 일치하는지 확인한다.
재현이 맞아야 오늘의 텍스트 기여 측정이 A2와 같은 기준에서 이루어진다.

1일차에는 최소 정형 3개 또는 7개를 기준선으로 썼기 때문에 텍스트 기여가
실제보다 크게 측정됐다. 오늘은 A의 실제 기준 모델 위에서 다시 잰다.

    python3 a_baseline.py

A의 다중 기간 이력 패널은 62,041행으로 test 6,046행이 이미 빠져 있다.
따라서 결합 결과에는 test가 구조적으로 포함될 수 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
B1_DIR = HERE.parent / "02_B_text_feature_validation"
A1_DIR = HERE.parent / "01_A_structured_timeseries"

sys.path.insert(0, str(B1_DIR))
from common_input import cfg, load_panel  # noqa: E402
from evaluation import run_model  # noqa: E402

# A1의 train_multiwindow_candidates.py는 subsample_freq=1을 함께 지정한다.
# LightGBM은 subsample_freq 기본값이 0이라, 이 옵션이 없으면 subsample=0.80이
# 실제로 적용되지 않고 전체 행으로 학습된다.
# experiment_config.LIGHTGBM_PARAMS에는 이 항목이 빠져 있어 여기서 보완한다.
cfg.LIGHTGBM_PARAMS = {**cfg.LIGHTGBM_PARAMS, "subsample_freq": 1}

STRUCTURED_PANEL = A1_DIR / "structured_modeling_panel.parquet"
HISTORY_PANEL = A1_DIR / "multiwindow_history_features_pretest.parquet"
GROUPS_PATH = B1_DIR / "text_feature_groups.csv"

# A1 README 3절. 현재 상태 7개.
A_CURRENT_7 = [
    "avg_rating",
    "low_rating_count",
    "low_rating_ratio",
    "mean_helpful_vote",
    "review_count",
    "text_available_ratio",
    "verified_purchase_ratio",
]

# A1이 확정한 공식 성능. 재현 검증에 사용한다.
A_OFFICIAL_PR_AUC = 0.231104
A_OFFICIAL_FEATURE_COUNT = 45
REPRODUCTION_TOLERANCE = 0.002

EXPECTED_ROWS_NO_TEST = 62_041
EXPECTED_SPLIT_NO_TEST = {"train": 54_813, "valid": 7_228}

# 1일차 클리핑 규칙을 그대로 유지한다.
CLIP_RULES = {"ttf_median_days": 730.0}


def load_history_features() -> tuple[pd.DataFrame, list[str]]:
    hist = pd.read_parquet(HISTORY_PANEL)
    feats = [c for c in hist.columns if c not in cfg.KEY_COLUMNS]
    assert len(feats) == 38, f"이력 피처 38개가 아님: {len(feats)}"
    return hist, feats


def load_combined(verbose: bool = True):
    """텍스트 패널과 A의 이력 패널을 공통 키로 결합한다.

    반환: (df, a_features_45, text_features_91)
    """
    text_df, _ = load_panel(verbose=False)
    hist, hist_feats = load_history_features()

    assert len(hist) == EXPECTED_ROWS_NO_TEST, f"이력 패널 행 수: {len(hist)}"
    assert hist.duplicated(subset=cfg.KEY_COLUMNS).sum() == 0, "이력 패널 키 중복"

    df = text_df.merge(hist, on=cfg.KEY_COLUMNS, how="inner", validate="one_to_one")
    assert len(df) == EXPECTED_ROWS_NO_TEST, f"결합 후 행 수: {len(df)}"

    counts = df["split"].value_counts().to_dict()
    assert counts == EXPECTED_SPLIT_NO_TEST, f"분할 행 수: {counts}"

    missing = [c for c in A_CURRENT_7 if c not in df.columns]
    assert not missing, f"현재 상태 피처 누락: {missing}"

    a_features = A_CURRENT_7 + hist_feats
    assert len(a_features) == A_OFFICIAL_FEATURE_COUNT, len(a_features)

    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    text_features = groups[groups["kept"]]["column"].tolist()

    for col, upper in CLIP_RULES.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=upper)

    if verbose:
        print("=" * 70)
        print("A 기준 모델 + 텍스트 결합 패널")
        print("=" * 70)
        print(f"  결합 패널       {len(df):,}행 x {df.shape[1]}열")
        print(f"  정형 피처       {len(a_features)}개 (현재 7 + 이력 38)")
        print(f"  텍스트 피처     {len(text_features)}개")
        for name in ["train", "valid"]:
            part = df[df["split"] == name]
            print(f"  {name:<6} {len(part):>6,}행  양성 {part[cfg.TARGET_COLUMN].mean():.2%}")
        print("  test는 A의 이력 패널에 애초에 포함되지 않는다")
        print("=" * 70)

    return df, a_features, text_features


def main() -> None:
    df, a_features, text_features = load_combined()

    print("\nA1 정형 기준 모델 재현 중...")
    row, _ = run_model(df, a_features, "A_candidate_long_1_3_6_12m", eval_split="valid")

    gap = abs(row["pr_auc"] - A_OFFICIAL_PR_AUC)
    print("\n" + "=" * 70)
    print("재현 검증")
    print("=" * 70)
    print(f"  A1 공식 PR-AUC    {A_OFFICIAL_PR_AUC:.6f}")
    print(f"  B 재현 PR-AUC     {row['pr_auc']:.6f}")
    print(f"  차이              {gap:.6f}")
    print(f"  ROC-AUC           {row['roc_auc']:.6f}  (A1 공식 0.685959)")
    print(f"  Recall@Top 5%     {row['recall_at_5pct']:.6f}  (A1 공식 0.114561)")
    print(f"  Recall@Top 10%    {row['recall_at_10pct']:.6f}  (A1 공식 0.213062)")
    print("-" * 70)
    if gap <= REPRODUCTION_TOLERANCE:
        print(f"  재현 성공. 오늘 텍스트 기여는 이 기준선 위에서 측정한다.")
    else:
        print(f"  차이가 허용치 {REPRODUCTION_TOLERANCE}를 넘는다.")
        print("  LightGBM 파라미터나 피처 순서가 A1과 다를 수 있으므로 확인이 필요하다.")
    print("=" * 70)

    print("\n참고: 1일차 기준선과의 비교")
    print(f"  최소 정형 7개        0.1666")
    print(f"  최소 정형 7개 + past 0.1822")
    print(f"  A 기준 모델 45개     {row['pr_auc']:.4f}")
    print("  1일차 텍스트 기여(+0.009~0.014)는 약한 기준선에서 잰 값이다.")


if __name__ == "__main__":
    main()
