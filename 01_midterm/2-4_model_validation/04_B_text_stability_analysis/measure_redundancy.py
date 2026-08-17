"""
B-2일차 3단계. 정형 피처와 텍스트 피처의 정보 중복 측정.

롤링 검증에서 텍스트 91개의 기여가 4개 fold 평균 모두 음수였다.
"텍스트가 나쁘다"가 아니라 "정형이 이미 담고 있다"를 보이려면
중복 자체를 직접 재야 한다.

방법
  각 텍스트 피처를 타깃으로 놓고 A의 정형 45개로 회귀한 뒤 R2를 구한다.
  R2가 높을수록 그 텍스트 피처는 정형 피처들의 함수로 재현 가능하다.

1일차에는 상관계수로 1:1 중복만 봤다. 실제로는 여러 정형 피처의
조합이 텍스트 하나를 재현할 수 있으므로 다변량으로 다시 잰다.

train 구간에서 학습하고 valid 구간에서 R2를 구한다.
train R2만 보면 과적합으로 중복이 과대평가된다.

    python3 measure_redundancy.py

출력: text_redundancy.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUPS_PATH = B1_DIR / "text_feature_groups.csv"
OUTPUT_PATH = HERE / "text_redundancy.csv"

# 회귀용 경량 설정. 예측이 목적이 아니라 재현 가능성 측정이 목적이다.
REGRESSOR_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

HIGH_REDUNDANCY = 0.50
LOW_REDUNDANCY = 0.20


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main() -> None:
    import lightgbm as lgb

    df, a_features, text_features = load_combined(verbose=False)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    meta = groups.set_index("column")[["family", "variant"]]

    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    Xtr, Xva = train[a_features], valid[a_features]

    print(f"정형 {len(a_features)}개로 텍스트 {len(text_features)}개를 각각 회귀")
    print(f"학습 {len(train):,}행 / 평가 {len(valid):,}행\n")

    rows = []
    for i, col in enumerate(text_features, 1):
        ytr, yva = train[col].to_numpy(float), valid[col].to_numpy(float)
        if ytr.std() == 0:
            continue
        model = lgb.LGBMRegressor(**REGRESSOR_PARAMS).fit(Xtr, ytr)
        score = r2(yva, model.predict(Xva))

        imp = pd.Series(model.booster_.feature_importance("gain"), index=a_features)
        rows.append({
            "column": col,
            "family": meta.loc[col, "family"],
            "variant": meta.loc[col, "variant"],
            "r2_valid": round(score, 4),
            "top_structured": imp.idxmax(),
            "top_share": round(float(imp.max() / imp.sum()), 4) if imp.sum() > 0 else None,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(text_features)} 완료")

    red = pd.DataFrame(rows).sort_values("r2_valid", ascending=False)
    red.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 84)
    print("정형 45개로 가장 잘 재현되는 텍스트 피처 20개")
    print("=" * 84)
    print(red.head(20).to_string(index=False))

    print("\n" + "=" * 84)
    print("정형이 재현하지 못하는 텍스트 피처 20개 (앞으로 유망한 방향)")
    print("=" * 84)
    print(red.tail(20).sort_values("r2_valid").to_string(index=False))

    print("\n" + "=" * 84)
    print("중복도 분포")
    print("=" * 84)
    n_high = int((red["r2_valid"] >= HIGH_REDUNDANCY).sum())
    n_low = int((red["r2_valid"] < LOW_REDUNDANCY).sum())
    print(f"  R2 >= {HIGH_REDUNDANCY}  {n_high:>3}개 ({n_high / len(red):.1%})  정형으로 대부분 재현됨")
    print(f"  R2 <  {LOW_REDUNDANCY}  {n_low:>3}개 ({n_low / len(red):.1%})  정형이 담지 못함")
    print(f"  전체 중앙값 R2  {red['r2_valid'].median():.4f}")

    print("\n  variant별 중앙 R2")
    print(red.groupby("variant")["r2_valid"]
          .agg(["count", "median", "max"]).round(4).to_string())

    print("\n  family별 중앙 R2")
    print(red.groupby("family")["r2_valid"]
          .agg(["count", "median", "max"]).round(4).to_string())

    print("\n  텍스트를 재현하는 데 가장 자주 쓰인 정형 피처")
    print(red["top_structured"].value_counts().head(10).to_string())

    print(f"\n저장: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
