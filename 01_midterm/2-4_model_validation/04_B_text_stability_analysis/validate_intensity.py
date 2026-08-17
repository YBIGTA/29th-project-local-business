"""
B-2일차 5단계. 텍스트 강도 피처 검증.

A2에서 추가한 강도 피처 17개를 오늘 정립한 세 척도로 평가한다.
  1. 정형 45개로 재현되는가            R2
  2. 라벨과 관련이 있는가              순위상관
  3. 여러 시간 구간에서 기여하는가      롤링 4-fold

기존 텍스트 91개는 3번에서 4개 fold 평균이 모두 음수였다.
그 이유는 "신호 있는 부분은 정형이 재현 가능하고, 정형이 못 담는
부분은 라벨과 무관하다"는 구조였다(R2 x 라벨상관 순위상관 +0.50).

강도 축이 이 구조를 벗어나는지 확인한다.
2번 사분면(신호 있고 정형이 재현 못함)에 들어가야 의미가 있다.

강도 패널은 원본을 덮어쓰지 않고 별도 경로에서 키로 결합한다.

    python3 validate_intensity.py

출력: intensity_diagnostics.csv
      intensity_rolling_metrics.csv
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
INTENSITY_PATH = REPO_ROOT / "data" / "external" / "product_month_text_intensity.parquet"
GROUPS_PATH = B1_DIR / "text_feature_groups.csv"
DIAG_PATH = HERE / "intensity_diagnostics.csv"
ROLLING_PATH = HERE / "intensity_rolling_metrics.csv"

FOLDS = [
    ("fold_1_2020_01_08", "2019-12", "2020-01", "2020-08"),
    ("fold_2_2020_09_2021_04", "2020-08", "2020-09", "2021-04"),
    ("fold_3_2021_05_12", "2021-04", "2021-05", "2021-12"),
    ("fold_4_2022_01_08", "2021-12", "2022-01", "2022-08"),
]

REGRESSOR_PARAMS = {
    "objective": "regression", "n_estimators": 200, "learning_rate": 0.05,
    "num_leaves": 15, "min_child_samples": 50, "subsample": 0.80,
    "subsample_freq": 1, "colsample_bytree": 0.80, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": -1, "verbose": -1,
}

# 기존 91개 텍스트 피처의 기준값. 강도 피처를 여기에 견준다.
BASE_MEDIAN_R2 = 0.0467
BASE_MEDIAN_SIGNAL = 0.0204


def r2(y_true, y_pred) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def attach_intensity(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """강도 패널을 공통 키로 결합한다. 원본 패널은 건드리지 않는다."""
    inten = pd.read_parquet(INTENSITY_PATH)
    new_cols = [c for c in inten.columns if c.startswith("intensity_")]
    assert len(new_cols) == 17, f"강도 피처 17개가 아님: {len(new_cols)}"

    sub = inten[cfg.KEY_COLUMNS + new_cols].copy()
    sub["year_month"] = sub["year_month"].astype(str)
    assert sub.duplicated(subset=cfg.KEY_COLUMNS).sum() == 0, "강도 패널 키 중복"

    before = len(df)
    out = df.merge(sub, on=cfg.KEY_COLUMNS, how="left", validate="one_to_one")
    assert len(out) == before, "결합 후 행 수 변화"
    assert out[new_cols].isna().sum().sum() == 0, "결합 후 결측 발생"
    return out, new_cols


def run_fold(df, features, train_end, valid_start, valid_end):
    import lightgbm as lgb

    train = df[df["year_month"] <= train_end]
    valid = df[(df["year_month"] >= valid_start) & (df["year_month"] <= valid_end)]
    params = dict(cfg.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(train[cfg.TARGET_COLUMN])
    params.setdefault("verbose", -1)
    model = lgb.LGBMClassifier(**params).fit(train[features], train[cfg.TARGET_COLUMN])
    return evaluate(valid, model.predict_proba(valid[features])[:, 1])


def main() -> None:
    import lightgbm as lgb

    df, a_features, text_features = load_combined(verbose=False)
    df, intensity = attach_intensity(df)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    kept = groups[groups["kept"]]["column"].tolist()

    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]

    # ---------------- 1. R2 와 라벨 신호 ----------------
    print("=" * 88)
    print("진단 1: 정형 45개로 재현 가능한가 / 라벨과 관련이 있는가")
    print("=" * 88)

    Xtr, Xva = train[a_features], valid[a_features]
    ranked = train[intensity + [cfg.TARGET_COLUMN]].rank(method="average")
    lab_corr = ranked.corr(method="pearson")[cfg.TARGET_COLUMN]

    rows = []
    for col in intensity:
        ytr, yva = train[col].to_numpy(float), valid[col].to_numpy(float)
        score = np.nan
        if ytr.std() > 0:
            m = lgb.LGBMRegressor(**REGRESSOR_PARAMS).fit(Xtr, ytr)
            score = r2(yva, m.predict(Xva))
        c = lab_corr[col]
        rows.append({
            "column": col,
            "nonzero_ratio": round(float((train[col] != 0).mean()), 4),
            "r2_valid": round(score, 4) if np.isfinite(score) else None,
            "abs_corr_label": round(abs(float(c)), 4) if pd.notna(c) else None,
        })

    diag = pd.DataFrame(rows)
    diag["quadrant"] = diag.apply(
        lambda r: (
            "2_신호O_정형이재현X"
            if (r["abs_corr_label"] or 0) >= BASE_MEDIAN_SIGNAL
            and (r["r2_valid"] or 0) < BASE_MEDIAN_R2
            else "1_신호O_정형이재현O"
            if (r["abs_corr_label"] or 0) >= BASE_MEDIAN_SIGNAL
            else "4_신호X"
        ), axis=1)
    diag = diag.sort_values("abs_corr_label", ascending=False)
    diag.to_csv(DIAG_PATH, index=False, encoding="utf-8-sig")
    print(diag.to_string(index=False))

    print(f"\n  기존 91개 중앙값:  R2 {BASE_MEDIAN_R2:.4f} / |라벨상관| {BASE_MEDIAN_SIGNAL:.4f}")
    print(f"  강도 17개 중앙값:  R2 {diag['r2_valid'].median():.4f} / "
          f"|라벨상관| {diag['abs_corr_label'].median():.4f}")
    q2 = int((diag["quadrant"] == "2_신호O_정형이재현X").sum())
    print(f"  2번 사분면(신호 있고 정형이 재현 못함): {q2}/17개")

    # ---------------- 2. 롤링 검증 ----------------
    sets = {
        "I1_intensity_only": intensity,
        "I2_intensity_plus_text91": kept + intensity,
        "I3_intensity_severe_only": [c for c in intensity
                                     if "severe" in c or "safety" in c],
        "I4_intensity_no_mild": [c for c in intensity if "mild" not in c],
    }

    print("\n" + "=" * 88)
    print(f"진단 2: 롤링 {len(FOLDS)}개 fold x 조합 {len(sets) + 1}개")
    print("=" * 88)

    recs = []
    for fold, tend, vs, ve in FOLDS:
        base = run_fold(df, a_features, tend, vs, ve)
        recs.append({"fold": fold, "model_name": "A_baseline_45", "n_text": 0, **base})
        print(f"\n[{fold}]  기준선 {base['pr_auc']:.5f}")
        for name, feats_text in sets.items():
            feats = list(dict.fromkeys(a_features + feats_text))
            m = run_fold(df, feats, tend, vs, ve)
            recs.append({"fold": fold, "model_name": name,
                         "n_text": len(feats_text), **m})
            gain = m["pr_auc"] - base["pr_auc"]
            print(f"    {'+' if gain > 0 else '-'} {name:<28} "
                  f"텍스트 {len(feats_text):>3}개  {m['pr_auc']:.5f}  ({gain:+.5f})")

    metrics = pd.DataFrame(recs)
    metrics.to_csv(ROLLING_PATH, index=False, encoding="utf-8-sig")

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

    print("\n" + "=" * 88)
    print("4개 fold 종합")
    print("=" * 88)
    print(summary.to_string())
    print("\n  참고: 기존 텍스트 91개는 4-fold 평균 -0.00710, 승수 1/4였다.")

    solid = summary[(summary["wins"] == 4) & (summary["gain_mean"] > 0)]
    print()
    if len(solid):
        print("  4개 fold 전부에서 기여한 조합:")
        print(solid.to_string())
    else:
        print("  4개 fold 전부에서 기여한 조합이 없다.")
        print("  강도 축을 추가해도 예측 성능은 개선되지 않는다.")

    print(f"\n저장: {DIAG_PATH.name}, {ROLLING_PATH.name}")


if __name__ == "__main__":
    main()
