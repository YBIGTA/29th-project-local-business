import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("01_midterm/2-4_model_validation")
OUT = ROOT / "03_A_combined_model_optimization"
PANEL_PATH = OUT / "combined_train_valid.parquet"
INTENSITY_PATH = Path("data/processed/product_month_text_features.parquet")
BASE_SETS_PATH = OUT / "standard_model_feature_sets.json"

OUTPUT_PANEL = OUT / "combined_train_valid_intensity.parquet"
METRICS_PATH = OUT / "intensity_candidate_metrics.csv"
PREDICTIONS_PATH = OUT / "intensity_candidate_predictions.parquet"
IMPORTANCE_PATH = OUT / "intensity_candidate_importance.csv"
SETS_PATH = OUT / "intensity_model_feature_sets.json"
SUMMARY_PATH = OUT / "intensity_experiment_summary.json"
MODEL_DIR = OUT / "models_intensity"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["parent_asin", "year_month"]
TARGET = "is_low_rating_surge"
EXPECTED_SPLITS = {"train": 54813, "valid": 7228}

CURRENT_INTENSITY = [
    "intensity_score_mean_t",
    "intensity_score_max_t",
    "intensity_safety_risk_rate_t",
    "intensity_severe_consequence_rate_t",
]
CHANGE_INTENSITY = [
    "intensity_score_mean_delta",
    "intensity_safety_risk_rate_delta",
    "intensity_severe_consequence_rate_delta",
    "intensity_severe_issue_rate_delta",
]

def unique_keep_order(values):
    return list(dict.fromkeys(str(value) for value in values))

def recall_at_fraction(frame, fraction):
    ranked = frame.sort_values(
        ["y_score", "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    k = max(1, math.ceil(len(ranked) * fraction))
    positives = int(ranked["y_true"].sum())
    return float(ranked.head(k)["y_true"].sum() / positives) if positives else np.nan

for required in [PANEL_PATH, INTENSITY_PATH, BASE_SETS_PATH]:
    if not required.exists():
        raise FileNotFoundError(f"필수 입력 파일 없음: {required}")

panel = pd.read_parquet(PANEL_PATH)
intensity_source = pd.read_parquet(INTENSITY_PATH)
panel["year_month"] = panel["year_month"].astype(str)
intensity_source["year_month"] = intensity_source["year_month"].astype(str)

all_intensity = sorted(
    column for column in intensity_source.columns if column.startswith("intensity_")
)
if len(all_intensity) != 17:
    raise ValueError(f"강도 피처 수 불일치: 기대 17, 실제 {len(all_intensity)}")

for frame_name, frame in {"base_panel": panel, "intensity_source": intensity_source}.items():
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{frame_name} 상품×월 키 중복")

intensity = intensity_source[KEYS + all_intensity].copy()
merged = panel.merge(intensity, on=KEYS, how="left", validate="one_to_one")
if len(merged) != len(panel):
    raise ValueError("강도 피처 결합 후 행 수 불일치")
if merged[all_intensity].isna().any().any():
    raise ValueError("결합 패널에 강도 피처 결측치 존재")
if merged["year_month"].max() > "2022-08":
    raise ValueError("Test 기간이 포함되었습니다")

split_counts = merged["split"].value_counts().to_dict()
if split_counts != EXPECTED_SPLITS:
    raise ValueError(f"분할 행 수 불일치: {split_counts}")

base_sets = json.loads(BASE_SETS_PATH.read_text(encoding="utf-8"))
baseline = base_sets.get("D_S45_plus_change_all")
if not isinstance(baseline, list):
    raise ValueError("기존 최고 모델 피처 목록을 찾지 못했습니다")

for name, features in {
    "baseline": baseline,
    "current": CURRENT_INTENSITY,
    "change": CHANGE_INTENSITY,
    "full": all_intensity,
}.items():
    missing = sorted(set(features) - set(merged.columns))
    if missing:
        raise KeyError(f"{name} 피처 누락: {missing}")

feature_sets = {
    "I0_reproduce_D_S45_plus_change_all": unique_keep_order(baseline),
    "I1_best_plus_intensity_current": unique_keep_order(baseline + CURRENT_INTENSITY),
    "I2_best_plus_intensity_change": unique_keep_order(baseline + CHANGE_INTENSITY),
    "I3_best_plus_intensity_all17": unique_keep_order(baseline + all_intensity),
}

train = merged.loc[merged["split"].eq("train")].copy()
valid = merged.loc[merged["split"].eq("valid")].copy()
y_train = train[TARGET].astype(int)
y_valid = valid[TARGET].astype(int)
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

params = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "scale_pos_weight": scale_pos_weight,
}

metric_rows = []
prediction_parts = []
importance_parts = []

print("===== 강도 피처 결합 검증 =====")
print(f"rows={len(merged):,}, train={len(train):,}, valid={len(valid):,}")
print(f"baseline_features={len(baseline)}, intensity_features={len(all_intensity)}")
print("Test 데이터 사용: False")

for index, (name, features) in enumerate(feature_sets.items(), start=1):
    started = time.perf_counter()
    x_train = train[features].replace([np.inf, -np.inf], np.nan).astype("float32")
    x_valid = valid[features].replace([np.inf, -np.inf], np.nan).astype("float32")

    model = LGBMClassifier(**params)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_valid)[:, 1]
    elapsed = time.perf_counter() - started

    pred = valid[KEYS].copy()
    pred["model_name"] = name
    pred["y_true"] = y_valid.to_numpy()
    pred["y_score"] = scores
    prediction_parts.append(pred)

    metric_rows.append({
        "model_name": name,
        "split": "valid",
        "pr_auc": average_precision_score(y_valid, scores),
        "roc_auc": roc_auc_score(y_valid, scores),
        "recall_at_5pct": recall_at_fraction(pred, 0.05),
        "recall_at_10pct": recall_at_fraction(pred, 0.10),
        "feature_count": len(features),
        "intensity_feature_count": len([f for f in features if f in all_intensity]),
        "n_rows": len(valid),
        "positive_rate": float(y_valid.mean()),
        "training_seconds": elapsed,
        "random_state": 42,
    })

    importance_parts.append(pd.DataFrame({
        "model_name": name,
        "feature": features,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }))
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    print(f"[{index}/{len(feature_sets)}] {name}: PR-AUC={metric_rows[-1]['pr_auc']:.6f}")

metrics = pd.DataFrame(metric_rows).sort_values(
    ["pr_auc", "recall_at_10pct"], ascending=False
).reset_index(drop=True)
predictions = pd.concat(prediction_parts, ignore_index=True)
importance = pd.concat(importance_parts, ignore_index=True).sort_values(
    ["model_name", "importance_gain"], ascending=[True, False]
)

baseline_pr = float(
    metrics.loc[metrics["model_name"].eq("I0_reproduce_D_S45_plus_change_all"), "pr_auc"].iloc[0]
)
metrics["pr_auc_vs_baseline"] = metrics["pr_auc"] - baseline_pr

merged.to_parquet(OUTPUT_PANEL, index=False)
metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
predictions.to_parquet(PREDICTIONS_PATH, index=False)
importance.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
SETS_PATH.write_text(json.dumps(feature_sets, ensure_ascii=False, indent=2), encoding="utf-8")
SUMMARY_PATH.write_text(json.dumps({
    "test_data_used": False,
    "input_rows": len(merged),
    "split_counts": split_counts,
    "baseline_model": "I0_reproduce_D_S45_plus_change_all",
    "baseline_pr_auc": baseline_pr,
    "intensity_feature_count": len(all_intensity),
    "current_intensity_features": CURRENT_INTENSITY,
    "change_intensity_features": CHANGE_INTENSITY,
    "best_model": str(metrics.iloc[0]["model_name"]),
    "best_pr_auc": float(metrics.iloc[0]["pr_auc"]),
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n===== Valid 결과 =====")
print(metrics.to_string(index=False))
print(f"\n저장: {METRICS_PATH}")
