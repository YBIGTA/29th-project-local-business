from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")
PANEL_PATH = BASE / "structured_modeling_panel.parquet"
SPEC_PATH = BASE / "structured_feature_spec.csv"
MODEL_DIR = BASE / "models"

KEYS = ["parent_asin", "year_month"]
LABEL = "is_low_rating_surge"

# 준비 단계에서 고정한 시간 분할
TRAIN_END = "2021-12"
VALID_START = "2022-01"
VALID_END = "2022-08"
TEST_START = "2022-09"  # 이 스크립트는 이 시점 이후 데이터를 사용하지 않는다.

LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

def recall_at_pct(frame: pd.DataFrame, pct: float) -> float:
    ordered = frame.sort_values(
        ["predicted_probability", "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    n_top = max(1, int(np.ceil(len(ordered) * pct)))
    total_positive = int(ordered[LABEL].sum())
    return 0.0 if total_positive == 0 else float(ordered.head(n_top)[LABEL].sum() / total_positive)

def evaluate(model_name: str, split_name: str, frame: pd.DataFrame) -> dict:
    y_true = frame[LABEL].astype(int)
    y_prob = frame["predicted_probability"]
    return {
        "model_name": model_name,
        "split": split_name,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "recall_at_5pct": round(recall_at_pct(frame, 0.05), 6),
        "recall_at_10pct": round(recall_at_pct(frame, 0.10), 6),
        "n_rows": len(frame),
        "positive_rate": round(float(y_true.mean()), 6),
        "feature_count": None,
        "random_state": 42,
    }

panel = pd.read_parquet(PANEL_PATH)
spec = pd.read_csv(SPEC_PATH)

if panel["year_month"].astype(str).ge(TEST_START).any():
    # 아래에서 Test를 즉시 제거한다. 이후 어떤 모델·지표에도 쓰지 않는다.
    modeling = panel.loc[panel["year_month"].astype(str) < TEST_START].copy()
else:
    modeling = panel.copy()

train = modeling.loc[modeling["year_month"].astype(str) <= TRAIN_END].copy()
valid = modeling.loc[
    (modeling["year_month"].astype(str) >= VALID_START)
    & (modeling["year_month"].astype(str) <= VALID_END)
].copy()

expected = {"train": 54813, "valid": 7228}
actual = {"train": len(train), "valid": len(valid)}
if actual != expected:
    raise ValueError(f"시간 분할 행 수 불일치: 기대 {expected}, 실제 {actual}")

current_features = spec.loc[
    (spec["decision"] == "use")
    & (spec["feature_group"] == "current_state"),
    "feature_name",
].tolist()

trend_features = spec.loc[
    (spec["decision"] == "use")
    & (spec["feature_group"] == "timeseries_trend"),
    "feature_name",
].tolist()

feature_sets = {
    "current_state": current_features,
    "timeseries": current_features + trend_features,
}

train_positive = int(train[LABEL].sum())
train_negative = int(len(train) - train_positive)
scale_pos_weight = train_negative / train_positive

results = []
prediction_frames = []
importance_frames = []

for feature_set_name, features in feature_sets.items():
    X_train = train[features]
    y_train = train[LABEL].astype(int)
    X_valid = valid[features]
    y_valid = valid[LABEL].astype(int)

    models = {
        f"{feature_set_name}_logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight={0: 1.0, 1: scale_pos_weight},
                        max_iter=2000,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        f"{feature_set_name}_lightgbm": LGBMClassifier(
            **LGBM_PARAMS,
            scale_pos_weight=scale_pos_weight,
        ),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]

        pred = valid[KEYS + [LABEL]].copy()
        pred["model_name"] = model_name
        pred["feature_set"] = feature_set_name
        pred["predicted_probability"] = probabilities
        prediction_frames.append(pred)

        metric = evaluate(model_name, "valid", pred)
        metric["feature_count"] = len(features)
        results.append(metric)

        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")

        if model_name.endswith("lightgbm"):
            importance_frames.append(
                pd.DataFrame(
                    {
                        "model_name": model_name,
                        "feature_name": features,
                        "importance_gain": model.booster_.feature_importance(
                            importance_type="gain"
                        ),
                        "importance_split": model.booster_.feature_importance(
                            importance_type="split"
                        ),
                    }
                ).sort_values("importance_gain", ascending=False)
            )

metrics = pd.DataFrame(results).sort_values(
    ["pr_auc", "recall_at_10pct"], ascending=False
)
predictions = pd.concat(prediction_frames, ignore_index=True)
importance = pd.concat(importance_frames, ignore_index=True)

metrics.to_csv(BASE / "validation_metrics.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(BASE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(BASE / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

summary = {
    "test_data_used": False,
    "train_period": f"~{TRAIN_END}",
    "valid_period": f"{VALID_START}~{VALID_END}",
    "excluded_test_period": f"{TEST_START}~2023-02",
    "train_rows": len(train),
    "valid_rows": len(valid),
    "train_positive_rate": round(float(train[LABEL].mean()), 6),
    "valid_positive_rate": round(float(valid[LABEL].mean()), 6),
    "scale_pos_weight_from_train_only": round(float(scale_pos_weight), 6),
    "current_state_feature_count": len(current_features),
    "timeseries_feature_count": len(current_features + trend_features),
}
(BASE / "baseline_training_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A1 정형 기준 모델 학습 완료 =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n[Valid 성능: PR-AUC 우선 정렬]")
print(metrics.to_string(index=False))
print("\n[LightGBM 중요도 상위 10개]")
print(
    importance.sort_values(["model_name", "importance_gain"], ascending=[True, False])
    .groupby("model_name", group_keys=False)
    .head(10)
    .to_string(index=False)
)
