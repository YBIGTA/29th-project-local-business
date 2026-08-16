from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")
KEYS = ["parent_asin", "year_month"]
LABEL = "is_low_rating_surge"

TRAIN_END = "2021-12"
VALID_START = "2022-01"
VALID_END = "2022-08"
TEST_START = "2022-09"

panel = pd.read_parquet(BASE / "structured_modeling_panel.parquet")
history = pd.read_parquet(BASE / "multiwindow_history_features_pretest.parquet")
spec = pd.read_csv(BASE / "structured_feature_spec.csv")

panel = panel.loc[panel["year_month"].astype(str) < TEST_START].copy()
df = panel.merge(history, on=KEYS, how="inner", validate="one_to_one")

if len(df) != 62041:
    raise ValueError(f"Train+Valid 행 수 불일치: 기대 62041, 실제 {len(df)}")

current_features = spec.loc[
    (spec["decision"] == "use")
    & (spec["feature_group"] == "current_state"),
    "feature_name",
].tolist()

existing_3m_features = spec.loc[
    (spec["decision"] == "use")
    & (spec["feature_group"] == "timeseries_trend"),
    "feature_name",
].tolist()

history_features = [c for c in history.columns if c not in KEYS]

def window_of(name: str) -> int | None:
    match = re.match(r"^history_(\d+)m_", name)
    return int(match.group(1)) if match else None

# 기간별 후보 비교에서는 장기 비교(delta) 피처가 단기 후보에 섞이지 않게 명시적으로 분리한다.
base_history_features = [c for c in history_features if "_vs_" not in c]

short_features = [
    c for c in base_history_features
    if window_of(c) in {1, 3}
]

medium_features = [
    c for c in base_history_features
    if window_of(c) in {1, 3, 6}
] + ["history_3m_vs_6m_low_rating_ratio_delta"]

long_features = history_features

feature_sets = {
    "baseline_existing_3m": current_features + existing_3m_features,
    "candidate_short_1_3m": current_features + short_features,
    "candidate_medium_1_3_6m": current_features + medium_features,
    "candidate_long_1_3_6_12m": current_features + long_features,
}

train = df.loc[df["year_month"].astype(str) <= TRAIN_END].copy()
valid = df.loc[
    (df["year_month"].astype(str) >= VALID_START)
    & (df["year_month"].astype(str) <= VALID_END)
].copy()

if (len(train), len(valid)) != (54813, 7228):
    raise ValueError(f"시간 분할 불일치: train={len(train)}, valid={len(valid)}")

train_positive = int(train[LABEL].sum())
scale_pos_weight = (len(train) - train_positive) / train_positive

def recall_at_pct(frame: pd.DataFrame, pct: float) -> float:
    ordered = frame.sort_values(
        ["predicted_probability", "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top_n = max(1, int(np.ceil(len(ordered) * pct)))
    positive_total = int(ordered[LABEL].sum())
    return 0.0 if positive_total == 0 else float(
        ordered.head(top_n)[LABEL].sum() / positive_total
    )

metric_rows = []
prediction_frames = []
importance_frames = []

for model_name, features in feature_sets.items():
    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(train[features], train[LABEL].astype(int))
    probability = model.predict_proba(valid[features])[:, 1]

    pred = valid[KEYS + [LABEL]].copy()
    pred["model_name"] = model_name
    pred["predicted_probability"] = probability
    prediction_frames.append(pred)

    metric_rows.append(
        {
            "model_name": model_name,
            "split": "valid",
            "pr_auc": round(float(average_precision_score(pred[LABEL], probability)), 6),
            "roc_auc": round(float(roc_auc_score(pred[LABEL], probability)), 6),
            "recall_at_5pct": round(recall_at_pct(pred, 0.05), 6),
            "recall_at_10pct": round(recall_at_pct(pred, 0.10), 6),
            "n_rows": len(pred),
            "positive_rate": round(float(pred[LABEL].mean()), 6),
            "feature_count": len(features),
            "random_state": 42,
        }
    )

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
    joblib.dump(model, BASE / "models" / f"{model_name}.joblib")

metrics = pd.DataFrame(metric_rows).sort_values(
    ["pr_auc", "recall_at_10pct"], ascending=False
)
importance = pd.concat(importance_frames, ignore_index=True)

metrics.to_csv(BASE / "multiwindow_validation_metrics.csv", index=False, encoding="utf-8-sig")
pd.concat(prediction_frames, ignore_index=True).to_csv(
    BASE / "multiwindow_validation_predictions.csv", index=False, encoding="utf-8-sig"
)
importance.to_csv(
    BASE / "multiwindow_feature_importance.csv", index=False, encoding="utf-8-sig"
)

summary = {
    "test_data_used": False,
    "train_period": "~2021-12",
    "valid_period": "2022-01~2022-08",
    "excluded_test_period": "2022-09~2023-02",
    "train_rows": len(train),
    "valid_rows": len(valid),
    "scale_pos_weight_train_only": round(float(scale_pos_weight), 6),
    "feature_counts": {name: len(features) for name, features in feature_sets.items()},
}
(BASE / "multiwindow_training_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A1 다중 기간 정형 후보 비교 완료 =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n[Valid 성능: PR-AUC 우선 정렬]")
print(metrics.to_string(index=False))

winner = metrics.iloc[0]["model_name"]
print(f"\n[현재 Valid 기준 1위 후보] {winner}")

print("\n[후보별 중요도 상위 5개]")
print(
    importance.sort_values(["model_name", "importance_gain"], ascending=[True, False])
    .groupby("model_name", group_keys=False)
    .head(5)
    .to_string(index=False)
)
