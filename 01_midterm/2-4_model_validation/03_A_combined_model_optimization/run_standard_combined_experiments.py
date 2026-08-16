
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
A1 = ROOT / "01_A_structured_timeseries"
B1 = ROOT / "02_B_text_feature_validation"
OUT = ROOT / "03_A_combined_model_optimization"
MODEL_DIR = OUT / "models_standard"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "combined_train_valid_canonical.parquet"
SETS_PATH = OUT / "combined_feature_sets.json"
GROUP_PATH = B1 / "text_feature_groups.csv"
SPEC_PATH = A1 / "multiwindow_history_feature_spec.csv"

TARGET = "is_low_rating_surge"
KEYS = ["parent_asin", "year_month"]
EXPECTED_SPLITS = {"train": 54813, "valid": 7228}

CURRENT_FEATURES = [
    "avg_rating",
    "low_rating_count",
    "low_rating_ratio",
    "mean_helpful_vote",
    "review_count",
    "text_available_ratio",
    "verified_purchase_ratio",
]

POLICY = {
    "declared_before_training": True,
    "primary_metric": "pr_auc",
    "secondary_metrics": ["recall_at_5pct", "recall_at_10pct", "roc_auc"],
    "candidate_models": ["C1_S45_plus_B1", "C5_S45_plus_B5", "C8_S45_plus_B8"],
    "diagnostic_models": [
        "D_S45_plus_past_p3",
        "D_S45_plus_delta",
        "D_S45_plus_change_all",
    ],
    "rules": [
        "결합 후보는 S45_structured_only보다 Valid PR-AUC가 높아야 한다.",
        "B1, B5, B8은 동일한 Train/Valid와 동일한 LightGBM 설정으로 비교한다.",
        "PR-AUC 차이가 0.003 이하라면 더 적은 피처를 우선한다.",
        "B5는 B1보다 PR-AUC 0.003 이상 또는 Recall@10% 0.01 이상 개선되어야 복잡성을 정당화한다.",
        "B8은 최고 후보와 PR-AUC 차이가 0.005 이내이거나 Recall@5%가 0.01 이상 좋을 때 유지한다.",
        "분해 모델은 텍스트 현재·과거·변화 정보의 역할 확인용이며 최종 후보로 바로 선택하지 않는다.",
        "최종 후보 확정에는 이후 롤링 검증 평균 개선 양수와 최소 3/4 구간 승리가 필요하다.",
        "Test 2022-09~2023-02는 사용하지 않는다.",
    ],
}
(OUT / "stage4_selection_policy.json").write_text(
    json.dumps(POLICY, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


def unique_keep_order(values):
    return list(dict.fromkeys(str(v) for v in values))


def get_candidate_entry(blob, name):
    if isinstance(blob, dict):
        if name in blob:
            return blob[name]
        for value in blob.values():
            found = get_candidate_entry(value, name)
            if found is not None:
                return found
    elif isinstance(blob, list):
        for item in blob:
            if isinstance(item, dict):
                item_name = (
                    item.get("candidate")
                    or item.get("candidate_name")
                    or item.get("name")
                )
                if item_name == name:
                    return item
    return None


def extract_text_features(config, name, structured, available):
    candidates = config.get("candidates", config)
    entry = get_candidate_entry(candidates, name)

    if entry is None:
        raise KeyError(f"후보를 JSON에서 찾지 못했습니다: {name}")

    if isinstance(entry, list):
        values = entry
    elif isinstance(entry, dict):
        values = None

        for field in [
            "text_features_for_combined_model",
            "text_features_original",
            "text_features",
            "retained_text_features",
            "selected_text_features",
        ]:
            if field in entry and isinstance(entry[field], list):
                values = entry[field]
                break

        if values is None:
            for field in [
                "all_features_for_combined_model",
                "combined_features",
                "features",
                "feature_names",
            ]:
                if field in entry and isinstance(entry[field], list):
                    values = entry[field]
                    break

        if values is None:
            raise KeyError(
                f"{name} 항목에서 피처 목록을 찾지 못했습니다. "
                f"키={list(entry.keys())}"
            )
    else:
        raise TypeError(f"{name} 후보 형식이 예상과 다릅니다: {type(entry)}")

    structured_set = set(structured)
    values = [
        str(v) for v in values
        if str(v) in available and str(v) not in structured_set
    ]
    return unique_keep_order(values)


def recall_at_fraction(frame, fraction):
    ranked = frame.sort_values(
        ["y_score", "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    k = max(1, math.ceil(len(ranked) * fraction))
    positives = int(ranked["y_true"].sum())
    if positives == 0:
        return np.nan
    return float(ranked.head(k)["y_true"].sum() / positives)


panel = pd.read_parquet(PANEL_PATH)
panel["year_month"] = panel["year_month"].astype(str)

actual_splits = panel["split"].value_counts().to_dict()
if actual_splits != EXPECTED_SPLITS:
    raise ValueError(
        f"Train/Valid 행 수 불일치: 기대={EXPECTED_SPLITS}, 실제={actual_splits}"
    )

if panel.duplicated(KEYS).any():
    raise ValueError("상품·월 키 중복이 있습니다.")

if panel["year_month"].max() > "2022-08":
    raise ValueError("Test 기간 데이터가 결합 패널에 포함됐습니다.")

config = json.loads(SETS_PATH.read_text(encoding="utf-8"))

spec = pd.read_csv(SPEC_PATH)
if "feature_name" not in spec.columns:
    raise KeyError(f"정형 피처 정의에 feature_name 열이 없습니다: {list(spec.columns)}")

history_features = unique_keep_order(spec["feature_name"].dropna().tolist())
structured = unique_keep_order(CURRENT_FEATURES + history_features)

if len(structured) != 45:
    raise ValueError(f"A1 정형 피처 수가 45가 아닙니다: {len(structured)}")

missing_structured = [f for f in structured if f not in panel.columns]
if missing_structured:
    raise KeyError(f"결합 패널에 없는 정형 피처: {missing_structured}")

available = set(panel.columns)

b1 = extract_text_features(
    config, "B1_clean_shrunk", structured, available
)
b5 = extract_text_features(
    config, "B5_clean_shrunk_change", structured, available
)
b8 = extract_text_features(
    config, "B8_no_shrunk", structured, available
)

groups = pd.read_csv(GROUP_PATH)
variant_map = dict(
    zip(groups["column"].astype(str), groups["variant"].astype(str))
)

b5_p3 = [
    f for f in b5
    if variant_map.get(f) == "past_p3" or f.endswith("_p3")
]
b5_delta = [
    f for f in b5
    if variant_map.get(f) == "delta" or f.endswith("_delta")
]
b5_change = unique_keep_order(b5_p3 + b5_delta)
b5_level = [f for f in b5 if f not in set(b5_change)]

expected_counts = {
    "structured": 45,
    "B1": 15,
    "B5": 57,
    "B8": 72,
    "B5_level": 15,
    "B5_past_p3": 21,
    "B5_delta": 21,
    "B5_change_all": 42,
}
actual_counts = {
    "structured": len(structured),
    "B1": len(b1),
    "B5": len(b5),
    "B8": len(b8),
    "B5_level": len(b5_level),
    "B5_past_p3": len(b5_p3),
    "B5_delta": len(b5_delta),
    "B5_change_all": len(b5_change),
}

if actual_counts != expected_counts:
    raise ValueError(
        f"후보 구성 수가 B 전달 내용과 다릅니다.\n"
        f"기대={expected_counts}\n실제={actual_counts}"
    )

if set(b5_level) != set(b1):
    raise ValueError("B5의 현재 상태 15개가 B1의 15개와 일치하지 않습니다.")

if set(b5_change) != (set(b5_p3) | set(b5_delta)):
    raise ValueError("B5 변화 피처 분해가 일치하지 않습니다.")

forbidden = {
    TARGET,
    "next_avg_rating",
    "next_low_rating_count",
    "next_low_rating_ratio",
    "next_review_count",
    "next_vs_past_3m_low_rating_change",
}
forbidden_used = sorted(
    forbidden.intersection(set(structured + b1 + b5 + b8))
)
if forbidden_used:
    raise ValueError(f"금지 피처가 모델 입력에 포함됐습니다: {forbidden_used}")

feature_sets = {
    "S45_structured_only": structured,
    "C1_S45_plus_B1": unique_keep_order(structured + b1),
    "C5_S45_plus_B5": unique_keep_order(structured + b5),
    "C8_S45_plus_B8": unique_keep_order(structured + b8),
    "D_S45_plus_past_p3": unique_keep_order(structured + b5_p3),
    "D_S45_plus_delta": unique_keep_order(structured + b5_delta),
    "D_S45_plus_change_all": unique_keep_order(structured + b5_change),
}

train = panel.loc[panel["split"] == "train"].copy()
valid = panel.loc[panel["split"] == "valid"].copy()

y_train = train[TARGET].astype(int)
y_valid = valid[TARGET].astype(int)
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

model_params = {
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

metrics_rows = []
prediction_parts = []
importance_parts = []
timings = {}

print("===== 입력·후보 검증 완료 =====")
print(json.dumps(actual_counts, ensure_ascii=False, indent=2))
print(f"Train 양성 가중치: {scale_pos_weight:.6f}")
print("Test 데이터 사용: False")
print()

total_models = len(feature_sets)

for index, (model_name, features) in enumerate(feature_sets.items(), start=1):
    print(
        f"[{index}/{total_models}] {model_name} 실제 학습 시작 "
        f"({len(features)}개 피처)"
    )

    start = time.perf_counter()

    x_train = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
    )
    x_valid = (
        valid[features]
        .replace([np.inf, -np.inf], np.nan)
    )

    model = LGBMClassifier(**model_params)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_valid)[:, 1]

    elapsed = time.perf_counter() - start
    timings[model_name] = round(elapsed, 3)

    pred = valid[KEYS].copy()
    pred["model_name"] = model_name
    pred["y_true"] = y_valid.to_numpy()
    pred["y_score"] = score

    row = {
        "model_name": model_name,
        "split": "valid",
        "pr_auc": average_precision_score(y_valid, score),
        "roc_auc": roc_auc_score(y_valid, score),
        "recall_at_5pct": recall_at_fraction(pred, 0.05),
        "recall_at_10pct": recall_at_fraction(pred, 0.10),
        "n_rows": len(valid),
        "positive_rate": y_valid.mean(),
        "feature_count": len(features),
        "text_feature_count": len(
            [f for f in features if f not in set(structured)]
        ),
        "training_seconds": elapsed,
        "random_state": 42,
    }
    metrics_rows.append(row)
    prediction_parts.append(pred)

    importance_parts.append(
        pd.DataFrame(
            {
                "model_name": model_name,
                "feature_name": features,
                "feature_type": [
                    "structured" if f in set(structured) else "text"
                    for f in features
                ],
                "importance_gain": model.booster_.feature_importance(
                    importance_type="gain"
                ),
                "importance_split": model.booster_.feature_importance(
                    importance_type="split"
                ),
            }
        )
    )

    joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
    print(
        f"[{index}/{total_models}] 완료: "
        f"PR-AUC={row['pr_auc']:.6f}, "
        f"Recall@10%={row['recall_at_10pct']:.6f}, "
        f"학습시간={elapsed:.2f}초"
    )

metrics = pd.DataFrame(metrics_rows)
predictions = pd.concat(prediction_parts, ignore_index=True)
importance = pd.concat(importance_parts, ignore_index=True)

baseline_pr = float(
    metrics.loc[
        metrics["model_name"] == "S45_structured_only", "pr_auc"
    ].iloc[0]
)
baseline_r5 = float(
    metrics.loc[
        metrics["model_name"] == "S45_structured_only", "recall_at_5pct"
    ].iloc[0]
)
baseline_r10 = float(
    metrics.loc[
        metrics["model_name"] == "S45_structured_only", "recall_at_10pct"
    ].iloc[0]
)

metrics["delta_pr_auc_vs_S45"] = metrics["pr_auc"] - baseline_pr
metrics["delta_recall_5pct_vs_S45"] = (
    metrics["recall_at_5pct"] - baseline_r5
)
metrics["delta_recall_10pct_vs_S45"] = (
    metrics["recall_at_10pct"] - baseline_r10
)

text_gain = (
    importance.assign(
        text_gain=lambda x: np.where(
            x["feature_type"].eq("text"),
            x["importance_gain"],
            0.0,
        )
    )
    .groupby("model_name", as_index=False)
    .agg(
        total_gain=("importance_gain", "sum"),
        text_gain=("text_gain", "sum"),
    )
)
text_gain["text_gain_share"] = np.where(
    text_gain["total_gain"] > 0,
    text_gain["text_gain"] / text_gain["total_gain"],
    0.0,
)

metrics = metrics.merge(
    text_gain[["model_name", "text_gain_share"]],
    on="model_name",
    how="left",
)

metrics.to_csv(
    OUT / "standard_candidate_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)
predictions.to_parquet(
    OUT / "standard_candidate_predictions.parquet",
    index=False,
)
importance.to_csv(
    OUT / "standard_candidate_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

(OUT / "standard_model_feature_sets.json").write_text(
    json.dumps(feature_sets, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

summary = {
    "test_data_used": False,
    "train_period": "2016-01~2021-12",
    "valid_period": "2022-01~2022-08",
    "excluded_test_period": "2022-09~2023-02",
    "train_rows": len(train),
    "valid_rows": len(valid),
    "scale_pos_weight_train_only": scale_pos_weight,
    "feature_counts": actual_counts,
    "model_training_seconds": timings,
    "selection_is_final": False,
    "next_required_check": "rolling time validation",
}
(OUT / "standard_experiment_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== 2-4-03 표준 결합 모델 비교 완료 =====")
print("[Valid 성능: PR-AUC 순]")
print(
    metrics.sort_values("pr_auc", ascending=False)[
        [
            "model_name",
            "pr_auc",
            "delta_pr_auc_vs_S45",
            "roc_auc",
            "recall_at_5pct",
            "recall_at_10pct",
            "feature_count",
            "text_feature_count",
            "text_gain_share",
            "training_seconds",
        ]
    ].round(6).to_string(index=False)
)

print("\n[B1·B5·B8 핵심 후보]")
print(
    metrics[
        metrics["model_name"].isin(
            [
                "S45_structured_only",
                "C1_S45_plus_B1",
                "C5_S45_plus_B5",
                "C8_S45_plus_B8",
            ]
        )
    ].sort_values("pr_auc", ascending=False)[
        [
            "model_name",
            "pr_auc",
            "delta_pr_auc_vs_S45",
            "recall_at_5pct",
            "recall_at_10pct",
            "feature_count",
            "text_gain_share",
        ]
    ].round(6).to_string(index=False)
)

print("\n[B5 변화 정보 분해 실험]")
print(
    metrics[
        metrics["model_name"].str.startswith("D_")
    ].sort_values("pr_auc", ascending=False)[
        [
            "model_name",
            "pr_auc",
            "delta_pr_auc_vs_S45",
            "recall_at_10pct",
            "text_feature_count",
        ]
    ].round(6).to_string(index=False)
)

print("\n[결합 후보별 텍스트 중요도 상위 8개]")
top_text = (
    importance[
        importance["model_name"].isin(
            ["C1_S45_plus_B1", "C5_S45_plus_B5", "C8_S45_plus_B8"]
        )
        & importance["feature_type"].eq("text")
    ]
    .sort_values(
        ["model_name", "importance_gain"],
        ascending=[True, False],
    )
    .groupby("model_name", group_keys=False)
    .head(8)
)
print(
    top_text[
        ["model_name", "feature_name", "importance_gain"]
    ].to_string(index=False)
)

print("\n주의: 아직 최종 후보를 확정하지 않았습니다.")
print("위 결과를 사전 고정 기준으로 해석한 뒤 5/8로 넘어갑니다.")
