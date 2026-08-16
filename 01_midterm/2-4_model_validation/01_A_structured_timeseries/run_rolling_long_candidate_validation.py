from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path("01_midterm/2-4_model_validation/01_A_structured_timeseries")
KEYS = ["parent_asin", "year_month"]
LABEL = "is_low_rating_surge"

panel = pd.read_parquet(BASE / "structured_modeling_panel.parquet")
history = pd.read_parquet(BASE / "multiwindow_history_features_pretest.parquet")
spec = pd.read_csv(BASE / "structured_feature_spec.csv")

# 최종 Test(2022-09~2023-02)는 처음부터 제외한다.
panel = panel.loc[panel["year_month"].astype(str) < "2022-09"].copy()
df = panel.merge(history, on=KEYS, how="inner", validate="one_to_one")

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

feature_sets = {
    "baseline_existing_3m": current_features + existing_3m_features,
    "candidate_long_1_3_6_12m": current_features + history_features,
}

folds = [
    ("fold_1_2020_01_08", "2019-12", "2020-01", "2020-08"),
    ("fold_2_2020_09_2021_04", "2020-08", "2020-09", "2021-04"),
    ("fold_3_2021_05_12", "2021-04", "2021-05", "2021-12"),
    ("fold_4_2022_01_08", "2021-12", "2022-01", "2022-08"),
]

def recall_at_pct(frame: pd.DataFrame, pct: float) -> float:
    ordered = frame.sort_values(
        ["predicted_probability", "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top_n = max(1, int(np.ceil(len(ordered) * pct)))
    positives = int(ordered[LABEL].sum())
    return 0.0 if positives == 0 else float(
        ordered.head(top_n)[LABEL].sum() / positives
    )

rows = []

for fold_name, train_end, valid_start, valid_end in folds:
    train = df.loc[df["year_month"].astype(str) <= train_end].copy()
    valid = df.loc[
        (df["year_month"].astype(str) >= valid_start)
        & (df["year_month"].astype(str) <= valid_end)
    ].copy()

    y_train = train[LABEL].astype(int)
    scale_pos_weight = (len(train) - y_train.sum()) / y_train.sum()

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
        model.fit(train[features], y_train)
        probability = model.predict_proba(valid[features])[:, 1]

        pred = valid[KEYS + [LABEL]].copy()
        pred["predicted_probability"] = probability

        rows.append(
            {
                "fold": fold_name,
                "train_end": train_end,
                "valid_period": f"{valid_start}~{valid_end}",
                "model_name": model_name,
                "pr_auc": round(float(average_precision_score(pred[LABEL], probability)), 6),
                "roc_auc": round(float(roc_auc_score(pred[LABEL], probability)), 6),
                "recall_at_5pct": round(recall_at_pct(pred, 0.05), 6),
                "recall_at_10pct": round(recall_at_pct(pred, 0.10), 6),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "feature_count": len(features),
                "scale_pos_weight_train_only": round(float(scale_pos_weight), 6),
            }
        )

metrics = pd.DataFrame(rows)
winner_by_fold = (
    metrics.sort_values(["fold", "pr_auc"], ascending=[True, False])
    .groupby("fold", as_index=False)
    .first()[["fold", "model_name"]]
)
wins = winner_by_fold["model_name"].value_counts().to_dict()

summary = (
    metrics.groupby("model_name", as_index=False)
    .agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        recall_at_10pct_mean=("recall_at_10pct", "mean"),
    )
)
summary["pr_auc_wins"] = summary["model_name"].map(wins).fillna(0).astype(int)

metrics.to_csv(
    BASE / "rolling_long_candidate_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)
summary.to_csv(
    BASE / "rolling_long_candidate_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("===== A1 장기 후보 롤링 시간 검증 완료 =====")
print("Test 기간(2022-09~2023-02)은 사용하지 않았습니다.\n")
print("[구간별 PR-AUC]")
print(
    metrics.pivot(index="fold", columns="model_name", values="pr_auc")
    .round(6)
    .to_string()
)
print("\n[모델별 평균 성능 및 PR-AUC 승리 횟수]")
print(summary.round(6).to_string(index=False))
