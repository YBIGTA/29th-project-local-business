from pathlib import Path
import json
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "data/processed/product_month_review_volume_2m_labeled.parquet"
OUT = Path(__file__).resolve().parent
TARGET = "is_review_volume_drop_2m"

data = pd.read_parquet(PANEL)
data["year_month"] = data["year_month"].astype(str)

train_valid = data[data["split"].isin(["train", "valid"])].copy()
test = data[
    (data["split"] == "test")
    & (data["year_month"] <= "2023-01")
].copy()

features = [
    c for c in data.columns
    if c not in {
        "parent_asin", "year_month", "split",
        "is_review_volume_drop",
        "is_review_volume_drop_2m",
        "next_review_count",
        "expected_next_review_count",
        "next_review_volume_ratio",
        "next_2m_review_count",
        "expected_next_2m_review_count",
    }
]

def recall_at(y, score, fraction):
    n = max(1, int(len(score) * fraction))
    top = pd.Series(score).nlargest(n).index
    return float(pd.Series(y).iloc[top].sum() / max(1, pd.Series(y).sum()))

def precision_at(y, score, fraction):
    n = max(1, int(len(score) * fraction))
    top = pd.Series(score).nlargest(n).index
    return float(pd.Series(y).iloc[top].mean())

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
    scale_pos_weight=(
        (len(train_valid) - train_valid[TARGET].sum())
        / train_valid[TARGET].sum()
    ),
)

model.fit(train_valid[features], train_valid[TARGET])
score = model.predict_proba(test[features])[:, 1]

metrics = {
    "target": TARGET,
    "target_definition": "t+1~t+2 리뷰 합계가 카테고리 보정 기대치의 50% 이하",
    "model_locked_before_test": True,
    "text_features_used": False,
    "train_valid_rows": int(len(train_valid)),
    "test_rows": int(len(test)),
    "test_period": f"{test['year_month'].min()}~{test['year_month'].max()}",
    "test_positive_count": int(test[TARGET].sum()),
    "test_positive_rate": float(test[TARGET].mean()),
    "test_pr_auc": float(average_precision_score(test[TARGET], score)),
    "test_roc_auc": float(roc_auc_score(test[TARGET], score)),
    "test_recall_at_5pct_alert": recall_at(test[TARGET], score, 0.05),
    "test_recall_at_10pct_alert": recall_at(test[TARGET], score, 0.10),
    "test_precision_at_5pct_alert": precision_at(test[TARGET], score, 0.05),
    "test_precision_at_10pct_alert": precision_at(test[TARGET], score, 0.10),
    "feature_count": len(features),
}

pd.DataFrame({
    "parent_asin": test["parent_asin"],
    "year_month": test["year_month"],
    "actual_2m_drop": test[TARGET],
    "risk_score": score,
}).sort_values("risk_score", ascending=False).to_parquet(
    OUT / "final_2m_test_predictions.parquet",
    index=False,
)

pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).to_csv(
    OUT / "final_2m_feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

(OUT / "final_2m_test_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
