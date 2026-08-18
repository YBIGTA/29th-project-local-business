from pathlib import Path
import json
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "data/processed/product_month_review_volume_labeled.parquet"
TEXT = ROOT / "data/processed/product_month_review_volume_text_features.parquet"
OUT = Path(__file__).resolve().parent
TARGET = "is_review_volume_drop"
KEYS = ["parent_asin", "year_month"]

def make_model(y):
    return LGBMClassifier(
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
        scale_pos_weight=(len(y) - y.sum()) / y.sum(),
    )

panel = pd.read_parquet(PANEL)
text = pd.read_parquet(TEXT)
panel["year_month"] = panel["year_month"].astype(str)
text["year_month"] = text["year_month"].astype(str)

merged = panel.merge(
    text.drop(columns=[TARGET, "split"], errors="ignore"),
    on=KEYS,
    how="left",
    validate="one_to_one",
)

structured = [
    c for c in panel.columns
    if c not in {
        "parent_asin", "year_month", "split", TARGET,
        "next_review_count", "expected_next_review_count",
        "next_review_volume_ratio",
    }
]

text_features = [
    c for c in text.columns
    if c not in KEYS + [TARGET, "split", "n_reviews_t", "n_reviews_p3", "low_volume_t"]
    and not c.startswith("next_")
    and not c.startswith("expected_")
    and not c.startswith("is_low_rating")
    and not c.startswith("auto_title")
]

merged[text_features] = merged[text_features].fillna(0)
train = merged[merged["split"] == "train"].copy()
valid = merged[merged["split"] == "valid"].copy()

selector = make_model(train[TARGET])
selector.fit(train[text_features], train[TARGET])
importance = pd.DataFrame({
    "feature": text_features,
    "importance": selector.feature_importances_,
}).sort_values("importance", ascending=False)

selected = importance.head(20)["feature"].tolist()

structured_model = make_model(train[TARGET])
structured_model.fit(train[structured], train[TARGET])

combined_model = make_model(train[TARGET])
combined_model.fit(train[structured + selected], train[TARGET])

structured_score = structured_model.predict_proba(valid[structured])[:, 1]
combined_score = combined_model.predict_proba(valid[structured + selected])[:, 1]

result = {
    "target": TARGET,
    "test_data_used": False,
    "train_rows": int(len(train)),
    "valid_rows": int(len(valid)),
    "text_feature_candidates": len(text_features),
    "selected_text_feature_count": len(selected),
    "structured_valid_pr_auc": float(average_precision_score(valid[TARGET], structured_score)),
    "structured_valid_roc_auc": float(roc_auc_score(valid[TARGET], structured_score)),
    "structured_plus_text_valid_pr_auc": float(average_precision_score(valid[TARGET], combined_score)),
    "structured_plus_text_valid_roc_auc": float(roc_auc_score(valid[TARGET], combined_score)),
}
result["pr_auc_delta"] = (
    result["structured_plus_text_valid_pr_auc"] - result["structured_valid_pr_auc"]
)

importance.to_csv(OUT / "review_volume_text_train_importance.csv", index=False, encoding="utf-8-sig")
(OUT / "review_volume_text_selected_features.json").write_text(
    json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "review_volume_text_increment_metrics.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
