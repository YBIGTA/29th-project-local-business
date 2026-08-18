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

def recall_at(y, score, fraction):
    n = max(1, int(len(score) * fraction))
    top = pd.Series(score).nlargest(n).index
    return float(pd.Series(y).iloc[top].sum() / max(1, pd.Series(y).sum()))

panel = pd.read_parquet(PANEL)
text = pd.read_parquet(TEXT)

panel["year_month"] = panel["year_month"].astype(str)
text["year_month"] = text["year_month"].astype(str)

data = panel.merge(
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

data[text_features] = data[text_features].fillna(0)

folds = [
    ("2019", "2019-01", "2019-12"),
    ("2020", "2020-01", "2020-12"),
    ("2021", "2021-01", "2021-12"),
    ("2022_01_08", "2022-01", "2022-08"),
]

rows = []
selected_by_fold = {}

for name, eval_start, eval_end in folds:
    train = data[data["year_month"] < eval_start].copy()
    valid = data[
        (data["year_month"] >= eval_start)
        & (data["year_month"] <= eval_end)
    ].copy()

    if valid["year_month"].max() > "2022-08":
        raise RuntimeError("봉인 Test 기간이 포함됐습니다.")

    selector = make_model(train[TARGET])
    selector.fit(train[text_features], train[TARGET])

    ranked = pd.DataFrame({
        "feature": text_features,
        "importance": selector.feature_importances_,
    }).sort_values("importance", ascending=False)

    selected = ranked.head(20)["feature"].tolist()
    selected_by_fold[name] = selected

    structured_model = make_model(train[TARGET])
    structured_model.fit(train[structured], train[TARGET])

    combined_model = make_model(train[TARGET])
    combined_model.fit(train[structured + selected], train[TARGET])

    structured_score = structured_model.predict_proba(valid[structured])[:, 1]
    combined_score = combined_model.predict_proba(valid[structured + selected])[:, 1]

    rows.append({
        "fold": name,
        "train_end": train["year_month"].max(),
        "eval_start": eval_start,
        "eval_end": eval_end,
        "train_rows": len(train),
        "eval_rows": len(valid),
        "eval_positive_rate": float(valid[TARGET].mean()),
        "structured_pr_auc": float(average_precision_score(valid[TARGET], structured_score)),
        "text_pr_auc": float(average_precision_score(valid[TARGET], combined_score)),
        "pr_auc_delta": float(
            average_precision_score(valid[TARGET], combined_score)
            - average_precision_score(valid[TARGET], structured_score)
        ),
        "structured_roc_auc": float(roc_auc_score(valid[TARGET], structured_score)),
        "text_roc_auc": float(roc_auc_score(valid[TARGET], combined_score)),
        "structured_recall_at_10pct": recall_at(valid[TARGET], structured_score, 0.10),
        "text_recall_at_10pct": recall_at(valid[TARGET], combined_score, 0.10),
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "rolling_validation_metrics.csv", index=False, encoding="utf-8-sig")

summary = {
    "test_data_used": False,
    "test_period_sealed": "2022-09~2023-02",
    "fold_count": int(len(result)),
    "mean_structured_pr_auc": float(result["structured_pr_auc"].mean()),
    "mean_text_pr_auc": float(result["text_pr_auc"].mean()),
    "mean_pr_auc_delta": float(result["pr_auc_delta"].mean()),
    "text_wins": int((result["pr_auc_delta"] > 0).sum()),
    "decision": (
        "텍스트 제외"
        if result["pr_auc_delta"].mean() <= 0
        else "텍스트 후속 검증 후보"
    ),
}

(OUT / "rolling_validation_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(OUT / "rolling_selected_text_features.json").write_text(
    json.dumps(selected_by_fold, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(result.to_string(index=False))
print()
print(json.dumps(summary, ensure_ascii=False, indent=2))
