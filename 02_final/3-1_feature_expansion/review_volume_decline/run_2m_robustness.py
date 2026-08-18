from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
ACTIVITY = ROOT / "data/interim/product_month_activity.parquet"
PANEL_1M = ROOT / "data/processed/product_month_review_volume_labeled.parquet"
PANEL_2M = ROOT / "data/processed/product_month_review_volume_2m_labeled.parquet"
OUT = Path(__file__).resolve().parent
TARGET = "is_review_volume_drop_2m"

activity = pd.read_parquet(ACTIVITY).copy()
panel = pd.read_parquet(PANEL_1M).copy()

activity["parent_asin"] = activity["parent_asin"].astype(str)
activity["year_month"] = activity["year_month"].astype(str)
activity["period"] = pd.PeriodIndex(activity["year_month"], freq="M")
panel["parent_asin"] = panel["parent_asin"].astype(str)
panel["year_month"] = panel["year_month"].astype(str)
panel["period"] = pd.PeriodIndex(panel["year_month"], freq="M")

lookup = {
    (row.parent_asin, row.period): float(row.review_count)
    for row in activity[["parent_asin", "period", "review_count"]].itertuples(index=False)
}
category = activity.groupby("period")["review_count"].sum().to_dict()

panel = panel[panel["period"] <= pd.Period("2023-01", freq="M")].copy()

next_2m = []
expected_2m = []

for row in panel.itertuples(index=False):
    t = row.period
    actual = (
        lookup.get((row.parent_asin, t + 1), 0.0)
        + lookup.get((row.parent_asin, t + 2), 0.0)
    )
    category_total = float(category.get(t + 1, 0) + category.get(t + 2, 0))
    share = row.past_3m_review_count / (
        float(category.get(t, 0))
        + float(category.get(t - 1, 0))
        + float(category.get(t - 2, 0))
    )
    next_2m.append(actual)
    expected_2m.append(share * category_total)

panel["next_2m_review_count"] = next_2m
panel["expected_next_2m_review_count"] = expected_2m
panel[TARGET] = (
    panel["next_2m_review_count"]
    <= 0.50 * panel["expected_next_2m_review_count"]
).astype(int)

panel = panel.drop(columns=["period"])
panel.to_parquet(PANEL_2M, index=False)

safe_features = [
    c for c in panel.columns
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

folds = [
    ("2019", "2019-01", "2019-12"),
    ("2020", "2020-01", "2020-12"),
    ("2021", "2021-01", "2021-12"),
    ("2022_01_08", "2022-01", "2022-08"),
]

rows = []

for name, start, end in folds:
    train = panel[panel["year_month"] < start].copy()
    valid = panel[
        (panel["year_month"] >= start)
        & (panel["year_month"] <= end)
    ].copy()

    if valid["year_month"].max() > "2022-08":
        raise RuntimeError("봉인 Test 기간이 포함됐습니다.")

    model = make_model(train[TARGET])
    model.fit(train[safe_features], train[TARGET])
    score = model.predict_proba(valid[safe_features])[:, 1]

    rows.append({
        "fold": name,
        "train_rows": len(train),
        "eval_rows": len(valid),
        "eval_positive_rate": float(valid[TARGET].mean()),
        "pr_auc": float(average_precision_score(valid[TARGET], score)),
        "roc_auc": float(roc_auc_score(valid[TARGET], score)),
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "rolling_2m_robustness_metrics.csv", index=False, encoding="utf-8-sig")

summary = {
    "target": TARGET,
    "definition": "t+1~t+2 리뷰 합계 <= 카테고리 보정 2개월 기대치의 50%",
    "test_data_used": False,
    "test_period_sealed": "2022-09~2023-01",
    "rows": int(len(panel)),
    "positive_count": int(panel[TARGET].sum()),
    "positive_rate": float(panel[TARGET].mean()),
    "mean_rolling_pr_auc": float(result["pr_auc"].mean()),
    "mean_rolling_roc_auc": float(result["roc_auc"].mean()),
}

(OUT / "rolling_2m_robustness_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(result.to_string(index=False))
print()
print(json.dumps(summary, ensure_ascii=False, indent=2))
