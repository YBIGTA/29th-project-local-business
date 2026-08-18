from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY = ROOT / "data/interim/product_month_activity.parquet"
PANEL = ROOT / "data/processed/product_month_review_volume_labeled.parquet"
OUT = ROOT / "02_final/3-1_feature_expansion/review_volume_decline"

START = pd.Period("2016-01", freq="M")
END = pd.Period("2023-02", freq="M")
TRAIN_END = "2021-12"
VALID_END = "2022-08"
CURRENT_MIN = 5
PAST3_MIN = 30
DROP_THRESHOLD = 0.50

def split_of(ym):
    if ym <= TRAIN_END:
        return "train"
    if ym <= VALID_END:
        return "valid"
    return "test"

def recall_at(y, score, fraction):
    n = max(1, int(len(score) * fraction))
    top = pd.Series(score).nlargest(n).index
    return float(pd.Series(y).iloc[top].sum() / max(1, pd.Series(y).sum()))

if not ACTIVITY.exists():
    raise SystemExit(f"없음: {ACTIVITY}")

OUT.mkdir(parents=True, exist_ok=True)
activity = pd.read_parquet(ACTIVITY).copy()
activity["parent_asin"] = activity["parent_asin"].astype(str)
activity["year_month"] = activity["year_month"].astype(str)
activity["period"] = pd.PeriodIndex(activity["year_month"], freq="M")

need = [
    "review_count", "avg_rating", "low_rating_count", "low_rating_ratio",
    "verified_purchase_ratio", "mean_helpful_vote", "text_available_ratio",
]
missing = set(need) - set(activity.columns)
if missing:
    raise SystemExit(f"activity 피처 누락: {sorted(missing)}")

lookup = {
    (row.parent_asin, row.period): row
    for row in activity[["parent_asin", "period"] + need].itertuples(index=False)
}
category = activity.groupby("period")["review_count"].sum().to_dict()

def get(asin, period, column, default=0.0):
    row = lookup.get((asin, period))
    return float(getattr(row, column)) if row is not None else default

records = []
candidates = []

for row in activity.itertuples(index=False):
    t = row.period
    if not (START <= t <= END):
        continue

    asin = row.parent_asin
    past3 = sum(get(asin, t - offset, "review_count") for offset in (0, 1, 2))
    cat_past3 = sum(float(category.get(t - offset, 0)) for offset in (0, 1, 2))
    next_count = get(asin, t + 1, "review_count")
    cat_next = float(category.get(t + 1, 0))
    expected_next = past3 / cat_past3 * cat_next if cat_past3 > 0 else np.nan

    candidates.append({
        "year_month": row.year_month,
        "current_review_count": float(row.review_count),
        "past_3m_review_count": past3,
        "next_review_count": next_count,
        "expected_next_review_count": expected_next,
    })

    if row.review_count < CURRENT_MIN or past3 < PAST3_MIN or not np.isfinite(expected_next) or expected_next <= 0:
        continue

    rec = {
        "parent_asin": asin,
        "year_month": row.year_month,
        "split": split_of(row.year_month),
        "review_count": float(row.review_count),
        "avg_rating": float(row.avg_rating),
        "low_rating_count": float(row.low_rating_count),
        "low_rating_ratio": float(row.low_rating_ratio),
        "verified_purchase_ratio": float(row.verified_purchase_ratio),
        "mean_helpful_vote": float(row.mean_helpful_vote),
        "text_available_ratio": float(row.text_available_ratio),
        "past_3m_review_count": past3,
        "next_review_count": next_count,
        "expected_next_review_count": expected_next,
        "next_review_volume_ratio": next_count / expected_next,
        "is_review_volume_drop": int(next_count <= DROP_THRESHOLD * expected_next),
    }

    for window in (1, 3, 6, 12):
        counts = [get(asin, t - offset, "review_count") for offset in range(1, window + 1)]
        lows = [get(asin, t - offset, "low_rating_count") for offset in range(1, window + 1)]
        total = sum(counts)
        rec[f"history_{window}m_review_count"] = total
        rec[f"history_{window}m_low_rating_count"] = sum(lows)
        rec[f"history_{window}m_low_rating_ratio"] = sum(lows) / total if total else np.nan
        rec[f"history_{window}m_active_month_count"] = float(sum(c > 0 for c in counts))
        rec[f"history_{window}m_coverage_ratio"] = float(sum(c > 0 for c in counts) / window)

        for col in ("avg_rating", "verified_purchase_ratio", "mean_helpful_vote", "text_available_ratio"):
            values = [get(asin, t - offset, col, np.nan) for offset in range(1, window + 1)]
            weighted = sum(v * c for v, c in zip(values, counts) if np.isfinite(v))
            rec[f"history_{window}m_{col}"] = weighted / total if total else np.nan

    rec["history_3m_vs_6m_low_rating_ratio_delta"] = (
        rec["history_3m_low_rating_ratio"] - rec["history_6m_low_rating_ratio"]
    )
    rec["history_3m_vs_12m_low_rating_ratio_delta"] = (
        rec["history_3m_low_rating_ratio"] - rec["history_12m_low_rating_ratio"]
    )
    records.append(rec)

panel = pd.DataFrame(records).sort_values(["year_month", "parent_asin"]).reset_index(drop=True)
panel.to_parquet(PANEL, index=False)

pd.DataFrame(candidates).to_csv(
    OUT / "label_candidates_pretest.csv", index=False, encoding="utf-8-sig"
)

summary = {
    "target": "is_review_volume_drop",
    "definition": "next_review_count <= 0.5 * category_adjusted_expected_next_review_count",
    "current_month_min_reviews": CURRENT_MIN,
    "past_3m_min_reviews": PAST3_MIN,
    "rows": int(len(panel)),
    "products": int(panel["parent_asin"].nunique()),
    "positive_count": int(panel["is_review_volume_drop"].sum()),
    "positive_rate": float(panel["is_review_volume_drop"].mean()),
    "test_rows_sealed": int((panel["split"] == "test").sum()),
    "test_data_used": False,
}
(OUT / "label_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

yearly = panel.assign(year=panel["year_month"].str[:4]).groupby("year").agg(
    rows=("parent_asin", "size"),
    positives=("is_review_volume_drop", "sum"),
    positive_rate=("is_review_volume_drop", "mean"),
).reset_index()
yearly.to_csv(OUT / "yearly_label_summary.csv", index=False, encoding="utf-8-sig")

features = [
    c for c in panel.columns
    if c not in {
        "parent_asin", "year_month", "split", "is_review_volume_drop",
        "next_review_count", "expected_next_review_count",
        "next_review_volume_ratio",
    }
]

train = panel[panel["split"] == "train"].copy()
valid = panel[panel["split"] == "valid"].copy()

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
    scale_pos_weight=(len(train) - train["is_review_volume_drop"].sum()) / train["is_review_volume_drop"].sum(),
)
model.fit(train[features], train["is_review_volume_drop"])
score = model.predict_proba(valid[features])[:, 1]

metrics = {
    **summary,
    "feature_count": len(features),
    "train_rows": int(len(train)),
    "valid_rows": int(len(valid)),
    "train_positive_rate": float(train["is_review_volume_drop"].mean()),
    "valid_positive_rate": float(valid["is_review_volume_drop"].mean()),
    "valid_pr_auc": float(average_precision_score(valid["is_review_volume_drop"], score)),
    "valid_roc_auc": float(roc_auc_score(valid["is_review_volume_drop"], score)),
    "valid_recall_at_5pct": recall_at(valid["is_review_volume_drop"], score, 0.05),
    "valid_recall_at_10pct": recall_at(valid["is_review_volume_drop"], score, 0.10),
}
(OUT / "structured_baseline_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)

pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).to_csv(
    OUT / "structured_feature_importance.csv", index=False, encoding="utf-8-sig"
)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
