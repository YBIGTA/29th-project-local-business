from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data/processed/product_month_review_volume_2m_labeled.parquet"
REVIEWS_PATH = ROOT / "data/interim/clean_reviews.parquet"
OUT = Path(__file__).resolve().parent
TARGET = "is_review_volume_drop_2m"
KEYS = ["parent_asin", "year_month"]
MAX_REVIEW_TRAIN = 300000

FOLDS = [
    ("2019", "2018-01", "2019-01", "2019-12"),
    ("2020", "2019-01", "2020-01", "2020-12"),
    ("2021", "2020-01", "2021-01", "2021-12"),
    ("2022_01_08", "2021-01", "2022-01", "2022-08"),
]

def make_lgbm(y, small=False):
    return LGBMClassifier(
        objective="binary",
        n_estimators=350 if small else 500,
        learning_rate=0.03,
        num_leaves=8 if small else 15,
        min_child_samples=80 if small else 50,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_lambda=2.0 if small else 1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=(len(y) - y.sum()) / y.sum(),
    )

def sample_reviews(frame):
    if len(frame) <= MAX_REVIEW_TRAIN:
        return frame
    positive = frame[frame["is_low_rating"] == 1]
    negative = frame[frame["is_low_rating"] == 0]
    n_pos = min(len(positive), int(MAX_REVIEW_TRAIN * 0.35))
    n_neg = MAX_REVIEW_TRAIN - n_pos
    return pd.concat([
        positive.sample(n=n_pos, random_state=42),
        negative.sample(n=n_neg, random_state=42),
    ], ignore_index=True).sample(frac=1, random_state=42)

def top3_mean(series):
    return float(series.nlargest(min(3, len(series))).mean())

def make_context_features(item_panel, scored_reviews):
    scored_reviews = scored_reviews.copy()
    scored_reviews["period"] = pd.PeriodIndex(scored_reviews["year_month"], freq="M")

    monthly = scored_reviews.groupby(["parent_asin", "period"]).agg(
        raw_review_count=("text_context_risk", "size"),
        raw_context_mean=("text_context_risk", "mean"),
        raw_context_max=("text_context_risk", "max"),
        raw_context_top3=("text_context_risk", top3_mean),
        raw_context_high_count=("text_context_high", "sum"),
    ).reset_index()

    mismatch = scored_reviews[scored_reviews["rating"] >= 4].groupby(
        ["parent_asin", "period"]
    ).agg(
        high_rating_context_mean=("text_context_risk", "mean"),
        high_rating_context_top3=("text_context_risk", top3_mean),
        high_rating_context_count=("text_context_risk", "size"),
    ).reset_index()

    monthly = monthly.merge(
        mismatch, on=["parent_asin", "period"], how="left"
    ).fillna(0)

    raw_base = [
        "raw_review_count",
        "raw_context_mean",
        "raw_context_max",
        "raw_context_top3",
        "raw_context_high_count",
        "high_rating_context_mean",
        "high_rating_context_top3",
        "high_rating_context_count",
    ]

    output = item_panel[["parent_asin", "year_month", "period"]].copy()
    output = output.merge(
        monthly, on=["parent_asin", "period"], how="left"
    ).fillna(0)

    past = []
    for offset in (1, 2, 3):
        shifted = monthly[["parent_asin", "period"] + raw_base].copy()
        shifted["period"] = shifted["period"] + offset
        past.append(shifted)

    past = pd.concat(past, ignore_index=True)
    past = past.groupby(["parent_asin", "period"]).agg(
        raw_p3_review_count=("raw_review_count", "sum"),
        raw_p3_context_mean=("raw_context_mean", "mean"),
        raw_p3_context_max=("raw_context_max", "max"),
        raw_p3_context_top3=("raw_context_top3", "mean"),
        raw_p3_high_count=("raw_context_high_count", "sum"),
        raw_p3_high_rating_mean=("high_rating_context_mean", "mean"),
        raw_p3_high_rating_top3=("high_rating_context_top3", "mean"),
        raw_p3_high_rating_count=("high_rating_context_count", "sum"),
    ).reset_index()

    output = output.merge(
        past, on=["parent_asin", "period"], how="left"
    ).fillna(0)

    output["raw_context_mean_delta"] = (
        output["raw_context_mean"] - output["raw_p3_context_mean"]
    )
    output["raw_context_top3_delta"] = (
        output["raw_context_top3"] - output["raw_p3_context_top3"]
    )
    output["high_rating_context_delta"] = (
        output["high_rating_context_mean"]
        - output["raw_p3_high_rating_mean"]
    )

    raw_features = [
        c for c in output.columns
        if c not in ["parent_asin", "year_month", "period"]
    ]
    return output.drop(columns=["period"]), raw_features

panel = pd.read_parquet(PANEL_PATH).copy()
panel["parent_asin"] = panel["parent_asin"].astype(str)
panel["year_month"] = panel["year_month"].astype(str)
panel["period"] = pd.PeriodIndex(panel["year_month"], freq="M")

reviews = pd.read_parquet(
    REVIEWS_PATH,
    columns=["parent_asin", "year_month", "text_norm", "rating"]
).copy()
reviews["parent_asin"] = reviews["parent_asin"].astype(str)
reviews["year_month"] = reviews["year_month"].astype(str)
reviews["text_norm"] = reviews["text_norm"].fillna("").astype(str)
reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
reviews = reviews[
    reviews["parent_asin"].isin(panel["parent_asin"].unique())
    & reviews["text_norm"].str.len().gt(2)
    & reviews["rating"].notna()
    & (reviews["year_month"] >= "2015-01")
    & (reviews["year_month"] <= "2022-08")
].copy()
reviews["is_low_rating"] = (reviews["rating"] <= 2).astype(int)

structured_features = [
    c for c in panel.columns
    if c not in {
        "parent_asin", "year_month", "period", "split",
        "is_review_volume_drop",
        "is_review_volume_drop_2m",
        "next_review_count",
        "expected_next_review_count",
        "next_review_volume_ratio",
        "next_2m_review_count",
        "expected_next_2m_review_count",
    }
]

all_results = []
all_evidence = []

for fold, inner_start, eval_start, eval_end in FOLDS:
    core = panel[panel["year_month"] < inner_start].copy()
    inner = panel[
        (panel["year_month"] >= inner_start)
        & (panel["year_month"] < eval_start)
    ].copy()
    valid = panel[
        (panel["year_month"] >= eval_start)
        & (panel["year_month"] <= eval_end)
    ].copy()

    if valid["year_month"].max() > "2022-08":
        raise RuntimeError("봉인 Test 기간이 포함됐습니다.")

    review_train = reviews[reviews["year_month"] < inner_start].copy()
    review_scoring = reviews[reviews["year_month"] <= eval_end].copy()
    review_train = sample_reviews(review_train)

    print(f"\n===== {fold} raw-context 모델 =====")
    print(f"리뷰 문맥 학습: {len(review_train):,}건")
    print(f"상품월 core/inner/valid: {len(core):,} / {len(inner):,} / {len(valid):,}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=40000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x_review_train = vectorizer.fit_transform(review_train["text_norm"])
    review_model = SGDClassifier(
        loss="log_loss",
        alpha=0.00003,
        class_weight="balanced",
        max_iter=40,
        tol=1e-3,
        random_state=42,
    )
    review_model.fit(x_review_train, review_train["is_low_rating"])

    x_review_scoring = vectorizer.transform(review_scoring["text_norm"])
    review_scoring["text_context_risk"] = review_model.predict_proba(
        x_review_scoring
    )[:, 1]
    review_scoring["text_context_high"] = (
        review_scoring["text_context_risk"] >= 0.70
    ).astype(int)

    fold_items = pd.concat([core, inner, valid], ignore_index=True)
    raw_frame, raw_features = make_context_features(fold_items, review_scoring)

    fold_items = fold_items.merge(
        raw_frame, on=KEYS, how="left", validate="one_to_one"
    )
    fold_items[raw_features] = fold_items[raw_features].fillna(0)

    core = fold_items[fold_items["year_month"] < inner_start].copy()
    inner = fold_items[
        (fold_items["year_month"] >= inner_start)
        & (fold_items["year_month"] < eval_start)
    ].copy()
    valid = fold_items[
        (fold_items["year_month"] >= eval_start)
        & (fold_items["year_month"] <= eval_end)
    ].copy()

    structured_model = make_lgbm(core[TARGET], small=False)
    raw_model = make_lgbm(core[TARGET], small=True)

    structured_model.fit(core[structured_features], core[TARGET])
    raw_model.fit(core[raw_features], core[TARGET])

    inner_structured = structured_model.predict_proba(inner[structured_features])[:, 1]
    inner_raw = raw_model.predict_proba(inner[raw_features])[:, 1]

    weight_candidates = np.arange(0.0, 1.01, 0.1)
    inner_scores = {
        float(weight): average_precision_score(
            inner[TARGET],
            (1 - weight) * inner_structured + weight * inner_raw,
        )
        for weight in weight_candidates
    }
    best_weight = max(inner_scores, key=inner_scores.get)

    valid_structured = structured_model.predict_proba(valid[structured_features])[:, 1]
    valid_raw = raw_model.predict_proba(valid[raw_features])[:, 1]
    valid_fusion = (
        (1 - best_weight) * valid_structured
        + best_weight * valid_raw
    )

    all_results.append({
        "fold": fold,
        "core_train_end": core["year_month"].max(),
        "inner_period": f"{inner['year_month'].min()}~{inner['year_month'].max()}",
        "valid_period": f"{valid['year_month'].min()}~{valid['year_month'].max()}",
        "core_rows": len(core),
        "inner_rows": len(inner),
        "valid_rows": len(valid),
        "best_raw_context_weight": float(best_weight),
        "structured_pr_auc": float(average_precision_score(valid[TARGET], valid_structured)),
        "raw_context_pr_auc": float(average_precision_score(valid[TARGET], valid_raw)),
        "late_fusion_pr_auc": float(average_precision_score(valid[TARGET], valid_fusion)),
        "fusion_minus_structured": float(
            average_precision_score(valid[TARGET], valid_fusion)
            - average_precision_score(valid[TARGET], valid_structured)
        ),
        "structured_roc_auc": float(roc_auc_score(valid[TARGET], valid_structured)),
        "late_fusion_roc_auc": float(roc_auc_score(valid[TARGET], valid_fusion)),
    })

    valid_evidence = valid[[
        "parent_asin", "year_month", TARGET
    ]].copy()
    valid_evidence["structured_score"] = valid_structured
    valid_evidence["raw_context_score"] = valid_raw
    valid_evidence["fusion_score"] = valid_fusion
    valid_evidence = valid_evidence.nlargest(15, "fusion_score")

    evidence_reviews = review_scoring.merge(
        valid_evidence[["parent_asin", "year_month", "fusion_score"]],
        on=KEYS,
        how="inner",
    )
    evidence_reviews = evidence_reviews.sort_values(
        ["fusion_score", "text_context_risk"],
        ascending=False,
    ).groupby(KEYS).head(3)

    evidence_reviews["fold"] = fold
    evidence_reviews["rating_text_mismatch"] = (
        evidence_reviews["rating"] >= 4
    ).astype(int)
    evidence_reviews["text_norm"] = evidence_reviews["text_norm"].str.slice(0, 450)

    all_evidence.append(evidence_reviews[[
        "fold", "parent_asin", "year_month", "rating",
        "rating_text_mismatch", "text_context_risk", "fusion_score",
        "text_norm",
    ]])

result = pd.DataFrame(all_results)
result.to_csv(
    OUT / "raw_context_late_fusion_rolling_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

evidence = pd.concat(all_evidence, ignore_index=True)
evidence.to_csv(
    OUT / "raw_context_top_evidence_reviews.csv",
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "target": TARGET,
    "test_data_used": False,
    "test_period_sealed": "2022-09~2023-01",
    "raw_text_method": (
        "TF-IDF 원문 문맥 점수 + 리뷰별 top-3 hard attention "
        "+ 평점-텍스트 불일치 + 3개월 변화량"
    ),
    "mean_structured_pr_auc": float(result["structured_pr_auc"].mean()),
    "mean_raw_context_pr_auc": float(result["raw_context_pr_auc"].mean()),
    "mean_late_fusion_pr_auc": float(result["late_fusion_pr_auc"].mean()),
    "mean_fusion_minus_structured": float(result["fusion_minus_structured"].mean()),
    "fusion_wins": int((result["fusion_minus_structured"] > 0).sum()),
    "decision": (
        "raw-context 후속 후보"
        if result["fusion_minus_structured"].mean() > 0
        else "raw-context 제외"
    ),
}

(OUT / "raw_context_late_fusion_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print()
print(result.to_string(index=False))
print()
print(json.dumps(summary, ensure_ascii=False, indent=2))
