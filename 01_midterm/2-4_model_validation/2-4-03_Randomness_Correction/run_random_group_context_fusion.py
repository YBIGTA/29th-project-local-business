"""Regime-balanced random grouped evaluation with adaptive text gating.

The split is random across years but grouped by parent_asin.  Therefore the
same product never appears in more than one of Train, Valid, and Test.  Seven
stratified group folds create an approximate 5:1:1 split while balancing the
year x target composition.

Two raw-text experts are trained independently:
1. Full-context expert: all t-month reviews concatenated into one document.
2. Specific-review expert: individual reviews receive weak product-month
   labels, then the most suspicious reviews are pooled at product-month level.

An adaptive gate is fitted on Train OOF predictions only.  It learns both the
specific-review share inside the text block and the total text weight relative
to S45.  Valid and Test labels never participate in gate fitting.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data/processed/product_month_review_volume_2m_labeled.parquet"
REVIEWS_PATH = ROOT / "data/interim/clean_reviews.parquet"
OUT = Path(__file__).resolve().parent
TARGET = "is_review_volume_drop_2m"
KEYS = ["parent_asin", "year_month"]
SEED = 42
MAX_REVIEWS_PER_MONTH = 8
MAX_TEXT_WEIGHT = 0.35


def make_s45(y: pd.Series) -> LGBMClassifier:
    positives = float(y.sum())
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
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=(len(y) - positives) / positives,
    )


def make_text_classifier() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=3e-5,
        penalty="l2",
        class_weight="balanced",
        max_iter=60,
        tol=1e-3,
        random_state=SEED,
    )


def assign_random_group_split(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["year"] = work["year_month"].str[:4]
    work["random_stratum"] = work["year"] + "_y" + work[TARGET].astype(str)
    work["random_fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    for fold_id, (_, held_index) in enumerate(
        splitter.split(work, y=work["random_stratum"], groups=work["parent_asin"])
    ):
        work.loc[work.index[held_index], "random_fold"] = fold_id
    work["random_split"] = np.select(
        [work["random_fold"].eq(0), work["random_fold"].eq(1)],
        ["test", "valid"],
        default="train",
    )
    if (work["random_fold"] < 0).any():
        raise RuntimeError("일부 행에 random fold가 배정되지 않았습니다.")
    product_sets = {
        split: set(work.loc[work["random_split"].eq(split), "parent_asin"])
        for split in ("train", "valid", "test")
    }
    if product_sets["train"] & product_sets["valid"]:
        raise RuntimeError("Train과 Valid에 같은 상품이 있습니다.")
    if product_sets["train"] & product_sets["test"]:
        raise RuntimeError("Train과 Test에 같은 상품이 있습니다.")
    if product_sets["valid"] & product_sets["test"]:
        raise RuntimeError("Valid와 Test에 같은 상품이 있습니다.")
    return work


def build_month_documents(panel: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    documents = reviews.groupby("row_id", sort=False).agg(
        full_context=("text_norm", " ".join),
        text_review_count=("text_norm", "size"),
    ).reset_index()
    output = panel.merge(documents, on="row_id", how="left", validate="one_to_one")
    output["full_context"] = output["full_context"].fillna("")
    output["text_review_count"] = output["text_review_count"].fillna(0).astype(int)
    return output


def fit_full_context(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
) -> tuple[np.ndarray, TfidfVectorizer, SGDClassifier]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_rows["full_context"])
    classifier = make_text_classifier()
    classifier.fit(x_train, train_rows[TARGET])
    score = classifier.predict_proba(vectorizer.transform(eval_rows["full_context"]))[:, 1]
    return score, vectorizer, classifier


def capped_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    return (
        reviews.sort_values(["row_id", "review_datetime_utc"])
        .groupby("row_id", sort=False)
        .tail(MAX_REVIEWS_PER_MONTH)
        .copy()
    )


def aggregate_specific_scores(scored: pd.DataFrame, eval_ids: pd.Series) -> pd.DataFrame:
    def top3_mean(values: pd.Series) -> float:
        return float(values.nlargest(min(3, len(values))).mean())

    agg = scored.groupby("row_id").agg(
        specific_mean=("specific_review_risk", "mean"),
        specific_max=("specific_review_risk", "max"),
        specific_top3=("specific_review_risk", top3_mean),
        specific_std=("specific_review_risk", "std"),
        specific_high_ratio=("specific_review_high", "mean"),
        specific_review_count=("specific_review_risk", "size"),
    ).reset_index()
    base = pd.DataFrame({"row_id": eval_ids.astype(int).to_numpy()})
    base = base.merge(agg, on="row_id", how="left")
    base["specific_mean"] = base["specific_mean"].fillna(0.5)
    base["specific_max"] = base["specific_max"].fillna(0.5)
    base["specific_top3"] = base["specific_top3"].fillna(0.5)
    base["specific_std"] = base["specific_std"].fillna(0.0)
    base["specific_high_ratio"] = base["specific_high_ratio"].fillna(0.0)
    base["specific_review_count"] = base["specific_review_count"].fillna(0.0)
    base["evidence_concentration"] = base["specific_top3"] - base["specific_mean"]
    return base


def fit_specific_reviews(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, TfidfVectorizer, SGDClassifier, pd.DataFrame]:
    train_reviews = reviews[reviews["row_id"].isin(train_rows["row_id"])].copy()
    eval_reviews = reviews[reviews["row_id"].isin(eval_rows["row_id"])].copy()
    train_reviews = capped_reviews(train_reviews)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train_reviews["text_norm"])
    per_month_count = train_reviews.groupby("row_id")["row_id"].transform("size")
    sample_weight = 1.0 / per_month_count.to_numpy()
    classifier = make_text_classifier()
    classifier.fit(
        x_train,
        train_reviews[TARGET],
        sample_weight=sample_weight,
    )

    eval_reviews["specific_review_risk"] = classifier.predict_proba(
        vectorizer.transform(eval_reviews["text_norm"])
    )[:, 1]
    eval_reviews["specific_review_high"] = (
        eval_reviews["specific_review_risk"] >= 0.70
    ).astype(int)
    aggregate = aggregate_specific_scores(eval_reviews, eval_rows["row_id"])
    return aggregate, vectorizer, classifier, eval_reviews


def predict_base_experts(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    structured_features: list[str],
) -> tuple[pd.DataFrame, dict]:
    s45_model = make_s45(train_rows[TARGET])
    s45_model.fit(train_rows[structured_features], train_rows[TARGET])
    s45_score = s45_model.predict_proba(eval_rows[structured_features])[:, 1]

    context_score, context_vectorizer, context_model = fit_full_context(train_rows, eval_rows)
    specific, specific_vectorizer, specific_model, scored_reviews = fit_specific_reviews(
        train_rows, eval_rows, reviews
    )

    prediction = eval_rows[["row_id", TARGET]].copy().reset_index(drop=True)
    prediction["s45_score"] = s45_score
    prediction["context_score"] = context_score
    prediction = prediction.merge(specific, on="row_id", how="left", validate="one_to_one")

    fitted = {
        "s45_model": s45_model,
        "context_vectorizer": context_vectorizer,
        "context_model": context_model,
        "specific_vectorizer": specific_vectorizer,
        "specific_model": specific_model,
        "scored_reviews": scored_reviews,
    }
    return prediction, fitted


GATE_COLUMNS = [
    "evidence_concentration",
    "specific_std",
    "specific_high_ratio",
    "log_review_count",
    "expert_disagreement",
    "text_confidence",
]


def gate_inputs(prediction: pd.DataFrame) -> np.ndarray:
    frame = pd.DataFrame(index=prediction.index)
    frame["evidence_concentration"] = prediction["evidence_concentration"]
    frame["specific_std"] = prediction["specific_std"]
    frame["specific_high_ratio"] = prediction["specific_high_ratio"]
    frame["log_review_count"] = np.log1p(prediction["specific_review_count"])
    frame["expert_disagreement"] = np.abs(
        prediction["context_score"] - prediction["specific_top3"]
    )
    frame["text_confidence"] = (
        np.abs(prediction["context_score"] - 0.5)
        + np.abs(prediction["specific_top3"] - 0.5)
    ) / 2
    return frame[GATE_COLUMNS].to_numpy(dtype=float)


def fit_adaptive_gate(oof: pd.DataFrame) -> dict:
    raw_x = gate_inputs(oof)
    mean = raw_x.mean(axis=0)
    std = raw_x.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (raw_x - mean) / std
    y = oof[TARGET].to_numpy(dtype=float)
    s45 = oof["s45_score"].to_numpy(dtype=float)
    context = oof["context_score"].to_numpy(dtype=float)
    specific = oof["specific_top3"].to_numpy(dtype=float)
    n_features = x.shape[1]

    def forward(params):
        spec_params = params[: n_features + 1]
        text_params = params[n_features + 1 :]
        specific_share = expit(spec_params[0] + x @ spec_params[1:])
        adaptive_text = (1 - specific_share) * context + specific_share * specific
        text_weight = MAX_TEXT_WEIGHT * expit(text_params[0] + x @ text_params[1:])
        final_score = (1 - text_weight) * s45 + text_weight * adaptive_text
        return np.clip(final_score, 1e-6, 1 - 1e-6), specific_share, text_weight

    def objective(params):
        score, _, _ = forward(params)
        penalty = 0.01 * float(np.square(params[1:]).sum())
        return log_loss(y, score) + penalty

    initial = np.zeros(2 * (n_features + 1), dtype=float)
    initial[n_features + 1] = -1.0
    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * len(initial),
        options={"maxiter": 300},
    )
    return {
        "params": fit.x,
        "mean": mean,
        "std": std,
        "optimization_success": bool(fit.success),
        "optimization_message": str(fit.message),
    }


def apply_adaptive_gate(prediction: pd.DataFrame, gate: dict) -> pd.DataFrame:
    output = prediction.copy()
    x = (gate_inputs(output) - gate["mean"]) / gate["std"]
    n_features = x.shape[1]
    params = gate["params"]
    spec_params = params[: n_features + 1]
    text_params = params[n_features + 1 :]
    output["specific_share_within_text"] = expit(
        spec_params[0] + x @ spec_params[1:]
    )
    output["adaptive_text_score"] = (
        (1 - output["specific_share_within_text"]) * output["context_score"]
        + output["specific_share_within_text"] * output["specific_top3"]
    )
    output["overall_text_weight"] = MAX_TEXT_WEIGHT * expit(
        text_params[0] + x @ text_params[1:]
    )
    output["adaptive_fusion_score"] = (
        (1 - output["overall_text_weight"]) * output["s45_score"]
        + output["overall_text_weight"] * output["adaptive_text_score"]
    )
    output["s45_plus_context_score"] = (
        (1 - output["overall_text_weight"]) * output["s45_score"]
        + output["overall_text_weight"] * output["context_score"]
    )
    output["s45_plus_specific_score"] = (
        (1 - output["overall_text_weight"]) * output["s45_score"]
        + output["overall_text_weight"] * output["specific_top3"]
    )
    return output


def metric_row(split: str, model_name: str, y, score, prediction=None) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    n10 = max(1, int(len(score) * 0.10))
    top = np.argsort(-score)[:n10]
    row = {
        "split": split,
        "model_name": model_name,
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_10pct": float(y[top].sum() / max(1, y.sum())),
        "precision_at_10pct": float(y[top].mean()),
    }
    if prediction is not None:
        row["mean_text_weight"] = float(prediction["overall_text_weight"].mean())
        row["mean_specific_share"] = float(
            prediction["specific_share_within_text"].mean()
        )
    return row


def evaluate_prediction(split: str, prediction: pd.DataFrame) -> list[dict]:
    y = prediction[TARGET].to_numpy()
    models = {
        "S45_structured_only": prediction["s45_score"],
        "full_context_text_only": prediction["context_score"],
        "specific_review_text_only": prediction["specific_top3"],
        "S45_plus_adaptive_full_context": prediction["s45_plus_context_score"],
        "S45_plus_adaptive_specific_review": prediction["s45_plus_specific_score"],
        "S45_plus_adaptive_both_texts": prediction["adaptive_fusion_score"],
    }
    rows = []
    for name, score in models.items():
        rows.append(metric_row(split, name, y, score, prediction))
    baseline = next(r["pr_auc"] for r in rows if r["model_name"] == "S45_structured_only")
    for row in rows:
        row["pr_auc_delta_vs_s45"] = row["pr_auc"] - baseline
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PANEL_PATH.exists() or not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"입력 확인 필요: {PANEL_PATH}, {REVIEWS_PATH}")

    panel = pd.read_parquet(PANEL_PATH).copy()
    panel["parent_asin"] = panel["parent_asin"].astype(str)
    panel["year_month"] = panel["year_month"].astype(str)
    panel = panel.reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=int)
    panel = assign_random_group_split(panel)

    future_columns = {
        "is_review_volume_drop",
        "is_review_volume_drop_2m",
        "next_review_count",
        "expected_next_review_count",
        "next_review_volume_ratio",
        "next_2m_review_count",
        "expected_next_2m_review_count",
    }
    non_features = {
        "parent_asin", "year_month", "split", "year", "random_stratum",
        "random_fold", "random_split", "row_id",
    } | future_columns
    structured_features = [
        column for column in panel.columns
        if column not in non_features
        and pd.api.types.is_numeric_dtype(panel[column])
    ]

    reviews = pd.read_parquet(
        REVIEWS_PATH,
        columns=[
            "parent_asin", "year_month", "review_datetime_utc", "text_norm"
        ],
    )
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews["year_month"] = reviews["year_month"].astype(str)
    reviews["text_norm"] = reviews["text_norm"].fillna("").astype(str)
    reviews = reviews[reviews["text_norm"].str.len().ge(3)].copy()
    reviews = reviews.merge(
        panel[["row_id", "parent_asin", "year_month", TARGET, "random_split"]],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )
    panel = build_month_documents(panel, reviews)

    balance = panel.groupby(["random_split", "year"]).agg(
        rows=("row_id", "size"),
        products=("parent_asin", "nunique"),
        positive_rate=(TARGET, "mean"),
    ).reset_index()
    balance.to_csv(OUT / "random_split_year_balance.csv", index=False, encoding="utf-8-sig")
    panel[["row_id", "parent_asin", "year_month", TARGET, "random_split", "random_fold"]].to_parquet(
        OUT / "random_group_split_assignments.parquet", index=False
    )

    train = panel[panel["random_split"].eq("train")].copy()
    valid = panel[panel["random_split"].eq("valid")].copy()
    test = panel[panel["random_split"].eq("test")].copy()
    print("랜덤 그룹 분할:")
    print(panel.groupby("random_split").agg(
        rows=("row_id", "size"),
        products=("parent_asin", "nunique"),
        positive_rate=(TARGET, "mean"),
    ).to_string())

    oof = pd.DataFrame(index=train.index)
    oof["row_id"] = train["row_id"]
    oof[TARGET] = train[TARGET]
    oof_columns = [
        "s45_score", "context_score", "specific_mean", "specific_max",
        "specific_top3", "specific_std", "specific_high_ratio",
        "specific_review_count", "evidence_concentration",
    ]
    for column in oof_columns:
        oof[column] = np.nan

    oof_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    for fold_id, (fit_position, held_position) in enumerate(
        oof_splitter.split(train, y=train[TARGET], groups=train["parent_asin"]), start=1
    ):
        fit_rows = train.iloc[fit_position].copy()
        held_rows = train.iloc[held_position].copy()
        print(f"OOF {fold_id}/5: 학습 {len(fit_rows):,}, 예측 {len(held_rows):,}")
        fold_prediction, _ = predict_base_experts(
            fit_rows, held_rows, reviews, structured_features
        )
        fold_prediction = fold_prediction.set_index("row_id")
        target_index = oof["row_id"].isin(held_rows["row_id"])
        row_ids = oof.loc[target_index, "row_id"]
        for column in oof_columns:
            oof.loc[target_index, column] = row_ids.map(fold_prediction[column])
        gc.collect()

    if oof[oof_columns].isna().any().any():
        raise RuntimeError("Train OOF 예측에 결측치가 있습니다.")
    gate = fit_adaptive_gate(oof.reset_index(drop=True))
    oof_gated = apply_adaptive_gate(oof.reset_index(drop=True), gate)

    print("Valid용 세 모델 학습")
    valid_prediction, _ = predict_base_experts(train, valid, reviews, structured_features)
    valid_prediction = apply_adaptive_gate(valid_prediction, gate)

    print("Test용 세 모델을 Train+Valid로 재학습")
    train_valid = pd.concat([train, valid], ignore_index=True)
    test_prediction, fitted_final = predict_base_experts(
        train_valid, test, reviews, structured_features
    )
    test_prediction = apply_adaptive_gate(test_prediction, gate)

    metrics = []
    metrics.extend(evaluate_prediction("train_oof", oof_gated))
    metrics.extend(evaluate_prediction("valid", valid_prediction))
    metrics.extend(evaluate_prediction("test", test_prediction))
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(
        OUT / "random_adaptive_fusion_metrics.csv", index=False, encoding="utf-8-sig"
    )

    test_output = test[["row_id", "parent_asin", "year_month", TARGET]].merge(
        test_prediction.drop(columns=[TARGET]), on="row_id", how="left", validate="one_to_one"
    )
    test_output.to_parquet(OUT / "random_test_predictions.parquet", index=False)

    top_months = test_output.nlargest(30, "adaptive_fusion_score")[[
        "row_id", "parent_asin", "year_month", TARGET,
        "s45_score", "context_score", "specific_top3",
        "specific_share_within_text", "overall_text_weight",
        "adaptive_fusion_score",
    ]]
    final_reviews = fitted_final["scored_reviews"].drop(
        columns=[TARGET], errors="ignore"
    ).copy()
    evidence = final_reviews.merge(top_months, on="row_id", how="inner")
    evidence = evidence.sort_values(
        ["adaptive_fusion_score", "specific_review_risk"], ascending=False
    ).groupby("row_id").head(3)
    evidence["text_norm"] = evidence["text_norm"].str.slice(0, 500)
    evidence[[
        "parent_asin_x", "year_month_x", TARGET,
        "s45_score", "context_score", "specific_top3",
        "specific_share_within_text", "overall_text_weight",
        "adaptive_fusion_score", "specific_review_risk", "text_norm",
    ]].rename(columns={
        "parent_asin_x": "parent_asin",
        "year_month_x": "year_month",
    }).to_csv(OUT / "random_test_top_evidence_reviews.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics_frame[metrics_frame["split"].eq("test")].copy()
    s45_pr = float(test_metrics.loc[
        test_metrics["model_name"].eq("S45_structured_only"), "pr_auc"
    ].iloc[0])
    combined_pr = float(test_metrics.loc[
        test_metrics["model_name"].eq("S45_plus_adaptive_both_texts"), "pr_auc"
    ].iloc[0])
    summary = {
        "experiment": "2-4-03_Randomness_Correction",
        "target": TARGET,
        "split_method": "7-fold year×label stratified random split grouped by parent_asin",
        "interpretation": "연도 환경을 섞은 미관측 상품 일반화 평가",
        "same_product_cross_split_count": 0,
        "structured_feature_count": len(structured_features),
        "full_context_definition": "t월 모든 리뷰 원문을 합친 word 1~2gram 문서 모델",
        "specific_review_definition": "리뷰별 char 3~5gram 위험도와 top-3 hard attention",
        "adaptive_weighting": "Train OOF에서 텍스트 신뢰도·불일치·증거 집중도에 따라 행별 가중치 학습",
        "maximum_text_weight": MAX_TEXT_WEIGHT,
        "gate_fit_success": gate["optimization_success"],
        "test_s45_pr_auc": s45_pr,
        "test_adaptive_fusion_pr_auc": combined_pr,
        "test_pr_auc_delta": combined_pr - s45_pr,
        "decision": "adaptive_text_pass" if combined_pr > s45_pr else "keep_s45",
    }
    (OUT / "randomness_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gate_json = {
        "gate_columns": GATE_COLUMNS,
        "maximum_text_weight": MAX_TEXT_WEIGHT,
        "optimization_success": gate["optimization_success"],
        "optimization_message": gate["optimization_message"],
        "parameters": gate["params"].tolist(),
        "normalization_mean": gate["mean"].tolist(),
        "normalization_std": gate["std"].tolist(),
    }
    (OUT / "adaptive_gate_parameters.json").write_text(
        json.dumps(gate_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nValid/Test 결과:")
    print(metrics_frame[metrics_frame["split"].isin(["valid", "test"])].to_string(index=False))
    print("\n최종 요약:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
