"""Hybrid raw-text + strong semantic-cluster experiment.

Raw TF-IDF is preserved for every review.  HDBSCAN clusters are used only
when a review is very close to a discovered cluster center.  Ambiguous/noise
reviews receive no cluster feature and remain represented by their original
words or character n-grams.

Three candidates are evaluated on the unchanged grouped random split:
1. S45 + hybrid full-month context
2. S45 + hybrid specific-review attention
3. S45 + OOF adaptive late fusion of both text experts
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import run_random_group_context_fusion as base
import run_semantic_cluster_tfidf_fusion as semantic


OUT = Path(__file__).resolve().parent / "hybrid_raw_strong_cluster"
STRONG_SIMILARITY_QUANTILE = 0.70
MIN_STRONG_COSINE = 0.60
MAX_CONTEXT_CLUSTER_SHARE = 0.35
SPECIFIC_CLUSTER_BLOCK_SCALE = 0.50
PREVIOUS_RAW_CONTEXT_TEST_PR_AUC = 0.353351
MIN_COMPLEXITY_GAIN = 0.003

_original_fit_cluster_basis = semantic.fit_cluster_basis


def build_raw_month_documents(
    panel: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    reviews["semantic_review_id"] = np.arange(len(reviews), dtype=np.int64)
    documents = reviews.groupby("row_id", sort=False).agg(
        full_context=("text_norm", " ".join),
        text_review_count=("text_norm", "size"),
    ).reset_index()
    output = panel.merge(documents, on="row_id", how="left", validate="one_to_one")
    output["full_context"] = output["full_context"].fillna("")
    output["text_review_count"] = (
        output["text_review_count"].fillna(0).astype(int)
    )
    return output


def fit_strong_cluster_basis(
    train_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
):
    basis, fit_index = _original_fit_cluster_basis(
        train_rows, reviews, embeddings
    )
    centers = semantic.normalized_centers(basis)
    similarity = embeddings[fit_index] @ centers.T
    nearest = similarity.argmax(axis=1)
    nearest_similarity = similarity.max(axis=1)
    thresholds = np.empty(len(centers), dtype=np.float32)
    strong_reference_counts = np.zeros(len(centers), dtype=np.int64)

    for cluster_id in range(len(centers)):
        values = nearest_similarity[nearest == cluster_id]
        if len(values) == 0:
            threshold = MIN_STRONG_COSINE
        else:
            threshold = max(
                MIN_STRONG_COSINE,
                float(np.quantile(values, STRONG_SIMILARITY_QUANTILE)),
            )
        thresholds[cluster_id] = threshold
        strong_reference_counts[cluster_id] = int((values >= threshold).sum())

    basis.cluster_similarity_thresholds_ = thresholds
    basis.strong_reference_counts_ = strong_reference_counts
    print(
        "강한 군집 기준: "
        f"cluster별 cosine {STRONG_SIMILARITY_QUANTILE:.0%} 분위수, "
        f"최저 {MIN_STRONG_COSINE:.2f}"
    )
    print(
        f"강한 군집 cosine 임계값 범위: "
        f"{thresholds.min():.3f}~{thresholds.max():.3f}"
    )
    return basis, fit_index


def strong_cluster_membership(
    embeddings: np.ndarray,
    centers: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    basis = semantic._active_cluster_basis
    thresholds = basis.cluster_similarity_thresholds_
    output = np.zeros((len(embeddings), len(centers)), dtype=np.float32)
    for start in range(0, len(embeddings), chunk_size):
        stop = min(start + chunk_size, len(embeddings))
        similarity = embeddings[start:stop] @ centers.T
        top_k = min(semantic.TOP_K_CLUSTERS, len(centers))
        top_index = np.argpartition(similarity, -top_k, axis=1)[:, -top_k:]
        top_similarity = np.take_along_axis(similarity, top_index, axis=1)
        top_threshold = thresholds[top_index]
        accepted = top_similarity >= top_threshold

        logits = top_similarity / semantic.SOFTMAX_TEMPERATURE
        logits = np.where(accepted, logits, -np.inf)
        has_cluster = accepted.any(axis=1)
        weight = np.zeros_like(top_similarity, dtype=np.float32)
        if has_cluster.any():
            valid_logits = logits[has_cluster]
            valid_logits -= np.max(valid_logits, axis=1, keepdims=True)
            valid_weight = np.exp(valid_logits)
            valid_weight /= np.clip(
                valid_weight.sum(axis=1, keepdims=True), 1e-8, None
            )
            weight[has_cluster] = valid_weight.astype(np.float32)

        local = np.zeros_like(similarity, dtype=np.float32)
        np.put_along_axis(local, top_index, weight, axis=1)
        output[start:stop] = local
    return output


def semantic_month_features(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_basis,
):
    semantic._active_cluster_basis = cluster_basis
    centers = semantic.normalized_centers(cluster_basis)
    train_reviews = semantic.select_reviews(reviews, train_rows)
    eval_reviews = semantic.select_reviews(reviews, eval_rows)
    train_tf, train_counts = semantic.month_semantic_tf(
        train_rows, train_reviews, embeddings, centers
    )
    eval_tf, eval_counts = semantic.month_semantic_tf(
        eval_rows, eval_reviews, embeddings, centers
    )

    document_frequency = (train_tf > 0).sum(axis=0)
    idf = np.log((len(train_rows) + 1) / (document_frequency + 1)) + 1.0
    x_train = normalize(train_tf * idf[None, :], norm="l2").astype(np.float32)
    x_eval = normalize(eval_tf * idf[None, :], norm="l2").astype(np.float32)

    train_coverage = train_tf.sum(axis=1).clip(0, 1)
    eval_coverage = eval_tf.sum(axis=1).clip(0, 1)
    train_extra = np.column_stack([
        train_coverage,
        train_tf.max(axis=1),
        np.log1p(train_counts),
    ]).astype(np.float32)
    eval_extra = np.column_stack([
        eval_coverage,
        eval_tf.max(axis=1),
        np.log1p(eval_counts),
    ]).astype(np.float32)
    x_train = np.hstack([x_train, train_extra]).astype(np.float32)
    x_eval = np.hstack([x_eval, eval_extra]).astype(np.float32)
    return (
        x_train,
        x_eval,
        idf.astype(np.float32),
        document_frequency,
        train_coverage.astype(np.float32),
        eval_coverage.astype(np.float32),
    )


def fit_hybrid_specific_reviews(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_basis,
    idf: np.ndarray,
):
    train_reviews = base.capped_reviews(
        semantic.select_reviews(reviews, train_rows)
    )
    eval_reviews = semantic.select_reviews(reviews, eval_rows)
    train_index = train_reviews["semantic_review_id"].to_numpy(dtype=np.int64)
    eval_index = eval_reviews["semantic_review_id"].to_numpy(dtype=np.int64)
    centers = semantic.normalized_centers(cluster_basis)
    semantic._active_cluster_basis = cluster_basis

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    raw_train = vectorizer.fit_transform(train_reviews["text_norm"])
    raw_eval = vectorizer.transform(eval_reviews["text_norm"])
    cluster_train = strong_cluster_membership(
        embeddings[train_index], centers
    ) * idf[None, :]
    cluster_eval = strong_cluster_membership(
        embeddings[eval_index], centers
    ) * idf[None, :]
    cluster_train = csr_matrix(normalize(cluster_train, norm="l2"))
    cluster_eval = csr_matrix(normalize(cluster_eval, norm="l2"))

    x_train = hstack([
        raw_train,
        cluster_train * SPECIFIC_CLUSTER_BLOCK_SCALE,
    ], format="csr")
    x_eval = hstack([
        raw_eval,
        cluster_eval * SPECIFIC_CLUSTER_BLOCK_SCALE,
    ], format="csr")

    per_month_count = train_reviews.groupby("row_id")["row_id"].transform("size")
    sample_weight = 1.0 / per_month_count.to_numpy(dtype=np.float32)
    model = base.make_text_classifier()
    model.fit(
        x_train,
        train_reviews[base.TARGET].to_numpy(),
        sample_weight=sample_weight,
    )
    eval_reviews["specific_review_risk"] = model.predict_proba(x_eval)[:, 1]
    eval_reviews["specific_review_high"] = (
        eval_reviews["specific_review_risk"] >= 0.70
    ).astype(int)
    aggregate = base.aggregate_specific_scores(eval_reviews, eval_rows["row_id"])
    strong_review_coverage = float((cluster_eval.getnnz(axis=1) > 0).mean())
    return aggregate, vectorizer, model, eval_reviews, strong_review_coverage


def predict_hybrid_experts(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    structured_features: list[str],
):
    embeddings = semantic.ensure_review_embeddings(reviews)

    s45_model = base.make_s45(train_rows[base.TARGET])
    s45_model.fit(train_rows[structured_features], train_rows[base.TARGET])
    s45_score = s45_model.predict_proba(eval_rows[structured_features])[:, 1]

    cluster_basis, cluster_fit_index = fit_strong_cluster_basis(
        train_rows, reviews, embeddings
    )
    (
        x_semantic_train,
        x_semantic_eval,
        idf,
        document_frequency,
        train_coverage,
        eval_coverage,
    ) = semantic_month_features(
        train_rows, eval_rows, reviews, embeddings, cluster_basis
    )

    raw_context_score, raw_context_vectorizer, raw_context_model = (
        base.fit_full_context(train_rows, eval_rows)
    )
    semantic_context_model = base.make_text_classifier()
    semantic_context_model.fit(x_semantic_train, train_rows[base.TARGET])
    semantic_context_score = semantic_context_model.predict_proba(
        x_semantic_eval
    )[:, 1]
    semantic_share = MAX_CONTEXT_CLUSTER_SHARE * eval_coverage
    hybrid_context_score = (
        (1 - semantic_share) * raw_context_score
        + semantic_share * semantic_context_score
    )

    (
        specific,
        specific_vectorizer,
        specific_model,
        scored_reviews,
        strong_review_coverage,
    ) = fit_hybrid_specific_reviews(
        train_rows,
        eval_rows,
        reviews,
        embeddings,
        cluster_basis,
        idf,
    )

    prediction = eval_rows[["row_id", base.TARGET]].copy().reset_index(drop=True)
    prediction["s45_score"] = s45_score
    prediction["context_score"] = hybrid_context_score
    prediction = prediction.merge(
        specific, on="row_id", how="left", validate="one_to_one"
    )
    fitted = {
        "s45_model": s45_model,
        "context_model": raw_context_model,
        "raw_context_vectorizer": raw_context_vectorizer,
        "raw_context_model": raw_context_model,
        "semantic_context_model": semantic_context_model,
        "specific_vectorizer": specific_vectorizer,
        "specific_model": specific_model,
        "cluster_basis": cluster_basis,
        "idf": idf,
        "document_frequency": document_frequency,
        "cluster_fit_index": cluster_fit_index,
        "scored_reviews": scored_reviews,
        "structured_features": structured_features,
        "mean_month_strong_cluster_coverage": float(eval_coverage.mean()),
        "mean_specific_strong_cluster_coverage": strong_review_coverage,
        "mean_semantic_share_in_context": float(semantic_share.mean()),
    }
    semantic._last_fitted = fitted
    return prediction, fitted


def save_hybrid_bundle() -> None:
    fitted = semantic._last_fitted
    gate = semantic._last_gate
    if fitted is None or gate is None:
        return
    bundle = {
        "method": "raw TF-IDF preserved + high-confidence HDBSCAN clusters only",
        "embedding_model_name": semantic.EMBEDDING_MODEL_NAME,
        "strong_similarity_quantile": STRONG_SIMILARITY_QUANTILE,
        "minimum_strong_cosine": MIN_STRONG_COSINE,
        "maximum_context_cluster_share": MAX_CONTEXT_CLUSTER_SHARE,
        "s45_model": fitted["s45_model"],
        "raw_context_vectorizer": fitted["raw_context_vectorizer"],
        "raw_context_model": fitted["raw_context_model"],
        "semantic_context_model": fitted["semantic_context_model"],
        "specific_vectorizer": fitted["specific_vectorizer"],
        "specific_model": fitted["specific_model"],
        "cluster_basis": fitted["cluster_basis"],
        "semantic_idf": fitted["idf"],
        "structured_features": fitted["structured_features"],
        "adaptive_gate": gate,
    }
    joblib.dump(bundle, OUT / "hybrid_raw_strong_cluster_model.joblib", compress=3)


def finalize_hybrid_summary() -> None:
    metrics = pd.read_csv(OUT / "random_adaptive_fusion_metrics.csv")
    valid = metrics[metrics["split"].eq("valid")]
    test = metrics[metrics["split"].eq("test")]
    candidates = [
        "S45_plus_adaptive_full_context",
        "S45_plus_adaptive_specific_review",
        "S45_plus_adaptive_both_texts",
    ]
    valid_scores = {
        name: float(valid.loc[valid["model_name"].eq(name), "pr_auc"].iloc[0])
        for name in candidates
    }
    test_scores = {
        name: float(test.loc[test["model_name"].eq(name), "pr_auc"].iloc[0])
        for name in candidates
    }
    best_single = max(candidates[:2], key=valid_scores.get)
    both = candidates[2]
    selected = (
        both
        if valid_scores[both] - valid_scores[best_single] >= MIN_COMPLEXITY_GAIN
        else best_single
    )
    selected_test = test_scores[selected]
    s45_test = float(test.loc[
        test["model_name"].eq("S45_structured_only"), "pr_auc"
    ].iloc[0])
    fitted = semantic._last_fitted
    previous_delta = selected_test - PREVIOUS_RAW_CONTEXT_TEST_PR_AUC
    if previous_delta >= MIN_COMPLEXITY_GAIN:
        decision = "하이브리드 강한 군집 모델 채택 후보"
    else:
        decision = "기존 전체 원문 문맥 모델 유지"
    summary = {
        "experiment": "2-4-03_Hybrid_Raw_Strong_Cluster",
        "target": base.TARGET,
        "raw_text_policy": "모든 리뷰의 원문 TF-IDF를 항상 보존",
        "cluster_policy": (
            "HDBSCAN 군집 중심과 cluster별 강한 cosine 임계값을 넘은 "
            "리뷰에만 semantic cluster 피처 추가"
        ),
        "noise_policy": "잡음·애매한 리뷰는 군집에 넣지 않고 원문으로만 사용",
        "strong_similarity_quantile": STRONG_SIMILARITY_QUANTILE,
        "minimum_strong_cosine": MIN_STRONG_COSINE,
        "maximum_context_cluster_share": MAX_CONTEXT_CLUSTER_SHARE,
        "final_cluster_count": (
            fitted["cluster_basis"].retained_cluster_count if fitted else None
        ),
        "hdbscan_noise_rate": (
            fitted["cluster_basis"].noise_rate if fitted else None
        ),
        "mean_month_strong_cluster_coverage": (
            fitted["mean_month_strong_cluster_coverage"] if fitted else None
        ),
        "mean_specific_strong_cluster_coverage": (
            fitted["mean_specific_strong_cluster_coverage"] if fitted else None
        ),
        "mean_semantic_share_in_context": (
            fitted["mean_semantic_share_in_context"] if fitted else None
        ),
        "valid_pr_auc_by_model": valid_scores,
        "test_pr_auc_by_model": test_scores,
        "selected_model_from_valid": selected,
        "test_s45_pr_auc": s45_test,
        "test_selected_model_pr_auc": selected_test,
        "test_delta_vs_s45": selected_test - s45_test,
        "previous_raw_context_test_pr_auc": PREVIOUS_RAW_CONTEXT_TEST_PR_AUC,
        "delta_vs_previous_raw_context": previous_delta,
        "selection_threshold": MIN_COMPLEXITY_GAIN,
        "decision": decision,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (OUT / "hybrid_raw_strong_cluster_summary.json").write_text(
        text, encoding="utf-8"
    )
    (OUT / "randomness_correction_summary.json").write_text(
        text, encoding="utf-8"
    )
    print("\n하이브리드 원문+강한 군집 최종 비교:")
    print(text)


def main() -> None:
    semantic.OUT = OUT
    semantic.fit_cluster_basis = fit_strong_cluster_basis
    semantic.soft_cluster_membership = strong_cluster_membership
    semantic.build_month_index = build_raw_month_documents
    semantic.predict_semantic_experts = predict_hybrid_experts
    semantic.save_model_bundle = save_hybrid_bundle
    semantic.finalize_summary = finalize_hybrid_summary
    semantic.main()


if __name__ == "__main__":
    main()