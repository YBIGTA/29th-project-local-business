"""Semantic-cluster TF-IDF reinforcement for the random grouped experiment.

The original raw-context experiment uses exact word/character n-grams.  This
variant embeds every review with a pretrained sentence encoder, groups similar
reviews by cosine geometry, and builds a product-month Semantic TF-IDF vector:

    semantic_tf(month, cluster) = mean soft membership of the month's reviews
    semantic_idf(cluster)       = log((N_train_months + 1) / (df + 1)) + 1

Each review contributes to its three nearest semantic clusters.  Consequently,
different expressions with similar meanings can reinforce the same feature.
Clusters, IDF values, classifiers, and adaptive gates are fitted with Train
data only.  Valid and Test labels never enter text fitting or gate fitting.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import run_random_group_context_fusion as base


OUT = Path(__file__).resolve().parent / "semantic_cluster_tfidf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CLUSTERS = 3
SOFTMAX_TEMPERATURE = float(os.environ.get("SEMANTIC_TEMPERATURE", "0.08"))
EMBED_BATCH_SIZE = int(os.environ.get("SEMANTIC_BATCH_SIZE", "128"))
MAX_CLUSTER_FIT_REVIEWS = int(os.environ.get("SEMANTIC_MAX_FIT_REVIEWS", "50000"))
MAX_RETAINED_CLUSTERS = int(os.environ.get("SEMANTIC_MAX_CLUSTERS", "256"))
PREVIOUS_FULL_CONTEXT_TEST_PR_AUC = 0.353351
MIN_COMPLEXITY_GAIN = 0.003

_review_embeddings: np.ndarray | None = None
_embedding_model: SentenceTransformer | None = None
_last_fitted: dict | None = None
_last_gate: dict | None = None
_cluster_fit_log: list[dict] = []


@dataclass
class AutoClusterBasis:
    cluster_centers_: np.ndarray
    discovered_cluster_count: int
    retained_cluster_count: int
    noise_rate: float
    min_cluster_size: int
    min_samples: int


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_month_index(panel: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Avoid concatenating huge raw documents; semantic aggregation uses rows."""
    reviews["semantic_review_id"] = np.arange(len(reviews), dtype=np.int64)
    counts = reviews.groupby("row_id", sort=False).size().rename(
        "text_review_count"
    ).reset_index()
    output = panel.merge(counts, on="row_id", how="left", validate="one_to_one")
    output["text_review_count"] = output["text_review_count"].fillna(0).astype(int)
    output["full_context"] = ""
    return output


def ensure_review_embeddings(reviews: pd.DataFrame) -> np.ndarray:
    global _review_embeddings, _embedding_model
    if _review_embeddings is not None:
        return _review_embeddings

    device = choose_device()
    print(
        f"臾몄옣 �꾨쿋��: {len(reviews):,}媛� 由щ럭 / {EMBEDDING_MODEL_NAME} / device={device}"
    )
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    _embedding_model.max_seq_length = 256
    _review_embeddings = _embedding_model.encode(
        reviews["text_norm"].tolist(),
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    return _review_embeddings


def normalized_centers(cluster_basis: AutoClusterBasis) -> np.ndarray:
    centers = cluster_basis.cluster_centers_.astype(np.float32, copy=False)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return centers / np.clip(norms, 1e-8, None)


def soft_cluster_membership(
    embeddings: np.ndarray,
    centers: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    output = np.zeros((len(embeddings), len(centers)), dtype=np.float32)
    for start in range(0, len(embeddings), chunk_size):
        stop = min(start + chunk_size, len(embeddings))
        similarity = embeddings[start:stop] @ centers.T
        top_k = min(TOP_K_CLUSTERS, len(centers))
        top_index = np.argpartition(similarity, -top_k, axis=1)[:, -top_k:]
        top_similarity = np.take_along_axis(similarity, top_index, axis=1)
        logits = top_similarity / SOFTMAX_TEMPERATURE
        logits -= logits.max(axis=1, keepdims=True)
        weight = np.exp(logits)
        weight /= np.clip(weight.sum(axis=1, keepdims=True), 1e-8, None)
        local = np.zeros_like(similarity, dtype=np.float32)
        np.put_along_axis(local, top_index, weight.astype(np.float32), axis=1)
        output[start:stop] = local
    return output


def select_reviews(reviews: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    return reviews[reviews["row_id"].isin(rows["row_id"])].copy()


def fit_cluster_basis(
    train_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[AutoClusterBasis, np.ndarray]:
    train_reviews = select_reviews(reviews, train_rows)
    balanced = base.capped_reviews(train_reviews)
    if len(balanced) > MAX_CLUSTER_FIT_REVIEWS:
        balanced = balanced.sample(MAX_CLUSTER_FIT_REVIEWS, random_state=base.SEED)
    fit_index = balanced["semantic_review_id"].to_numpy(dtype=np.int64)
    fit_embeddings = embeddings[fit_index]
    n_components = min(32, fit_embeddings.shape[1], len(fit_embeddings) - 1)
    pca = PCA(n_components=n_components, random_state=base.SEED)
    reduced = pca.fit_transform(fit_embeddings).astype(np.float32)
    reduced = normalize(reduced, norm="l2").astype(np.float32)

    min_cluster_size = max(60, int(round(np.sqrt(len(reduced)))))
    min_samples = max(10, min_cluster_size // 4)
    print(
        f"臾몃㎘ 援곗쭛 �먮룞 �먯깋: {len(fit_index):,}媛� 由щ럭 / "
        f"HDBSCAN min_cluster_size={min_cluster_size}"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=False,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(reduced)
    valid_labels = labels[labels >= 0]
    discovered = int(len(np.unique(valid_labels)))
    noise_rate = float((labels < 0).mean())
    if discovered < 3:
        print("�먮룞 援곗쭛�� �덈Т �곸뼱 min_cluster_size瑜� �덈컲�쇰줈 ��떠 �� 踰� �ы깘��")
        min_cluster_size = max(30, min_cluster_size // 2)
        min_samples = max(5, min_cluster_size // 4)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=False,
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(reduced)
        valid_labels = labels[labels >= 0]
        discovered = int(len(np.unique(valid_labels)))
        noise_rate = float((labels < 0).mean())
    if discovered < 2:
        raise RuntimeError(
            "HDBSCAN�� �섎� �덈뒗 臾몃㎘ 援곗쭛�� 李얠� 紐삵뻽�듬땲��. "
            "SEMANTIC_MAX_FIT_REVIEWS瑜� �섎젮 �ㅼ떆 �ㅽ뻾�� 二쇱꽭��."
        )

    label_counts = pd.Series(valid_labels).value_counts()
    retained_labels = label_counts.head(MAX_RETAINED_CLUSTERS).index.to_numpy()
    centers = []
    for label in retained_labels:
        center = fit_embeddings[labels == label].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-8)
        centers.append(center.astype(np.float32))
    centers_array = np.vstack(centers)
    basis = AutoClusterBasis(
        cluster_centers_=centers_array,
        discovered_cluster_count=discovered,
        retained_cluster_count=len(centers_array),
        noise_rate=noise_rate,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    _cluster_fit_log.append({
        "train_month_rows": int(len(train_rows)),
        "cluster_fit_reviews": int(len(fit_index)),
        "discovered_clusters": discovered,
        "retained_clusters": int(len(centers_array)),
        "noise_rate": noise_rate,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    })
    print(
        f"�먮룞 寃곗젙 寃곌낵: 諛쒓껄 {discovered}媛� / �ъ슜 {len(centers_array)}媛� / "
        f"noise {noise_rate:.1%}"
    )
    return basis, fit_index


def month_semantic_tf(
    rows: pd.DataFrame,
    selected_reviews: pd.DataFrame,
    embeddings: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_position = pd.Series(
        np.arange(len(rows), dtype=np.int64), index=rows["row_id"].to_numpy()
    )
    review_position = selected_reviews["row_id"].map(row_position)
    valid = review_position.notna().to_numpy()
    selected_reviews = selected_reviews.loc[valid]
    positions = review_position.loc[valid].to_numpy(dtype=np.int64)
    review_index = selected_reviews["semantic_review_id"].to_numpy(dtype=np.int64)

    totals = np.zeros((len(rows), len(centers)), dtype=np.float32)
    counts = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(review_index), 8192):
        stop = min(start + 8192, len(review_index))
        membership = soft_cluster_membership(
            embeddings[review_index[start:stop]], centers
        )
        np.add.at(totals, positions[start:stop], membership)
        np.add.at(counts, positions[start:stop], 1.0)
    semantic_tf = totals / np.clip(counts[:, None], 1.0, None)
    return semantic_tf, counts


def semantic_tfidf_features(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_basis: AutoClusterBasis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = normalized_centers(cluster_basis)
    train_reviews = select_reviews(reviews, train_rows)
    eval_reviews = select_reviews(reviews, eval_rows)
    train_tf, train_counts = month_semantic_tf(
        train_rows, train_reviews, embeddings, centers
    )
    eval_tf, eval_counts = month_semantic_tf(
        eval_rows, eval_reviews, embeddings, centers
    )

    document_frequency = (train_tf > 0).sum(axis=0)
    idf = np.log((len(train_rows) + 1) / (document_frequency + 1)) + 1.0
    x_train = normalize(train_tf * idf[None, :], norm="l2").astype(np.float32)
    x_eval = normalize(eval_tf * idf[None, :], norm="l2").astype(np.float32)

    train_extra = np.column_stack([
        train_tf.max(axis=1),
        -(train_tf * np.log(np.clip(train_tf, 1e-8, None))).sum(axis=1),
        np.log1p(train_counts),
    ]).astype(np.float32)
    eval_extra = np.column_stack([
        eval_tf.max(axis=1),
        -(eval_tf * np.log(np.clip(eval_tf, 1e-8, None))).sum(axis=1),
        np.log1p(eval_counts),
    ]).astype(np.float32)
    x_train = np.hstack([x_train, train_extra]).astype(np.float32)
    x_eval = np.hstack([x_eval, eval_extra]).astype(np.float32)
    return x_train, x_eval, idf.astype(np.float32), document_frequency


def fit_semantic_specific_reviews(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
    centers: np.ndarray,
    idf: np.ndarray,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    train_reviews = base.capped_reviews(select_reviews(reviews, train_rows))
    eval_reviews = select_reviews(reviews, eval_rows)
    train_index = train_reviews["semantic_review_id"].to_numpy(dtype=np.int64)
    eval_index = eval_reviews["semantic_review_id"].to_numpy(dtype=np.int64)

    train_membership = soft_cluster_membership(embeddings[train_index], centers)
    eval_membership = soft_cluster_membership(embeddings[eval_index], centers)
    train_cluster = normalize(train_membership * idf[None, :], norm="l2")
    eval_cluster = normalize(eval_membership * idf[None, :], norm="l2")
    x_train = np.hstack([embeddings[train_index], train_cluster]).astype(np.float32)
    x_eval = np.hstack([embeddings[eval_index], eval_cluster]).astype(np.float32)

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
    return aggregate, model, eval_reviews


def predict_semantic_experts(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    reviews: pd.DataFrame,
    structured_features: list[str],
) -> tuple[pd.DataFrame, dict]:
    global _last_fitted
    embeddings = ensure_review_embeddings(reviews)

    s45_model = base.make_s45(train_rows[base.TARGET])
    s45_model.fit(train_rows[structured_features], train_rows[base.TARGET])
    s45_score = s45_model.predict_proba(eval_rows[structured_features])[:, 1]

    cluster_basis, cluster_fit_index = fit_cluster_basis(
        train_rows, reviews, embeddings
    )
    x_train, x_eval, idf, document_frequency = semantic_tfidf_features(
        train_rows, eval_rows, reviews, embeddings, cluster_basis
    )
    context_model = base.make_text_classifier()
    context_model.fit(x_train, train_rows[base.TARGET])
    context_score = context_model.predict_proba(x_eval)[:, 1]

    centers = normalized_centers(cluster_basis)
    specific, specific_model, scored_reviews = fit_semantic_specific_reviews(
        train_rows, eval_rows, reviews, embeddings, centers, idf
    )

    prediction = eval_rows[["row_id", base.TARGET]].copy().reset_index(drop=True)
    prediction["s45_score"] = s45_score
    prediction["context_score"] = context_score
    prediction = prediction.merge(
        specific, on="row_id", how="left", validate="one_to_one"
    )
    fitted = {
        "s45_model": s45_model,
        "context_model": context_model,
        "specific_model": specific_model,
        "cluster_basis": cluster_basis,
        "idf": idf,
        "document_frequency": document_frequency,
        "cluster_fit_index": cluster_fit_index,
        "scored_reviews": scored_reviews,
        "structured_features": structured_features,
    }
    _last_fitted = fitted
    return prediction, fitted


def capture_gate(oof: pd.DataFrame) -> dict:
    global _last_gate
    gate = _original_fit_gate(oof)
    _last_gate = gate
    return gate


def write_cluster_audit(reviews: pd.DataFrame) -> None:
    if _last_fitted is None or _review_embeddings is None:
        return
    cluster_basis = _last_fitted["cluster_basis"]
    centers = normalized_centers(cluster_basis)
    fit_index = _last_fitted["cluster_fit_index"]
    fit_embeddings = _review_embeddings[fit_index]
    fit_reviews = reviews.set_index("semantic_review_id").loc[fit_index].reset_index()
    similarity = fit_embeddings @ centers.T
    hard_cluster = similarity.argmax(axis=1)
    records = []
    global_rate = float(fit_reviews[base.TARGET].mean())
    for cluster_id in range(len(centers)):
        member = np.flatnonzero(hard_cluster == cluster_id)
        if len(member) == 0:
            continue
        member_similarity = similarity[member, cluster_id]
        keep = member[np.argsort(-member_similarity)[:5]]
        cluster_rate = float(fit_reviews.iloc[member][base.TARGET].mean())
        for rank, position in enumerate(keep, start=1):
            review = fit_reviews.iloc[position]
            records.append({
                "cluster_id": cluster_id,
                "example_rank": rank,
                "cosine_similarity": float(similarity[position, cluster_id]),
                "cluster_review_count": int(len(member)),
                "cluster_positive_rate": cluster_rate,
                "global_positive_rate": global_rate,
                "semantic_idf": float(_last_fitted["idf"][cluster_id]),
                "parent_asin": review["parent_asin"],
                "year_month": review["year_month"],
                "text": str(review["text_norm"])[:500],
            })
    pd.DataFrame(records).to_csv(
        OUT / "semantic_cluster_examples.csv", index=False, encoding="utf-8-sig"
    )


def finalize_summary() -> None:
    summary_path = OUT / "randomness_correction_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUT / "random_adaptive_fusion_metrics.csv")
    valid_metrics = metrics[metrics["split"].eq("valid")]
    test_metrics = metrics[metrics["split"].eq("test")]
    full_name = "S45_plus_adaptive_full_context"
    specific_name = "S45_plus_adaptive_specific_review"
    both_name = "S45_plus_adaptive_both_texts"
    candidate_names = [full_name, specific_name, both_name]
    valid_scores = {
        name: float(valid_metrics.loc[
            valid_metrics["model_name"].eq(name), "pr_auc"
        ].iloc[0])
        for name in candidate_names
    }
    test_scores = {
        name: float(test_metrics.loc[
            test_metrics["model_name"].eq(name), "pr_auc"
        ].iloc[0])
        for name in candidate_names
    }
    best_single = max([full_name, specific_name], key=valid_scores.get)
    if valid_scores[both_name] - valid_scores[best_single] >= MIN_COMPLEXITY_GAIN:
        selected_model = both_name
    else:
        selected_model = best_single
    semantic_pr = test_scores[selected_model]
    s45_pr = float(test_metrics.loc[
        test_metrics["model_name"].eq("S45_structured_only"),
        "pr_auc",
    ].iloc[0])
    semantic_delta = semantic_pr - PREVIOUS_FULL_CONTEXT_TEST_PR_AUC
    if semantic_delta >= MIN_COMPLEXITY_GAIN:
        decision = "Semantic TF-IDF 蹂닿컯 紐⑤뜽 梨꾪깮 �꾨낫"
    elif semantic_delta > 0:
        decision = "媛쒖꽑 �� 0.003 誘몃쭔: 湲곗〈 �꾩껜 臾몃㎘ 紐⑤뜽 �좎�"
    else:
        decision = "湲곗〈 �꾩껜 臾몃㎘ 紐⑤뜽 �좎�"
    summary.update({
        "experiment": "2-4-03_Semantic_Cluster_TFIDF",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "full_context_definition": (
            "臾몄옣 �꾨쿋�� cosine 援곗쭛�� �곹뭹�� soft TF 횞 Train cluster IDF"
        ),
        "specific_review_definition": (
            "臾몄옣 �꾨쿋��+semantic cluster �좎궗�� 湲곕컲 由щ럭 �꾪뿕�꾩� top-3 attention"
        ),
        "clustering_algorithm": (
            "PCA 32李⑥썝 異뺤냼 �� HDBSCAN density clustering; 援곗쭛 �� �ъ쟾 吏��� �놁쓬"
        ),
        "semantic_cluster_count_final": (
            _last_fitted["cluster_basis"].retained_cluster_count
            if _last_fitted is not None else None
        ),
        "semantic_cluster_noise_rate_final": (
            _last_fitted["cluster_basis"].noise_rate
            if _last_fitted is not None else None
        ),
        "models_compared": candidate_names,
        "valid_pr_auc_by_model": valid_scores,
        "test_pr_auc_by_model": test_scores,
        "selected_semantic_model_from_valid": selected_model,
        "semantic_assignment": "媛� 由щ럭瑜� cosine similarity �곸쐞 3媛� 援곗쭛�� soft assignment",
        "semantic_tfidf": "�곹뭹�붾퀎 soft cluster TF 횞 Train �곹뭹�� 湲곗� cluster IDF",
        "previous_full_context_test_pr_auc": PREVIOUS_FULL_CONTEXT_TEST_PR_AUC,
        "test_s45_pr_auc": s45_pr,
        "test_final_model_pr_auc": semantic_pr,
        "test_pr_auc_delta_vs_s45": semantic_pr - s45_pr,
        "test_selected_semantic_model_pr_auc": semantic_pr,
        "semantic_minus_previous_full_context": semantic_delta,
        "selection_threshold": MIN_COMPLEXITY_GAIN,
        "decision": decision,
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "semantic_cluster_tfidf_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSemantic TF-IDF 理쒖쥌 鍮꾧탳:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def save_model_bundle() -> None:
    if _last_fitted is None or _last_gate is None:
        return
    bundle = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "clustering_algorithm": "PCA + HDBSCAN",
        "discovered_clusters": _last_fitted[
            "cluster_basis"
        ].discovered_cluster_count,
        "retained_clusters": _last_fitted[
            "cluster_basis"
        ].retained_cluster_count,
        "top_k_clusters": TOP_K_CLUSTERS,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "s45_model": _last_fitted["s45_model"],
        "context_model": _last_fitted["context_model"],
        "specific_model": _last_fitted["specific_model"],
        "cluster_basis": _last_fitted["cluster_basis"],
        "semantic_idf": _last_fitted["idf"],
        "structured_features": _last_fitted["structured_features"],
        "adaptive_gate": _last_gate,
    }
    joblib.dump(bundle, OUT / "semantic_cluster_model_bundle.joblib", compress=3)


_original_fit_gate = base.fit_adaptive_gate


def main() -> None:
    base.OUT = OUT
    base.EXPERIMENT_NAME = "2-4-03_Semantic_Cluster_TFIDF"
    base.FULL_CONTEXT_DEFINITION = (
        "臾몄옣 �꾨쿋�� cosine 援곗쭛�� �곹뭹�� soft TF 횞 Train cluster IDF"
    )
    base.SPECIFIC_REVIEW_DEFINITION = (
        "臾몄옣 �꾨쿋��+semantic cluster �좎궗�� 湲곕컲 由щ럭 �꾪뿕�꾩� top-3 attention"
    )
    base.ADAPTIVE_WEIGHTING_DEFINITION = (
        "Train OOF�먯꽌 semantic context �좊ː�꽷룻듅�� 由щ럭 利앷굅 吏묒쨷�꾩뿉 �곕씪 �됰퀎 媛�以묒튂 �숈뒿"
    )
    base.MODEL_SELECTION_RULE = (
        "�숈씪 �쒕뜡 �곹뭹 遺꾪븷�먯꽌 湲곗〈 �꾩껜 臾몃㎘ Test PR-AUC蹂대떎 0.003 �댁긽 �믪쓣 �뚮쭔 "
        "異붽� 蹂듭옟�깆쓣 �뺣떦�뷀븳��."
    )
    base.build_month_documents = build_month_index
    base.predict_base_experts = predict_semantic_experts
    base.fit_adaptive_gate = capture_gate
    OUT.mkdir(parents=True, exist_ok=True)
    base.main()

    pd.DataFrame(_cluster_fit_log).to_csv(
        OUT / "semantic_cluster_fit_log.csv", index=False, encoding="utf-8-sig"
    )

    reviews = pd.read_parquet(
        base.REVIEWS_PATH,
        columns=["parent_asin", "year_month", "review_datetime_utc", "text_norm"],
    )
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews["year_month"] = reviews["year_month"].astype(str)
    reviews["text_norm"] = reviews["text_norm"].fillna("").astype(str)
    reviews = reviews[reviews["text_norm"].str.len().ge(3)].copy()
    panel_keys = pd.read_parquet(
        OUT / "random_group_split_assignments.parquet",
        columns=["row_id", "parent_asin", "year_month", base.TARGET],
    )
    reviews = reviews.merge(
        panel_keys, on=base.KEYS, how="inner", validate="many_to_one"
    )
    reviews["semantic_review_id"] = np.arange(len(reviews), dtype=np.int64)
    write_cluster_audit(reviews)
    save_model_bundle()
    finalize_summary()


if __name__ == "__main__":
    main()