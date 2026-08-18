"""Train/Valid-only phrase-cluster burst experiment for review-volume decline.

This experiment deliberately does NOT read or score the held-out random test
fold.  It keeps the existing S45 + raw full-context expert, then trains a
separate product-month expert from stable semantic complaint clusters.  All
vectorizers, encoder adaptation, graph clusters, cluster thresholds, IDF,
history statistics, and gates are fitted inside the relevant training fold.

Main idea
---------
* A review is split into short clauses instead of being embedded as one vector.
* A phrase becomes a graph node only when dense cosine neighbours and sparse
  TF-IDF neighbours agree.  Leiden chooses the number of communities.
* Bootstrapped graph reruns retain only stable communities.  Unassigned phrases
  stay available to the original raw TF-IDF model; they are never force-fit.
* Cluster shares are converted to causal product-history novelty/change/burst
  features and used as a separate LightGBM expert.

Fast first run (recommended):
    PHRASE_DOMAIN_ADAPT=0 PHRASE_MAX_CLUSTER_FIT=40000 python run_phrase_dualview_burst_fusion.py

Rigorous domain-adaptation ablation (slow: repeated in each OOF fold):
    PHRASE_DOMAIN_ADAPT=1 PHRASE_ADAPT_MAX=60000 python run_phrase_dualview_burst_fusion.py
"""
from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from pynndescent import NNDescent
from scipy.optimize import minimize
from scipy.special import expit
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

import run_random_group_context_fusion as base


TARGET = base.TARGET
KEYS = base.KEYS
SEED = base.SEED
MAX_TEXT_WEIGHT = base.MAX_TEXT_WEIGHT

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOMAIN_ADAPT = os.getenv("PHRASE_DOMAIN_ADAPT", "0") == "1"
ADAPT_MAX_PHRASES = int(os.getenv("PHRASE_ADAPT_MAX", "60000"))
MAX_CLUSTER_FIT = int(os.getenv("PHRASE_MAX_CLUSTER_FIT", "40000"))
MAX_RETAINED_CLUSTERS = int(os.getenv("PHRASE_MAX_RETAINED_CLUSTERS", "40"))
EMBED_BATCH_SIZE = int(os.getenv("PHRASE_EMBED_BATCH", "128"))
K_NEIGHBORS = int(os.getenv("PHRASE_K_NEIGHBORS", "30"))
LEIDEN_RESOLUTION = float(os.getenv("PHRASE_LEIDEN_RESOLUTION", "1.0"))
BOOTSTRAPS = int(os.getenv("PHRASE_BOOTSTRAPS", "3"))
MIN_STABILITY = float(os.getenv("PHRASE_MIN_STABILITY", "0.50"))
MIN_CLUSTER_SIZE = int(os.getenv("PHRASE_MIN_CLUSTER_SIZE", "30"))
MIN_DENSE_COS = float(os.getenv("PHRASE_MIN_DENSE_COS", "0.45"))
MIN_SPARSE_COS = float(os.getenv("PHRASE_MIN_SPARSE_COS", "0.10"))
COMPLEXITY_THRESHOLD = 0.003
OUT = Path(__file__).resolve().parent / (
    "phrase_dualview_burst_dapt" if DOMAIN_ADAPT else "phrase_dualview_burst"
)


@dataclass
class ClusterBasis:
    dense_centers: np.ndarray
    sparse_centers: np.ndarray
    dense_threshold: np.ndarray
    sparse_threshold: np.ndarray
    cluster_ids: list[int]
    cluster_sizes: list[int]
    stability: list[float]
    fit_phrase_count: int
    raw_cluster_count: int


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_phrases(text: str) -> list[str]:
    """Keep meaningful complaint-sized clauses without pretending every comma is a boundary."""
    text = str(text).strip().lower()
    if not text:
        return []
    parts = re.split(r"(?:[.!?;]+|\bbut\b|\bhowever\b|\balthough\b|\byet\b)", text)
    cleaned = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" ,:-")
        tokens = part.split()
        if 3 <= len(tokens) <= 80:
            cleaned.append(part)
    return cleaned[:3]


def build_phrase_table(reviews: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for review_idx, row in reviews.reset_index(drop=True).iterrows():
        for phrase in split_phrases(row["text_norm"]):
            records.append({
                "review_idx": review_idx,
                "row_id": int(row["row_id"]),
                "parent_asin": str(row["parent_asin"]),
                "year_month": str(row["year_month"]),
                "rating": float(row["rating"]),
                "phrase": phrase,
            })
    phrases = pd.DataFrame(records)
    if phrases.empty:
        raise RuntimeError("문장/절 분리 뒤 남은 텍스트가 없습니다.")
    phrases["phrase_id"] = np.arange(len(phrases), dtype=np.int64)
    return phrases


def maybe_domain_adapt(train_texts: list[str], fold_tag: str) -> SentenceTransformer:
    device = choose_device()
    model = SentenceTransformer(ENCODER_NAME, device=device)
    model.max_seq_length = 128
    if not DOMAIN_ADAPT:
        return model
    sample = train_texts[:]
    if len(sample) > ADAPT_MAX_PHRASES:
        rng = np.random.default_rng(SEED + abs(hash(fold_tag)) % 10000)
        sample = rng.choice(sample, size=ADAPT_MAX_PHRASES, replace=False).tolist()
    print(f"[{fold_tag}] Train 문장만으로 unsupervised SimCSE 적응: {len(sample):,}개")
    examples = [InputExample(texts=[text, text]) for text in sample]
    loader = DataLoader(examples, shuffle=True, batch_size=32, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    try:
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=1,
            warmup_steps=max(10, len(loader) // 10),
            show_progress_bar=True,
        )
    except Exception as exc:  # sentence-transformers v5 changed the legacy trainer in some installs
        raise RuntimeError(
            "SimCSE 적응 단계가 실행되지 않았습니다. 먼저 PHRASE_DOMAIN_ADAPT=0으로 "
            "기본 군집 ablation을 끝내거나 sentence-transformers 3.x를 설치하세요."
        ) from exc
    return model


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)


def balanced_phrase_sample(phrases: pd.DataFrame, train_ids: set[int], fold_tag: str) -> pd.DataFrame:
    train = phrases[phrases["row_id"].isin(train_ids)].copy()
    # No high-volume product-month is allowed to dominate the geometry.
    train = train.groupby("row_id", group_keys=False, sort=False).head(6)
    if len(train) > MAX_CLUSTER_FIT:
        train = train.sample(MAX_CLUSTER_FIT, random_state=SEED + abs(hash(fold_tag)) % 10000)
    return train.reset_index(drop=True)


def dual_view_edges(dense: np.ndarray, sparse) -> tuple[list[tuple[int, int]], list[float]]:
    """Approximate dense kNN, then retain only pairs with TF-IDF agreement."""
    index = NNDescent(
        dense,
        n_neighbors=K_NEIGHBORS + 1,
        metric="cosine",
        random_state=SEED,
        n_jobs=-1,
    )
    neighbours, distances = index.neighbor_graph
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(dense)):
        for j, dist in zip(neighbours[i, 1:], distances[i, 1:]):
            j = int(j)
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in seen:
                continue
            dense_cos = float(1.0 - dist)
            if dense_cos < MIN_DENSE_COS:
                continue
            sparse_cos = float(sparse[i].multiply(sparse[j]).sum())
            if sparse_cos < MIN_SPARSE_COS:
                continue
            seen.add(pair)
            edges.append(pair)
            weights.append((dense_cos + sparse_cos) / 2.0)
    return edges, weights


def leiden_labels(n_nodes: int, edges: list[tuple[int, int]], weights: list[float], seed: int) -> np.ndarray:
    if not edges:
        return np.full(n_nodes, -1, dtype=int)
    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=LEIDEN_RESOLUTION,
        seed=seed,
    )
    labels = np.asarray(partition.membership, dtype=int)
    counts = pd.Series(labels).value_counts()
    tiny = set(counts[counts < MIN_CLUSTER_SIZE].index.astype(int))
    labels[np.isin(labels, list(tiny))] = -1
    return labels


def bootstrap_stability(dense: np.ndarray, sparse, base_labels: np.ndarray) -> dict[int, float]:
    """Mean best Jaccard overlap of each base community in bootstrap reruns."""
    retained = [int(x) for x in np.unique(base_labels) if x >= 0]
    scores = {cluster: [] for cluster in retained}
    rng = np.random.default_rng(SEED + 91)
    for boot in range(BOOTSTRAPS):
        take = np.sort(rng.choice(len(dense), size=int(0.80 * len(dense)), replace=False))
        edges, weights = dual_view_edges(dense[take], sparse[take])
        labels = leiden_labels(len(take), edges, weights, SEED + 100 + boot)
        for cluster in retained:
            # Both partitions are indexed in this bootstrap's local coordinates.
            original = set(np.flatnonzero(base_labels[take] == cluster))
            if len(original) < MIN_CLUSTER_SIZE:
                scores[cluster].append(0.0)
                continue
            best = 0.0
            for candidate in np.unique(labels):
                if candidate < 0:
                    continue
                other = set(np.flatnonzero(labels == candidate))
                overlap = len(original & other) / max(1, len(original | other))
                best = max(best, overlap)
            scores[cluster].append(best)
    return {cluster: float(np.mean(values)) for cluster, values in scores.items()}


def fit_cluster_basis(train_phrases: pd.DataFrame, fold_tag: str) -> tuple[ClusterBasis, TfidfVectorizer, SentenceTransformer, pd.DataFrame]:
    sample = balanced_phrase_sample(train_phrases, set(train_phrases["row_id"]), fold_tag)
    model = maybe_domain_adapt(sample["phrase"].tolist(), fold_tag)
    dense = embed(model, sample["phrase"].tolist())
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.98,
        max_features=50000, sublinear_tf=True, dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(sample["phrase"])
    print(f"[{fold_tag}] 이중 관점 Leiden 그래프: {len(sample):,}개 문장/절")
    edges, weights = dual_view_edges(dense, sparse)
    labels = leiden_labels(len(sample), edges, weights, SEED)
    stability = bootstrap_stability(dense, sparse, labels)
    counts = pd.Series(labels[labels >= 0]).value_counts()
    keep = [
        int(cluster) for cluster in counts.index
        if stability.get(int(cluster), 0.0) >= MIN_STABILITY
    ][:MAX_RETAINED_CLUSTERS]
    if len(keep) < 2:
        raise RuntimeError(
            "안정 군집이 2개 미만입니다. PHRASE_MIN_DENSE_COS/PHRASE_MIN_SPARSE_COS를 "
            "낮추거나 PHRASE_MAX_CLUSTER_FIT을 늘리세요."
        )
    dense_centers, sparse_centers, dense_threshold, sparse_threshold = [], [], [], []
    for cluster in keep:
        member = np.flatnonzero(labels == cluster)
        dc = normalize(dense[member].mean(axis=0, keepdims=True))[0]
        sc = normalize(np.asarray(sparse[member].mean(axis=0)), norm="l2")[0]
        d_sim = dense[member] @ dc
        s_sim = np.asarray(sparse[member] @ sc).ravel()
        dense_centers.append(dc)
        sparse_centers.append(sc)
        dense_threshold.append(float(np.quantile(d_sim, 0.10)))
        sparse_threshold.append(float(np.quantile(s_sim, 0.10)))
    basis = ClusterBasis(
        dense_centers=np.vstack(dense_centers).astype(np.float32),
        sparse_centers=np.vstack(sparse_centers).astype(np.float32),
        dense_threshold=np.asarray(dense_threshold, dtype=np.float32),
        sparse_threshold=np.asarray(sparse_threshold, dtype=np.float32),
        cluster_ids=keep,
        cluster_sizes=[int(counts[c]) for c in keep],
        stability=[float(stability[c]) for c in keep],
        fit_phrase_count=len(sample),
        raw_cluster_count=int(len(counts)),
    )
    sample["raw_cluster"] = labels
    return basis, vectorizer, model, sample


def assign_stable_clusters(phrases: pd.DataFrame, basis: ClusterBasis, vectorizer: TfidfVectorizer, encoder: SentenceTransformer) -> pd.DataFrame:
    out = phrases.copy()
    dense = embed(encoder, out["phrase"].tolist())
    sparse = vectorizer.transform(out["phrase"])
    dense_sim = dense @ basis.dense_centers.T
    sparse_sim = sparse @ basis.sparse_centers.T
    combined = (dense_sim + sparse_sim) / 2.0
    chosen = combined.argmax(axis=1)
    row = np.arange(len(out))
    strong = (
        dense_sim[row, chosen] >= basis.dense_threshold[chosen]
    ) & (
        sparse_sim[row, chosen] >= basis.sparse_threshold[chosen])
    out["cluster_pos"] = np.where(strong, chosen, -1).astype(int)
    out["dense_cosine"] = dense_sim[row, chosen]
    out["sparse_cosine"] = sparse_sim[row, chosen]
    return out


def make_cluster_features(rows: pd.DataFrame, assigned: pd.DataFrame, split_tag: str) -> pd.DataFrame:
    k = int(assigned.loc[assigned["cluster_pos"] >= 0, "cluster_pos"].max() + 1) if (assigned["cluster_pos"] >= 0).any() else 0
    if k < 2:
        raise RuntimeError("상품월 피처를 만들 안정 군집이 부족합니다.")
    base_rows = rows[["row_id", "parent_asin", "year_month"]].copy().sort_values(["parent_asin", "year_month"])
    all_phrase_count = assigned.groupby("row_id").size().rename("phrase_count")
    strong = assigned[assigned["cluster_pos"] >= 0].copy()
    pivot_count = pd.crosstab(strong["row_id"], strong["cluster_pos"]).reindex(columns=range(k), fill_value=0)
    feature = base_rows.set_index("row_id")
    feature["phrase_count"] = all_phrase_count
    feature["phrase_count"] = feature["phrase_count"].fillna(0.0)
    for c in range(k):
        count = pivot_count[c] if c in pivot_count else pd.Series(dtype=float)
        feature[f"cluster_{c}_share"] = count / feature["phrase_count"].clip(lower=1)
        feature[f"cluster_{c}_count"] = count.reindex(feature.index).fillna(0.0)
        # distinct-review count: multiple phrases from one review should not look like consensus.
        tmp = strong[strong["cluster_pos"].eq(c)].groupby("row_id")["review_idx"].nunique()
        feature[f"cluster_{c}_review_n"] = tmp.reindex(feature.index).fillna(0.0)
        mean_rating = strong[strong["cluster_pos"].eq(c)].groupby("row_id")["rating"].mean()
        low_ratio = strong.assign(low=(strong["rating"] <= 2).astype(float)).query("cluster_pos == @c").groupby("row_id")["low"].mean()
        feature[f"cluster_{c}_mean_rating"] = mean_rating.reindex(feature.index).fillna(3.0)
        feature[f"cluster_{c}_low_ratio"] = low_ratio.reindex(feature.index).fillna(0.0)
    assigned_count = strong.groupby("row_id").size().reindex(feature.index).fillna(0.0)
    feature["stable_cluster_coverage"] = assigned_count / feature["phrase_count"].clip(lower=1)
    feature["unassigned_phrase_ratio"] = 1.0 - feature["stable_cluster_coverage"]
    shares = feature[[f"cluster_{c}_share" for c in range(k)]].to_numpy(dtype=float)
    feature["cluster_entropy"] = -(shares * np.log(np.clip(shares, 1e-8, None))).sum(axis=1)
    feature["cluster_top_share"] = shares.max(axis=1)

    # Causal per-product history; the current month's value never enters its own history.
    for c in range(k):
        col = f"cluster_{c}_share"
        group = feature.groupby("parent_asin", sort=False)[col]
        p3 = group.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
        p6 = group.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean()).fillna(0.0)
        previous = group.transform(lambda s: s.shift(1).expanding().max()).fillna(0.0)
        feature[f"cluster_{c}_delta_p3"] = feature[col] - p3
        feature[f"cluster_{c}_delta_p6"] = feature[col] - p6
        feature[f"cluster_{c}_first_seen"] = ((feature[col] > 0) & (previous <= 0)).astype(float)
        n = feature["phrase_count"].clip(lower=1).to_numpy(dtype=float)
        observed = feature[f"cluster_{c}_count"].to_numpy(dtype=float)
        expected = np.clip(p3.to_numpy(dtype=float), 1e-5, 1 - 1e-5)
        rate = observed / n
        llr = 2 * (observed * np.log(np.clip(rate / expected, 1e-8, None)) + (n - observed) * np.log(np.clip((1 - rate) / (1 - expected), 1e-8, None)))
        feature[f"cluster_{c}_burst"] = np.where(rate > expected, np.sqrt(np.clip(llr, 0, None)), 0.0)

    # Appliances is a single category. This is therefore the category-wide past expectation.
    feature["month_order"] = pd.PeriodIndex(feature["year_month"], freq="M").astype(int)
    for c in range(k):
        col = f"cluster_{c}_share"
        month_mean = feature.groupby("year_month")[col].mean().sort_index()
        prior = month_mean.shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
        feature[f"cluster_{c}_category_excess"] = feature[col] - feature["year_month"].map(prior).fillna(0.0)
    feature = feature.drop(columns=["month_order"])
    feature = feature.replace([np.inf, -np.inf], 0.0).fillna(0.0).reset_index()
    feature["feature_split"] = split_tag
    return feature


def cluster_expert(train_rows: pd.DataFrame, eval_rows: pd.DataFrame, phrases: pd.DataFrame, fold_tag: str) -> tuple[np.ndarray, dict, pd.DataFrame]:
    train_phrase = phrases[phrases["row_id"].isin(train_rows["row_id"])].copy()
    eval_phrase = phrases[phrases["row_id"].isin(eval_rows["row_id"])].copy()
    basis, vectorizer, encoder, sample = fit_cluster_basis(train_phrase, fold_tag)
    assigned_train = assign_stable_clusters(train_phrase, basis, vectorizer, encoder)
    assigned_eval = assign_stable_clusters(eval_phrase, basis, vectorizer, encoder)
    x_train = make_cluster_features(train_rows, assigned_train, f"{fold_tag}_train")
    x_eval = make_cluster_features(eval_rows, assigned_eval, f"{fold_tag}_eval")
    feature_cols = [c for c in x_train.columns if c not in {"row_id", "parent_asin", "year_month", "feature_split"}]
    aligned_eval = eval_rows[["row_id"]].merge(x_eval[["row_id"] + feature_cols], on="row_id", how="left")
    aligned_train = train_rows[["row_id", TARGET]].merge(x_train[["row_id"] + feature_cols], on="row_id", how="left")
    model = base.make_s45(aligned_train[TARGET])
    model.fit(aligned_train[feature_cols], aligned_train[TARGET])
    score = model.predict_proba(aligned_eval[feature_cols])[:, 1]
    fitted = {
        "basis": basis, "vectorizer": vectorizer, "encoder": encoder, "model": model,
        "feature_cols": feature_cols, "sample": sample, "assigned_train": assigned_train,
    }
    return score, fitted, x_eval


GATE_COLUMNS = ["raw_confidence", "cluster_confidence", "coverage", "novelty", "burst", "review_count"]


def gate_x(frame: pd.DataFrame) -> np.ndarray:
    x = pd.DataFrame(index=frame.index)
    x["raw_confidence"] = np.abs(frame["raw_full_score"] - 0.5)
    x["cluster_confidence"] = np.abs(frame["cluster_score"] - 0.5)
    x["coverage"] = frame["stable_cluster_coverage"]
    first = [c for c in frame.columns if c.endswith("_first_seen")]
    burst = [c for c in frame.columns if c.endswith("_burst")]
    x["novelty"] = frame[first].max(axis=1) if first else 0.0
    x["burst"] = frame[burst].max(axis=1) if burst else 0.0
    x["review_count"] = np.log1p(frame["phrase_count"])
    return x[GATE_COLUMNS].to_numpy(dtype=float)


def fit_gate(oof: pd.DataFrame) -> dict:
    raw = gate_x(oof)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (raw - mean) / std
    y = oof[TARGET].to_numpy(dtype=float)
    base_score = oof["raw_full_score"].to_numpy(dtype=float)
    cluster = oof["cluster_score"].to_numpy(dtype=float)
    def forward(p):
        w = MAX_TEXT_WEIGHT * expit(p[0] + x @ p[1:])
        score = (1 - w) * base_score + w * cluster
        return np.clip(score, 1e-6, 1 - 1e-6), w
    def objective(p):
        score, _ = forward(p)
        return log_loss(y, score) + 0.01 * np.square(p[1:]).sum()
    result = minimize(objective, np.r_[-1.0, np.zeros(x.shape[1])], method="L-BFGS-B", bounds=[(-4, 4)] * (x.shape[1] + 1))
    return {"params": result.x, "mean": mean, "std": std, "success": bool(result.success), "message": str(result.message)}


def apply_gate(frame: pd.DataFrame, gate: dict) -> pd.DataFrame:
    out = frame.copy()
    x = (gate_x(out) - gate["mean"]) / gate["std"]
    out["cluster_text_weight"] = MAX_TEXT_WEIGHT * expit(gate["params"][0] + x @ gate["params"][1:])
    out["phrase_dualview_fusion_score"] = (1 - out["cluster_text_weight"]) * out["raw_full_score"] + out["cluster_text_weight"] * out["cluster_score"]
    return out


def metric(y, score) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    top = np.argsort(-score)[:max(1, int(0.1 * len(y)))]
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_10pct": float(y[top].sum() / max(1, y.sum())),
        "precision_at_10pct": float(y[top].mean()),
    }


def structured_feature_names(panel: pd.DataFrame) -> list[str]:
    future = {"is_review_volume_drop", "is_review_volume_drop_2m", "next_review_count", "expected_next_review_count", "next_review_volume_ratio", "next_2m_review_count", "expected_next_2m_review_count"}
    excluded = {"parent_asin", "year_month", "row_id", "random_fold", "random_split", "random_stratum", "year"} | future
    return [c for c in panel.columns if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]


def save_catalog(fitted: dict) -> None:
    basis: ClusterBasis = fitted["basis"]
    terms = fitted["vectorizer"].get_feature_names_out()
    sample = fitted["sample"]
    records = []
    for pos, raw_id in enumerate(basis.cluster_ids):
        center = basis.sparse_centers[pos]
        top = np.argsort(-center)[:12]
        examples = sample[sample["raw_cluster"].eq(raw_id)]["phrase"].head(3).tolist()
        records.append({
            "cluster_pos": pos, "raw_leiden_cluster": raw_id, "fit_phrase_count": basis.cluster_sizes[pos],
            "bootstrap_stability": basis.stability[pos], "dense_threshold": float(basis.dense_threshold[pos]),
            "sparse_threshold": float(basis.sparse_threshold[pos]),
            "top_terms": " | ".join(terms[top]), "example_phrases": " || ".join(examples),
        })
    pd.DataFrame(records).to_csv(OUT / "phrase_cluster_catalog.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(base.PANEL_PATH).copy().reset_index(drop=True)
    panel["parent_asin"] = panel["parent_asin"].astype(str)
    panel["year_month"] = panel["year_month"].astype(str)
    panel["row_id"] = np.arange(len(panel), dtype=int)
    panel = base.assign_random_group_split(panel)
    # The test fold is removed before any review text, phrase, vectorizer,
    # embedding, graph, feature, gate, or metric object is constructed.
    panel = panel[panel["random_split"].ne("test")].copy()
    structured = structured_feature_names(panel)
    reviews = pd.read_parquet(base.REVIEWS_PATH, columns=["parent_asin", "year_month", "review_datetime_utc", "text_norm", "rating"])
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews["year_month"] = reviews["year_month"].astype(str)
    reviews["text_norm"] = reviews["text_norm"].fillna("").astype(str)
    reviews = reviews[reviews["text_norm"].str.len().ge(3)].merge(panel[["row_id", "parent_asin", "year_month", TARGET, "random_split"]], on=KEYS, how="inner", validate="many_to_one")
    panel = base.build_month_documents(panel, reviews)
    phrases = build_phrase_table(reviews)
    train = panel[panel["random_split"].eq("train")].copy()
    valid = panel[panel["random_split"].eq("valid")].copy()
    print(f"문장/절: {len(phrases):,}개 / Train 상품월: {len(train):,} / Valid: {len(valid):,}")
    print("Test는 split 배정 직후 제거했으며, 이후 텍스트/피처/모델/평가에 사용하지 않습니다.")

    oof_rows = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(train, train[TARGET], groups=train["parent_asin"]), 1):
        fit, held = train.iloc[fit_idx].copy(), train.iloc[held_idx].copy()
        tag = f"oof{fold}"
        print(f"\n===== {tag}: train {len(fit):,}, held {len(held):,} =====")
        raw_pred, _ = base.predict_base_experts(fit, held, reviews, structured)
        cluster_score, _, cluster_feature = cluster_expert(fit, held, phrases, tag)
        raw_pred = raw_pred.merge(cluster_feature, on="row_id", how="left", validate="one_to_one")
        raw_pred["cluster_score"] = cluster_score
        oof_rows.append(raw_pred)
        gc.collect()
    oof = pd.concat(oof_rows, ignore_index=True)
    raw_gate = base.fit_adaptive_gate(oof)
    raw_oof = base.apply_adaptive_gate(oof, raw_gate)
    oof["raw_full_score"] = raw_oof["s45_plus_context_score"]
    gate = fit_gate(oof)
    oof = apply_gate(oof, gate)
    oof.to_parquet(OUT / "train_oof_phrase_dualview_predictions.parquet", index=False)

    print("\n===== Valid: Train 전체로 raw 및 phrase expert 재학습 =====")
    valid_raw, _ = base.predict_base_experts(train, valid, reviews, structured)
    valid_cluster, fitted, valid_feature = cluster_expert(train, valid, phrases, "valid")
    valid_pred = valid_raw.merge(valid_feature, on="row_id", how="left", validate="one_to_one")
    valid_pred["cluster_score"] = valid_cluster
    valid_raw_gated = base.apply_adaptive_gate(valid_pred, raw_gate)
    valid_pred["raw_full_score"] = valid_raw_gated["s45_plus_context_score"]
    valid_pred = apply_gate(valid_pred, gate)
    valid_pred.to_parquet(OUT / "valid_phrase_dualview_predictions.parquet", index=False)
    save_catalog(fitted)

    results = []
    for split, frame in [("train_oof", oof), ("valid", valid_pred)]:
        for name, column in [("S45_structured_only", "s45_score"), ("raw_full_context", "raw_full_score"), ("phrase_cluster_only", "cluster_score"), ("S45_raw_plus_phrase_dualview", "phrase_dualview_fusion_score")]:
            row = {"split": split, "model_name": name, **metric(frame[TARGET], frame[column])}
            results.append(row)
    metrics = pd.DataFrame(results)
    for split in metrics["split"].unique():
        base_pr = float(metrics.loc[(metrics["split"] == split) & (metrics["model_name"] == "raw_full_context"), "pr_auc"].iloc[0])
        metrics.loc[metrics["split"] == split, "pr_auc_delta_vs_raw_full"] = metrics.loc[metrics["split"] == split, "pr_auc"] - base_pr
    metrics.to_csv(OUT / "phrase_dualview_burst_metrics.csv", index=False, encoding="utf-8-sig")
    valid_rows = metrics[metrics["split"].eq("valid")].set_index("model_name")
    raw_metric = valid_rows.loc["raw_full_context"]
    final_metric = valid_rows.loc["S45_raw_plus_phrase_dualview"]
    passed = (final_metric["pr_auc"] - raw_metric["pr_auc"] >= COMPLEXITY_THRESHOLD) or (final_metric["recall_at_10pct"] - raw_metric["recall_at_10pct"] >= 0.01)
    summary = {
        "experiment": "2-4-03_phrase_dualview_stable_cluster_burst",
        "test_data_used": False,
        "target": TARGET,
        "split": "existing random grouped Train/Valid only; parent_asin never crosses split",
        "domain_adaptation": "unsupervised SimCSE on each fit fold only" if DOMAIN_ADAPT else "off (required ablation baseline)",
        "clustering": "dense MiniLM + sparse TF-IDF mutual-kNN graph; Leiden auto community count; bootstrap stable clusters only",
        "stable_cluster_count": len(fitted["basis"].cluster_ids),
        "raw_leiden_cluster_count": fitted["basis"].raw_cluster_count,
        "stable_cluster_coverage_valid": float(valid_pred["stable_cluster_coverage"].mean()),
        "valid_raw_full_pr_auc": float(raw_metric["pr_auc"]),
        "valid_final_pr_auc": float(final_metric["pr_auc"]),
        "valid_pr_auc_delta_vs_raw_full": float(final_metric["pr_auc"] - raw_metric["pr_auc"]),
        "valid_recall10_delta_vs_raw_full": float(final_metric["recall_at_10pct"] - raw_metric["recall_at_10pct"]),
        "selection_threshold_pr_auc": COMPLEXITY_THRESHOLD,
        "decision": "candidate_pass" if passed else "keep_raw_full_context",
    }
    (OUT / "phrase_dualview_burst_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== Train OOF / Valid 결과 =====")
    print(metrics.to_string(index=False))
    print("\n===== 최종 판정 (Test 미사용) =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
