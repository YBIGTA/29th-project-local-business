
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path(
    "01_midterm/2-4_model_validation/"
    "03_A_combined_model_optimization"
)
MODEL_DIR = OUT / "models_stage5_oof"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "combined_train_valid_canonical.parquet"
FEATURE_SET_PATH = OUT / "standard_model_feature_sets.json"
STAGE4_METRICS_PATH = OUT / "standard_candidate_metrics.csv"

TARGET = "is_low_rating_surge"
KEYS = ["parent_asin", "year_month"]

FOLDS = [
    {
        "fold": "oof_2019",
        "train_end": "2018-12",
        "valid_start": "2019-01",
        "valid_end": "2019-12",
    },
    {
        "fold": "oof_2020",
        "train_end": "2019-12",
        "valid_start": "2020-01",
        "valid_end": "2020-12",
    },
    {
        "fold": "oof_2021",
        "train_end": "2020-12",
        "valid_start": "2021-01",
        "valid_end": "2021-12",
    },
]

PROTOCOL = {
    "test_data_used": False,
    "train_period": "2016-01~2021-12",
    "official_valid_period": "2022-01~2022-08",
    "test_period_excluded": "2022-09~2023-02",
    "oof_folds": FOLDS,
    "text_blocks": {
        "change_all": "B5의 past_p3 21개와 delta 21개",
        "B8": "축소추정을 사용하지 않는 텍스트 72개",
    },
    "fusion_methods": [
        "Train OOF에서 선택한 가중평균",
        "Train OOF 예측으로 학습한 로지스틱 스태킹",
    ],
    "valid_used_for_weight_selection": False,
    "selection_is_final": False,
}

(OUT / "stage5_protocol.json").write_text(
    json.dumps(PROTOCOL, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


def prepare_x(frame, features):
    return (
        frame[features]
        .replace([np.inf, -np.inf], np.nan)
    )


def make_model(y):
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())

    if positive == 0 or negative == 0:
        raise ValueError("학습 구간에 한쪽 라벨만 존재합니다.")

    return LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=negative / positive,
    )


def fit_predict(fit_frame, score_frame, features):
    y_fit = fit_frame[TARGET].astype(int)
    model = make_model(y_fit)
    model.fit(prepare_x(fit_frame, features), y_fit)
    score = model.predict_proba(
        prepare_x(score_frame, features)
    )[:, 1]
    return model, score


def logit(values):
    values = np.clip(
        np.asarray(values, dtype=float),
        1e-6,
        1 - 1e-6,
    )
    return np.log(values / (1 - values))


def recall_at_fraction(prediction, fraction):
    ranked = prediction.sort_values(
        ["y_score", "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    k = max(1, math.ceil(len(ranked) * fraction))
    positives = int(ranked["y_true"].sum())

    if positives == 0:
        return np.nan

    return float(
        ranked.head(k)["y_true"].sum() / positives
    )


def make_prediction_frame(frame, model_name, score):
    result = frame[KEYS].copy()
    result["model_name"] = model_name
    result["y_true"] = frame[TARGET].astype(int).to_numpy()
    result["y_score"] = np.asarray(score)
    return result


def evaluate(frame, model_name, score, method, feature_count):
    prediction = make_prediction_frame(
        frame, model_name, score
    )
    y_true = prediction["y_true"]

    return {
        "model_name": model_name,
        "method": method,
        "split": "valid",
        "pr_auc": average_precision_score(
            y_true, prediction["y_score"]
        ),
        "roc_auc": roc_auc_score(
            y_true, prediction["y_score"]
        ),
        "recall_at_5pct": recall_at_fraction(
            prediction, 0.05
        ),
        "recall_at_10pct": recall_at_fraction(
            prediction, 0.10
        ),
        "n_rows": len(prediction),
        "positive_rate": float(y_true.mean()),
        "feature_count": feature_count,
    }, prediction


panel = pd.read_parquet(PANEL_PATH)
panel["year_month"] = panel["year_month"].astype(str)

if panel.duplicated(KEYS).any():
    raise ValueError("상품·월 키 중복이 있습니다.")

if panel["year_month"].max() > "2022-08":
    raise ValueError("Test 기간 데이터가 포함됐습니다.")

split_counts = panel["split"].value_counts().to_dict()
expected_counts = {"train": 54813, "valid": 7228}

if split_counts != expected_counts:
    raise ValueError(
        f"분할 행 수 불일치: {split_counts}"
    )

train = panel.loc[
    panel["split"] == "train"
].copy()
valid = panel.loc[
    panel["split"] == "valid"
].copy()

feature_sets = json.loads(
    FEATURE_SET_PATH.read_text(encoding="utf-8")
)

structured = feature_sets["S45_structured_only"]
change_combined = feature_sets["D_S45_plus_change_all"]
b8_combined = feature_sets["C8_S45_plus_B8"]

structured_set = set(structured)
change_text = [
    feature for feature in change_combined
    if feature not in structured_set
]
b8_text = [
    feature for feature in b8_combined
    if feature not in structured_set
]

if len(structured) != 45:
    raise ValueError(
        f"정형 피처 수 오류: {len(structured)}"
    )

if len(change_text) != 42:
    raise ValueError(
        f"change_all 텍스트 수 오류: {len(change_text)}"
    )

if len(b8_text) != 72:
    raise ValueError(
        f"B8 텍스트 수 오류: {len(b8_text)}"
    )

all_used = set(structured + change_text + b8_text)
forbidden_used = sorted(
    feature for feature in all_used
    if feature == TARGET
    or feature.startswith("next_")
    or feature.startswith("auto_title_")
)

if forbidden_used:
    raise ValueError(
        f"금지 피처 포함: {forbidden_used}"
    )

print("===== OOF 입력 검증 완료 =====")
print(f"정형 피처: {len(structured)}개")
print(f"change_all 텍스트: {len(change_text)}개")
print(f"B8 텍스트: {len(b8_text)}개")
print("Test 데이터 사용: False")
print()

structured_oof_parts = []
fold_summary = []
start_total = time.perf_counter()

for index, fold in enumerate(FOLDS, start=1):
    fit_frame = train.loc[
        train["year_month"] <= fold["train_end"]
    ]
    hold_frame = train.loc[
        train["year_month"].between(
            fold["valid_start"], fold["valid_end"]
        )
    ]

    if len(fit_frame) == 0 or len(hold_frame) == 0:
        raise ValueError(
            f"{fold['fold']}의 학습 또는 OOF 구간이 비었습니다."
        )

    print(
        f"[정형 OOF {index}/3] {fold['fold']} "
        f"학습={len(fit_frame):,}, 예측={len(hold_frame):,}"
    )

    model, score = fit_predict(
        fit_frame, hold_frame, structured
    )

    part = hold_frame[KEYS + [TARGET]].copy()
    part["fold"] = fold["fold"]
    part["p_structured"] = score
    structured_oof_parts.append(part)

    fold_summary.append(
        {
            **fold,
            "train_rows": len(fit_frame),
            "oof_rows": len(hold_frame),
            "train_positive_rate": float(
                fit_frame[TARGET].mean()
            ),
            "oof_positive_rate": float(
                hold_frame[TARGET].mean()
            ),
        }
    )

structured_oof = pd.concat(
    structured_oof_parts,
    ignore_index=True,
)

print("[정형 전체] Train 전체 학습 후 공식 Valid 예측")
structured_model, structured_valid_score = fit_predict(
    train, valid, structured
)
joblib.dump(
    structured_model,
    MODEL_DIR / "structured_full.joblib",
)

metrics_rows = []
valid_prediction_parts = []
oof_output_parts = []
weight_grid_parts = []
stacking_summary = {}

row, pred = evaluate(
    valid,
    "S45_structured_only",
    structured_valid_score,
    "structured_only",
    len(structured),
)
metrics_rows.append(row)
valid_prediction_parts.append(pred)

blocks = {
    "change_all": change_text,
    "B8": b8_text,
}

for block_index, (block_name, text_features) in enumerate(
    blocks.items(),
    start=1,
):
    print()
    print(
        f"===== 텍스트 블록 {block_index}/2: "
        f"{block_name} ({len(text_features)}개) ====="
    )

    text_oof_parts = []

    for fold_index, fold in enumerate(FOLDS, start=1):
        fit_frame = train.loc[
            train["year_month"] <= fold["train_end"]
        ]
        hold_frame = train.loc[
            train["year_month"].between(
                fold["valid_start"], fold["valid_end"]
            )
        ]

        print(
            f"[{block_name} OOF {fold_index}/3] "
            f"{fold['fold']}"
        )

        model, score = fit_predict(
            fit_frame,
            hold_frame,
            text_features,
        )

        part = hold_frame[KEYS + [TARGET]].copy()
        part["fold"] = fold["fold"]
        part["p_text"] = score
        text_oof_parts.append(part)

    text_oof = pd.concat(
        text_oof_parts,
        ignore_index=True,
    )

    merged_oof = structured_oof.merge(
        text_oof,
        on=KEYS + [TARGET, "fold"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged_oof) != len(structured_oof):
        raise ValueError(
            f"{block_name} OOF 결합 행 수가 다릅니다."
        )

    grid_rows = []

    for structured_weight in np.linspace(0, 1, 21):
        text_weight = 1 - structured_weight
        blend_score = (
            structured_weight
            * merged_oof["p_structured"].to_numpy()
            + text_weight
            * merged_oof["p_text"].to_numpy()
        )
        grid_rows.append(
            {
                "text_block": block_name,
                "structured_weight": float(
                    structured_weight
                ),
                "text_weight": float(text_weight),
                "oof_pr_auc": average_precision_score(
                    merged_oof[TARGET],
                    blend_score,
                ),
            }
        )

    grid = pd.DataFrame(grid_rows).sort_values(
        ["oof_pr_auc", "structured_weight"],
        ascending=[False, False],
    )
    best = grid.iloc[0]
    best_structured_weight = float(
        best["structured_weight"]
    )
    best_text_weight = float(best["text_weight"])

    weight_grid_parts.append(grid)

    x_meta_oof = np.column_stack(
        [
            logit(merged_oof["p_structured"]),
            logit(merged_oof["p_text"]),
        ]
    )
    y_meta_oof = merged_oof[TARGET].astype(int)

    stack_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    )
    stack_model.fit(x_meta_oof, y_meta_oof)

    merged_oof["p_blend"] = (
        best_structured_weight
        * merged_oof["p_structured"]
        + best_text_weight
        * merged_oof["p_text"]
    )
    merged_oof["p_stack"] = stack_model.predict_proba(
        x_meta_oof
    )[:, 1]
    merged_oof["text_block"] = block_name
    oof_output_parts.append(merged_oof)

    print(
        f"OOF 최적 가중치: "
        f"정형={best_structured_weight:.2f}, "
        f"텍스트={best_text_weight:.2f}"
    )
    print(
        f"OOF 가중평균 PR-AUC: "
        f"{best['oof_pr_auc']:.6f}"
    )

    print(
        f"[{block_name} 전체] "
        f"Train 전체 학습 후 공식 Valid 예측"
    )
    text_model, text_valid_score = fit_predict(
        train,
        valid,
        text_features,
    )

    joblib.dump(
        text_model,
        MODEL_DIR / f"text_{block_name}_full.joblib",
    )
    joblib.dump(
        stack_model,
        MODEL_DIR / f"stack_{block_name}.joblib",
    )

    blend_valid_score = (
        best_structured_weight
        * structured_valid_score
        + best_text_weight
        * text_valid_score
    )

    x_meta_valid = np.column_stack(
        [
            logit(structured_valid_score),
            logit(text_valid_score),
        ]
    )
    stack_valid_score = stack_model.predict_proba(
        x_meta_valid
    )[:, 1]

    candidate_results = [
        (
            f"T_{block_name}_only",
            text_valid_score,
            "text_only",
        ),
        (
            f"OOF_blend_{block_name}",
            blend_valid_score,
            "oof_weighted_blend",
        ),
        (
            f"OOF_stack_{block_name}",
            stack_valid_score,
            "oof_logistic_stack",
        ),
    ]

    for model_name, score, method in candidate_results:
        result_row, result_prediction = evaluate(
            valid,
            model_name,
            score,
            method,
            len(structured) + len(text_features)
            if method != "text_only"
            else len(text_features),
        )
        metrics_rows.append(result_row)
        valid_prediction_parts.append(
            result_prediction
        )

    logistic = stack_model.named_steps[
        "logisticregression"
    ]

    stacking_summary[block_name] = {
        "text_feature_count": len(text_features),
        "oof_rows": len(merged_oof),
        "selected_structured_weight": (
            best_structured_weight
        ),
        "selected_text_weight": best_text_weight,
        "oof_blend_pr_auc": float(
            best["oof_pr_auc"]
        ),
        "stack_scaled_logit_coefficients": {
            "structured": float(logistic.coef_[0][0]),
            "text": float(logistic.coef_[0][1]),
        },
        "stack_intercept": float(
            logistic.intercept_[0]
        ),
    }

metrics = pd.DataFrame(metrics_rows)
valid_predictions = pd.concat(
    valid_prediction_parts,
    ignore_index=True,
)
oof_predictions = pd.concat(
    oof_output_parts,
    ignore_index=True,
)
weight_grid = pd.concat(
    weight_grid_parts,
    ignore_index=True,
)

baseline_pr = float(
    metrics.loc[
        metrics["model_name"] == "S45_structured_only",
        "pr_auc",
    ].iloc[0]
)
baseline_r5 = float(
    metrics.loc[
        metrics["model_name"] == "S45_structured_only",
        "recall_at_5pct",
    ].iloc[0]
)
baseline_r10 = float(
    metrics.loc[
        metrics["model_name"] == "S45_structured_only",
        "recall_at_10pct",
    ].iloc[0]
)

metrics["delta_pr_auc_vs_S45"] = (
    metrics["pr_auc"] - baseline_pr
)
metrics["delta_recall_5pct_vs_S45"] = (
    metrics["recall_at_5pct"] - baseline_r5
)
metrics["delta_recall_10pct_vs_S45"] = (
    metrics["recall_at_10pct"] - baseline_r10
)

metrics.to_csv(
    OUT / "stage5_oof_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)
valid_predictions.to_parquet(
    OUT / "stage5_valid_predictions.parquet",
    index=False,
)
oof_predictions.to_parquet(
    OUT / "stage5_oof_predictions.parquet",
    index=False,
)
weight_grid.to_csv(
    OUT / "stage5_weight_search.csv",
    index=False,
    encoding="utf-8-sig",
)

stage4 = pd.read_csv(STAGE4_METRICS_PATH)
comparison_columns = [
    "model_name",
    "pr_auc",
    "roc_auc",
    "recall_at_5pct",
    "recall_at_10pct",
    "feature_count",
]

stage4_comparison = stage4[
    stage4["model_name"].isin(
        [
            "S45_structured_only",
            "D_S45_plus_change_all",
            "C8_S45_plus_B8",
        ]
    )
][comparison_columns].copy()
stage4_comparison["experiment"] = "stage4_early_fusion"

stage5_comparison = metrics[
    comparison_columns
].copy()
stage5_comparison["experiment"] = "stage5_oof_late_fusion"

comparison = pd.concat(
    [stage4_comparison, stage5_comparison],
    ignore_index=True,
)
comparison.to_csv(
    OUT / "stage5_all_model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "test_data_used": False,
    "oof_is_time_ordered": True,
    "official_valid_used_for_weight_selection": False,
    "folds": fold_summary,
    "stacking": stacking_summary,
    "total_training_seconds": round(
        time.perf_counter() - start_total, 3
    ),
    "selection_is_final": False,
    "next_required_check": (
        "제한적 튜닝 후 롤링 시간 안정성 검증"
    ),
}
(OUT / "stage5_stacking_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print()
print("===== 2-4-03 단계 5/8 OOF 스태킹 완료 =====")
print("Test 데이터 사용: False")
print()
print("[Stage5 공식 Valid 결과]")
print(
    metrics.sort_values("pr_auc", ascending=False)[
        [
            "model_name",
            "method",
            "pr_auc",
            "delta_pr_auc_vs_S45",
            "roc_auc",
            "recall_at_5pct",
            "recall_at_10pct",
            "feature_count",
        ]
    ].round(6).to_string(index=False)
)

print()
print("[Stage4 조기결합과 Stage5 후반결합 전체 비교]")
print(
    comparison.sort_values(
        "pr_auc", ascending=False
    ).round(6).to_string(index=False)
)

print()
print("[OOF에서 선택한 결합 가중치]")
for block_name, values in stacking_summary.items():
    print(
        f"{block_name}: "
        f"정형={values['selected_structured_weight']:.2f}, "
        f"텍스트={values['selected_text_weight']:.2f}, "
        f"텍스트 스태킹 계수="
        f"{values['stack_scaled_logit_coefficients']['text']:.6f}"
    )
