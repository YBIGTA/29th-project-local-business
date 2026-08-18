from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("01_midterm/2-4_model_validation")
A1 = ROOT / "01_A_structured_timeseries"
B1 = ROOT / "02_B_text_feature_validation"
OUT = ROOT / "03_A_combined_model_optimization"

STRUCTURED_PATH = A1 / "structured_modeling_panel.parquet"
HISTORY_PATH = A1 / "multiwindow_history_features_pretest.parquet"
STRUCTURED_SPEC_PATH = A1 / "structured_feature_spec.csv"

TEXT_PATH = Path("data/processed/product_month_text_features.parquet")
CANDIDATE_PATH = B1 / "candidate_feature_sets.json"
TEXT_GROUP_PATH = B1 / "text_feature_groups.csv"

KEYS = ["parent_asin", "year_month"]
LABEL = "is_low_rating_surge"

TRAIN_END = "2021-12"
VALID_START = "2022-01"
VALID_END = "2022-08"
TEST_START = "2022-09"

EXPECTED_TRAIN = 54813
EXPECTED_VALID = 7228
EXPECTED_PRETEST = EXPECTED_TRAIN + EXPECTED_VALID

OVERLAP_THRESHOLD = 0.95

for path in [
    STRUCTURED_PATH,
    HISTORY_PATH,
    STRUCTURED_SPEC_PATH,
    TEXT_PATH,
    CANDIDATE_PATH,
    TEXT_GROUP_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"필수 입력 파일 없음: {path}")

structured = pd.read_parquet(STRUCTURED_PATH)
history = pd.read_parquet(HISTORY_PATH)
text = pd.read_parquet(TEXT_PATH)
structured_spec = pd.read_csv(STRUCTURED_SPEC_PATH)
text_groups = pd.read_csv(TEXT_GROUP_PATH)
candidate_data = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

for frame in [structured, history, text]:
    frame["year_month"] = frame["year_month"].astype(str)

for name, frame in {
    "structured": structured,
    "history": history,
    "text": text,
}.items():
    missing_keys = [column for column in KEYS if column not in frame.columns]
    if missing_keys:
        raise ValueError(f"{name} 키 누락: {missing_keys}")

    duplicate_count = int(frame.duplicated(KEYS).sum())
    if duplicate_count:
        raise ValueError(f"{name} 상품×월 키 중복: {duplicate_count}건")

if LABEL not in structured.columns:
    raise ValueError(f"정형 패널에 라벨 {LABEL}이 없습니다.")

# 텍스트 패널에도 라벨이 있으면 값 일치만 검사하고 입력에서는 제거한다.
text_label_mismatch = 0
if LABEL in text.columns:
    label_check = structured[KEYS + [LABEL]].merge(
        text[KEYS + [LABEL]],
        on=KEYS,
        how="inner",
        suffixes=("_structured", "_text"),
        validate="one_to_one",
    )
    text_label_mismatch = int(
        (
            label_check[f"{LABEL}_structured"].astype(int)
            != label_check[f"{LABEL}_text"].astype(int)
        ).sum()
    )
    if text_label_mismatch:
        raise ValueError(f"정형·텍스트 라벨 불일치: {text_label_mismatch}건")

current_features = structured_spec.loc[
    (structured_spec["decision"] == "use")
    & (structured_spec["feature_group"] == "current_state"),
    "feature_name",
].astype(str).tolist()

history_features = [
    column for column in history.columns
    if column not in KEYS
]

structured_features = current_features + history_features

if len(current_features) != 7:
    raise ValueError(f"현재 상태 피처 수 불일치: 기대 7, 실제 {len(current_features)}")

if len(history_features) != 38:
    raise ValueError(f"이력 피처 수 불일치: 기대 38, 실제 {len(history_features)}")

if len(structured_features) != 45:
    raise ValueError(f"최종 정형 피처 수 불일치: 기대 45, 실제 {len(structured_features)}")

candidates = candidate_data.get("candidates", [])
if len(candidates) != 3:
    raise ValueError(f"B 최종 후보 수 불일치: 기대 3, 실제 {len(candidates)}")

candidate_names = [candidate["name"] for candidate in candidates]
expected_names = {
    "B5_clean_shrunk_change",
    "B1_clean_shrunk",
    "B8_no_shrunk",
}
if set(candidate_names) != expected_names:
    raise ValueError(f"B 후보 이름 불일치: {candidate_names}")

candidate_text_features = {
    candidate["name"]: list(map(str, candidate["text_features"]))
    for candidate in candidates
}

all_candidate_text = sorted(
    set().union(*candidate_text_features.values())
)

missing_text_features = sorted(set(all_candidate_text) - set(text.columns))
if missing_text_features:
    raise ValueError(
        f"텍스트 패널에 후보 피처가 없습니다: {missing_text_features}"
    )

forbidden_pattern = re.compile(
    r"(^is_low_rating_surge$)|(^next_)|(^auto_title_)|(^target$)|(^label$)",
    flags=re.IGNORECASE,
)

forbidden_text_features = [
    feature for feature in all_candidate_text
    if forbidden_pattern.search(feature)
]
if forbidden_text_features:
    raise ValueError(
        f"후보에 금지 텍스트 피처 포함: {forbidden_text_features}"
    )

forbidden_structured_features = [
    feature for feature in structured_features
    if forbidden_pattern.search(feature)
]
if forbidden_structured_features:
    raise ValueError(
        f"정형 후보에 금지 피처 포함: {forbidden_structured_features}"
    )

# B가 정한 클리핑을 그대로 적용한다.
clip_rules = {}
if "clip_upper" in text_groups.columns:
    for row in text_groups.itertuples(index=False):
        feature = str(row.column)
        upper = getattr(row, "clip_upper", np.nan)
        if pd.notna(upper) and feature in text.columns:
            upper = float(upper)
            text[feature] = pd.to_numeric(
                text[feature], errors="coerce"
            ).clip(upper=upper)
            clip_rules[feature] = upper

# A2에서는 Test를 처음부터 제거한다.
structured_pretest = structured.loc[
    structured["year_month"] < TEST_START,
    KEYS + [LABEL] + current_features,
].copy()

history_pretest = history.loc[
    history["year_month"] < TEST_START,
    KEYS + history_features,
].copy()

text_pretest = text.loc[
    text["year_month"] < TEST_START,
    KEYS + all_candidate_text,
].copy()

combined = structured_pretest.merge(
    history_pretest,
    on=KEYS,
    how="inner",
    validate="one_to_one",
).merge(
    text_pretest,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

if len(combined) != EXPECTED_PRETEST:
    raise ValueError(
        f"결합 행 수 불일치: 기대 {EXPECTED_PRETEST:,}, 실제 {len(combined):,}"
    )

if combined.duplicated(KEYS).any():
    raise ValueError("결합 패널의 상품×월 키가 중복됩니다.")

if combined["year_month"].ge(TEST_START).any():
    raise ValueError("결합 패널에 Test 기간이 포함됐습니다.")

combined["split"] = np.select(
    [
        combined["year_month"] <= TRAIN_END,
        combined["year_month"].between(VALID_START, VALID_END),
    ],
    ["train", "valid"],
    default="forbidden",
)

if (combined["split"] == "forbidden").any():
    raise ValueError("Train·Valid 외 기간이 결합 패널에 포함됐습니다.")

split_counts = combined["split"].value_counts().to_dict()
if split_counts.get("train") != EXPECTED_TRAIN:
    raise ValueError(f"Train 행 수 불일치: {split_counts}")
if split_counts.get("valid") != EXPECTED_VALID:
    raise ValueError(f"Valid 행 수 불일치: {split_counts}")

# A의 장기 정형 피처와 B 텍스트의 추가 중복을 Train에서만 검사한다.
train = combined.loc[combined["split"] == "train"].copy()
correlation_columns = structured_features + all_candidate_text

ranked = train[correlation_columns].rank(
    method="average",
    pct=True,
).astype("float32")

rank_matrix = ranked.to_numpy(dtype=np.float32, copy=True)
rank_matrix -= np.nanmean(rank_matrix, axis=0, keepdims=True)
rank_matrix = np.nan_to_num(rank_matrix, nan=0.0)

norms = np.sqrt(np.sum(rank_matrix * rank_matrix, axis=0))
norms[norms == 0] = 1.0
normalized = rank_matrix / norms

n_structured = len(structured_features)
structured_matrix = normalized[:, :n_structured]
text_matrix = normalized[:, n_structured:]

correlation_matrix = text_matrix.T @ structured_matrix

audit_rows = []
for text_index, text_feature in enumerate(all_candidate_text):
    correlations = correlation_matrix[text_index]
    best_index = int(np.argmax(np.abs(correlations)))
    best_correlation = float(correlations[best_index])
    closest_structured = structured_features[best_index]

    group_match = text_groups.loc[
        text_groups["column"].astype(str) == text_feature
    ]

    if group_match.empty:
        family = ""
        axis = ""
        variant = ""
    else:
        row = group_match.iloc[0]
        family = str(row.get("family", ""))
        axis = str(row.get("axis", ""))
        variant = str(row.get("variant", ""))

    audit_rows.append(
        {
            "text_feature": text_feature,
            "family": family,
            "axis": axis,
            "variant": variant,
            "closest_structured_feature": closest_structured,
            "spearman_corr_train": round(best_correlation, 6),
            "abs_spearman_corr_train": round(abs(best_correlation), 6),
            "flag_duplicate_with_a1": abs(best_correlation) >= OVERLAP_THRESHOLD,
        }
    )

overlap_audit = pd.DataFrame(audit_rows).sort_values(
    "abs_spearman_corr_train",
    ascending=False,
)

duplicate_with_a1 = set(
    overlap_audit.loc[
        overlap_audit["flag_duplicate_with_a1"],
        "text_feature",
    ].astype(str)
)

combined_feature_sets = {}
candidate_summary_rows = []

for candidate in candidates:
    name = candidate["name"]
    original_text = candidate_text_features[name]
    removed = sorted(set(original_text) & duplicate_with_a1)
    retained = [
        feature for feature in original_text
        if feature not in duplicate_with_a1
    ]

    combined_feature_sets[name] = {
        "rank_from_B1": int(candidate["rank"]),
        "structured_features": structured_features,
        "structured_feature_count": len(structured_features),
        "text_features_original": original_text,
        "text_feature_count_original": len(original_text),
        "removed_due_to_A1_overlap": removed,
        "text_features_for_combined_model": retained,
        "text_feature_count_for_combined_model": len(retained),
        "all_features_for_combined_model": structured_features + retained,
        "total_feature_count": len(structured_features) + len(retained),
        "B1_selection_reason": candidate.get("selection_reason", ""),
        "B1_caution": candidate.get("caution", ""),
    }

    candidate_summary_rows.append(
        {
            "candidate": name,
            "B1_rank": int(candidate["rank"]),
            "original_text_count": len(original_text),
            "removed_A1_overlap_count": len(removed),
            "retained_text_count": len(retained),
            "structured_count": len(structured_features),
            "combined_feature_count": len(structured_features) + len(retained),
        }
    )

candidate_summary = pd.DataFrame(candidate_summary_rows).sort_values("B1_rank")

final_text_union = sorted(
    set().union(
        *[
            value["text_features_for_combined_model"]
            for value in combined_feature_sets.values()
        ]
    )
)

final_columns = (
    KEYS
    + [LABEL, "split"]
    + structured_features
    + final_text_union
)

combined_output = combined[final_columns].copy()

validation = {
    "validation": "passed",
    "test_data_used": False,
    "rows": len(combined_output),
    "unique_products": int(combined_output["parent_asin"].nunique()),
    "key_duplicate_count": int(combined_output.duplicated(KEYS).sum()),
    "split_counts": {
        key: int(value)
        for key, value in split_counts.items()
    },
    "label_positive_count": int(combined_output[LABEL].sum()),
    "label_positive_rate": round(float(combined_output[LABEL].mean()), 6),
    "structured_feature_count": len(structured_features),
    "candidate_text_union_before_overlap_check": len(all_candidate_text),
    "A1_overlap_duplicate_count": len(duplicate_with_a1),
    "A1_overlap_duplicate_features": sorted(duplicate_with_a1),
    "candidate_text_union_after_overlap_check": len(final_text_union),
    "forbidden_feature_count": 0,
    "text_label_mismatch_count": text_label_mismatch,
    "clip_rules": clip_rules,
}

OUT.mkdir(parents=True, exist_ok=True)

combined_output.to_parquet(
    OUT / "combined_train_valid.parquet",
    index=False,
)

overlap_audit.to_csv(
    OUT / "text_structured_overlap_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

candidate_summary.to_csv(
    OUT / "combined_candidate_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

(OUT / "combined_feature_sets.json").write_text(
    json.dumps(
        {
            "generated_by": "03_A_combined_model_optimization/build_combined_input.py",
            "test_data_used": False,
            "train_period": "2016-01~2021-12",
            "valid_period": "2022-01~2022-08",
            "test_period_excluded": "2022-09~2023-02",
            "overlap_threshold_train_only": OVERLAP_THRESHOLD,
            "candidates": combined_feature_sets,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

(OUT / "combined_input_validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== 2-4-03 결합 입력 생성·검증 완료 =====")
print(json.dumps(validation, ensure_ascii=False, indent=2))

print("\n[결합 후보 구성]")
print(candidate_summary.to_string(index=False))

print("\n[A1 정형 피처와 상관이 높은 텍스트 피처 상위 15개]")
print(
    overlap_audit[
        [
            "text_feature",
            "closest_structured_feature",
            "spearman_corr_train",
            "flag_duplicate_with_a1",
        ]
    ].head(15).to_string(index=False)
)
