from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment_config import (
    ANALYSIS_END,
    ANALYSIS_START,
    EXPECTED_POSITIVES,
    EXPECTED_PRODUCTS,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    FORBIDDEN_AUTO_TITLE_COLUMNS,
    FORBIDDEN_EXACT_COLUMNS,
    FORBIDDEN_PREFIXES,
    FORBIDDEN_TEXT_FAMILIES,
    KEY_COLUMNS,
    LABELED_DATA_PATH,
    TARGET_COLUMN,
    TEXT_DATA_PATH,
    TEXT_DICTIONARY_PATH,
    split_2_4,
)

ROOT = Path(__file__).resolve().parents[3]
PREP_DIR = Path(__file__).resolve().parent


def main() -> None:
    labeled = pd.read_parquet(ROOT / LABELED_DATA_PATH)
    text = pd.read_parquet(ROOT / TEXT_DATA_PATH)
    dictionary = pd.read_csv(ROOT / TEXT_DICTIONARY_PATH, encoding="utf-8-sig")

    for name, frame in [("labeled panel", labeled), ("text panel", text)]:
        missing_keys = sorted(set(KEY_COLUMNS) - set(frame.columns))
        assert not missing_keys, f"{name}에 키가 없음: {missing_keys}"
        duplicate_count = int(frame.duplicated(KEY_COLUMNS).sum())
        assert duplicate_count == 0, f"{name} 키 중복: {duplicate_count:,}건"

    assert len(labeled) == EXPECTED_ROWS, f"행 수 불일치: {len(labeled):,}"
    assert labeled["parent_asin"].nunique() == EXPECTED_PRODUCTS, "상품 수 불일치"
    assert int(labeled[TARGET_COLUMN].sum()) == EXPECTED_POSITIVES, "양성 수 불일치"
    assert str(labeled["year_month"].min()) == ANALYSIS_START, "시작월 불일치"
    assert str(labeled["year_month"].max()) == ANALYSIS_END, "종료월 불일치"

    joined = labeled[KEY_COLUMNS + [TARGET_COLUMN]].merge(
        text[KEY_COLUMNS + [TARGET_COLUMN]],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_labeled", "_text"),
        indicator=True,
    )
    assert (joined["_merge"] == "both").all(), "두 패널의 키 집합이 다름"
    assert (
        joined[f"{TARGET_COLUMN}_labeled"]
        == joined[f"{TARGET_COLUMN}_text"]
    ).all(), "두 패널의 라벨이 다름"

    next_columns = [
        column for column in text.columns
        if column.startswith(FORBIDDEN_PREFIXES)
    ]
    assert not next_columns, f"텍스트 패널에 미래 열이 포함됨: {next_columns}"

    safe_text = dictionary[
        ~dictionary["family"].isin(FORBIDDEN_TEXT_FAMILIES)
        & ~dictionary["column"].isin(FORBIDDEN_AUTO_TITLE_COLUMNS)
    ].copy()
    safe_text = safe_text.sort_values(["family", "column"]).reset_index(drop=True)

    missing_safe_columns = sorted(set(safe_text["column"]) - set(text.columns))
    assert not missing_safe_columns, f"텍스트 패널에 없는 피처: {missing_safe_columns}"
    assert len(safe_text) == 105, f"안전 텍스트 피처 수 불일치: {len(safe_text)}"

    split = labeled["year_month"].astype(str).map(split_2_4)
    split_counts = {
        str(name): int(count)
        for name, count in split.value_counts().sort_index().items()
    }
    assert split_counts == EXPECTED_SPLIT_COUNTS, f"분할 행 수 불일치: {split_counts}"

    family_counts = {
        str(name): int(count)
        for name, count in safe_text["family"].value_counts().sort_index().items()
    }

    safe_text.to_csv(
        PREP_DIR / "safe_text_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "validation": "passed",
        "labeled_rows": int(len(labeled)),
        "unique_products": int(labeled["parent_asin"].nunique()),
        "analysis_start": str(labeled["year_month"].min()),
        "analysis_end": str(labeled["year_month"].max()),
        "key_duplicate_count": 0,
        "label_positive_count": int(labeled[TARGET_COLUMN].sum()),
        "label_positive_rate": round(float(labeled[TARGET_COLUMN].mean()), 6),
        "text_total_columns": int(len(text.columns)),
        "safe_text_feature_count": int(len(safe_text)),
        "safe_text_feature_count_by_family": family_counts,
        "split_counts": split_counts,
        "forbidden_exact_columns": sorted(FORBIDDEN_EXACT_COLUMNS),
        "forbidden_auto_title_columns": sorted(FORBIDDEN_AUTO_TITLE_COLUMNS),
    }

    output_path = PREP_DIR / "common_input_validation.json"
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"저장: {output_path.relative_to(ROOT)}")
    print(f"저장: {(PREP_DIR / 'safe_text_features.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
