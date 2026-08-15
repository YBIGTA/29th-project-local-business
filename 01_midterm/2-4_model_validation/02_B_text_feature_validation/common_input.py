"""
B 파트 공통 입력 로더.

00_preparation/README.md의 합류 기준(10절)을 assert로 강제한다.
검증을 통과한 경우에만 결합 패널과 안전 텍스트 피처 목록을 돌려준다.

    from common_input import load_panel
    df, safe = load_panel()

기준이 하나라도 어긋나면 AssertionError로 즉시 중단한다.
성능을 비교하기 전에 데이터 버전 불일치를 잡기 위함이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# 00_preparation의 공통 설정을 그대로 사용한다.
# 담당자가 자기 폴더에서 분할·금지 피처·지표 정의를 바꾸지 않기 위함이다.
# ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / "00_preparation"
REPO_ROOT = HERE.parents[2]

sys.path.insert(0, str(PREP_DIR))
import experiment_config as cfg  # noqa: E402

SAFE_FEATURES_PATH = PREP_DIR / "safe_text_features.csv"

# 텍스트 패널에서 결합 전에 반드시 떼어내는 열.
# split은 2-3에서 2020/2021 기준으로 만든 구버전이므로 2-4에서 쓰지 않는다.
TEXT_PANEL_DROP = ["is_low_rating_surge", "split"]


def _read(path_str: str) -> pd.DataFrame:
    path = REPO_ROOT / path_str
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    return pd.read_parquet(path)


def load_safe_features() -> pd.DataFrame:
    """안전 텍스트 피처 목록. column, family, dtype 3열."""
    safe = pd.read_csv(SAFE_FEATURES_PATH, encoding="utf-8-sig")
    assert list(safe.columns[:2]) == ["column", "family"], safe.columns.tolist()
    assert len(safe) == 105, f"안전 텍스트 피처 105개가 아님: {len(safe)}"
    return safe


def load_panel(verbose: bool = True):
    """정형·라벨 패널과 텍스트 패널을 공통 키로 결합해 돌려준다."""
    labeled = _read(cfg.LABELED_DATA_PATH)
    text = _read(cfg.TEXT_DATA_PATH)
    keys = cfg.KEY_COLUMNS

    # --- 1. 행 수 / 상품 수 / 양성 수 ---
    assert len(labeled) == cfg.EXPECTED_ROWS, f"라벨 패널 행 수: {len(labeled)}"
    assert len(text) == cfg.EXPECTED_ROWS, f"텍스트 패널 행 수: {len(text)}"
    n_products = labeled["parent_asin"].nunique()
    assert n_products == cfg.EXPECTED_PRODUCTS, f"상품 수: {n_products}"
    n_pos = int(labeled[cfg.TARGET_COLUMN].sum())
    assert n_pos == cfg.EXPECTED_POSITIVES, f"양성 수: {n_pos}"

    # --- 2. 키 중복 0건 ---
    assert labeled.duplicated(subset=keys).sum() == 0, "라벨 패널 키 중복"
    assert text.duplicated(subset=keys).sum() == 0, "텍스트 패널 키 중복"

    # --- 3. 두 패널의 키 집합과 라벨 값이 완전히 같은지 ---
    lk = set(map(tuple, labeled[keys].values))
    tk = set(map(tuple, text[keys].values))
    assert lk == tk, f"키 집합 불일치: 라벨만 {len(lk - tk)}건, 텍스트만 {len(tk - lk)}건"

    if cfg.TARGET_COLUMN in text.columns:
        cmp = labeled[keys + [cfg.TARGET_COLUMN]].merge(
            text[keys + [cfg.TARGET_COLUMN]], on=keys, suffixes=("_l", "_t")
        )
        mismatch = int(
            (cmp[f"{cfg.TARGET_COLUMN}_l"] != cmp[f"{cfg.TARGET_COLUMN}_t"]).sum()
        )
        assert mismatch == 0, f"두 패널의 라벨 불일치 {mismatch}건"

    # --- 4. 분석 기간 ---
    assert labeled["year_month"].min() == cfg.ANALYSIS_START
    assert labeled["year_month"].max() == cfg.ANALYSIS_END

    # --- 5. 결합 ---
    text_use = text.drop(columns=[c for c in TEXT_PANEL_DROP if c in text.columns])
    df = labeled.merge(text_use, on=keys, how="inner", validate="one_to_one")
    assert len(df) == cfg.EXPECTED_ROWS, f"결합 후 행 수: {len(df)}"

    # --- 6. 시간 분할은 split_2_4()로 다시 계산 ---
    df["split"] = df["year_month"].map(cfg.split_2_4)
    counts = df["split"].value_counts().to_dict()
    assert counts == cfg.EXPECTED_SPLIT_COUNTS, f"분할 행 수: {counts}"

    # --- 7. 안전 텍스트 피처가 전부 결합 패널에 있는지 ---
    safe = load_safe_features()
    missing = [c for c in safe["column"] if c not in df.columns]
    assert not missing, f"결합 패널에 없는 안전 피처: {missing}"

    # --- 8. 금지 피처가 안전 목록에 섞이지 않았는지 ---
    banned = set(cfg.FORBIDDEN_EXACT_COLUMNS) | set(cfg.FORBIDDEN_AUTO_TITLE_COLUMNS)
    leaked = [c for c in safe["column"] if c in banned or c.startswith(cfg.FORBIDDEN_PREFIXES)]
    assert not leaked, f"안전 목록에 금지 피처 포함: {leaked}"

    if verbose:
        print("=" * 62)
        print("공통 입력 검증 통과")
        print("=" * 62)
        print(f"  결합 패널        {len(df):,}행 x {df.shape[1]}열")
        print(f"  고유 상품        {n_products:,}개")
        print(f"  양성            {n_pos:,}행 ({df[cfg.TARGET_COLUMN].mean():.4%})")
        print(f"  분석 기간        {cfg.ANALYSIS_START} ~ {cfg.ANALYSIS_END}")
        print(f"  안전 텍스트 피처  {len(safe)}개")
        print("-" * 62)
        for name in ["train", "valid", "test"]:
            part = df[df["split"] == name]
            print(f"  {name:<6} {len(part):>6,}행   양성 {part[cfg.TARGET_COLUMN].mean():.2%}")
        print("=" * 62)
        print("  주의: test는 3일차 A의 최종 평가 전까지 열지 않는다.")
        print("=" * 62)

    return df, safe


if __name__ == "__main__":
    load_panel()
