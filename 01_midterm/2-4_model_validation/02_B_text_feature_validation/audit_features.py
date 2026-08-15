"""
B-1. 안전 텍스트 피처 감사.

safe_text_features.csv의 105개 피처를 train 구간에서만 진단한다.
valid는 후보 선택에 쓰므로 통계 산출에 사용하지 않는다.

진단 항목
  1. 상수 / 준상수      변별력이 없는 열
  2. 정형 피처 중복      2-1 정형 컬럼과 사실상 같은 값
  3. 텍스트 내부 중복    _rate_t 와 _rate_t_shrunk 처럼 같은 정보의 두 버전
  4. 극단값             파싱 오류로 튀는 값
  5. 희소도             참고용. 단독 제외 근거로 쓰지 않는다
  6. 라벨 단변량 신호    train 기준 순위상관

분산이 0인 열은 순위상관이 정의되지 않아 NaN이 된다.
이 경우 상관 관련 값을 비워 두고 상수 플래그로 따로 표시한다.

    python3 audit_features.py

출력: feature_audit.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common_input import cfg, load_panel

HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "feature_audit.csv"

# 2-1 라벨 패널의 정형 컬럼. 텍스트 피처가 이들과 겹치는지 본다.
STRUCTURED_COLUMNS = [
    "review_count",
    "avg_rating",
    "low_rating_count",
    "low_rating_ratio",
    "verified_purchase_ratio",
    "mean_helpful_vote",
    "text_available_ratio",
    "past_3m_review_count",
    "past_3m_low_rating_count",
    "past_3m_low_rating_ratio",
]

# README 6절의 접미사 규칙.
LEVEL_SUFFIXES = ("_rate_t_shrunk", "_rate_t", "_mean")
CHANGE_SUFFIXES = ("_rate_p3", "_delta")

NEAR_CONSTANT_SHARE = 0.99   # 한 값이 이 비율 이상이면 준상수
SPARSE_THRESHOLD = 0.05      # 참고용 희소 기준
DUP_CORR_THRESHOLD = 0.95    # 이 이상이면 중복 의심
OUTLIER_RATIO = 20.0         # max / p99 가 이 배수 이상이면 극단값 의심


def classify_axis(col: str):
    """접미사로 현재 상태(level)와 변화 정보(change)를 구분한다."""
    for s in LEVEL_SUFFIXES:
        if col.endswith(s):
            return "level", s
    for s in CHANGE_SUFFIXES:
        if col.endswith(s):
            return "change", s
    return "static", "(none)"


def best_abs(series: pd.Series):
    """절댓값이 가장 큰 항목을 (이름, 값)으로 돌려준다.

    전부 NaN이면 (None, nan)을 돌려준다. 분산이 0인 열은 상관이
    정의되지 않으므로 이 경우가 발생한다.
    """
    s = series.dropna()
    if s.empty:
        return None, np.nan
    return s.idxmax(), float(s.max())


def as_round(value, digits=4):
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


def main() -> None:
    df, safe = load_panel()
    feats = safe["column"].tolist()
    fam = dict(zip(safe["column"], safe["family"]))

    train = df[df["split"] == "train"]
    print(f"\n감사 대상: train {len(train):,}행 x 피처 {len(feats)}개\n")

    struct_present = [c for c in STRUCTURED_COLUMNS if c in train.columns]

    # --- 순위 상관 일괄 계산 ---
    ranked = train[feats + struct_present + [cfg.TARGET_COLUMN]].rank(method="average")
    corr = ranked.corr(method="pearson")

    rows = []
    for col in feats:
        s = train[col]
        axis, suffix = classify_axis(col)

        vc = s.value_counts(normalize=True, dropna=False)
        top_share = float(vc.iloc[0]) if len(vc) else 1.0
        n_uniq = int(s.nunique(dropna=False))

        p99 = float(s.quantile(0.99))
        mx = float(s.max())
        outlier_ratio = mx / p99 if p99 > 0 else np.nan

        struct_col, struct_corr = best_abs(corr.loc[col, struct_present].abs())
        label_corr = corr.loc[col, cfg.TARGET_COLUMN]

        rows.append({
            "column": col,
            "family": fam[col],
            "axis": axis,
            "suffix": suffix,
            "n_unique": n_uniq,
            "nonzero_ratio": round(float((s != 0).mean()), 4),
            "top_value_share": round(top_share, 4),
            "na_ratio": round(float(s.isna().mean()), 4),
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "p99": round(p99, 6),
            "max": round(mx, 6),
            "max_over_p99": as_round(outlier_ratio, 2),
            "corr_label": as_round(label_corr),
            "max_corr_structured": as_round(struct_corr),
            "closest_structured": struct_col,
        })

    audit = pd.DataFrame(rows)

    # --- 텍스트 내부 중복쌍 ---
    tvals = corr.loc[feats, feats].abs().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(tvals, np.nan)
    tcorr = pd.DataFrame(tvals, index=feats, columns=feats)
    pairs = [best_abs(tcorr.loc[c]) for c in feats]
    audit["closest_text"] = [p[0] for p in pairs]
    audit["max_corr_text"] = [as_round(p[1]) for p in pairs]

    # --- 플래그 ---
    audit["flag_constant"] = audit["n_unique"] <= 1
    audit["flag_near_constant"] = (~audit["flag_constant"]) & (
        audit["top_value_share"] >= NEAR_CONSTANT_SHARE
    )
    audit["flag_dup_structured"] = (
        audit["max_corr_structured"].fillna(0) >= DUP_CORR_THRESHOLD
    )
    audit["flag_dup_text"] = audit["max_corr_text"].fillna(0) >= DUP_CORR_THRESHOLD
    audit["flag_outlier"] = audit["max_over_p99"].fillna(0) >= OUTLIER_RATIO
    audit["flag_sparse"] = audit["nonzero_ratio"] < SPARSE_THRESHOLD

    audit = audit.sort_values(["family", "column"])
    audit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------
    # 요약 출력
    # ---------------------------------------------------------------
    def show(title, sub, cols):
        print("=" * 78)
        print(title)
        print("=" * 78)
        if len(sub) == 0:
            print("  해당 없음")
        else:
            print(sub[cols].to_string(index=False))
        print()

    show("1. 상수 / 준상수 (변별력 없음)",
         audit[audit["flag_constant"] | audit["flag_near_constant"]],
         ["column", "family", "n_unique", "top_value_share", "corr_label"])

    show(f"2. 정형 피처와 중복 의심 (|순위상관| >= {DUP_CORR_THRESHOLD})",
         audit[audit["flag_dup_structured"]].sort_values(
             "max_corr_structured", ascending=False),
         ["column", "family", "closest_structured", "max_corr_structured"])

    show("2-b. 정형 상관 상위 10개 (중복 임계 미만 포함)",
         audit.sort_values("max_corr_structured", ascending=False).head(10),
         ["column", "family", "closest_structured", "max_corr_structured"])

    show(f"3. 극단값 의심 (max / p99 >= {OUTLIER_RATIO})",
         audit[audit["flag_outlier"]].sort_values("max_over_p99", ascending=False),
         ["column", "family", "p99", "max", "max_over_p99"])

    show(f"4. 텍스트 내부 중복쌍 (|순위상관| >= {DUP_CORR_THRESHOLD})",
         audit[audit["flag_dup_text"]].sort_values("max_corr_text", ascending=False),
         ["column", "closest_text", "max_corr_text"])

    ranked_by_label = audit.reindex(
        audit["corr_label"].abs().sort_values(ascending=False).index)
    show("5. 라벨 단변량 신호 상위 20개 (train 순위상관)",
         ranked_by_label.head(20),
         ["column", "family", "axis", "corr_label", "nonzero_ratio"])

    print("=" * 78)
    print("그룹 x 축 분포")
    print("=" * 78)
    print(pd.crosstab(audit["family"], audit["axis"]))
    print()
    n_flagged = int((audit["flag_constant"] | audit["flag_near_constant"]
                     | audit["flag_dup_structured"]).sum())
    print(f"제외 검토 대상(상수/준상수/정형중복): {n_flagged}개")
    print(f"희소(비영 < {SPARSE_THRESHOLD:.0%}) 피처: {int(audit['flag_sparse'].sum())}개")
    print("  희소도는 참고 지표다. 단독으로 제외 근거로 삼지 않는다.")
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
