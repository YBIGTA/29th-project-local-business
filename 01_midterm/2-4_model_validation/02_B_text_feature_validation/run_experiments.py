"""
B-4. 텍스트 피처 조합 실험.

최소 정형 기준(review_count, low_rating_ratio, past_3m_low_rating_ratio)에
텍스트 그룹을 얹어가며 valid 성능을 비교한다.

핵심 대조군
  V1  축소추정 계열만        리뷰 수 대리 가설을 검증한다
  V2  원값 계열만
  V3  변화 계열만
  V4  축소를 뺀 전부

V1이 S0 대비 거의 오르지 않고 V4가 오른다면
축소값의 예측력은 리뷰 수의 재탕이라는 해석이 뒷받침된다.

    python3 run_experiments.py

출력: valid_results.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_input import cfg, load_panel
from evaluation import run_model, to_result_frame

HERE = Path(__file__).resolve().parent
GROUPS_PATH = HERE / "text_feature_groups.csv"
RESULT_PATH = HERE / "valid_results.csv"

# 리뷰 수 오염이 이 값 이상인 축소 피처를 오염 하위군으로 본다.
CONTAMINATION_THRESHOLD = 0.85

# 2026-08-15 A가 재확정한 최소 정형 기준 7개.
# experiment_config.MINIMUM_STRUCTURED_FEATURES_FOR_B(3개)를 대체한다.
S0_SEVEN = [
    "avg_rating",
    "low_rating_count",
    "low_rating_ratio",
    "mean_helpful_vote",
    "review_count",
    "text_available_ratio",
    "verified_purchase_ratio",
]

# 새 기준에는 과거 기준선이 없다.
# 시간 정보 부재가 텍스트 변화 피처의 기여를 부풀리는지 진단하기 위한 열.
PAST_BASELINE = "past_3m_low_rating_ratio"


def apply_clipping(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """파싱 오류로 튀는 값에 상한을 적용한다."""
    rules = groups.dropna(subset=["clip_upper"])
    for _, r in rules.iterrows():
        col, upper = r["column"], float(r["clip_upper"])
        if col in df.columns:
            before = float(df[col].max())
            df[col] = df[col].clip(upper=upper)
            print(f"  클리핑 {col}: {before:,.0f} -> {upper:,.0f}")
    return df


def build_feature_sets(groups: pd.DataFrame, contamination: dict) -> dict[str, list[str]]:
    kept = groups[groups["kept"]]

    def pick(**cond) -> list[str]:
        sub = kept
        for key, values in cond.items():
            values = values if isinstance(values, (list, tuple)) else [values]
            sub = sub[sub[key].isin(values)]
        return sub["column"].tolist()

    S0 = list(S0_SEVEN)

    shrunk = pick(variant="shrunk")
    raw = pick(variant=["raw_t", "raw_mean"])
    change = pick(variant=["past_p3", "delta"])
    static = pick(variant="static")
    all_text = kept["column"].tolist()

    # 축소 계열을 리뷰 수 오염 정도로 둘로 나눈다.
    contaminated = [c for c in shrunk if contamination.get(c, 0.0) >= CONTAMINATION_THRESHOLD]
    clean = [c for c in shrunk if c not in contaminated]

    sets: dict[str, list[str]] = {
        "S0_minimum_structured": list(S0),
        "T_text_only_all": all_text,
        "T_text_only_no_shrunk": raw + change + static,
        "V1_S0_shrunk": S0 + shrunk,
        "V2_S0_raw": S0 + raw,
        "V3_S0_change": S0 + change,
        "V4_S0_raw_change": S0 + raw + change + static,
        "V5_S0_all_text": S0 + all_text,
        "V6_S0_shrunk_contaminated": S0 + contaminated,
        "V7_S0_shrunk_clean": S0 + clean,
    }

    for family in ["T_cause", "T_consequence", "T_narrative", "T_cooc", "T_basic"]:
        cols = pick(family=family)
        if cols:
            sets[f"F_{family}"] = S0 + cols

    sets["C_cause_consequence"] = S0 + pick(family=["T_cause", "T_consequence"])
    sets["C_cause_narrative"] = S0 + pick(family=["T_cause", "T_narrative"])

    # ---- T_cause 내부 정밀화 ----
    # 1차 실험에서 T_cause만으로 전체 텍스트의 기여 대부분을 설명했다.
    # 어느 하위 축이 실제로 기여하는지 좁힌다.
    cause = pick(family="T_cause")
    cause_raw = [c for c in cause if c in raw]
    cause_change = [c for c in cause if c in change]
    cause_shrunk = [c for c in cause if c in shrunk]
    cause_clean_shrunk = [c for c in cause_shrunk if c in clean]
    cause_no_bad = [c for c in cause if c not in contaminated]

    sets["M1_cause_change"] = S0 + cause_change
    sets["M2_cause_raw"] = S0 + cause_raw
    sets["M3_cause_no_contaminated"] = S0 + cause_no_bad
    sets["M4_cause_change_clean_shrunk"] = S0 + cause_change + cause_clean_shrunk
    sets["M5_cause_narrative_change"] = S0 + cause_change + [
        c for c in pick(family="T_narrative") if c in change
    ]

    # ---- 진단: 과거 기준선 유무 ----
    # 새 최소 정형 기준에는 시간 정보가 없다.
    # 텍스트 변화 피처의 기여가 시간 정보 부재 때문에 부풀려졌는지 본다.
    sets["D1_S0_plus_past"] = S0 + [PAST_BASELINE]
    sets["D2_S0_past_change"] = S0 + [PAST_BASELINE] + change
    sets["D3_S0_past_cause"] = S0 + [PAST_BASELINE] + cause
    sets["D4_S0_past_all_text"] = S0 + [PAST_BASELINE] + all_text

    # ---- 참고: 이전 3개 기준 ----
    sets["R_legacy_S0_three"] = list(cfg.MINIMUM_STRUCTURED_FEATURES_FOR_B)
    return sets


def measure_contamination(df: pd.DataFrame, columns: list[str]) -> dict:
    """축소 피처가 리뷰 수를 얼마나 대리하는지 train 순위상관 절댓값으로 잰다."""
    train = df[df["split"] == "train"]
    ranked = train[columns + ["review_count"]].rank(method="average")
    corr = ranked.corr(method="pearson")["review_count"]
    return {c: abs(float(corr[c])) for c in columns if pd.notna(corr[c])}


def main() -> None:
    df, _ = load_panel(verbose=False)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")

    print("클리핑 적용")
    df = apply_clipping(df, groups)

    kept_all = groups[groups["kept"]]["column"].tolist()
    shrunk_all = [c for c in kept_all if c.endswith("_rate_t_shrunk")]
    contamination = measure_contamination(df, shrunk_all)
    n_bad = sum(1 for v in contamination.values() if v >= CONTAMINATION_THRESHOLD)
    print(f"\n축소 피처 {len(shrunk_all)}개 중 리뷰 수 오염 "
          f"|상관| >= {CONTAMINATION_THRESHOLD}: {n_bad}개")

    sets = build_feature_sets(groups, contamination)
    print(f"\n실험 조합 {len(sets)}개\n")

    rows = []
    for name, feats in sets.items():
        feats = list(dict.fromkeys(feats))  # 순서 유지 중복 제거
        row, _ = run_model(df, feats, name, eval_split="valid")
        rows.append(row)
        print(f"  {name:<34} 피처 {len(feats):>3}개  "
              f"PR-AUC {row['pr_auc']:.4f}  ROC {row['roc_auc']:.4f}  "
              f"R@5% {row['recall_at_5pct']:.4f}")

    result = to_result_frame(rows)
    result.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    base = result.loc[result["model_name"] == "S0_minimum_structured", "pr_auc"].iloc[0]
    result = result.assign(pr_auc_gain=(result["pr_auc"] - base).round(5))

    print("\n" + "=" * 78)
    print(f"S0(정형 7개) 대비 PR-AUC 개선 (기준선 {base:.4f}, valid 양성률 12.92%)")
    print("=" * 78)
    print(result.sort_values("pr_auc_gain", ascending=False)[
        ["model_name", "feature_count", "pr_auc", "pr_auc_gain", "recall_at_5pct"]
    ].to_string(index=False))

    # ---- 진단: 과거 기준선이 텍스트 기여를 얼마나 흡수하는가 ----
    def pr(name):
        hit = result.loc[result["model_name"] == name, "pr_auc"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    print("\n" + "=" * 78)
    print("진단: 과거 기준선(past_3m_low_rating_ratio) 유무에 따른 텍스트 기여")
    print("=" * 78)
    pairs = [
        ("변화 계열", "S0", "V3_S0_change", "S0+past", "D2_S0_past_change"),
        ("T_cause", "S0", "F_T_cause", "S0+past", "D3_S0_past_cause"),
        ("전체 텍스트", "S0", "V5_S0_all_text", "S0+past", "D4_S0_past_all_text"),
    ]
    b2 = pr("D1_S0_plus_past")
    print(f"  기준선 없는 S0 = {base:.4f} / 기준선 있는 S0+past = {b2:.4f}\n")
    print(f"  {'텍스트 그룹':<12} {'S0 기준 기여':>12} {'S0+past 기준 기여':>18} {'차이':>10}")
    for label, _, a_name, _, b_name in pairs:
        ga, gb = pr(a_name) - base, pr(b_name) - b2
        print(f"  {label:<12} {ga:>+12.4f} {gb:>+18.4f} {gb - ga:>+10.4f}")
    print("\n  차이가 크게 음수면 텍스트 변화 기여의 일부는 시간 정보 부재 때문이다.")
    print(f"\n저장: {RESULT_PATH.name}")


if __name__ == "__main__":
    main()
