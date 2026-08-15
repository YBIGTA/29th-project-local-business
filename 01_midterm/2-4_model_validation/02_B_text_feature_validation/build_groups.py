"""
B-2. 텍스트 피처 그룹 정의와 제외 목록 확정.

audit_features.py의 진단 결과를 근거로
  (1) 축소추정 피처가 리뷰 수를 인코딩하는지 부호까지 확인하고
  (2) 제외 피처와 제외 이유를 파일로 고정하며
  (3) 실험에 사용할 그룹 정의를 만든다.

    python3 build_groups.py

출력: excluded_features.csv
      text_feature_groups.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_input import cfg, load_panel

HERE = Path(__file__).resolve().parent
AUDIT_PATH = HERE / "feature_audit.csv"
EXCLUDED_PATH = HERE / "excluded_features.csv"
GROUPS_PATH = HERE / "text_feature_groups.csv"

# ttf_median_days는 리뷰의 기간 표현 파싱 오류로 최대 31,755일(약 87년)까지 튄다.
# 제외 대신 상한을 두어 트리 분할이 극단값에 끌려가지 않게 한다.
CLIP_RULES = {"ttf_median_days": 730.0}  # 2년

# 규칙만으로는 판정할 수 없어 근거를 명시해 직접 제외하는 피처.
MANUAL_EXCLUSIONS = {
    "n_reviews_p3": (
        "past_3m_review_count와 순위상관 0.9384. "
        "2-2가 corr 0.940 기준으로 past_3m_review_count를 제외했으므로 동일 기준 적용"
    ),
    "cause_topic_count_mean": "any_cause_topic_rate_t와 순위상관 0.9920. 비율 쪽을 남김",
    "consequence_topic_count_mean": "any_consequence_topic_rate_t와 순위상관 0.9984. 비율 쪽을 남김",
    "emotional_count_mean": "emotional_flag_rate_t와 순위상관 0.9979. 비율 쪽을 남김",
}

VARIANT_BY_SUFFIX = {
    "_rate_t_shrunk": "shrunk",
    "_rate_t": "raw_t",
    "_mean": "raw_mean",
    "_rate_p3": "past_p3",
    "_delta": "delta",
    "(none)": "static",
}


def verify_shrunk_sign(df, audit) -> pd.DataFrame:
    """축소추정 피처가 리뷰 수와 음의 관계인지 부호까지 확인한다.

    희소 토픽의 축소값은 k*prior/(n+k)로 수렴하므로 리뷰 수가 늘면 작아진다.
    이 예측이 맞다면 corr(shrunk, review_count) < 0 이고
    라벨과는 양의 상관으로 나타난다.
    """
    train = df[df["split"] == "train"]
    shrunk = [c for c in audit["column"] if c.endswith("_rate_t_shrunk")]
    cols = shrunk + ["review_count", cfg.TARGET_COLUMN]
    corr = train[cols].rank(method="average").corr(method="pearson")

    out = pd.DataFrame({
        "column": shrunk,
        "corr_review_count": [round(float(corr.loc[c, "review_count"]), 4) for c in shrunk],
        "corr_label": [round(float(corr.loc[c, cfg.TARGET_COLUMN]), 4) for c in shrunk],
    }).sort_values("corr_review_count")

    print("=" * 78)
    print("축소추정 피처의 부호 검증 (train 순위상관)")
    print("=" * 78)
    print(out.to_string(index=False))
    n_neg = int((out["corr_review_count"] < 0).sum())
    print(f"\n  리뷰 수와 음의 상관: {n_neg}/{len(out)}개")
    print(f"  |상관| 0.85 이상   : {int((out['corr_review_count'].abs() >= 0.85).sum())}개")
    print("  음의 상관이 다수이면 축소값이 리뷰 수를 대리한다는 해석과 일치한다.\n")
    return out


def main() -> None:
    df, safe = load_panel(verbose=False)
    audit = pd.read_csv(AUDIT_PATH, encoding="utf-8-sig")

    sign_check = verify_shrunk_sign(df, audit)

    # ---------------- 제외 판정 ----------------
    reasons = {}

    for _, r in audit.iterrows():
        col = r["column"]
        if r["flag_constant"]:
            reasons[col] = f"상수. 고유값 {int(r['n_unique'])}개로 변별력 없음"
        elif r["flag_near_constant"]:
            reasons[col] = f"준상수. 단일값이 {r['top_value_share']:.2%}를 차지"
        elif r["flag_dup_structured"]:
            reasons[col] = (
                f"정형 피처 중복. {r['closest_structured']}와 "
                f"순위상관 {r['max_corr_structured']:.4f}"
            )

    for col, why in MANUAL_EXCLUSIONS.items():
        reasons.setdefault(col, why)

    excluded = (
        audit[audit["column"].isin(reasons)]
        [["column", "family", "axis", "nonzero_ratio", "corr_label"]]
        .assign(reason=lambda d: d["column"].map(reasons))
        .sort_values(["family", "column"])
    )
    excluded.to_csv(EXCLUDED_PATH, index=False, encoding="utf-8-sig")

    # ---------------- 그룹 정의 ----------------
    groups = audit[["column", "family", "axis", "suffix", "nonzero_ratio", "corr_label"]].copy()
    groups["variant"] = groups["suffix"].map(VARIANT_BY_SUFFIX).fillna("static")
    groups["kept"] = ~groups["column"].isin(reasons)
    groups["exclusion_reason"] = groups["column"].map(reasons).fillna("")
    groups["clip_upper"] = groups["column"].map(CLIP_RULES)
    groups = groups.sort_values(["family", "variant", "column"])
    groups.to_csv(GROUPS_PATH, index=False, encoding="utf-8-sig")

    kept = groups[groups["kept"]]

    print("=" * 78)
    print(f"제외 확정: {len(excluded)}개")
    print("=" * 78)
    print(excluded[["column", "family", "reason"]].to_string(index=False))
    print()

    print("=" * 78)
    print(f"유지: {len(kept)}개  (105 - {len(excluded)})")
    print("=" * 78)
    print(pd.crosstab(kept["family"], kept["variant"], margins=True))
    print()

    print("실험용 축 요약")
    for v in ["shrunk", "raw_t", "raw_mean", "past_p3", "delta", "static"]:
        n = int((kept["variant"] == v).sum())
        if n:
            print(f"  {v:<10} {n:>3}개")
    print()
    print("클리핑 적용 대상")
    for col, upper in CLIP_RULES.items():
        row = audit[audit["column"] == col]
        if len(row):
            print(f"  {col}: 최대 {float(row['max'].iloc[0]):,.0f} -> 상한 {upper:,.0f}")

    print(f"\n저장: {EXCLUDED_PATH.name}, {GROUPS_PATH.name}")


if __name__ == "__main__":
    main()
