"""
B-2일차 1단계. A 기준 모델 위에서 텍스트 기여 재측정.

1일차 후보(B1, B5, B8)는 최소 정형 3~7개 기준선에서 선정했다.
A의 실제 기준 모델은 정형 45개에 Valid PR-AUC 0.2311로 훨씬 강하다.
같은 후보를 A 기준선 위에서 다시 재고, 순위가 왜 뒤집혔는지 설명한다.

검증할 가설
  축소추정(_rate_t_shrunk)은 값 자체에 리뷰 수 정보를 담고 있다.
  A의 45개 중 18개가 이미 리뷰 볼륨을 인코딩하므로 중복이 되고,
  약한 기준선에서 도움이 되던 축소 계열이 여기서는 기여하지 못한다.

분해할 축
  p3 단독 / delta 단독 / p3+delta
  A2에서 관측된 시너지를 정량화한다.

    python3 measure_on_a_baseline.py

출력: text_gain_on_a_baseline.csv
      shrinkage_overlap.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from a_baseline import A_CURRENT_7, B1_DIR, load_combined

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402
from evaluation import run_model, to_result_frame  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUPS_PATH = B1_DIR / "text_feature_groups.csv"
RESULT_PATH = HERE / "text_gain_on_a_baseline.csv"
OVERLAP_PATH = HERE / "shrinkage_overlap.csv"

CONTAMINATION_THRESHOLD = 0.85

# A의 45개 중 리뷰 볼륨을 인코딩하는 피처.
# 축소추정의 분모(n)와 정보가 겹치는지 확인하는 대상이다.
A_VOLUME_FEATURES = (
    ["review_count", "low_rating_count"]
    + [f"history_{w}m_review_count" for w in (1, 3, 6, 12)]
    + [f"history_{w}m_low_rating_count" for w in (1, 3, 6, 12)]
    + [f"history_{w}m_active_month_count" for w in (1, 3, 6, 12)]
    + [f"history_{w}m_coverage_ratio" for w in (1, 3, 6, 12)]
)


def measure_shrinkage_overlap(df: pd.DataFrame, shrunk: list[str]) -> pd.DataFrame:
    """축소 피처가 A의 볼륨 피처와 얼마나 겹치는지 train 순위상관으로 잰다."""
    train = df[df["split"] == "train"]
    vols = [c for c in A_VOLUME_FEATURES if c in train.columns]
    ranked = train[shrunk + vols].rank(method="average")
    corr = ranked.corr(method="pearson").loc[shrunk, vols].abs()

    out = pd.DataFrame({
        "column": shrunk,
        "max_corr_volume": corr.max(axis=1).round(4).values,
        "closest_volume": corr.idxmax(axis=1).values,
        "mean_corr_volume": corr.mean(axis=1).round(4).values,
    }).sort_values("max_corr_volume", ascending=False)
    out.to_csv(OVERLAP_PATH, index=False, encoding="utf-8-sig")

    print("=" * 82)
    print(f"축소추정 피처와 A의 볼륨 피처 {len(vols)}개의 중복도 (train 순위상관 절댓값)")
    print("=" * 82)
    print(out.to_string(index=False))
    n_high = int((out["max_corr_volume"] >= 0.85).sum())
    print(f"\n  |상관| 0.85 이상: {n_high}/{len(out)}개")
    print(f"  평균 최대상관   : {out['max_corr_volume'].mean():.4f}")
    print("  값이 높을수록 축소값이 A의 정형 피처와 같은 정보를 담고 있다는 뜻이다.\n")
    return out


def build_sets(groups: pd.DataFrame, df: pd.DataFrame) -> dict[str, list[str]]:
    kept = groups[groups["kept"]]

    def pick(**cond) -> list[str]:
        sub = kept
        for key, values in cond.items():
            values = values if isinstance(values, (list, tuple)) else [values]
            sub = sub[sub[key].isin(values)]
        return sub["column"].tolist()

    shrunk = pick(variant="shrunk")
    raw = pick(variant=["raw_t", "raw_mean"])
    p3 = pick(variant="past_p3")
    delta = pick(variant="delta")
    static = pick(variant="static")
    all_text = kept["column"].tolist()

    # 1일차 오염 기준으로 나눈 축소 하위군 (review_count 단독 기준)
    train = df[df["split"] == "train"]
    rc = train[shrunk + ["review_count"]].rank(method="average").corr()["review_count"]
    clean_shrunk = [c for c in shrunk if abs(float(rc[c])) < CONTAMINATION_THRESHOLD]

    return {
        # 1일차 후보 재측정
        "D1_B1_clean_shrunk": clean_shrunk,
        "D2_B5_clean_shrunk_change": sorted(set(clean_shrunk + p3 + delta)),
        "D3_B8_no_shrunk": raw + p3 + delta + static,
        "D4_all_text": all_text,
        # 축소 가설 검증
        "S1_shrunk_only": shrunk,
        "S2_all_text_minus_shrunk": [c for c in all_text if c not in shrunk],
        # p3 x delta 시너지 분해
        "P1_p3_only": p3,
        "P2_delta_only": delta,
        "P3_p3_and_delta": p3 + delta,
        "P4_raw_only": raw,
        "P5_raw_p3_delta": raw + p3 + delta,
        # 의미 그룹
        "G_cause_no_shrunk": [c for c in pick(family="T_cause") if c not in shrunk],
        "G_consequence_no_shrunk": [c for c in pick(family="T_consequence") if c not in shrunk],
        "G_narrative_no_shrunk": [c for c in pick(family="T_narrative") if c not in shrunk],
    }


def main() -> None:
    df, a_features, _ = load_combined()
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")

    shrunk = groups[(groups["kept"]) & (groups["variant"] == "shrunk")]["column"].tolist()
    measure_shrinkage_overlap(df, shrunk)

    sets = build_sets(groups, df)

    print("=" * 82)
    print(f"A 기준 모델(정형 45개) + 텍스트 조합 {len(sets) + 1}회 학습")
    print("=" * 82)

    rows = []
    base_row, _ = run_model(df, a_features, "A_baseline_45", eval_split="valid")
    base_row["_n_text"] = 0
    rows.append(base_row)
    base = base_row["pr_auc"]
    print(f"  {'A_baseline_45':<30} 텍스트   0개  PR-AUC {base:.5f}")

    for name, text_feats in sets.items():
        feats = list(dict.fromkeys(a_features + text_feats))
        row, _ = run_model(df, feats, name, eval_split="valid")
        row["_n_text"] = len(text_feats)
        rows.append(row)
        print(f"  {name:<30} 텍스트 {len(text_feats):>3}개  "
              f"PR-AUC {row['pr_auc']:.5f}  ({row['pr_auc'] - base:+.5f})  "
              f"R@10% {row['recall_at_10pct']:.4f}")

    result = to_result_frame(rows)
    result.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    full = pd.DataFrame(rows)
    full["gain"] = (full["pr_auc"] - base).round(5)

    print("\n" + "=" * 82)
    print(f"A 기준선 {base:.5f} 대비 텍스트 기여")
    print("=" * 82)
    print(full[full["model_name"] != "A_baseline_45"]
          .sort_values("gain", ascending=False)
          [["model_name", "_n_text", "pr_auc", "gain", "recall_at_5pct", "recall_at_10pct"]]
          .to_string(index=False))

    def g(name):
        hit = full.loc[full["model_name"] == name, "gain"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    print("\n" + "=" * 82)
    print("p3 x delta 시너지 분해")
    print("=" * 82)
    p1, p2, p3g = g("P1_p3_only"), g("P2_delta_only"), g("P3_p3_and_delta")
    print(f"  p3 단독        {p1:+.5f}")
    print(f"  delta 단독     {p2:+.5f}")
    print(f"  p3 + delta     {p3g:+.5f}")
    print(f"  단독 합        {p1 + p2:+.5f}")
    print(f"  시너지         {p3g - (p1 + p2):+.5f}")
    print("  시너지가 양수면 두 정보가 함께 있을 때만 작동하는 신호가 있다는 뜻이다.")

    print("\n" + "=" * 82)
    print("축소추정 가설")
    print("=" * 82)
    print(f"  축소만              {g('S1_shrunk_only'):+.5f}")
    print(f"  전체 텍스트         {g('D4_all_text'):+.5f}")
    print(f"  전체 텍스트 - 축소  {g('S2_all_text_minus_shrunk'):+.5f}")
    print("  축소를 뺐을 때가 더 좋으면 축소 계열이 A 기준선에서 방해가 된다는 뜻이다.")

    print(f"\n저장: {RESULT_PATH.name}, {OVERLAP_PATH.name}")


if __name__ == "__main__":
    main()
