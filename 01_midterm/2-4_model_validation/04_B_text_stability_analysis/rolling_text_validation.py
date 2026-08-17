"""
B-2일차 2단계. 텍스트 기여의 시간 구간 안정성 검증.

valid 단일 구간(2022-01~08)은 7,228행에 양성 934개뿐이다.
이 규모에서 PR-AUC 0.005 차이는 양성 몇 개의 순위 변동으로도 생긴다.
따라서 A1이 정형 모델에 적용한 것과 동일한 4개 fold에서 다시 잰다.

fold 정의는 A1의 rolling_long_candidate_metrics.csv와 같다.
train 행 수와 scale_pos_weight까지 일치하는 것을 확인했다.

판정 기준
  기여가 4개 fold에서 일관되게 양수인가
  fold 간 표준편차가 평균 기여보다 작은가
  둘 다 만족해야 실제 신호로 본다

    python3 rolling_text_validation.py

출력: rolling_text_metrics.csv
      rolling_text_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from a_baseline import B1_DIR, load_combined

import sys
sys.path.insert(0, str(B1_DIR))
from common_input import cfg  # noqa: E402
from evaluation import compute_scale_pos_weight, evaluate  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUPS_PATH = B1_DIR / "text_feature_groups.csv"
SAFE_PATH = B1_DIR.parent / "00_preparation" / "safe_text_features.csv"
METRICS_PATH = HERE / "rolling_text_metrics.csv"
SUMMARY_PATH = HERE / "rolling_text_summary.csv"

# A1 rolling_long_candidate_metrics.csv와 동일한 구간 정의.
FOLDS = [
    ("fold_1_2020_01_08", "2019-12", "2020-01", "2020-08"),
    ("fold_2_2020_09_2021_04", "2020-08", "2020-09", "2021-04"),
    ("fold_3_2021_05_12", "2021-04", "2021-05", "2021-12"),
    ("fold_4_2022_01_08", "2021-12", "2022-01", "2022-08"),
]

CONTAMINATION_THRESHOLD = 0.85


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
    all_kept = kept["column"].tolist()

    train = df[df["split"] == "train"]
    rc = train[shrunk + ["review_count"]].rank(method="average").corr()["review_count"]
    clean_shrunk = [c for c in shrunk if abs(float(rc[c])) < CONTAMINATION_THRESHOLD]

    # A가 제외 14개를 반영하지 않고 105개를 그대로 썼을 경우를 재현한다.
    safe105 = pd.read_csv(SAFE_PATH, encoding="utf-8-sig")["column"].tolist()
    safe105 = [c for c in safe105 if c in df.columns]

    return {
        "T1_raw_only": raw,
        "T2_cause_no_shrunk": [c for c in pick(family="T_cause") if c not in shrunk],
        "T3_all_minus_shrunk": [c for c in all_kept if c not in shrunk],
        "T4_B5_clean_shrunk_change": sorted(set(clean_shrunk + p3 + delta)),
        "T5_B8_no_shrunk": raw + p3 + delta + static,
        "T6_p3_and_delta": p3 + delta,
        "T7_all_kept_91": all_kept,
        "T8_safe105_unfiltered": safe105,
    }


def run_fold(df, features, train_end, valid_start, valid_end):
    import lightgbm as lgb

    train = df[df["year_month"] <= train_end]
    valid = df[(df["year_month"] >= valid_start) & (df["year_month"] <= valid_end)]

    params = dict(cfg.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(train[cfg.TARGET_COLUMN])
    params.setdefault("verbose", -1)

    model = lgb.LGBMClassifier(**params)
    model.fit(train[features], train[cfg.TARGET_COLUMN])
    proba = model.predict_proba(valid[features])[:, 1]
    m = evaluate(valid, proba)
    m.update(n_train=len(train), n_valid=len(valid),
             positive_rate=float(valid[cfg.TARGET_COLUMN].mean()),
             feature_count=len(features))
    return m


def main() -> None:
    df, a_features, _ = load_combined(verbose=False)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    sets = build_sets(groups, df)

    print("=" * 88)
    print(f"롤링 검증: fold {len(FOLDS)}개 x 조합 {len(sets) + 1}개 = "
          f"{len(FOLDS) * (len(sets) + 1)}회 학습")
    print("=" * 88)

    rows = []
    for fold, tend, vs, ve in FOLDS:
        base = run_fold(df, a_features, tend, vs, ve)
        rows.append({"fold": fold, "model_name": "A_baseline_45",
                     "n_text": 0, **base})
        print(f"\n[{fold}]  기준선 PR-AUC {base['pr_auc']:.5f}  "
              f"(train {base['n_train']:,} / valid {base['n_valid']:,})")

        for name, text_feats in sets.items():
            feats = list(dict.fromkeys(a_features + text_feats))
            m = run_fold(df, feats, tend, vs, ve)
            rows.append({"fold": fold, "model_name": name,
                         "n_text": len(text_feats), **m})
            gain = m["pr_auc"] - base["pr_auc"]
            mark = "+" if gain > 0 else "-"
            print(f"    {mark} {name:<28} 텍스트 {len(text_feats):>3}개  "
                  f"PR-AUC {m['pr_auc']:.5f}  ({gain:+.5f})")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    # ---------------- fold별 기여 요약 ----------------
    base_by_fold = (metrics[metrics["model_name"] == "A_baseline_45"]
                    .set_index("fold")["pr_auc"])
    work = metrics[metrics["model_name"] != "A_baseline_45"].copy()
    work["gain"] = work.apply(
        lambda r: r["pr_auc"] - base_by_fold[r["fold"]], axis=1)

    summary = (work.groupby("model_name")
               .agg(n_text=("n_text", "first"),
                    gain_mean=("gain", "mean"),
                    gain_std=("gain", "std"),
                    gain_min=("gain", "min"),
                    gain_max=("gain", "max"),
                    wins=("gain", lambda s: int((s > 0).sum())),
                    r10_mean=("recall_at_10pct", "mean"))
               .round(5)
               .sort_values("gain_mean", ascending=False))
    # 평균 기여를 변동성으로 나눈 값. 클수록 안정적으로 기여한다.
    summary["stability"] = (summary["gain_mean"] / summary["gain_std"]).round(3)
    summary.to_csv(SUMMARY_PATH, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("4개 fold 종합 (gain = A 기준선 대비 PR-AUC 차이)")
    print("=" * 88)
    print(summary.to_string())

    print("\n" + "=" * 88)
    print("판정")
    print("=" * 88)
    solid = summary[(summary["wins"] == 4) & (summary["gain_mean"] > 0)]
    if len(solid):
        print("  4개 fold 전부에서 기여한 조합:")
        for name, r in solid.iterrows():
            print(f"    {name:<28} 평균 {r['gain_mean']:+.5f}  "
                  f"최소 {r['gain_min']:+.5f}  표준편차 {r['gain_std']:.5f}")
    else:
        print("  4개 fold 전부에서 기여한 조합이 없다.")
        print("  단일 valid 구간에서 관측된 개선은 우연일 가능성이 높다.")

    print("\n  참고: fold별 승수 분포")
    for w in range(5):
        names = summary[summary["wins"] == w].index.tolist()
        if names:
            print(f"    {w}/4 승: {', '.join(names)}")

    print(f"\n저장: {METRICS_PATH.name}, {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
