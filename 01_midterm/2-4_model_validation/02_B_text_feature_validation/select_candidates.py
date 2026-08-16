"""
B-5. 최종 텍스트 후보 선정.

1차 실험에서 리뷰 수 오염을 제거한 축소 계열(clean shrunk)이
텍스트 전체보다 적은 피처로 더 좋은 성능을 냈다.
여기서 그 조합에 무엇을 더할 때 개선되는지 좁혀 후보 2~3개를 확정한다.

각 후보는 두 기준선에서 모두 평가한다.
  S0        A가 지정한 최소 정형 7개. 시간 정보 없음
  S0+past   3개월 기준선을 더한 조건. A의 시계열 모델에 가까운 환경

A의 확정 모델은 1/3/6/12개월 이력을 쓰므로 실제 기여는 S0+past 쪽
수치에 더 가까울 것으로 본다. 두 값을 모두 넘겨 A가 판단하게 한다.

    python3 select_candidates.py

출력: candidate_results.csv
      candidate_feature_sets.json
      candidate_importance.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common_input import cfg, load_panel
from evaluation import run_model, to_result_frame
from run_experiments import (
    CONTAMINATION_THRESHOLD,
    PAST_BASELINE,
    S0_SEVEN,
    apply_clipping,
    measure_contamination,
)

HERE = Path(__file__).resolve().parent
GROUPS_PATH = HERE / "text_feature_groups.csv"
RESULT_PATH = HERE / "candidate_results.csv"
SETS_PATH = HERE / "candidate_feature_sets.json"
IMPORTANCE_PATH = HERE / "candidate_importance.csv"

N_FINAL = 3


def build_candidates(groups: pd.DataFrame, contamination: dict) -> dict[str, list[str]]:
    kept = groups[groups["kept"]]

    def pick(**cond) -> list[str]:
        sub = kept
        for key, values in cond.items():
            values = values if isinstance(values, (list, tuple)) else [values]
            sub = sub[sub[key].isin(values)]
        return sub["column"].tolist()

    shrunk = pick(variant="shrunk")
    clean_shrunk = [c for c in shrunk
                    if contamination.get(c, 0.0) < CONTAMINATION_THRESHOLD]
    raw = pick(variant=["raw_t", "raw_mean"])
    change = pick(variant=["past_p3", "delta"])
    static = pick(variant="static")
    all_text = kept["column"].tolist()

    cause_change = [c for c in pick(family="T_cause") if c in change]
    narrative = pick(family="T_narrative")
    content_families = pick(family=["T_cause", "T_consequence"])

    return {
        # 기준
        "B0_baseline_only": [],
        # 핵심 후보
        "B1_clean_shrunk": clean_shrunk,
        "B2_clean_shrunk_cause_change": clean_shrunk + cause_change,
        "B3_clean_shrunk_narrative": sorted(set(clean_shrunk + narrative)),
        "B4_clean_shrunk_raw": sorted(set(clean_shrunk + raw)),
        "B5_clean_shrunk_change": sorted(set(clean_shrunk + change)),
        "B6_clean_shrunk_content": sorted(
            set(clean_shrunk) | (set(content_families) - set(change))
        ),
        # 참고
        "B7_all_text": all_text,
        "B8_no_shrunk": raw + change + static,
    }


def main() -> None:
    df, _ = load_panel(verbose=False)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    df = apply_clipping(df, groups)

    kept_all = groups[groups["kept"]]["column"].tolist()
    shrunk_all = [c for c in kept_all if c.endswith("_rate_t_shrunk")]
    contamination = measure_contamination(df, shrunk_all)

    candidates = build_candidates(groups, contamination)
    print(f"\n후보 {len(candidates)}개 x 기준선 2종 = {len(candidates) * 2}회 학습\n")

    rows, models = [], {}
    for base_name, base_feats in [
        ("S0", list(S0_SEVEN)),
        ("S0past", list(S0_SEVEN) + [PAST_BASELINE]),
    ]:
        for name, text_feats in candidates.items():
            feats = list(dict.fromkeys(base_feats + text_feats))
            model_name = f"{name}__{base_name}"
            row, _ = run_model(df, feats, model_name, eval_split="valid")
            row["_candidate"] = name
            row["_baseline"] = base_name
            row["_n_text"] = len(text_feats)
            rows.append(row)
            models[model_name] = feats
            print(f"  {model_name:<38} 텍스트 {len(text_feats):>3}개  "
                  f"PR-AUC {row['pr_auc']:.4f}  R@5% {row['recall_at_5pct']:.4f}")

    full = pd.DataFrame(rows)
    to_result_frame(rows).to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    # ---------------- 후보 비교표 ----------------
    wide = full.pivot(index="_candidate", columns="_baseline", values="pr_auc")
    n_text = full.groupby("_candidate")["_n_text"].first()
    r5 = full[full["_baseline"] == "S0past"].set_index("_candidate")["recall_at_5pct"]

    base_s0 = wide.loc["B0_baseline_only", "S0"]
    base_past = wide.loc["B0_baseline_only", "S0past"]

    table = pd.DataFrame({
        "n_text": n_text,
        "pr_S0": wide["S0"].round(5),
        "gain_S0": (wide["S0"] - base_s0).round(5),
        "pr_S0past": wide["S0past"].round(5),
        "gain_S0past": (wide["S0past"] - base_past).round(5),
        "r5_S0past": r5.round(5),
    }).drop(index="B0_baseline_only")
    table["gain_per_10feat"] = (table["gain_S0past"] / table["n_text"] * 10).round(5)

    print("\n" + "=" * 86)
    print(f"후보 비교  (S0 기준선 {base_s0:.4f} / S0+past 기준선 {base_past:.4f})")
    print("=" * 86)
    print(table.sort_values("gain_S0past", ascending=False).to_string())

    # 선정: S0+past 기준 개선폭 우선, 동률이면 피처 수가 적은 쪽
    ranked = table.sort_values(
        ["gain_S0past", "n_text"], ascending=[False, True]
    ).head(N_FINAL)

    print("\n" + "=" * 86)
    print(f"최종 후보 {N_FINAL}개")
    print("=" * 86)
    for i, (name, r) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {name}")
        print(f"     텍스트 {int(r['n_text'])}개  "
              f"S0 기준 {r['gain_S0']:+.4f}  S0+past 기준 {r['gain_S0past']:+.4f}  "
              f"R@5% {r['r5_S0past']:.4f}")

    payload = {
        "generated_by": "02_B_text_feature_validation/select_candidates.py",
        "evaluated_on": "valid (2022-01~2022-08)",
        "baseline_S0": list(S0_SEVEN),
        "baseline_S0past": list(S0_SEVEN) + [PAST_BASELINE],
        "contamination_threshold": CONTAMINATION_THRESHOLD,
        "candidates": {
            name: {
                "text_features": candidates[name],
                "n_text": int(table.loc[name, "n_text"]),
                "pr_auc_S0": float(table.loc[name, "pr_S0"]),
                "pr_auc_S0past": float(table.loc[name, "pr_S0past"]),
                "gain_S0": float(table.loc[name, "gain_S0"]),
                "gain_S0past": float(table.loc[name, "gain_S0past"]),
            }
            for name in ranked.index
        },
    }
    SETS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---------------- 1순위 후보의 피처 중요도 ----------------
    import lightgbm as lgb

    top = ranked.index[0]
    feats = models[f"{top}__S0past"]
    train = df[df["split"] == "train"]
    params = dict(cfg.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = (
        (train[cfg.TARGET_COLUMN] == 0).sum() / (train[cfg.TARGET_COLUMN] == 1).sum()
    )
    params.setdefault("verbose", -1)
    m = lgb.LGBMClassifier(**params).fit(train[feats], train[cfg.TARGET_COLUMN])

    imp = (pd.DataFrame({"feature": feats, "gain": m.booster_.feature_importance("gain")})
           .sort_values("gain", ascending=False))
    imp["share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 86)
    print(f"1순위 후보 {top} 의 피처 중요도 상위 15개")
    print("=" * 86)
    print(imp.head(15).to_string(index=False))

    print(f"\n저장: {RESULT_PATH.name}, {SETS_PATH.name}, {IMPORTANCE_PATH.name}")


if __name__ == "__main__":
    main()
