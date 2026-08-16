"""
B-6. 최종 후보 확정.

select_candidates.py의 자동 정렬은 개선폭만 사용하므로
B7_all_text처럼 성능이 낮으면서 피처가 더 많은 열등 후보가 섞인다.
여기서 열등 후보를 걸러내고 A에게 넘길 3개를 확정한다.

선정 근거
  B5  S0+past 기준 개선폭 1위
  B1  텍스트 10개당 개선폭 1위. 현재 시점 토픽 비율만 사용해
      A의 장기 이력 모델과 정보가 겹치지 않는다
  B8  Recall@Top 5% 1위. 축소추정을 쓰지 않은 대안

재학습하지 않고 candidate_results.csv를 다시 읽어 정리한다.

    python3 finalize_candidates.py

출력: candidate_feature_sets.json (덮어씀)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common_input import load_panel
from run_experiments import (
    CONTAMINATION_THRESHOLD,
    PAST_BASELINE,
    S0_SEVEN,
    apply_clipping,
    measure_contamination,
)
from select_candidates import build_candidates

HERE = Path(__file__).resolve().parent
GROUPS_PATH = HERE / "text_feature_groups.csv"
RESULT_PATH = HERE / "candidate_results.csv"
SETS_PATH = HERE / "candidate_feature_sets.json"

FINAL = {
    "B5_clean_shrunk_change": {
        "rank": 1,
        "why": (
            "S0+past 기준 개선폭 1위(+0.0136). 다만 텍스트 기여 상위가 "
            "_rate_p3, _delta 등 3개월 창 변화 피처라 A의 1/3/6/12개월 이력과 "
            "겹칠 여지가 있다"
        ),
        "watch": "A의 장기 이력 모델에서 기여가 줄어드는지 우선 확인 필요",
    },
    "B1_clean_shrunk": {
        "rank": 2,
        "why": (
            "텍스트 10개당 개선폭 1위(0.0077, B5의 3.2배). 15개로 "
            "91개 전체(+0.0119)에 근접한 +0.0115를 낸다"
        ),
        "watch": (
            "전부 현재 시점 토픽 비율이라 별점 이력이 담을 수 없는 정보다. "
            "장기 이력 모델에서 가장 잘 살아남을 것으로 예상"
        ),
    },
    "B8_no_shrunk": {
        "rank": 3,
        "why": (
            "Recall@Top 5% 1위(0.0964). 축소추정을 전혀 쓰지 않아 "
            "리뷰 수 오염 문제에서 자유롭다"
        ),
        "watch": "PR-AUC는 셋 중 가장 낮다. 상위 경보 위주 운영일 때 유리",
    },
}

EXCLUDED_NOTE = {
    "B7_all_text": (
        "B5보다 텍스트 34개를 더 쓰면서 PR-AUC가 낮다(0.19416 < 0.19580). "
        "열등 후보라 제외"
    ),
}


def main() -> None:
    df, _ = load_panel(verbose=False)
    groups = pd.read_csv(GROUPS_PATH, encoding="utf-8-sig")
    df = apply_clipping(df, groups)

    kept_all = groups[groups["kept"]]["column"].tolist()
    shrunk_all = [c for c in kept_all if c.endswith("_rate_t_shrunk")]
    contamination = measure_contamination(df, shrunk_all)
    candidates = build_candidates(groups, contamination)

    res = pd.read_csv(RESULT_PATH, encoding="utf-8-sig")
    res["_candidate"] = res["model_name"].str.split("__").str[0]
    res["_baseline"] = res["model_name"].str.split("__").str[1]
    piv = res.pivot(index="_candidate", columns="_baseline", values="pr_auc")
    r5 = res.pivot(index="_candidate", columns="_baseline", values="recall_at_5pct")
    base_s0 = float(piv.loc["B0_baseline_only", "S0"])
    base_past = float(piv.loc["B0_baseline_only", "S0past"])

    payload = {
        "generated_by": "02_B_text_feature_validation/finalize_candidates.py",
        "evaluated_on": "valid 2022-01~2022-08, test 미사용",
        "primary_metric": "pr_auc",
        "baselines": {
            "S0": {"features": list(S0_SEVEN), "pr_auc": round(base_s0, 5)},
            "S0past": {
                "features": list(S0_SEVEN) + [PAST_BASELINE],
                "pr_auc": round(base_past, 5),
                "note": "A가 지정한 S0에는 시간 정보가 없어 진단용으로 추가한 기준선",
            },
        },
        "contamination_threshold": CONTAMINATION_THRESHOLD,
        "safe_text_features_total": len(kept_all),
        "candidates": [],
        "excluded_candidates": EXCLUDED_NOTE,
    }

    for name, meta in sorted(FINAL.items(), key=lambda kv: kv[1]["rank"]):
        feats = candidates[name]
        payload["candidates"].append({
            "rank": meta["rank"],
            "name": name,
            "n_text_features": len(feats),
            "pr_auc_S0": round(float(piv.loc[name, "S0"]), 5),
            "gain_S0": round(float(piv.loc[name, "S0"]) - base_s0, 5),
            "pr_auc_S0past": round(float(piv.loc[name, "S0past"]), 5),
            "gain_S0past": round(float(piv.loc[name, "S0past"]) - base_past, 5),
            "recall_at_5pct_S0past": round(float(r5.loc[name, "S0past"]), 5),
            "selection_reason": meta["why"],
            "caution": meta["watch"],
            "text_features": feats,
        })

    SETS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print("=" * 78)
    print("최종 후보 확정")
    print("=" * 78)
    for c in payload["candidates"]:
        print(f"  {c['rank']}. {c['name']}  텍스트 {c['n_text_features']}개")
        print(f"     S0 {c['gain_S0']:+.4f} / S0+past {c['gain_S0past']:+.4f} / "
              f"R@5% {c['recall_at_5pct_S0past']:.4f}")
        print(f"     {c['selection_reason']}")
        print()
    print(f"제외: {list(EXCLUDED_NOTE)}")
    print(f"저장: {SETS_PATH.name}")


if __name__ == "__main__":
    main()
