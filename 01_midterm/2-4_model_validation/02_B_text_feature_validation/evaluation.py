"""
B-3. 공통 평가 모듈.

00_preparation/README.md 8절(공통 모델 설정)과 9절(공통 평가 지표)을 강제한다.
A와 B가 같은 방식으로 성능을 계산해야 결과를 비교할 수 있다.

핵심 규칙
  - LightGBM 파라미터는 experiment_config.LIGHTGBM_PARAMS 고정
  - scale_pos_weight는 train 구간의 음성/양성 비로만 계산한다
  - 확률 동점은 parent_asin, year_month 오름차순으로 순위를 고정한다
  - 결과 열 순서는 experiment_config.RESULT_COLUMNS를 따른다
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common_input import cfg


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """train 구간의 음성 행 수를 양성 행 수로 나눈 값.

    valid나 test의 라벨 비율을 쓰면 검증 정보가 학습에 새어 들어간다.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    assert n_pos > 0, "train 구간에 양성이 없다"
    return n_neg / n_pos


def recall_at_top(frame: pd.DataFrame, proba_col: str, pct: float) -> float:
    """예측 확률 상위 pct 비율 안에 포함된 전체 양성의 비율.

    확률이 같을 때 순서가 실행마다 달라지면 지표가 흔들리므로
    parent_asin, year_month 오름차순으로 동점을 고정한다.
    """
    n_top = math.ceil(len(frame) * pct)
    ordered = frame.sort_values(
        [proba_col, "parent_asin", "year_month"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    total_pos = int(frame[cfg.TARGET_COLUMN].sum())
    if total_pos == 0:
        return float("nan")
    hit = int(ordered.head(n_top)[cfg.TARGET_COLUMN].sum())
    return hit / total_pos


def evaluate(frame: pd.DataFrame, proba: np.ndarray) -> dict:
    """PR-AUC, ROC-AUC, Recall@Top 5%, Recall@Top 10%."""
    work = frame[cfg.KEY_COLUMNS + [cfg.TARGET_COLUMN]].copy()
    work["proba"] = proba
    y = work[cfg.TARGET_COLUMN]
    return {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "recall_at_5pct": recall_at_top(work, "proba", 0.05),
        "recall_at_10pct": recall_at_top(work, "proba", 0.10),
    }


def run_model(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    eval_split: str = "valid",
) -> tuple[dict, np.ndarray]:
    """train으로 학습하고 지정한 구간에서 평가한다.

    eval_split의 기본값이 valid인 이유는 1일 차와 2일 차에 test를
    열지 않기 때문이다. test 평가는 3일 차 A의 작업이다.
    """
    import lightgbm as lgb

    assert eval_split != "test", (
        "test는 3일 차 최종 평가 전까지 사용할 수 없다"
    )
    missing = [c for c in features if c not in df.columns]
    assert not missing, f"패널에 없는 피처: {missing}"

    train = df[df["split"] == "train"]
    holdout = df[df["split"] == eval_split]

    params = dict(cfg.LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(train[cfg.TARGET_COLUMN])
    params.setdefault("verbose", -1)

    model = lgb.LGBMClassifier(**params)
    model.fit(train[features], train[cfg.TARGET_COLUMN])
    proba = model.predict_proba(holdout[features])[:, 1]

    metrics = evaluate(holdout, proba)
    row = {
        "model_name": model_name,
        "split": eval_split,
        **{k: round(v, 5) for k, v in metrics.items()},
        "n_rows": len(holdout),
        "positive_rate": round(float(holdout[cfg.TARGET_COLUMN].mean()), 5),
        "feature_count": len(features),
        "random_state": cfg.LIGHTGBM_PARAMS["random_state"],
    }
    return row, proba


def to_result_frame(rows: list[dict]) -> pd.DataFrame:
    """README 10절의 결과 전달 형식으로 열 순서를 맞춘다."""
    out = pd.DataFrame(rows)
    missing = [c for c in cfg.RESULT_COLUMNS if c not in out.columns]
    assert not missing, f"결과 열 누락: {missing}"
    return out[cfg.RESULT_COLUMNS]
