from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path("01_midterm/2-4_model_validation")
OUT = BASE / "01_A_structured_timeseries"
LABELED_PATH = Path("data/processed/product_month_labeled.parquet")
SAFE_TEXT_PATH = BASE / "00_preparation/safe_text_features.csv"

KEY_COLUMNS = ["parent_asin", "year_month"]
LABEL_COLUMNS = [
    "is_low_rating_surge",
    "next_avg_rating",
    "next_low_rating_count",
    "next_low_rating_ratio",
    "next_review_count",
    "next_vs_past_3m_low_rating_change",
]

# 미래·라벨 복제 가능성이 있는 이름은 정형 모델에서도 절대 제외한다.
FORBIDDEN_PATTERNS = [
    r"(^|_)next(_|$)",
    r"(^|_)future(_|$)",
    r"(^|_)target(_|$)",
    r"(^|_)label(_|$)",
    r"(^|_)surge(_|$)",
    r"auto_title",
]

# 날짜 자체·ID성 숫자는 모델 입력에서 제외한다.
ID_OR_DATE_PATTERNS = [
    r"(^|_)(asin|product_id|user_id|month|year|timestamp|time)(_|$)",
]

# 과거 평균·윈도우·변화량은 시계열군, 그 외 현재 시점 값은 현재 상태군.
TREND_PATTERNS = [
    r"past",
    r"prior",
    r"rolling",
    r"window",
    r"lag",
    r"delta",
    r"change",
    r"trend",
    r"growth",
    r"ratio_p",
    r"_p3",
    r"_p6",
    r"_p12",
    r"3m",
    r"6m",
    r"12m",
]

df = pd.read_parquet(LABELED_PATH)
safe_text = pd.read_csv(SAFE_TEXT_PATH)
# 텍스트 피처는 별도 parquet에 있으므로, 안전 피처 목록의 명시적 `column` 열을 사용한다.
if "column" not in safe_text.columns:
    raise ValueError(f"safe_text_features.csv에 column 열이 없습니다: {safe_text.columns.tolist()}")

safe_text_columns = set(safe_text["column"].dropna().astype(str))
print(f"안전 텍스트 피처 수: {len(safe_text_columns)}")

missing = [c for c in KEY_COLUMNS + ["is_low_rating_surge"] if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

if df.duplicated(KEY_COLUMNS).any():
    raise ValueError("product_month_labeled에 (parent_asin, year_month) 중복이 있습니다.")

numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
rows = []

for col in numeric_columns:
    name = str(col).lower()

    if col in KEY_COLUMNS:
        decision, group, reason = "exclude", "excluded", "공통 키"
    elif col in LABEL_COLUMNS:
        decision, group, reason = "exclude", "excluded", "라벨 또는 미래 결과"
    elif col in safe_text_columns or name.startswith("t_"):
        decision, group, reason = "exclude", "excluded", "텍스트 피처는 B 작업 영역"
    elif any(re.search(p, name) for p in FORBIDDEN_PATTERNS):
        decision, group, reason = "exclude", "excluded", "미래 정보·라벨 복제 위험"
    elif any(re.search(p, name) for p in ID_OR_DATE_PATTERNS):
        decision, group, reason = "exclude", "excluded", "ID 또는 시간 식별자"
    elif df[col].nunique(dropna=True) <= 1:
        decision, group, reason = "exclude", "excluded", "상수 컬럼"
    elif any(token in name for token in TREND_PATTERNS):
        decision, group, reason = "use", "timeseries_trend", "과거 누적·변화·윈도우 기반 정형 피처"
    else:
        decision, group, reason = "use", "current_state", "예측 시점 현재 상태의 정형 피처"

    rows.append(
        {
            "feature_name": col,
            "dtype": str(df[col].dtype),
            "missing_rate": round(float(df[col].isna().mean()), 6),
            "n_unique": int(df[col].nunique(dropna=True)),
            "decision": decision,
            "feature_group": group,
            "reason": reason,
        }
    )

spec = pd.DataFrame(rows).sort_values(
    ["decision", "feature_group", "feature_name"],
    ascending=[True, True, True],
)

current_features = spec.query(
    "decision == 'use' and feature_group == 'current_state'"
)["feature_name"].tolist()

trend_features = spec.query(
    "decision == 'use' and feature_group == 'timeseries_trend'"
)["feature_name"].tolist()

all_features = current_features + trend_features

if not current_features:
    raise ValueError("현재 상태 정형 피처가 0개입니다. 피처명을 확인해야 합니다.")
if not trend_features:
    raise ValueError("시계열 변화 정형 피처가 0개입니다. 피처명을 확인해야 합니다.")

panel = df[KEY_COLUMNS + ["is_low_rating_surge"] + all_features].copy()
panel.to_parquet(OUT / "structured_modeling_panel.parquet", index=False)
spec.to_csv(OUT / "structured_feature_spec.csv", index=False, encoding="utf-8-sig")

summary = {
    "source": str(LABELED_PATH),
    "rows": int(len(panel)),
    "unique_products": int(panel["parent_asin"].nunique()),
    "key_duplicate_count": int(panel.duplicated(KEY_COLUMNS).sum()),
    "positive_count": int(panel["is_low_rating_surge"].sum()),
    "positive_rate": round(float(panel["is_low_rating_surge"].mean()), 6),
    "current_state_feature_count": len(current_features),
    "timeseries_trend_feature_count": len(trend_features),
    "total_structured_feature_count": len(all_features),
    "excluded_column_count": int((spec["decision"] == "exclude").sum()),
    "forbidden_feature_in_use": sorted(
        set(all_features) & (set(LABEL_COLUMNS) | safe_text_columns)
    ),
}
if summary["forbidden_feature_in_use"]:
    raise ValueError(f"사용 피처에 금지 컬럼 포함: {summary['forbidden_feature_in_use']}")

(OUT / "structured_panel_validation.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

readme = f"""# A1 정형 시계열 모델링 기반

## 1. 목적
상품×월 시점의 정형 정보만으로 다음 달 저평점 급증(`is_low_rating_surge`)을 예측하는 기준 모델을 만든다.

## 2. 공통 입력
- 원본: `{LABELED_PATH}`
- 행 수: {summary["rows"]:,}
- 상품 수: {summary["unique_products"]:,}
- 공통 키: `parent_asin`, `year_month`
- 라벨: `is_low_rating_surge`
- 양성 비율: {summary["positive_rate"]:.2%}

## 3. 피처 구성
| 구분 | 피처 수 | 역할 |
|---|---:|---|
| 현재 상태 | {len(current_features)} | t월의 리뷰·평점·활동량 등 현재 상품 상태 |
| 시계열 변화 | {len(trend_features)} | t월 이전 누적·변화·이동 구간 기반 신호 |
| 전체 정형 피처 | {len(all_features)} | 시계열 모델 후보 입력 |

상세 정의와 제외 이유는 `structured_feature_spec.csv`에 기록했다.

## 4. 누수 방지
다음은 모델 입력에서 제외한다.
- 라벨과 다음 달 결과: `is_low_rating_surge`, `next_*`
- 텍스트 피처: `T_*` 및 공통 안전 텍스트 피처
- 제목 기반 자동 라벨 위험 피처: `auto_title_*`
- 식별자·시간 식별값·상수 컬럼

## 5. 생성 파일
- `structured_modeling_panel.parquet`: A1 모델 학습용 정형 패널
- `structured_feature_spec.csv`: 컬럼별 사용 여부·그룹·제외 이유
- `structured_panel_validation.json`: 행·키·라벨·피처 검증 결과

## 6. 다음 단계
이 패널로 현재 상태 모델과 시계열 변화 모델을 동일한 시간 분할에서 학습·검증한다. Test 데이터는 A1에서 사용하지 않는다.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

print("===== 정형 피처 패널 생성 완료 =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n[현재 상태 피처]")
print(", ".join(current_features))
print("\n[시계열 변화 피처]")
print(", ".join(trend_features))
print(f"\n저장: {OUT / 'structured_modeling_panel.parquet'}")
print(f"저장: {OUT / 'structured_feature_spec.csv'}")
print(f"저장: {OUT / 'README.md'}")
