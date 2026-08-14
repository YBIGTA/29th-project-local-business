"""
L4. 검수 결과 집계.

topic_quality_sample.csv의 is_correct를 채운 뒤 실행한다.
토픽별 정밀도와, 저평점 구간과 고평점 구간의 정밀도 차이를 계산한다.

두 구간의 차이가 크면 사전이 별점을 대신 보고 있다는 뜻이므로 그 토픽의
표현을 손봐야 한다. 저평점에서만 잘 맞는 사전은 별점을 이미 아는 상황에서만
쓸모가 있는데, 우리는 다음 달을 예측해야 하므로 그런 사전은 도움이 되지 않는다.

is_correct가 비어 있는 행은 경고만 남기고 제외하므로 일부만 채워도 실행된다.

    python3 src/nlp/build_quality_report.py

출력: 01_midterm/2-3_nlp_features/topic_quality_report.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPORT_DIR  # noqa: E402

SAMPLE_PATH = REPORT_DIR / "topic_quality_sample.csv"

if not SAMPLE_PATH.exists():
    raise SystemExit(
        f"검수 파일이 없다: {SAMPLE_PATH}\nbuild_quality_sample.py를 먼저 실행한다."
    )

data = pd.read_csv(SAMPLE_PATH)
data["is_correct"] = pd.to_numeric(data["is_correct"], errors="coerce")

unlabeled = data["is_correct"].isna().sum()
if unlabeled:
    print(f"경고: is_correct가 비어 있는 행 {unlabeled}건은 집계에서 제외한다.")

data = data.dropna(subset=["is_correct"])
if data.empty:
    raise SystemExit("채워진 행이 없다. is_correct에 1 또는 0을 입력한 뒤 다시 실행한다.")
data["is_correct"] = data["is_correct"].astype(int)

report = (
    data.groupby("topic")
    .agg(n_reviewed=("is_correct", "size"), precision=("is_correct", "mean"))
    .round(3)
)
low = (
    data[data["rating"] <= 2].groupby("topic")["is_correct"]
    .mean().rename("precision_low_rating")
)
high = (
    data[data["rating"] >= 4].groupby("topic")["is_correct"]
    .mean().rename("precision_high_rating")
)
report = report.join(low).join(high).round(3)
report["gap"] = (report["precision_low_rating"] - report["precision_high_rating"]).round(3)
report = report.sort_values("precision")

report.to_csv(REPORT_DIR / "topic_quality_report.csv", encoding="utf-8-sig")

print("=== 토픽별 정밀도 ===")
print(report.to_string())
print(f"\n전체 정밀도: {data['is_correct'].mean():.3f} ({len(data)}건 검토)")
if (data["rating"] <= 2).any():
    print(f"저평점 구간: {data[data['rating'] <= 2]['is_correct'].mean():.3f}")
if (data["rating"] >= 4).any():
    print(f"고평점 구간: {data[data['rating'] >= 4]['is_correct'].mean():.3f}")
print("\ngap이 큰 토픽부터 표현을 손보면 된다.")
print(f"생성 파일: {REPORT_DIR / 'topic_quality_report.csv'}")
