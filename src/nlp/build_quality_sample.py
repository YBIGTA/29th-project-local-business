"""
L3. 토픽 사전 정밀도 검수용 표본 추출.

사전이 잡아낸 리뷰가 실제로 그 불만을 말하고 있는지 사람이 확인할 표본을
뽑는다. 토픽마다 저평점과 고평점을 절반씩 뽑는 것이 핵심이다. 두 가지를 동시에
확인할 수 있기 때문이다.

  1) 정밀도 - 잡힌 것 중 실제로 맞는 비율
  2) 별점 의존 여부 - 고평점인데 토픽이 잡힌 리뷰가 진짜 불만인지 오탐인지

사전을 고친 뒤 다시 측정할 때는 SEED를 바꿔 새 표본을 뽑아야 한다. 사전을
고치는 데 쓴 표본으로 다시 재면 정밀도가 부풀려진다.

    python3 src/nlp/build_quality_sample.py

출력: 01_midterm/2-3_nlp_features/topic_quality_sample.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CLEAN_REVIEWS, DICTIONARY_PATH, REPORT_DIR, REVIEW_FEATURES,
)

PER_TOPIC = 25          # 토픽당 표본 수
SEED = 42               # 재측정 시 반드시 바꾼다
TEXT_LIMIT = 400

with open(DICTIONARY_PATH, encoding="utf-8") as file:
    dictionary = json.load(file)
TOPIC_KEYS = [
    key for layer in ("cause", "consequence") for key in dictionary[layer]
]

print("=== 검수 표본 추출 ===")
reviews = pd.read_parquet(REVIEW_FEATURES)
raw = pd.read_parquet(CLEAN_REVIEWS, columns=["review_id", "text_norm"])
data = reviews.merge(raw, on="review_id", how="left")
print(f"리뷰 {len(data):,}건")

samples = []
for key in TOPIC_KEYS:
    matched = data[data[f"topic_{key}"] == 1]
    if matched.empty:
        print(f"  {key}: 매칭 없음")
        continue

    half = PER_TOPIC // 2
    low = matched[matched["rating"] <= 2]
    high = matched[matched["rating"] >= 4]
    picked = pd.concat([
        low.sample(min(half, len(low)), random_state=SEED),
        high.sample(min(PER_TOPIC - half, len(high)), random_state=SEED),
    ])
    # 한쪽이 모자라면 나머지에서 채운다
    if len(picked) < PER_TOPIC:
        rest = matched.drop(picked.index)
        shortfall = min(PER_TOPIC - len(picked), len(rest))
        if shortfall:
            picked = pd.concat([picked, rest.sample(shortfall, random_state=SEED)])

    other = [
        k for k in TOPIC_KEYS if k != key
    ]
    for _, review in picked.iterrows():
        also = [k for k in other if review[f"topic_{k}"] == 1]
        samples.append({
            "topic": key,
            "rating": review["rating"],
            "review_text": str(review["text_norm"])[:TEXT_LIMIT],
            "matched_other_topics": ",".join(also),
            "is_correct": "",
            "note": "",
        })

sample = pd.DataFrame(samples).sample(frac=1, random_state=SEED).reset_index(drop=True)
sample.to_csv(REPORT_DIR / "topic_quality_sample.csv",
              index=False, encoding="utf-8-sig")

print(f"\n=== 추출 완료: {len(sample)}건 ===")
print(sample.groupby("topic").size().to_string())
print(f"저평점(2점 이하) 비율: {(sample['rating'] <= 2).mean():.1%}")
print("\n작성 방법")
print("  is_correct 열에 1(해당 불만이 실제로 리뷰에 있음) 또는")
print("  0(사전이 잘못 잡음)을 채운다.")
print("  note 열에 오분류 사유를 짧게 적으면 사전 개선의 근거가 된다.")
print(f"\n생성 파일: {REPORT_DIR / 'topic_quality_sample.csv'}")
