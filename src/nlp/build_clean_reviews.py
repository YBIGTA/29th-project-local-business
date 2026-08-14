"""
L0. 리뷰 정제 및 정규화.

2-1이 만든 정제 리뷰에서 텍스트 분석에 필요한 형태를 만든다.
사전 매칭이 소문자 토큰 기준으로 동작하므로 여기서 표기를 통일해 둔다.

주요 처리
  1) 제목과 본문 결합. 제목에만 불만이 적힌 리뷰가 적지 않다.
  2) HTML 조각(<br />) 제거.
  3) 축약형 전개. n't를 not으로 바꿔야 doesn't fit이 not fit으로 잡힌다.
  4) 반복 오타 교정. 저평점 리뷰에서 자주 나타난 것만 최소한으로 넣었다.
  5) 자동 생성 제목 판별. 2018년 중반 이전 아마존은 별점을 제목으로 자동
     생성했다(One Star, Five Stars). 이는 별점의 문자열 사본이므로 텍스트
     피처로 쓰면 정답을 미리 보는 것이 된다. 제거하지 않고 플래그로 남겨
     2-4가 처리 방식을 정하게 한다.
  6) 스페인어 추정 플래그. 소수의 스페인어 리뷰가 섞여 있어 영어 사전이
     걸리지 않는다.

    python3 src/nlp/build_clean_reviews.py

출력: data/interim/clean_reviews.parquet
      01_midterm/2-3_nlp_features/clean_reviews_summary.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BASELINE_START, CLEAN_REVIEWS, REPORT_DIR, REVIEWS_CLEAN,
)

# ---------------------------------------------------------------
# 정규화 규칙
# ---------------------------------------------------------------
CONTRACTIONS = {
    r"n't\b": " not",
    r"'re\b": " are",
    r"'ve\b": " have",
    r"'ll\b": " will",
    r"'m\b": " am",
    r"'d\b": " would",
}

TYPO_MAP = {
    "stoped": "stopped",
    "stopd": "stopped",
    "brake": "break",
    "brakes": "breaks",
    "waist": "waste",
    "recieved": "received",
    "recieve": "receive",
    "dosent": "does not",
    "doesnt": "does not",
    "didnt": "did not",
    "dont": "do not",
    "wont": "will not",
    "cant": "can not",
    "isnt": "is not",
    "wasnt": "was not",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "havent": "have not",
    "hasnt": "has not",
    "defectiv": "defective",
    "guarentee": "guarantee",
    "warrenty": "warranty",
    "warrantee": "warranty",
}

# 2018년 중반 이전 아마존이 별점으로 자동 생성한 제목
AUTO_TITLE = re.compile(
    r"^\s*(one|two|three|four|five)\s+stars?\s*$", re.IGNORECASE
)

# 영어 리뷰에는 거의 나오지 않는 스페인어 기능어
SPANISH_HINTS = {
    "que", "para", "pero", "muy", "esta", "este", "con", "por", "los",
    "las", "una", "sirve", "mala", "malo", "bueno", "producto", "compre",
}


def normalize(text):
    text = text.lower()
    text = text.replace("<br />", " ").replace("<br>", " ")
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\[\[videoid:[^\]]*\]\]", " ", text)
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [TYPO_MAP.get(token, token) for token in text.split()]
    return " ".join(tokens)


def pick_column(data, candidates, required=True):
    for name in candidates:
        if name in data.columns:
            return name
    if required:
        raise KeyError(f"필요한 열을 찾을 수 없다: {candidates}")
    return None


print("=== 리뷰 정제 시작 ===")
print(f"입력: {REVIEWS_CLEAN}")
if not REVIEWS_CLEAN.exists():
    raise SystemExit(
        f"입력 파일이 없다: {REVIEWS_CLEAN}\n"
        "2-1 파이프라인을 먼저 실행해 정제 리뷰를 만들어야 한다."
    )

if REVIEWS_CLEAN.suffix == ".csv":
    raw = pd.read_csv(REVIEWS_CLEAN)
else:
    raw = pd.read_parquet(REVIEWS_CLEAN)
print(f"원본 {len(raw):,}건, 열 {len(raw.columns)}개")

col_asin = pick_column(raw, ["parent_asin", "asin"])
col_rating = pick_column(raw, ["rating", "overall", "star_rating"])
col_text = pick_column(raw, ["text", "review_text", "review_body"])
col_title = pick_column(raw, ["title", "review_title", "summary"], required=False)
col_time = pick_column(
    raw, ["review_datetime_utc", "timestamp", "review_date", "date"]
)

data = pd.DataFrame({
    "parent_asin": raw[col_asin].astype(str),
    "rating": pd.to_numeric(raw[col_rating], errors="coerce"),
})

timestamp = raw[col_time]
if pd.api.types.is_numeric_dtype(timestamp):
    # 아마존 원본은 밀리초 단위 정수다
    unit = "ms" if timestamp.max() > 1e11 else "s"
    data["review_datetime_utc"] = pd.to_datetime(
        timestamp, unit=unit, utc=True, errors="coerce"
    )
else:
    data["review_datetime_utc"] = pd.to_datetime(
        timestamp, utc=True, errors="coerce"
    )
data["year_month"] = data["review_datetime_utc"].dt.strftime("%Y-%m")

for name, column in (
    ("verified_purchase", "verified_purchase"),
    ("helpful_vote", "helpful_vote"),
):
    if column in raw.columns:
        data[name] = raw[column]
    else:
        data[name] = 0

title = raw[col_title].fillna("") if col_title else ""
body = raw[col_text].fillna("")
title = title if isinstance(title, pd.Series) else pd.Series([""] * len(raw))

data["auto_title_flag"] = title.astype(str).str.match(AUTO_TITLE).astype(int)
# 자동 생성 제목은 본문 앞에 붙이지 않는다. 별점의 사본이기 때문이다.
combined = title.astype(str).where(data["auto_title_flag"] == 0, "") + " " + body.astype(str)

print("정규화 중...")
data["text_norm"] = [normalize(t) for t in combined]

tokens = data["text_norm"].str.split()
data["word_count"] = tokens.str.len().fillna(0).astype(int)
data["char_count"] = data["text_norm"].str.len()
# 문장부호는 정규화에서 사라지므로 원문에서 센다
data["sentence_count"] = (
    combined.astype(str).str.count(r"[.!?]+").clip(lower=1)
)
data["exclaim_count"] = combined.astype(str).str.count("!")
data["question_count"] = combined.astype(str).str.count(r"\?")
letters = combined.astype(str).str.count(r"[A-Za-z]").replace(0, 1)
data["caps_ratio"] = (combined.astype(str).str.count(r"[A-Z]") / letters).round(4)
data["is_likely_spanish"] = tokens.apply(
    lambda t: int(len(SPANISH_HINTS.intersection(t or [])) >= 3)
)

data["is_low_rating"] = (data["rating"] <= 2).astype(int)

before = len(data)
data = data[data["year_month"] >= BASELINE_START]
data = data.dropna(subset=["rating", "year_month"])
print(f"기준선 시작({BASELINE_START}) 이후만 유지: {before:,} -> {len(data):,}건")

data = data.reset_index(drop=True)
data["review_id"] = data.index.astype(str)

data.to_parquet(CLEAN_REVIEWS, index=False)

# ---------------------------------------------------------------
# 요약
# ---------------------------------------------------------------
by_year = (
    data.assign(year=data["year_month"].str[:4])
    .groupby("year")
    .agg(
        n_reviews=("rating", "size"),
        auto_title_ratio=("auto_title_flag", "mean"),
        low_rating_ratio=("is_low_rating", "mean"),
        mean_word_count=("word_count", "mean"),
    )
    .round(4)
)
by_year.to_csv(REPORT_DIR / "clean_reviews_summary.csv", encoding="utf-8-sig")

print("\n=== 연도별 요약 ===")
print(by_year.to_string())
print(f"\n1단어 이하 리뷰: {(data['word_count'] <= 1).sum():,}건")
print(f"스페인어 추정: {data['is_likely_spanish'].sum():,}건")
print(f"자동 생성 제목: {data['auto_title_flag'].sum():,}건 "
      f"({data['auto_title_flag'].mean():.1%})")
print(f"\n생성 파일: {CLEAN_REVIEWS}")
print(f"          {REPORT_DIR / 'clean_reviews_summary.csv'}")
