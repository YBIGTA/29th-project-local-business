import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data/raw/Appliances.jsonl.gz"
INTERIM_DIR = ROOT / "data/interim"
REPORT_DIR = ROOT / "01_midterm/2-1_data_label"

ANALYSIS_END = datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc)

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

stats = defaultdict(lambda: {
    "review_count": 0,
    "rating_sum": 0.0,
    "low_rating_count": 0,
    "verified_purchase_count": 0,
    "helpful_vote_sum": 0,
    "text_available_count": 0,
})

seen_signatures = set()
raw_count = 0
tail_excluded_count = 0
duplicate_excluded_count = 0
invalid_rating_count = 0

with gzip.open(REVIEW_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        raw_count += 1

        rating = row.get("rating")
        parent_asin = row.get("parent_asin")
        timestamp = row.get("timestamp")

        if rating not in {1.0, 2.0, 3.0, 4.0, 5.0} or not parent_asin:
            invalid_rating_count += 1
            continue

        review_datetime = datetime.fromtimestamp(
            timestamp / 1000, tz=timezone.utc
        )

        if review_datetime > ANALYSIS_END:
            tail_excluded_count += 1
            continue

        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()

        signature = hash((
            parent_asin,
            row.get("user_id"),
            timestamp,
            rating,
            title,
            text,
        ))
        if signature in seen_signatures:
            duplicate_excluded_count += 1
            continue
        seen_signatures.add(signature)

        year_month = review_datetime.strftime("%Y-%m")
        key = (parent_asin, year_month)
        item = stats[key]

        item["review_count"] += 1
        item["rating_sum"] += rating
        item["low_rating_count"] += int(rating <= 2.0)
        item["verified_purchase_count"] += int(
            row.get("verified_purchase") is True
        )
        item["helpful_vote_sum"] += int(row.get("helpful_vote") or 0)
        item["text_available_count"] += int(bool(text))

rows = []
for (parent_asin, year_month), item in stats.items():
    count = item["review_count"]
    rows.append({
        "parent_asin": parent_asin,
        "year_month": year_month,
        "review_count": count,
        "avg_rating": item["rating_sum"] / count,
        "low_rating_count": item["low_rating_count"],
        "low_rating_ratio": item["low_rating_count"] / count,
        "verified_purchase_ratio": item["verified_purchase_count"] / count,
        "mean_helpful_vote": item["helpful_vote_sum"] / count,
        "text_available_ratio": item["text_available_count"] / count,
    })

activity = pd.DataFrame(rows).sort_values(
    ["parent_asin", "year_month"]
)

activity.to_parquet(
    INTERIM_DIR / "product_month_activity.parquet",
    index=False,
)

monthly_summary = (
    activity.groupby("year_month", as_index=False)
    .agg(
        active_product_count=("parent_asin", "nunique"),
        product_month_count=("parent_asin", "size"),
        total_review_count=("review_count", "sum"),
        product_months_ge_5=(
            "review_count", lambda x: int((x >= 5).sum())
        ),
        product_months_ge_10=(
            "review_count", lambda x: int((x >= 10).sum())
        ),
    )
)
monthly_summary.to_csv(
    REPORT_DIR / "monthly_activity_summary.csv",
    index=False,
)

threshold_summary = pd.DataFrame([
    {
        "minimum_monthly_review_count": threshold,
        "eligible_product_month_count": int(
            (activity["review_count"] >= threshold).sum()
        ),
        "eligible_ratio": round(
            (activity["review_count"] >= threshold).mean(), 4
        ),
    }
    for threshold in [1, 2, 3, 5, 10, 20]
])
threshold_summary.to_csv(
    REPORT_DIR / "monthly_review_thresholds.csv",
    index=False,
)

print("=== 상품×월 활동량 테이블 생성 완료 ===")
print(f"원본 리뷰 수: {raw_count:,}")
print(f"2023-04 이후 제외: {tail_excluded_count:,}")
print(f"중복 후보 제외: {duplicate_excluded_count:,}")
print(f"잘못된 평점 또는 상품 키 제외: {invalid_rating_count:,}")
print(f"최종 상품×월 행 수: {len(activity):,}")
print("\n=== 월별 최소 리뷰 수 기준별 유지 행 수 ===")
print(threshold_summary.to_string(index=False))

print("\n생성 파일")
print(INTERIM_DIR / "product_month_activity.parquet")
print(REPORT_DIR / "monthly_activity_summary.csv")
print(REPORT_DIR / "monthly_review_thresholds.csv")
