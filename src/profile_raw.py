import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data/raw/Appliances.jsonl.gz"
META_PATH = ROOT / "data/raw/meta_Appliances.jsonl.gz"
OUTPUT_DIR = ROOT / "01_midterm/2-1_data_label"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

review_count = 0
invalid_timestamp_count = 0
empty_text_count = 0
empty_title_count = 0
duplicate_candidate_count = 0

parent_asins = set()
month_counts = Counter()
rating_counts = Counter()
seen_signatures = set()

min_datetime = None
max_datetime = None

with gzip.open(REVIEW_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        review_count += 1

        parent_asin = row.get("parent_asin")
        if parent_asin:
            parent_asins.add(parent_asin)

        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()

        if not title:
            empty_title_count += 1
        if not text:
            empty_text_count += 1

        signature = hash((
            parent_asin,
            row.get("user_id"),
            row.get("timestamp"),
            row.get("rating"),
            title,
            text,
        ))
        if signature in seen_signatures:
            duplicate_candidate_count += 1
        else:
            seen_signatures.add(signature)

        rating_counts[str(row.get("rating"))] += 1

        timestamp = row.get("timestamp")
        try:
            review_datetime = datetime.fromtimestamp(
                timestamp / 1000, tz=timezone.utc
            )
        except (TypeError, ValueError, OSError):
            invalid_timestamp_count += 1
            continue

        month = review_datetime.strftime("%Y-%m")
        month_counts[month] += 1

        if min_datetime is None or review_datetime < min_datetime:
            min_datetime = review_datetime
        if max_datetime is None or review_datetime > max_datetime:
            max_datetime = review_datetime

meta_count = 0
meta_parent_asins = set()
meta_missing_parent_asin_count = 0

with gzip.open(META_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        meta_count += 1

        parent_asin = row.get("parent_asin")
        if parent_asin:
            meta_parent_asins.add(parent_asin)
        else:
            meta_missing_parent_asin_count += 1

matched_parent_asins = parent_asins & meta_parent_asins

profile = {
    "review_row_count": review_count,
    "unique_review_parent_asin_count": len(parent_asins),
    "review_start_utc": min_datetime.isoformat() if min_datetime else None,
    "review_end_utc": max_datetime.isoformat() if max_datetime else None,
    "empty_title_count": empty_title_count,
    "empty_text_count": empty_text_count,
    "invalid_timestamp_count": invalid_timestamp_count,
    "duplicate_candidate_count": duplicate_candidate_count,
    "rating_counts": dict(sorted(rating_counts.items())),
    "metadata_row_count": meta_count,
    "unique_metadata_parent_asin_count": len(meta_parent_asins),
    "metadata_missing_parent_asin_count": meta_missing_parent_asin_count,
    "matched_parent_asin_count": len(matched_parent_asins),
    "metadata_match_rate": round(
        len(matched_parent_asins) / len(parent_asins), 6
    ) if parent_asins else None,
}

with open(OUTPUT_DIR / "raw_profile.json", "w", encoding="utf-8") as file:
    json.dump(profile, file, ensure_ascii=False, indent=2)

with open(
    OUTPUT_DIR / "monthly_review_count.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)
    writer.writerow(["year_month", "review_count"])
    for month, count in sorted(month_counts.items()):
        writer.writerow([month, count])

print("=== 원본 전수 점검 완료 ===")
for key, value in profile.items():
    print(f"{key}: {value}")

print("\n생성 파일")
print(OUTPUT_DIR / "raw_profile.json")
print(OUTPUT_DIR / "monthly_review_count.csv")
