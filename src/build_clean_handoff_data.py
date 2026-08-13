import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data/raw/Appliances.jsonl.gz"
META_PATH = ROOT / "data/raw/meta_Appliances.jsonl.gz"
INTERIM_DIR = ROOT / "data/interim"
REPORT_DIR = ROOT / "01_midterm/2-1_data_label"

REVIEW_OUTPUT = INTERIM_DIR / "reviews_clean.parquet"
META_OUTPUT = INTERIM_DIR / "products_meta_clean.parquet"

START = datetime(2015, 11, 1, tzinfo=timezone.utc)
END = datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
BATCH_SIZE = 50_000

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if REVIEW_OUTPUT.exists():
    REVIEW_OUTPUT.unlink()

seen_signatures = set()
writer = None
buffer = []

raw_count = 0
period_excluded_count = 0
duplicate_excluded_count = 0
written_count = 0

with gzip.open(REVIEW_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        raw_count += 1

        parent_asin = row.get("parent_asin")
        rating = row.get("rating")
        timestamp = row.get("timestamp")

        if not parent_asin or rating not in {1.0, 2.0, 3.0, 4.0, 5.0}:
            continue

        review_datetime = datetime.fromtimestamp(
            timestamp / 1000, tz=timezone.utc
        )
        if review_datetime < START or review_datetime > END:
            period_excluded_count += 1
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

        review_text = " ".join(value for value in [title, text] if value)

        buffer.append(
            {
                "parent_asin": parent_asin,
                "asin": row.get("asin"),
                "user_id": row.get("user_id"),
                "review_datetime_utc": review_datetime,
                "year_month": review_datetime.strftime("%Y-%m"),
                "rating": rating,
                "title": title,
                "text": text,
                "review_text": review_text,
                "text_missing": not bool(text),
                "verified_purchase": bool(row.get("verified_purchase")),
                "helpful_vote": int(row.get("helpful_vote") or 0),
            }
        )

        if len(buffer) >= BATCH_SIZE:
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(REVIEW_OUTPUT, table.schema)
            writer.write_table(table)
            written_count += len(buffer)
            buffer = []

if buffer:
    table = pa.Table.from_pylist(buffer)
    if writer is None:
        writer = pq.ParquetWriter(REVIEW_OUTPUT, table.schema)
    writer.write_table(table)
    written_count += len(buffer)

if writer is not None:
    writer.close()

meta_rows = []
with gzip.open(META_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        meta_rows.append(
            {
                "parent_asin": row.get("parent_asin"),
                "product_title": row.get("title"),
                "store": row.get("store"),
                "main_category": row.get("main_category"),
                "categories": json.dumps(
                    row.get("categories") or [],
                    ensure_ascii=False,
                ),
            }
        )

meta = pd.DataFrame(meta_rows)
meta_duplicate_count = int(meta.duplicated("parent_asin").sum())
meta = meta.drop_duplicates("parent_asin", keep="first")
meta.to_parquet(META_OUTPUT, index=False)

summary = pd.DataFrame(
    {
        "metric": [
            "review_period_start_utc",
            "review_period_end_utc",
            "raw_review_row_count",
            "review_rows_outside_period",
            "duplicate_review_rows_removed",
            "clean_review_row_count",
            "clean_metadata_product_count",
            "duplicate_metadata_rows_removed",
        ],
        "value": [
            START.isoformat(),
            END.isoformat(),
            raw_count,
            period_excluded_count,
            duplicate_excluded_count,
            written_count,
            len(meta),
            meta_duplicate_count,
        ],
    }
)
summary.to_csv(
    REPORT_DIR / "clean_handoff_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=== 2-3 인수인계용 정제 데이터 생성 완료 ===")
print(summary.to_string(index=False))
print("\n생성 파일")
print(REVIEW_OUTPUT)
print(META_OUTPUT)
print(REPORT_DIR / "clean_handoff_summary.csv")
