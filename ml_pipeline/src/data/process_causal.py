"""Processor for causal_data.csv (in-store displays and flyer circulars).

Processes the 36.8M row causal_data table into a compact, indexed Parquet file
containing active store-level promotional features per product-store-week.

Methodological Notes:
- Preserves raw categorical strings 'display_code' and 'mailer_code' exactly as observed.
- Provides 'has_display_placement' and 'has_mailer_placement' as explicitly PROVISIONAL HEURISTICS:
    has_display_placement = display_code not in ('0', '')
    has_mailer_placement = mailer_code not in ('0', 'A', '')
  These heuristic flags are NOT authoritative and should not be treated as verified ground truth
  until official retailer codebook documentation is established.
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml_pipeline.src.config.settings import ml_settings


def process_causal_promotions(
    raw_path: Path = ml_settings.RAW_DATA_DIR / "causal_data.csv",
    output_path: Path = ml_settings.INTERIM_DATA_DIR / "store_promotions.parquet",
    chunk_size: int = 1_000_000,
):
    print(f"Streaming and filtering {raw_path}...")

    dtypes = {
        "PRODUCT_ID": "int64",
        "STORE_ID": "int32",
        "WEEK_NO": "int16",
        "display": "str",
        "mailer": "str",
    }

    schema = pa.schema(
        [
            ("product_id", pa.int64()),
            ("store_id", pa.int32()),
            ("week_no", pa.int16()),
            ("display_code", pa.string()),
            ("mailer_code", pa.string()),
            ("has_display_placement", pa.bool_()),
            ("has_mailer_placement", pa.bool_()),
        ]
    )

    writer = pq.ParquetWriter(output_path, schema=schema, compression="snappy")

    total_rows = 0
    promoted_rows = 0

    try:
        for chunk in pd.read_csv(raw_path, dtype=dtypes, chunksize=chunk_size):
            total_rows += len(chunk)

            # Raw observed code preservation
            display_raw = chunk["display"].fillna("").astype(str)
            mailer_raw = chunk["mailer"].fillna("").astype(str)

            # Provisional heuristic rules:
            # display: '0' denotes unpromoted/no display; codes 1-9, A denote placement
            # mailer: '0' and 'A' denote baseline/unfeatured circular; C-Z denote placement
            has_display = ~display_raw.isin(["0", ""])
            has_mailer = ~mailer_raw.isin(["0", "A", ""])

            active_mask = has_display | has_mailer
            active_chunk = chunk[active_mask].copy()

            if not active_chunk.empty:
                active_chunk["display_code"] = display_raw[active_mask]
                active_chunk["mailer_code"] = mailer_raw[active_mask]
                active_chunk["has_display_placement"] = has_display[active_mask]
                active_chunk["has_mailer_placement"] = has_mailer[active_mask]

                active_chunk = active_chunk.rename(
                    columns={
                        "PRODUCT_ID": "product_id",
                        "STORE_ID": "store_id",
                        "WEEK_NO": "week_no",
                    }
                )

                cols_to_write = [
                    "product_id",
                    "store_id",
                    "week_no",
                    "display_code",
                    "mailer_code",
                    "has_display_placement",
                    "has_mailer_placement",
                ]
                table = pa.Table.from_pandas(
                    active_chunk[cols_to_write], schema=schema, preserve_index=False
                )
                writer.write_table(table)
                promoted_rows += len(active_chunk)

            if total_rows % 5_000_000 == 0:
                print(
                    f"  Processed {total_rows / 1_000_000:.1f}M rows... ({promoted_rows:,} active promos written)"
                )

        print(
            f"Finished scanning {total_rows:,} rows. Total active promo rows saved: {promoted_rows:,}"
        )
    finally:
        writer.close()

    print(f"Saved active store promotions to {output_path}")


if __name__ == "__main__":
    process_causal_promotions()
