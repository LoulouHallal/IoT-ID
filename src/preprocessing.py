#!/usr/bin/env python3
"""
Memory-conscious preprocessing for the CICIoT2023 merged CSV dataset.

What it does:
- Reads each Merged*.csv in chunks.
- Drops rows with missing/blank labels.
- Preserves original label names (after trimming surrounding whitespace).
- Converts feature columns to numeric; unexpected non-numeric feature values become NaN.
- Converts +inf/-inf in feature columns to NaN.
- Does NOT impute missing feature values; imputation should be fitted on training data only.
- Writes cleaned Parquet parts and CSV reports.
- Never modifies the original CSV files.

Example:
    python preprocessing.py
    python preprocessing.py --input-dir /content/cic_temp/MERGED_CSV \
        --output-dir /content/cic_temp/cleaned_parquet --chunksize 200000
"""

import argparse
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/content/cic_temp/MERGED_CSV",
        help="Directory containing Merged*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/cic_temp/cleaned_parquet",
        help="Directory for cleaned Parquet parts and reports",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per input chunk",
    )
    args = parser.parse_args()

    csv_files = sorted(
        glob.glob(os.path.join(args.input_dir, "Merged*.csv"))
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No Merged*.csv files found in: {args.input_dir}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    parts_dir = os.path.join(args.output_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyArrow is required to write Parquet. In Colab run: "
            "!pip install pyarrow"
        ) from exc

    total_rows_read = 0
    total_rows_written = 0
    total_missing_label_rows = 0
    total_infinite_values = 0
    total_new_nans_from_conversion = 0
    missing_by_column = Counter()
    label_counts = Counter()
    report_rows = []

    for file_index, csv_path in enumerate(csv_files, start=1):
        file_name = os.path.basename(csv_path)
        file_stem = os.path.splitext(file_name)[0]
        file_output_dir = os.path.join(parts_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)

        file_rows_read = 0
        file_rows_written = 0
        file_missing_labels = 0
        file_infinities = 0
        file_conversion_nans = 0
        part_index = 0
        feature_columns = None

        for chunk in pd.read_csv(csv_path, chunksize=args.chunksize):
            rows_before = len(chunk)
            file_rows_read += rows_before
            total_rows_read += rows_before

            if "Label" not in chunk.columns:
                raise ValueError(f"'Label' column missing in {csv_path}")

            # Preserve labels while normalizing accidental surrounding whitespace.
            chunk["Label"] = chunk["Label"].astype("string").str.strip()
            invalid_label = chunk["Label"].isna() | chunk["Label"].eq("")
            dropped_labels = int(invalid_label.sum())
            file_missing_labels += dropped_labels
            total_missing_label_rows += dropped_labels
            chunk = chunk.loc[~invalid_label].copy()

            if chunk.empty:
                continue

            if feature_columns is None:
                feature_columns = [c for c in chunk.columns if c != "Label"]

            # Convert all features to numeric. Count unexpected strings that
            # become NaN separately from NaNs already present in the source.
            for col in feature_columns:
                original = chunk[col]
                converted = pd.to_numeric(original, errors="coerce")
                newly_missing = int((original.notna() & converted.isna()).sum())
                file_conversion_nans += newly_missing
                total_new_nans_from_conversion += newly_missing

                # Count infinities before replacing them with NaN.
                numeric_array = converted.to_numpy(dtype="float64", na_value=np.nan)
                infinite_mask = np.isinf(numeric_array)
                n_infinite = int(infinite_mask.sum())
                file_infinities += n_infinite
                total_infinite_values += n_infinite

                converted = converted.replace([np.inf, -np.inf], np.nan)
                # Use a consistent dtype across chunks/Parquet parts.
                chunk[col] = converted.astype("float64")

            # Keep original labels; do not resolve conflicting labels here.
            label_counts.update(chunk["Label"].astype(str).value_counts().to_dict())
            for col in chunk.columns:
                n_missing = int(chunk[col].isna().sum())
                if n_missing:
                    missing_by_column[col] += n_missing

            part_path = os.path.join(
                file_output_dir, f"part-{part_index:04d}.parquet"
            )
            chunk.to_parquet(part_path, index=False, compression="snappy")
            part_index += 1

            rows_written = len(chunk)
            file_rows_written += rows_written
            total_rows_written += rows_written

        report_rows.append(
            {
                "file": file_name,
                "rows_read": file_rows_read,
                "rows_written": file_rows_written,
                "rows_dropped_missing_label": file_missing_labels,
                "infinite_feature_values_replaced_with_nan": file_infinities,
                "unexpected_non_numeric_feature_values_coerced_to_nan":
                    file_conversion_nans,
            }
        )
        print(
            f"[{file_index}/{len(csv_files)}] {file_name}: "
            f"{file_rows_read:,} read, {file_rows_written:,} written"
        )

    pd.DataFrame(report_rows).to_csv(
        os.path.join(args.output_dir, "file_cleaning_report.csv"), index=False
    )
    pd.DataFrame(
        sorted(missing_by_column.items()),
        columns=["column", "missing_values_after_cleaning"],
    ).to_csv(os.path.join(args.output_dir, "missing_values_report.csv"), index=False)
    pd.DataFrame(
        sorted(label_counts.items(), key=lambda item: item[0]),
        columns=["Label", "count"],
    ).to_csv(os.path.join(args.output_dir, "label_distribution.csv"), index=False)

    summary = pd.DataFrame(
        [
            ("csv_files_processed", len(csv_files)),
            ("rows_read", total_rows_read),
            ("rows_written", total_rows_written),
            ("rows_dropped_missing_or_blank_label", total_missing_label_rows),
            ("infinite_feature_values_replaced_with_nan", total_infinite_values),
            ("unexpected_non_numeric_feature_values_coerced_to_nan",
             total_new_nans_from_conversion),
        ],
        columns=["metric", "value"],
    )
    summary.to_csv(os.path.join(args.output_dir, "preprocessing_summary.csv"), index=False)

    print("\nPreprocessing complete.")
    print(summary.to_string(index=False))
    print(f"\nCleaned Parquet parts: {parts_dir}")
    print(f"Reports: {args.output_dir}")
    print(
        "\nNote: feature NaNs are intentionally retained. Fit any imputer "
        "on the training split only to avoid data leakage."
    )


if __name__ == "__main__":
    main()
