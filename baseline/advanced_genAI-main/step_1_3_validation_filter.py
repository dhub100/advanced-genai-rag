#!/usr/bin/env python3
"""
Script: step_1_3_validation_filter.py

Purpose:
  - Reads JSON files from a 'BSD_advanced' folder.
  - Removes any document where 'paragraphs_cleaned' is empty.
  - Writes the remaining (valid) records to a new folder 'BSD_advanced_validated'.

Usage:
    python step_1_3_validation_filter.py [BSD_advanced_dir] [BSD_advanced_validated_dir]
Example:
    python Code/step_1_3_validation_filter.py data_cleaned/BSD_advanced data_cleaned/BSD_advanced_validated
"""

import os
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def has_nonempty_cleaned_paragraphs(record: dict) -> bool:
    """
    Check that paragraphs_cleaned is a non-empty list.
    """
    paragraphs_cleaned = record.get("paragraphs_cleaned", [])
    # If it's not a list or if it's empty, we discard the record
    return isinstance(paragraphs_cleaned, list) and len(paragraphs_cleaned) > 0

def main():
    parser = argparse.ArgumentParser(
        description="Filter out records that have empty paragraphs_cleaned and write valid ones to a new folder."
    )
    parser.add_argument("input_dir", help="Path to the BSD_advanced folder containing JSON files.")
    parser.add_argument("output_dir", help="Path to the BSD_advanced_validated folder for valid records.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_records = []
    count_files = 0

    # Traverse all .json files in input_dir
    for path in input_dir.rglob("*.json"):
        if not path.is_file():
            continue
        count_files += 1

        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception as e:
            logging.warning(f"Error reading {path}: {e}")
            continue

        # If it's not a dict, skip
        if not isinstance(record, dict):
            logging.warning(f"Invalid record format in {path}; skipping.")
            continue

        # Apply the single rule: remove if paragraphs_cleaned is empty
        if has_nonempty_cleaned_paragraphs(record):
            valid_records.append(record)

    logging.info(f"Scanned {count_files} JSON files in '{input_dir}'.")
    logging.info(f"{len(valid_records)} records passed the paragraphs_cleaned check.")

    # Write valid records to output_dir
    for rec in valid_records:
        doc_id = rec.get("doc_id", "unknown")
        out_path = output_dir / f"{doc_id}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Could not write file {out_path}: {e}")

    logging.info(f"Wrote {len(valid_records)} valid records to '{output_dir}'.")

if __name__ == "__main__":
    main()
