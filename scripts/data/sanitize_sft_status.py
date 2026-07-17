#!/usr/bin/env python3
"""Exclude known environment failures from an evaluator status JSONL."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded-output", required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    args = parser.parse_args()

    excluded_ids = set(args.exclude_id)
    records = []
    excluded = []
    with open(args.input, "r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["instance_id"] in excluded_ids:
                original_status = record.get("status")
                record["status"] = "excluded_preprocess_failure"
                record["original_status"] = original_status
                excluded.append(record)
            records.append(record)

    missing = excluded_ids - {record["instance_id"] for record in excluded}
    if missing:
        raise ValueError(f"Excluded IDs not found: {sorted(missing)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for path, items in (
        (Path(args.output), records),
        (Path(args.excluded_output), excluded),
    ):
        with path.open("w", encoding="utf-8") as output:
            for item in items:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} statuses; excluded {len(excluded)} preprocess failures")


if __name__ == "__main__":
    main()
