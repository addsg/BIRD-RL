#!/usr/bin/env python3
"""Build database-disjoint VERL multi-turn SFT parquet files."""

import argparse
import collections
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bird_rl.data.prepare_multi_turn_sft_data import convert_trajectory_to_multiturn


def load_by_id(path):
    records = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                instance_id = item["instance_id"]
                if instance_id in records:
                    raise ValueError(f"Duplicate {instance_id} in {path}")
                records[instance_id] = item
    return records


def has_final_submit(item):
    trajectory = item.get("trajectory") or []
    return bool(
        trajectory
        and trajectory[-1].get("end_flag") is True
        and '"name": "submit_solution"' in trajectory[-1].get("action", "")
    )


def save_parquet(path, examples):
    table = pa.table({"messages": [item["messages"] for item in examples]})
    pq.write_table(table, path)


def counts(examples, key):
    return dict(sorted(collections.Counter(x[key] for x in examples).items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--schema-data", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-db", action="append", required=True)
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    train = load_by_id(args.train_data)
    schemas = load_by_id(args.schema_data)
    trajectories = load_by_id(args.trajectory_file)
    statuses = load_by_id(args.status_file)
    validation_dbs = set(args.validation_db)
    examples = []
    rejected = []

    for instance_id, status in statuses.items():
        if status.get("status") != "success":
            rejected.append((instance_id, "evaluation_failed"))
            continue
        sample = train.get(instance_id)
        schema = schemas.get(instance_id)
        trajectory = trajectories.get(instance_id)
        if not sample or not schema or not trajectory:
            rejected.append((instance_id, "missing_join_record"))
            continue
        if not has_final_submit(trajectory):
            rejected.append((instance_id, "invalid_or_missing_submit"))
            continue

        messages = convert_trajectory_to_multiturn(
            trajectory["trajectory"],
            sample,
            schema["after_preprocess_schema"],
            max_turns=args.max_turns,
            use_think_tags=True,
        )
        if [message["role"] for message in messages[:2]] != ["system", "user"]:
            rejected.append((instance_id, "invalid_initial_roles"))
            continue
        if "Ground Truth Solution" in messages[1]["content"]:
            rejected.append((instance_id, "ground_truth_prompt_leak"))
            continue
        examples.append(
            {
                "instance_idx": schema["instance_idx"],
                "instance_id": instance_id,
                "db_id": sample["db_id"],
                "category": sample.get("category", ""),
                "turns": len(trajectory["trajectory"]),
                "messages": messages,
            }
        )

    examples.sort(key=lambda x: x["instance_idx"])
    training = [x for x in examples if x["db_id"] not in validation_dbs]
    validation = [x for x in examples if x["db_id"] in validation_dbs]
    train_dbs = {x["db_id"] for x in training}
    val_dbs = {x["db_id"] for x in validation}
    if not training or not validation:
        raise ValueError("Both splits must be non-empty")
    if train_dbs & val_dbs:
        raise ValueError("A db_id appears in both train and validation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(output_dir / "train.parquet", training)
    save_parquet(output_dir / "validation.parquet", validation)

    with (output_dir / "split_manifest.jsonl").open("w", encoding="utf-8") as output:
        for split, records in (("train", training), ("validation", validation)):
            for item in records:
                output.write(
                    json.dumps(
                        {key: item[key] for key in (
                            "instance_idx", "instance_id", "db_id", "category", "turns"
                        )} | {"split": split},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "format": {"column": "messages", "messages_key": "messages"},
        "accepted": len(examples),
        "rejected": len(rejected),
        "rejection_reasons": counts(
            [{"reason": reason} for _, reason in rejected], "reason"
        ),
        "train": {
            "count": len(training),
            "db_ids": sorted(train_dbs),
            "categories": counts(training, "category"),
            "turns": counts(training, "turns"),
        },
        "validation": {
            "count": len(validation),
            "db_ids": sorted(val_dbs),
            "categories": counts(validation, "category"),
            "turns": counts(validation, "turns"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
