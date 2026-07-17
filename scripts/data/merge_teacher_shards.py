#!/usr/bin/env python3
"""Merge teacher-trajectory shards and restore global dataset indices."""

import argparse
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def final_trajectory_path(shard_dir: Path) -> Path:
    candidates = []
    for path in (shard_dir / "trajectories").glob("traj_*.jsonl"):
        match = re.fullmatch(r"traj_(\d+)\.jsonl", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No trajectory files found in {shard_dir}")
    return max(candidates)[1]


def add_unique(target: dict[str, dict], record: dict, source: Path) -> None:
    instance_id = record.get("instance_id")
    if not instance_id:
        raise ValueError(f"Missing instance_id in {source}")
    if instance_id in target:
        raise ValueError(f"Duplicate instance_id {instance_id} from {source}")
    target[instance_id] = record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    train_records = load_jsonl(Path(args.train_data))
    global_indices = {
        record["instance_id"]: index for index, record in enumerate(train_records)
    }

    trajectories = {}
    statuses = {}
    evaluation_inputs = {}
    source_summary = []

    for raw_shard_dir in args.shard_dir:
        shard_dir = Path(raw_shard_dir)
        trajectory_path = final_trajectory_path(shard_dir)
        status_path = shard_dir / "eval_ready_simple_output_with_status.jsonl"
        evaluation_path = shard_dir / "eval_ready.jsonl"

        shard_trajectories = load_jsonl(trajectory_path)
        shard_statuses = load_jsonl(status_path)
        shard_evaluations = load_jsonl(evaluation_path)

        for record in shard_trajectories:
            add_unique(trajectories, record, trajectory_path)
        for record in shard_statuses:
            add_unique(statuses, record, status_path)
        for record in shard_evaluations:
            add_unique(evaluation_inputs, record, evaluation_path)

        source_summary.append(
            {
                "directory": str(shard_dir),
                "trajectory_file": str(trajectory_path),
                "trajectory_count": len(shard_trajectories),
                "status_count": len(shard_statuses),
            }
        )

    instance_ids = set(trajectories)
    if instance_ids != set(statuses) or instance_ids != set(evaluation_inputs):
        raise ValueError(
            "Trajectory, status, and evaluation instance_id sets do not match"
        )
    unknown = sorted(instance_ids - set(global_indices))
    if unknown:
        raise ValueError(f"Instances not found in train data: {unknown}")

    ordered_ids = sorted(instance_ids, key=global_indices.__getitem__)
    merged_trajectories = []
    merged_statuses = []
    merged_evaluations = []
    successful_trajectories = []
    quarantined = []

    for instance_id in ordered_ids:
        global_index = global_indices[instance_id]

        trajectory = dict(trajectories[instance_id])
        trajectory["instance_idx"] = global_index
        merged_trajectories.append(trajectory)

        status = dict(statuses[instance_id])
        status["instance_idx"] = global_index
        merged_statuses.append(status)

        evaluation = dict(evaluation_inputs[instance_id])
        evaluation["instance_idx"] = global_index
        merged_evaluations.append(evaluation)

        if status.get("status") == "success":
            successful_trajectories.append(trajectory)
        else:
            quarantined.append(
                {
                    "instance_idx": global_index,
                    "instance_id": instance_id,
                    "status": status.get("status"),
                    "error_message": status.get("error_message", ""),
                    "pred_sqls": status.get("pred_sqls", []),
                    "sol_sql": status.get("sol_sql", []),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "trajectories.jsonl", merged_trajectories)
    write_jsonl(output_dir / "successful_trajectories.jsonl", successful_trajectories)
    write_jsonl(output_dir / "evaluation_status.jsonl", merged_statuses)
    write_jsonl(output_dir / "eval_ready.jsonl", merged_evaluations)
    write_jsonl(output_dir / "quarantine.jsonl", quarantined)

    manifest = {
        "train_data": args.train_data,
        "sources": source_summary,
        "total": len(merged_trajectories),
        "successful": len(successful_trajectories),
        "quarantined": len(quarantined),
        "first_instance": ordered_ids[0] if ordered_ids else None,
        "last_instance": ordered_ids[-1] if ordered_ids else None,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
