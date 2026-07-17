#!/usr/bin/env python3
"""Validate VERL multi-turn SFT parquet structure and token lengths."""

import argparse
import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def validate_file(path, tokenizer, max_length):
    table = pq.read_table(path)
    if table.column_names != ["messages"]:
        raise ValueError(f"{path}: expected only messages column, got {table.column_names}")
    rows = table.to_pylist()
    lengths = []
    assistant_turns = 0

    for row_index, row in enumerate(rows):
        messages = row["messages"]
        roles = [message["role"] for message in messages]
        if roles[:2] != ["system", "user"]:
            raise ValueError(f"{path}:{row_index}: invalid initial roles {roles[:2]}")
        if roles[-1] != "assistant":
            raise ValueError(f"{path}:{row_index}: final role is not assistant")
        for index, role in enumerate(roles[2:], start=2):
            expected = "assistant" if index % 2 == 0 else "user"
            if role != expected:
                raise ValueError(
                    f"{path}:{row_index}: role {role} at {index}, expected {expected}"
                )
        if '"name": "submit_solution"' not in messages[-1]["content"]:
            raise ValueError(f"{path}:{row_index}: final assistant has no submit_solution")
        initial_prompt = messages[0]["content"] + "\n" + messages[1]["content"]
        for marker in ("Ground Truth Solution", "For Reference Only"):
            if marker in initial_prompt:
                raise ValueError(f"{path}:{row_index}: teacher marker leaked: {marker}")
        for message in messages:
            if not message["content"].strip():
                raise ValueError(f"{path}:{row_index}: empty {message['role']} message")

        assistant_turns += roles.count("assistant")
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(len(token_ids))

    return {
        "rows": len(rows),
        "assistant_turns": assistant_turns,
        "tokens": {
            "min": min(lengths),
            "median": int(statistics.median(lengths)),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "over_max_length": sum(length > max_length for length in lengths),
            "max_length": max_length,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    report = {
        "tokenizer": args.tokenizer,
        "train": validate_file(Path(args.train), tokenizer, args.max_length),
        "validation": validate_file(
            Path(args.validation), tokenizer, args.max_length
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
