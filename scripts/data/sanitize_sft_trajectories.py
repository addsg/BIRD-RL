#!/usr/bin/env python3
"""Bound teacher thoughts and tool observations without truncating actions."""

import argparse
import json
from pathlib import Path


def truncate(text, limit, label):
    if not text or len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return text[:limit] + f"\n... <{label} truncated, {omitted} chars omitted>", True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-observation-chars", type=int, default=4000)
    parser.add_argument("--max-thought-chars", type=int, default=4000)
    args = parser.parse_args()

    records = []
    observation_truncations = 0
    thought_truncations = 0
    with open(args.input, "r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for turn in record.get("trajectory", []):
                turn["observation"], changed = truncate(
                    turn.get("observation", ""),
                    args.max_observation_chars,
                    "observation",
                )
                observation_truncations += int(changed)
                turn["thought"], changed = truncate(
                    turn.get("thought", ""),
                    args.max_thought_chars,
                    "thought",
                )
                thought_truncations += int(changed)
            records.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"Saved {len(records)} trajectories; "
        f"truncated {observation_truncations} observations and "
        f"{thought_truncations} thoughts"
    )


if __name__ == "__main__":
    main()
