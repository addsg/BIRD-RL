#!/usr/bin/env python3
"""Generate critic prompts with hidden gold-SQL guidance for SFT trajectories."""

import argparse
import json
import os
from pathlib import Path

from bird_rl.inference.critic.generate_prompts import (
    build_history_from_trajectory,
    get_schema_from_db,
)
from bird_rl.inference.critic_session import session_database_path
from bird_rl.prompts.sft_generation import (
    SFT_GENERATION_SYSTEM_PROMPT,
    SFT_GENERATION_USER_TEMPLATE,
)


def process_dataset(args: argparse.Namespace) -> int:
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        data = data[:args.limit]

    trajectories = {}
    if args.turn > 0:
        previous_path = Path(args.traj_dir) / f"traj_{args.turn - 1}.jsonl"
        with previous_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    trajectories[item["instance_idx"]] = item.get("trajectory", [])

    prompts = []
    schema_cache = {}
    for instance_idx, item in enumerate(data):
        trajectory = trajectories.get(instance_idx, [])
        if args.turn > 0 and (
            not trajectory or trajectory[-1].get("end_flag", False)
        ):
            continue

        query = item.get("query", "")
        db_id = item.get("db_id", "")
        issue_sql = item.get("issue_sql", "")
        solution_sql = item.get("sol_sql", "")
        if not query or not db_id or not issue_sql or not solution_sql:
            continue

        if instance_idx not in schema_cache:
            if args.session_dir:
                db_path = str(session_database_path(args.session_dir, instance_idx))
            else:
                db_path = os.path.join(args.db_dir, db_id, f"{db_id}.sqlite")
            schema_cache[instance_idx] = get_schema_from_db(db_path)

        issue_sql = "\n".join(issue_sql) if isinstance(issue_sql, list) else str(issue_sql)
        solution_sql = (
            "\n".join(solution_sql) if isinstance(solution_sql, list) else str(solution_sql)
        )
        system_prompt = SFT_GENERATION_SYSTEM_PROMPT.format(
            max_turns=args.max_turns,
            prev_turns=args.max_turns - 1,
        )
        user_prompt = SFT_GENERATION_USER_TEMPLATE.format(
            query=query.strip(),
            schema=schema_cache[instance_idx].strip(),
            issue_sql=issue_sql.strip(),
            solution_sql=solution_sql.strip(),
            max_turns=args.max_turns,
        )
        history = build_history_from_trajectory(trajectory)
        if history:
            user_prompt += "\n\n" + history

        prompts.append(
            {
                "idx": len(prompts),
                "instance_idx": instance_idx,
                "instance_id": item.get("instance_id", str(instance_idx)),
                "db_id": db_id,
                "current_turn": args.turn,
                "max_turns": args.max_turns,
                "system_prompt": system_prompt,
                "prompt": user_prompt,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(prompts)} teacher prompts -> {output_path}")
    return len(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gold-guided critic prompts")
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--input", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--session-dir")
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    process_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
