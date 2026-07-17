#!/usr/bin/env python3
"""Generate gold-guided critic trajectories with an OpenAI-compatible API."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run([str(arg) for arg in args], check=True)


def module(name: str, *args: str) -> None:
    run(sys.executable, "-m", name, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--skip_evaluation", action="store_true")
    args = parser.parse_args()

    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if not os.environ.get(name):
            parser.error(f"{name} is not set")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    trajectory_dir = output_dir / "trajectories"
    session_dir = output_dir / "session_dbs"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    limit_args = ("--limit", str(args.limit))
    final_trajectory = None

    try:
        module(
            "bird_rl.inference.critic_session",
            "init",
            "--input", args.input,
            "--db-dir", args.db_dir,
            "--session-dir", str(session_dir),
            "--force",
            *limit_args,
        )

        for turn in range(args.max_turns):
            prompt = output_dir / f"prompts_turn_{turn}.jsonl"
            response = output_dir / f"responses_turn_{turn}.jsonl"
            parsed = output_dir / f"parsed_turn_{turn}.jsonl"
            observation = output_dir / f"observations_turn_{turn}.jsonl"
            trajectory = trajectory_dir / f"traj_{turn}.jsonl"

            module(
                "bird_rl.inference.critic.generate_teacher_prompts",
                "--turn", str(turn),
                "--max-turns", str(args.max_turns),
                "--input", args.input,
                "--db-dir", args.db_dir,
                "--session-dir", str(session_dir),
                "--traj-dir", str(trajectory_dir),
                "--output", str(prompt),
                *limit_args,
            )
            if not prompt.exists() or prompt.stat().st_size == 0:
                break

            module(
                "bird_rl.inference.openai_compatible_infer",
                "--prompt_path", str(prompt),
                "--output_path", str(response),
                "--num_threads", str(args.num_threads),
                "--max_tokens", str(args.max_tokens),
                "--temperature", str(args.temperature),
            )
            module(
                "bird_rl.inference.parse_responses",
                "--input", str(response),
                "--output", str(parsed),
            )
            module(
                "bird_rl.inference.execute_sql_observations",
                "--input", str(parsed),
                "--output", str(observation),
                "--db-dir", args.db_dir,
                "--session-dir", str(session_dir),
            )
            module(
                "bird_rl.inference.build_trajectory",
                "--turn", str(turn),
                "--traj-dir", str(trajectory_dir),
                "--observations", str(observation),
                "--output", str(trajectory),
                "--submit-format", "sql_list",
            )
            final_trajectory = trajectory

        if final_trajectory:
            evaluation_input = output_dir / "eval_ready.jsonl"
            module(
                "bird_rl.inference.critic.evaluate",
                "--trajectory", str(final_trajectory),
                "--original-data", args.input,
                "--output", str(evaluation_input),
                *limit_args,
            )
            if not args.skip_evaluation:
                run(
                    "bash",
                    str(repo_root / "evaluation/critic/run/run_eval.sh"),
                    "--jsonl_file", str(evaluation_input),
                    "--db_dir", args.db_dir,
                    "--mode", "pred",
                    "--num_threads", "1",
                    "--batch_size", "1",
                )
    finally:
        module(
            "bird_rl.inference.critic_session",
            "cleanup",
            "--session-dir", str(session_dir),
        )

    print(f"Pipeline complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
