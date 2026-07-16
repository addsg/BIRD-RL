#!/usr/bin/env python3
"""Manage isolated, persistent SQLite databases for critic trajectories."""

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path


_TEMP_OBJECT_RE = re.compile(
    r"^(\s*CREATE\s+)(?:TEMP|TEMPORARY)(\s+)"
    r"(TABLE|VIEW|INDEX|TRIGGER)\b",
    flags=re.IGNORECASE,
)


def session_database_path(session_dir: str, instance_idx: int) -> Path:
    """Return the deterministic database path for one trajectory."""
    return Path(session_dir) / f"instance_{instance_idx}.sqlite"


def _source_database_path(db_dir: str, db_id: str) -> Path:
    db_root = Path(db_dir) / db_id
    candidates = [
        db_root / f"{db_id}_template.sqlite",
        db_root / f"{db_id}.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No SQLite database found for {db_id}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def materialize_temporary_object(sql: str) -> tuple[str, bool]:
    """Convert CREATE TEMP objects to persistent objects in an isolated DB."""
    rewritten, count = _TEMP_OBJECT_RE.subn(r"\1\3 ", sql, count=1)
    return rewritten, bool(count)


def initialize_sessions(
    input_path: str,
    db_dir: str,
    session_dir: str,
    limit: int | None = None,
    force: bool = False,
) -> list[dict]:
    """Create and preprocess one independent SQLite copy per trajectory."""
    session_root = Path(session_dir)
    if session_root.exists():
        if not force:
            raise FileExistsError(
                f"Session directory already exists: {session_root}. "
                "Use --force to replace it."
            )
        shutil.rmtree(session_root)
    session_root.mkdir(parents=True)

    records = []
    try:
        with open(input_path, "r", encoding="utf-8") as source:
            for instance_idx, line in enumerate(source):
                if limit is not None and len(records) >= limit:
                    break
                if not line.strip():
                    continue

                sample = json.loads(line)
                db_id = sample.get("db_id", "")
                instance_id = sample.get("instance_id", str(instance_idx))
                if not db_id:
                    raise ValueError(f"Missing db_id for instance {instance_id}")

                source_db = _source_database_path(db_dir, db_id)
                target_db = session_database_path(session_dir, instance_idx)
                shutil.copy2(source_db, target_db)

                preprocess_sql = sample.get("preprocess_sql") or []
                materialized_count = 0
                conn = sqlite3.connect(target_db)
                try:
                    for statement in preprocess_sql:
                        prepared_sql, materialized = materialize_temporary_object(
                            str(statement)
                        )
                        materialized_count += int(materialized)
                        conn.execute(prepared_sql)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

                records.append(
                    {
                        "instance_idx": instance_idx,
                        "instance_id": instance_id,
                        "db_id": db_id,
                        "database_path": str(target_db),
                        "preprocess_statements": len(preprocess_sql),
                        "materialized_temp_objects": materialized_count,
                    }
                )

        manifest_path = session_root / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            for record in records:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        shutil.rmtree(session_root, ignore_errors=True)
        raise

    return records


def cleanup_sessions(session_dir: str) -> None:
    """Remove all trajectory database copies."""
    shutil.rmtree(session_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage isolated SQLite databases for critic trajectories"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--input", required=True)
    init_parser.add_argument("--db-dir", required=True)
    init_parser.add_argument("--session-dir", required=True)
    init_parser.add_argument("--limit", type=int, default=None)
    init_parser.add_argument("--force", action="store_true")

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--session-dir", required=True)

    args = parser.parse_args()
    if args.command == "init":
        records = initialize_sessions(
            input_path=args.input,
            db_dir=args.db_dir,
            session_dir=args.session_dir,
            limit=args.limit,
            force=args.force,
        )
        preprocess_count = sum(r["preprocess_statements"] for r in records)
        print(
            f"Initialized {len(records)} trajectory databases; "
            f"executed {preprocess_count} preprocess statements"
        )
    else:
        cleanup_sessions(args.session_dir)
        print(f"Cleaned trajectory databases: {args.session_dir}")


if __name__ == "__main__":
    main()
