#!/usr/bin/env python3
"""Generate one bounded after-preprocess SQLite schema snapshot per sample."""

import argparse
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

from bird_rl.inference.critic_session import (
    _shadow_main_table,
    _source_database_path,
    materialize_temporary_object,
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def format_value(value, max_cell_chars: int) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    text = str(value).replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > max_cell_chars:
        return text[:max_cell_chars] + f"... <{len(text)} chars>"
    return text


def extract_schema(
    db_path: Path,
    sample_rows: int,
    max_cell_chars: int,
    max_schema_chars: int,
) -> str:
    parts = []
    connection = sqlite3.connect(db_path)
    try:
        objects = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE '__critic_base_%' "
            "ORDER BY type, name"
        ).fetchall()
        for object_type, name, ddl in objects:
            if ddl:
                parts.append(ddl.rstrip(";") + ";")
            if object_type == "table" and sample_rows > 0:
                columns = [
                    row[1]
                    for row in connection.execute(
                        f"PRAGMA table_info({quote_identifier(name)})"
                    ).fetchall()
                ]
                try:
                    rows = connection.execute(
                        f"SELECT * FROM {quote_identifier(name)} LIMIT ?",
                        (sample_rows,),
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                if rows:
                    parts.append(f"First {len(rows)} rows:")
                    parts.append("  " + "\t".join(columns))
                    for row in rows:
                        parts.append(
                            "  "
                            + "\t".join(
                                format_value(value, max_cell_chars)
                                for value in row
                            )
                        )
            parts.append("")
    finally:
        connection.close()

    schema = "\n".join(parts).strip()
    if len(schema) > max_schema_chars:
        schema = schema[:max_schema_chars] + "\n... <schema truncated>"
    return schema


def apply_preprocess(db_path: Path, statements: list) -> None:
    connection = sqlite3.connect(db_path)
    statement = None
    try:
        for statement in statements:
            sql = str(statement)
            _shadow_main_table(connection, sql)
            prepared_sql, _ = materialize_temporary_object(sql)
            connection.execute(prepared_sql)
        connection.commit()
    except Exception as exc:
        connection.rollback()
        raise RuntimeError(f"preprocess failed at {statement!r}: {exc}") from exc
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--max-cell-chars", type=int, default=200)
    parser.add_argument("--max-schema-chars", type=int, default=30000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(prefix="schema-snapshot-") as temp_dir:
        working_db = Path(temp_dir) / "working.sqlite"
        with open(args.input, "r", encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8"
        ) as output:
            for instance_idx, line in enumerate(source):
                if args.limit is not None and count >= args.limit:
                    break
                if not line.strip():
                    continue
                sample = json.loads(line)
                instance_id = sample.get("instance_id", str(instance_idx))
                db_id = sample["db_id"]
                source_db = _source_database_path(args.db_dir, db_id)
                shutil.copy2(source_db, working_db)
                apply_preprocess(working_db, sample.get("preprocess_sql") or [])
                schema = extract_schema(
                    working_db,
                    args.sample_rows,
                    args.max_cell_chars,
                    args.max_schema_chars,
                )
                output.write(
                    json.dumps(
                        {
                            "instance_idx": instance_idx,
                            "instance_id": instance_id,
                            "db_id": db_id,
                            "after_preprocess_schema": schema,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                working_db.unlink(missing_ok=True)
                count += 1
                if count % 25 == 0:
                    print(f"Generated {count} schema snapshots", flush=True)

    print(f"Saved {count} schema snapshots to {output_path}")


if __name__ == "__main__":
    main()
