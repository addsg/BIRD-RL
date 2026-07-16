#!/usr/bin/env python3
"""Execute SQL tool calls against persistent per-trajectory SQLite copies."""

import argparse
import json
import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bird_rl.inference.critic_session import session_database_path


def _format_rows_as_json(col_names, rows, max_rows=50):
    """Format query results as readable JSON lines."""
    if not rows:
        return "(No results)"
    lines = []
    for row in rows[:max_rows]:
        row_dict = {
            col: (str(value) if value is not None else "NULL")
            for col, value in zip(col_names, row)
        }
        lines.append(json.dumps(row_dict, ensure_ascii=False))
    result = "\n".join(lines)
    if len(rows) > max_rows:
        result += f"\n... (more rows available, showing first {max_rows})"
    return result


def _execute_sql_worker(queue, sql_statements, db_path):
    """Run a tool call in a subprocess so it can be terminated on timeout."""
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        col_names = []
        rows = []
        for sql in sql_statements:
            cursor.execute(sql)
            col_names = (
                [description[0] for description in cursor.description]
                if cursor.description
                else []
            )
            rows = cursor.fetchmany(51) if cursor.description else []
        conn.commit()
        queue.put(("success", col_names, rows))
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        queue.put(("error", str(exc), None))
    finally:
        if conn is not None:
            conn.close()


def execute_sql_safe(sql, db_path: str, timeout: int = 30) -> dict:
    """Execute one tool call transactionally and return a model observation."""
    statements = sql if isinstance(sql, list) else [sql]
    statements = [
        str(statement).strip()
        for statement in statements
        if statement is not None and str(statement).strip()
    ]
    if not statements:
        return {"exec_flag": False, "exec_results": "Error: Empty SQL"}
    if not os.path.exists(db_path):
        return {
            "exec_flag": False,
            "exec_results": f"Error: Database not found: {db_path}",
        }

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_sql_worker, args=(queue, statements, db_path)
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        return {
            "exec_flag": False,
            "exec_results": f"Error: Query timed out after {timeout}s",
        }

    try:
        if not queue.empty():
            status, *rest = queue.get_nowait()
            if status == "success":
                col_names, rows = rest
                observation = (
                    _format_rows_as_json(col_names, rows)
                    if col_names
                    else "Statement executed successfully."
                )
                return {"exec_flag": True, "exec_results": observation}
            return {"exec_flag": False, "exec_results": f"Error: {rest[0]}"}
    except Exception as exc:
        return {"exec_flag": False, "exec_results": f"Error: {exc}"}
    finally:
        queue.close()

    return {"exec_flag": False, "exec_results": "Error: No result returned"}


def process_single_instance(
    item: dict,
    db_dir: str,
    timeout: int = 30,
    session_dir: str = None,
) -> dict:
    """Execute one parsed action without allowing bad output to stop the batch."""
    tool_name = item.get("tool_name")
    pred_sqls = item.get("pred_sqls") or []
    end_flag = item.get("end_flag", False)

    if tool_name is None:
        item.update(
            exec_flag=False,
            error_type="invalid_tool_call",
            exec_results=(
                "Error: Invalid tool call. Use exactly one <tool_call> JSON object "
                'with execute_sql {"sql": "..."} or submit_solution '
                '{"sql_list": ["..."]}.'
            ),
        )
        return item

    if end_flag:
        if tool_name != "submit_solution" or not pred_sqls:
            item.update(
                end_flag=False,
                exec_flag=False,
                error_type="empty_submission",
                exec_results=(
                    "Error: submit_solution requires at least one SQL statement."
                ),
            )
            return item
        item.update(
            exec_flag=True,
            submission_ready=True,
            exec_results="Solution accepted for final evaluation.",
        )
        return item

    if tool_name != "execute_sql":
        item.update(
            exec_flag=False,
            error_type="invalid_tool_call",
            exec_results=f"Error: Unsupported tool: {tool_name}",
        )
        return item
    if not pred_sqls:
        item.update(exec_flag=False, exec_results="Error: No SQL to execute")
        return item

    if session_dir:
        db_path = str(session_database_path(session_dir, item["instance_idx"]))
    else:
        db_id = item.get("db_id", "")
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    item.update(execute_sql_safe(pred_sqls, db_path, timeout=timeout))
    return item


def process_observations(
    input_path: str,
    output_path: str,
    db_dir: str,
    num_threads: int = 8,
    timeout: int = 30,
    session_dir: str = None,
):
    """Execute all parsed responses while isolating failures per instance."""
    print(f"Loading parsed responses from: {input_path}")
    data = []
    with open(input_path, "r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} instances")

    results = [None] * len(data)
    completed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                process_single_instance, item, db_dir, timeout, session_dir
            ): index
            for index, item in enumerate(data)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result(timeout=timeout + 30)
            except Exception as exc:
                data[index].update(
                    exec_flag=False,
                    error_type="executor_error",
                    exec_results=f"Error: {exc}",
                )
                results[index] = data[index]
            completed += 1
            if completed % 100 == 0 or completed == len(data):
                print(f"  Progress: {completed}/{len(data)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    exec_count = sum(1 for result in results if result.get("exec_flag"))
    invalid_count = sum(
        1 for result in results if result.get("error_type") == "invalid_tool_call"
    )
    print(f"  Successful actions: {exec_count}/{len(results)}")
    print(f"  Invalid tool calls recovered: {invalid_count}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute SQL and collect observations")
    parser.add_argument("--input", required=True, help="Input parsed-response JSONL")
    parser.add_argument("--output", required=True, help="Output observation JSONL")
    parser.add_argument("--db-dir", required=True, help="Base database directory")
    parser.add_argument("--session-dir", default=None,
                        help="Per-trajectory SQLite database directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    process_observations(
        args.input,
        args.output,
        args.db_dir,
        args.threads,
        args.timeout,
        args.session_dir,
    )


if __name__ == "__main__":
    main()
