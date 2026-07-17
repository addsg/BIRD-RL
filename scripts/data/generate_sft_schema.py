#!/usr/bin/env python3
"""Generate SFT schemas while matching critic-session preprocess rollback."""

import sqlite3

from scripts.data import generate_after_preprocess_schema as generator


def tolerant_preprocess(db_path, statements):
    connection = sqlite3.connect(db_path)
    statement = None
    try:
        for statement in statements:
            sql = str(statement)
            generator._shadow_main_table(connection, sql)
            prepared_sql, _ = generator.materialize_temporary_object(sql)
            connection.execute(prepared_sql)
        connection.commit()
    except Exception as exc:
        connection.rollback()
        print(
            f"WARNING: preprocess rolled back: {type(exc).__name__}: {exc}; "
            f"statement={statement!r}",
            flush=True,
        )
    finally:
        connection.close()


generator.apply_preprocess = tolerant_preprocess


if __name__ == "__main__":
    generator.main()
