"""DuckDB results store (per CLAUDE.md tech constraints)."""
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).parent / "results.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    run_id VARCHAR,
    instance_id VARCHAR,
    precision VARCHAR,
    resolved BOOLEAN,
    turns INTEGER,
    submitted BOOLEAN,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    wall_clock_seconds DOUBLE,
    error BOOLEAN,
    recorded_at TIMESTAMP DEFAULT current_timestamp
);
"""


def get_conn():
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(SCHEMA)
    return conn


def record_episode(run_id: str, precision: str, episode_result: dict, resolved: bool):
    conn = get_conn()
    conn.execute(
        """INSERT INTO episodes
        (run_id, instance_id, precision, resolved, turns, submitted,
         prompt_tokens, completion_tokens, wall_clock_seconds, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            run_id,
            episode_result["instance_id"],
            precision,
            resolved,
            episode_result["turns"],
            episode_result["submitted"],
            episode_result["prompt_tokens"],
            episode_result["completion_tokens"],
            episode_result["wall_clock_seconds"],
            episode_result.get("error", False),
        ],
    )
    conn.close()
