"""Persistent history of classification runs.

Each classification is saved so past uploads and generated reports stay
accessible through the app, and so their summary stats can feed an aggregate
dashboard. The enriched workbook and the row-level data are written to a mounted
disk; lightweight summary stats go in a small SQLite database.

Storage layout (under DATA_DIR — a Render persistent disk in production):
    DATA_DIR/history.db          - SQLite metadata + stats (one row per report)
    DATA_DIR/reports/<id>.xlsx   - the enriched, downloadable workbook
    DATA_DIR/reports/<id>.pkl.gz - the classified DataFrame (gzip pickle)

DATA_DIR defaults to /var/data (the disk mount). If that isn't writable — e.g.
the Render persistent disk isn't mounted (free tier, or the Starter blueprint
hasn't synced yet) — we fall back to the first writable candidate so the app
keeps working instead of crashing with PermissionError. On an ephemeral
fallback the history simply doesn't survive restarts. Everything stored here can
contain driver PII, so it lives only on the password-gated server and is never
committed to the repo.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from llm_classifier import (
    ACCEPTABLE,
    DRIVER_GUIDANCE,
    MANUAL_REVIEW,
    NOT_ACCEPTABLE,
    POTENTIALLY_ACCEPTABLE,
)

# n_potentially predates the July-2026 taxonomy change; it now holds the
# "Manual Review Required" count (plus legacy "Potentially Acceptable" rows)
# so old report rows stay comparable. n_guidance is the new
# "Acceptable - Driver Guidance" count.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id               TEXT PRIMARY KEY,
    created_at       TEXT    NOT NULL,
    file_name        TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    quick_test       INTEGER NOT NULL DEFAULT 0,
    total_rows       INTEGER NOT NULL DEFAULT 0,
    classified_rows  INTEGER NOT NULL DEFAULT 0,
    distinct_reasons INTEGER NOT NULL DEFAULT 0,
    excluded_rows    INTEGER NOT NULL DEFAULT 0,
    n_acceptable     INTEGER NOT NULL DEFAULT 0,
    n_guidance       INTEGER NOT NULL DEFAULT 0,
    n_potentially    INTEGER NOT NULL DEFAULT 0,
    n_not_acceptable INTEGER NOT NULL DEFAULT 0
);
"""


_resolved_dir: Path | None = None


def _is_writable(path: Path) -> bool:
    """True if we can create `path` and write a file inside it."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_test_{uuid.uuid4().hex[:8]}"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def data_dir() -> Path:
    """First writable storage dir among the configured/default candidates.

    Prefers DATA_DIR (the mounted disk in production) but falls back to a local
    or temp dir if that mount is missing or read-only, so history never crashes
    the app. The choice is cached for the process once resolved.
    """
    global _resolved_dir
    if _resolved_dir is not None:
        return _resolved_dir

    candidates: list[Path] = []
    env = os.environ.get("DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/var/data"))
    candidates.append(Path.cwd() / ".data")
    fallback = Path(tempfile.gettempdir()) / "trip_reason_variance"
    candidates.append(fallback)

    for candidate in candidates:
        if _is_writable(candidate):
            _resolved_dir = candidate
            return candidate

    # Nothing was writable (very unlikely); use the temp dir and let the real
    # write surface its own error.
    _resolved_dir = fallback
    return fallback


def is_persistent() -> bool:
    """True when history is stored on the configured/mounted data dir.

    False means we fell back to an ephemeral directory (no disk mounted, or
    DATA_DIR unwritable) — saved reports will NOT survive a restart or
    redeploy, and on Render's free tier they vanish on every idle spin-down.
    The app uses this to warn users instead of losing data silently.
    """
    resolved = data_dir()
    intended = [Path(os.environ["DATA_DIR"])] if os.environ.get("DATA_DIR") else []
    intended.append(Path("/var/data"))
    if resolved in intended:
        return True
    # Local dev: ./.data lives in the repo and does persist. On Render (which
    # sets RENDER=true) the same path is on an ephemeral filesystem.
    return resolved == Path.cwd() / ".data" and not os.environ.get("RENDER")


def _reports_dir() -> Path:
    d = data_dir() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path() -> Path:
    data_dir().mkdir(parents=True, exist_ok=True)
    return data_dir() / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    """Create the database + schema if they don't exist yet (idempotent)."""
    with _connect() as conn:
        conn.execute(_SCHEMA)
        # Migrate pre-guidance databases in place.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(reports)")}
        if "n_guidance" not in cols:
            conn.execute(
                "ALTER TABLE reports ADD COLUMN n_guidance INTEGER NOT NULL DEFAULT 0"
            )


def _xlsx_path(report_id: str) -> Path:
    return _reports_dir() / f"{report_id}.xlsx"


def _data_path(report_id: str) -> Path:
    return _reports_dir() / f"{report_id}.pkl.gz"


def save_report(
    *,
    df: pd.DataFrame,
    xlsx_bytes: bytes,
    file_name: str,
    model: str,
    quick_test: bool,
    distinct_reasons: int,
    excluded_rows: int,
) -> str:
    """Persist one classified report (files + stats row) and return its id."""
    init()
    report_id = uuid.uuid4().hex[:12]

    done = df[df["Classification"] != ""]
    counts = done["Classification"].value_counts()

    _xlsx_path(report_id).write_bytes(xlsx_bytes)
    df.to_pickle(_data_path(report_id), compression="gzip")

    with _connect() as conn:
        conn.execute(
            "INSERT INTO reports (id, created_at, file_name, model, quick_test, "
            "total_rows, classified_rows, distinct_reasons, excluded_rows, "
            "n_acceptable, n_guidance, n_potentially, n_not_acceptable) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                file_name,
                model,
                int(quick_test),
                int(len(df)),
                int(len(done)),
                int(distinct_reasons),
                int(excluded_rows),
                int(counts.get(ACCEPTABLE, 0)),
                int(counts.get(DRIVER_GUIDANCE, 0)),
                int(counts.get(MANUAL_REVIEW, 0))
                + int(counts.get(POTENTIALLY_ACCEPTABLE, 0)),
                int(counts.get(NOT_ACCEPTABLE, 0)),
            ),
        )
    return report_id


def list_reports() -> pd.DataFrame:
    """All saved reports, newest first (stats only — no row-level PII loaded)."""
    init()
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM reports ORDER BY created_at DESC", conn
        )


def load_df(report_id: str) -> pd.DataFrame:
    """Load a saved report's full classified rows for re-viewing."""
    return pd.read_pickle(_data_path(report_id), compression="gzip")


def load_xlsx(report_id: str) -> bytes:
    """Load a saved report's enriched workbook bytes for re-download."""
    return _xlsx_path(report_id).read_bytes()


def delete_report(report_id: str) -> None:
    """Remove a report's stats row and its files."""
    for p in (_xlsx_path(report_id), _data_path(report_id)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    with _connect() as conn:
        conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
