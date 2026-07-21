"""Persistent history of classification runs.

Each classification is saved so past uploads and generated reports stay
accessible through the app, and so their summary stats can feed an aggregate
dashboard.

Two storage backends, chosen by environment:

S3-compatible object storage (production on Render's free tier)
    Set S3_BUCKET, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID and
    AWS_SECRET_ACCESS_KEY (e.g. a Backblaze B2 or Cloudflare R2 bucket).
    Every report becomes three objects, so nothing depends on the server's
    filesystem and history survives restarts, redeploys, and free-tier idle
    spin-downs:
        reports/<id>.json    - summary stats (one small JSON per report)
        reports/<id>.xlsx    - the enriched, downloadable workbook
        reports/<id>.pkl.gz  - the classified DataFrame (gzip pickle)

Local filesystem (development, or when no bucket is configured)
    DATA_DIR/history.db          - SQLite metadata + stats (one row per report)
    DATA_DIR/reports/<id>.xlsx   - the enriched, downloadable workbook
    DATA_DIR/reports/<id>.pkl.gz - the classified DataFrame (gzip pickle)

DATA_DIR defaults to /var/data (a mounted disk). If that isn't writable we
fall back to the first writable candidate so the app keeps working instead of
crashing with PermissionError — but on an ephemeral fallback (e.g. Render free
tier without a bucket) the history doesn't survive restarts; is_persistent()
lets the app warn about that. Everything stored here can contain driver PII,
so it lives only behind the password gate and is never committed to the repo.
"""

from __future__ import annotations

import io
import json
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
COLUMNS = [
    "id", "created_at", "file_name", "model", "quick_test",
    "total_rows", "classified_rows", "distinct_reasons", "excluded_rows",
    "n_acceptable", "n_guidance", "n_potentially", "n_not_acceptable",
]

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


# --- S3-compatible object storage backend -----------------------------------

_s3_client = None


def _bucket() -> str | None:
    return os.environ.get("S3_BUCKET") or None


def _s3():
    """A cached boto3 client when a bucket is configured, else None."""
    global _s3_client
    if not _bucket():
        return None
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None
        )
    return _s3_client


def _s3_key(report_id: str, ext: str) -> str:
    return f"reports/{report_id}.{ext}"


def _s3_get(key: str) -> bytes:
    return _s3().get_object(Bucket=_bucket(), Key=key)["Body"].read()


def _s3_put(key: str, body: bytes) -> None:
    _s3().put_object(Bucket=_bucket(), Key=key, Body=body)


# --- Local filesystem backend ------------------------------------------------

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

    Prefers DATA_DIR (a mounted disk) but falls back to a local or temp dir if
    that mount is missing or read-only, so history never crashes the app. The
    choice is cached for the process once resolved.
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
    """True when saved reports will survive a restart/redeploy.

    Object storage always persists. On the filesystem it depends on where we
    ended up: the configured/mounted data dir persists; an ephemeral fallback
    (no disk mounted, DATA_DIR unwritable) does not — on Render's free tier it
    is wiped on every idle spin-down. The app uses this to warn users instead
    of losing data silently.
    """
    if _bucket():
        return True
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
    """Create the local database + schema if needed (idempotent; no-op on S3)."""
    if _bucket():
        return
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


# --- Public API ---------------------------------------------------------------

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
    """Persist one classified report (stats + files) and return its id."""
    init()
    report_id = uuid.uuid4().hex[:12]

    done = df[df["Classification"] != ""]
    counts = done["Classification"].value_counts()
    stats = {
        "id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_name": file_name,
        "model": model,
        "quick_test": int(quick_test),
        "total_rows": int(len(df)),
        "classified_rows": int(len(done)),
        "distinct_reasons": int(distinct_reasons),
        "excluded_rows": int(excluded_rows),
        "n_acceptable": int(counts.get(ACCEPTABLE, 0)),
        "n_guidance": int(counts.get(DRIVER_GUIDANCE, 0)),
        "n_potentially": int(counts.get(MANUAL_REVIEW, 0))
        + int(counts.get(POTENTIALLY_ACCEPTABLE, 0)),
        "n_not_acceptable": int(counts.get(NOT_ACCEPTABLE, 0)),
    }

    if _bucket():
        buf = io.BytesIO()
        df.to_pickle(buf, compression="gzip")
        _s3_put(_s3_key(report_id, "xlsx"), xlsx_bytes)
        _s3_put(_s3_key(report_id, "pkl.gz"), buf.getvalue())
        # Stats go last so a half-uploaded report never appears in listings.
        _s3_put(_s3_key(report_id, "json"), json.dumps(stats).encode())
        return report_id

    _xlsx_path(report_id).write_bytes(xlsx_bytes)
    df.to_pickle(_data_path(report_id), compression="gzip")
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO reports ({', '.join(COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(COLUMNS))})",
            [stats[c] for c in COLUMNS],
        )
    return report_id


def list_reports() -> pd.DataFrame:
    """All saved reports, newest first (stats only — no row-level PII loaded)."""
    init()
    if _bucket():
        rows = []
        paginator = _s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_bucket(), Prefix="reports/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    rows.append(json.loads(_s3_get(obj["Key"])))
        frame = pd.DataFrame(rows, columns=COLUMNS)
        return frame.sort_values("created_at", ascending=False).reset_index(drop=True)
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM reports ORDER BY created_at DESC", conn
        )


def load_df(report_id: str) -> pd.DataFrame:
    """Load a saved report's full classified rows for re-viewing."""
    if _bucket():
        return pd.read_pickle(
            io.BytesIO(_s3_get(_s3_key(report_id, "pkl.gz"))), compression="gzip"
        )
    return pd.read_pickle(_data_path(report_id), compression="gzip")


def load_xlsx(report_id: str) -> bytes:
    """Load a saved report's enriched workbook bytes for re-download."""
    if _bucket():
        return _s3_get(_s3_key(report_id, "xlsx"))
    return _xlsx_path(report_id).read_bytes()


def delete_report(report_id: str) -> None:
    """Remove a report's stats and its files."""
    if _bucket():
        # Stats first so the report disappears from listings immediately.
        for ext in ("json", "xlsx", "pkl.gz"):
            _s3().delete_object(Bucket=_bucket(), Key=_s3_key(report_id, ext))
        return
    for p in (_xlsx_path(report_id), _data_path(report_id)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    with _connect() as conn:
        conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
