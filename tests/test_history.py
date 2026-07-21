"""Round-trip tests for the report history store."""

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_classifier import ACCEPTABLE, DRIVER_GUIDANCE, MANUAL_REVIEW, NOT_ACCEPTABLE


def _hist(tmp_path, monkeypatch):
    """Import history with DATA_DIR pointed at an isolated temp dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import history
    return importlib.reload(history)


def _frame():
    return pd.DataFrame(
        {
            "Parent Name": ["Acme", "Acme", "Beta", "Beta"],
            "vcReason": ["google maps", "personal trip", "garage road test", "pick up parts"],
            "_reason": ["google maps", "personal trip", "garage road test", "pick up parts"],
            "Classification": [ACCEPTABLE, NOT_ACCEPTABLE, MANUAL_REVIEW, DRIVER_GUIDANCE],
            "Rationale": ["route choice", "personal", "customer contact to decide", "unlogged stop"],
        }
    )


def test_save_then_load_round_trip(tmp_path, monkeypatch):
    hist = _hist(tmp_path, monkeypatch)
    rid = hist.save_report(
        df=_frame(), xlsx_bytes=b"PK\x03\x04 fake xlsx",
        file_name="report.xlsx", model="anthropic/claude-haiku-4.5",
        quick_test=True, distinct_reasons=4, excluded_rows=1,
    )

    reports = hist.list_reports()
    assert len(reports) == 1
    row = reports.iloc[0]
    assert row["id"] == rid
    assert row["file_name"] == "report.xlsx"
    assert row["classified_rows"] == 4
    assert row["n_acceptable"] == 1
    assert row["n_not_acceptable"] == 1
    assert row["n_guidance"] == 1
    assert row["n_potentially"] == 1
    assert row["quick_test"] == 1
    assert row["excluded_rows"] == 1

    loaded = hist.load_df(rid)
    assert list(loaded["Classification"]) == [
        ACCEPTABLE, NOT_ACCEPTABLE, MANUAL_REVIEW, DRIVER_GUIDANCE,
    ]
    assert hist.load_xlsx(rid) == b"PK\x03\x04 fake xlsx"


def test_delete_removes_row_and_files(tmp_path, monkeypatch):
    hist = _hist(tmp_path, monkeypatch)
    rid = hist.save_report(
        df=_frame(), xlsx_bytes=b"x", file_name="r.xlsx",
        model="m", quick_test=False, distinct_reasons=3, excluded_rows=0,
    )
    assert not hist.list_reports().empty

    hist.delete_report(rid)
    assert hist.list_reports().empty
    assert not hist._xlsx_path(rid).exists()
    assert not hist._data_path(rid).exists()


class _FakeS3:
    """Minimal in-memory stand-in for the boto3 S3 client surface we use."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objects = self.objects

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [k for k in sorted(objects) if k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}

        return _Paginator()


def _s3_hist(monkeypatch):
    """Import history in S3 mode with the fake client injected."""
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    import history
    hist = importlib.reload(history)
    fake = _FakeS3()
    monkeypatch.setattr(hist, "_s3", lambda: fake)
    return hist, fake


def test_s3_round_trip(monkeypatch):
    hist, fake = _s3_hist(monkeypatch)
    assert hist.is_persistent()

    rid = hist.save_report(
        df=_frame(), xlsx_bytes=b"PK\x03\x04 fake xlsx",
        file_name="report.xlsx", model="anthropic/claude-haiku-4.5",
        quick_test=True, distinct_reasons=4, excluded_rows=1,
    )
    assert set(fake.objects) == {
        f"reports/{rid}.json", f"reports/{rid}.xlsx", f"reports/{rid}.pkl.gz",
    }

    reports = hist.list_reports()
    assert len(reports) == 1
    row = reports.iloc[0]
    assert row["id"] == rid
    assert row["file_name"] == "report.xlsx"
    assert row["classified_rows"] == 4
    assert row["n_acceptable"] == 1
    assert row["n_guidance"] == 1
    assert row["n_potentially"] == 1
    assert row["n_not_acceptable"] == 1
    assert row["excluded_rows"] == 1

    loaded = hist.load_df(rid)
    assert list(loaded["Classification"]) == [
        ACCEPTABLE, NOT_ACCEPTABLE, MANUAL_REVIEW, DRIVER_GUIDANCE,
    ]
    assert hist.load_xlsx(rid) == b"PK\x03\x04 fake xlsx"


def test_s3_delete_and_empty_listing(monkeypatch):
    hist, fake = _s3_hist(monkeypatch)
    assert hist.list_reports().empty  # empty frame still has the schema columns
    assert list(hist.list_reports().columns) == hist.COLUMNS

    rid = hist.save_report(
        df=_frame(), xlsx_bytes=b"x", file_name="r.xlsx",
        model="m", quick_test=False, distinct_reasons=3, excluded_rows=0,
    )
    assert not hist.list_reports().empty
    hist.delete_report(rid)
    assert hist.list_reports().empty
    assert fake.objects == {}


def test_s3_lists_newest_first(monkeypatch):
    hist, fake = _s3_hist(monkeypatch)
    import json
    for i, ts in enumerate(["2026-07-01T10:00:00+00:00", "2026-07-21T10:00:00+00:00"]):
        stats = {c: 0 for c in hist.COLUMNS}
        stats.update(id=f"id{i}", created_at=ts, file_name=f"f{i}.xlsx", model="m")
        fake.objects[f"reports/id{i}.json"] = json.dumps(stats).encode()
    reports = hist.list_reports()
    assert list(reports["id"]) == ["id1", "id0"]


def test_excluded_count_persists_unique_per_report(tmp_path, monkeypatch):
    hist = _hist(tmp_path, monkeypatch)
    hist.save_report(df=_frame(), xlsx_bytes=b"a", file_name="one.xlsx",
                     model="m", quick_test=False, distinct_reasons=3, excluded_rows=2)
    hist.save_report(df=_frame(), xlsx_bytes=b"b", file_name="two.xlsx",
                     model="m", quick_test=False, distinct_reasons=3, excluded_rows=5)
    reports = hist.list_reports()
    assert len(reports) == 2
    assert set(reports["file_name"]) == {"one.xlsx", "two.xlsx"}
    assert set(reports["excluded_rows"]) == {2, 5}
