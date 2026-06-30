"""Round-trip tests for the report history store."""

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_classifier import ACCEPTABLE, NOT_ACCEPTABLE, POTENTIALLY_ACCEPTABLE


def _hist(tmp_path, monkeypatch):
    """Import history with DATA_DIR pointed at an isolated temp dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import history
    return importlib.reload(history)


def _frame():
    return pd.DataFrame(
        {
            "Parent Name": ["Acme", "Acme", "Beta"],
            "vcReason": ["client meeting", "personal trip", "site visit"],
            "_reason": ["client meeting", "personal trip", "site visit"],
            "Classification": [ACCEPTABLE, NOT_ACCEPTABLE, POTENTIALLY_ACCEPTABLE],
            "Rationale": ["ok", "personal", "vague"],
        }
    )


def test_save_then_load_round_trip(tmp_path, monkeypatch):
    hist = _hist(tmp_path, monkeypatch)
    rid = hist.save_report(
        df=_frame(), xlsx_bytes=b"PK\x03\x04 fake xlsx",
        file_name="report.xlsx", model="anthropic/claude-haiku-4.5",
        quick_test=True, distinct_reasons=3, excluded_rows=1,
    )

    reports = hist.list_reports()
    assert len(reports) == 1
    row = reports.iloc[0]
    assert row["id"] == rid
    assert row["file_name"] == "report.xlsx"
    assert row["classified_rows"] == 3
    assert row["n_acceptable"] == 1
    assert row["n_not_acceptable"] == 1
    assert row["n_potentially"] == 1
    assert row["quick_test"] == 1
    assert row["excluded_rows"] == 1

    loaded = hist.load_df(rid)
    assert list(loaded["Classification"]) == [ACCEPTABLE, NOT_ACCEPTABLE, POTENTIALLY_ACCEPTABLE]
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
