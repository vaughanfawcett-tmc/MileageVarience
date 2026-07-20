"""Tests for the variance-percentage layer and workbook output cleaning."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import classify_report as cr
from llm_classifier import ACCEPTABLE, DRIVER_GUIDANCE, MANUAL_REVIEW, NOT_ACCEPTABLE


def _frame():
    return pd.DataFrame(
        {
            "vcReason": [
                "google maps",
                "Inc return journey",
                "personal trip",
                "sat nav",
            ],
            "BusinessMileage": [66, 88.8, 30, 40],
            "SystemCalculatedMileage": [60, 34.08, 20, 0],
            "Classification": [ACCEPTABLE, DRIVER_GUIDANCE, NOT_ACCEPTABLE, ACCEPTABLE],
            "Rationale": ["route choice", "unlogged return leg", "personal", "route choice"],
        }
    )


def test_add_variance_pct():
    df = _frame()
    assert cr.add_variance_pct(df) == "Variance %"
    assert df["Variance %"].iloc[0] == 10.0
    assert df["Variance %"].iloc[1] == 160.6
    # Zero/invalid system distance must not divide: NaN, not inf.
    assert pd.isna(df["Variance %"].iloc[3])


def test_high_variance_bumps_acceptable_to_manual_review():
    df = _frame()
    cr.add_variance_pct(df)
    bumped = cr.apply_variance_review(df)
    assert bumped == 1
    # Only the 160% row moves; its original rationale is preserved in brackets.
    assert df["Classification"].tolist() == [
        ACCEPTABLE, MANUAL_REVIEW, NOT_ACCEPTABLE, ACCEPTABLE,
    ]
    assert "161%" in df["Rationale"].iloc[1]
    assert "unlogged return leg" in df["Rationale"].iloc[1]


def test_not_acceptable_rows_are_never_bumped():
    df = _frame()
    df.loc[1, "Classification"] = NOT_ACCEPTABLE
    cr.add_variance_pct(df)
    assert cr.apply_variance_review(df) == 0


def test_workbook_cleans_reason_and_adds_high_variance_sheet(tmp_path):
    df = _frame()
    df.loc[0, "vcReason"] = "2 mile from BH25NW_x000D_\n38.6 Mile from BH11EN"
    cr.add_variance_pct(df)
    out = tmp_path / "out.xlsx"
    cr.write_workbook(df, out)

    xl = pd.ExcelFile(out)
    assert xl.sheet_names == ["High Variance 50%+", "Classified"]
    main = xl.parse("Classified")
    assert "_x000D_" not in main["vcReason"].iloc[0]
    high = xl.parse("High Variance 50%+")
    assert len(high) == 1
    assert high["vcReason"].iloc[0] == "Inc return journey"
