"""Unit tests for TripReasonClassifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier import (
    ACCEPTABLE,
    NOT_ACCEPTABLE,
    POTENTIALLY_ACCEPTABLE,
    TripReasonClassifier,
)


RULES = Path(__file__).resolve().parent.parent / "rules.yaml"


def make():
    return TripReasonClassifier(RULES)


def test_within_tolerance_is_acceptable_regardless_of_reason():
    c = make()
    # 2% variance -> within tolerance, classified Acceptable even with bad text.
    r = c.classify("personal trip", claimed_miles=51, expected_miles=50)
    assert r.category == ACCEPTABLE
    assert "tolerance" in r.rationale.lower()


def test_blank_reason_when_variance_exceeds_tolerance():
    c = make()
    r = c.classify("", claimed_miles=60, expected_miles=40)
    assert r.category == NOT_ACCEPTABLE


def test_generic_reason_is_not_acceptable():
    c = make()
    r = c.classify("n/a", claimed_miles=60, expected_miles=40)
    assert r.category == NOT_ACCEPTABLE


def test_personal_keyword_is_not_acceptable():
    c = make()
    r = c.classify("personal errand", claimed_miles=60, expected_miles=40)
    assert r.category == NOT_ACCEPTABLE
    assert r.matched_term == "personal"


def test_roadworks_is_acceptable():
    c = make()
    r = c.classify("Roadworks on M62 diversion", claimed_miles=60, expected_miles=40)
    assert r.category == ACCEPTABLE


def test_multiple_stops_is_acceptable():
    c = make()
    r = c.classify("Multiple stops at client sites", claimed_miles=60, expected_miles=40)
    assert r.category == ACCEPTABLE


def test_vague_meeting_is_potentially_acceptable():
    c = make()
    r = c.classify("Meeting", claimed_miles=60, expected_miles=40)
    assert r.category == POTENTIALLY_ACCEPTABLE


def test_unknown_reason_defaults_to_potentially_acceptable():
    c = make()
    r = c.classify("Asdf qwerty something", claimed_miles=60, expected_miles=40)
    assert r.category == POTENTIALLY_ACCEPTABLE
    assert r.matched_term is None


def test_not_acceptable_takes_priority_over_acceptable():
    c = make()
    # Contains "delivery" (acceptable) but also "personal" (not acceptable) — NA wins.
    r = c.classify(
        "personal delivery to family", claimed_miles=60, expected_miles=40
    )
    assert r.category == NOT_ACCEPTABLE


def test_case_insensitive():
    c = make()
    r = c.classify("DIVERSION DUE TO ROAD CLOSED", claimed_miles=60, expected_miles=40)
    assert r.category == ACCEPTABLE


def test_school_run_phrase_matches():
    c = make()
    r = c.classify("Had to do the school run", claimed_miles=30, expected_miles=20)
    assert r.category == NOT_ACCEPTABLE
