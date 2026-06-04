"""Trip reason classifier.

Loads rules from rules.yaml and classifies free-text trip reasons into:
  - Acceptable
  - Potentially Acceptable
  - Not Acceptable
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ACCEPTABLE = "Acceptable"
POTENTIALLY_ACCEPTABLE = "Potentially Acceptable"
NOT_ACCEPTABLE = "Not Acceptable"


@dataclass
class ClassificationResult:
    category: str
    matched_term: str | None
    rationale: str


class TripReasonClassifier:
    def __init__(self, rules_path: str | Path = "rules.yaml"):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        self.variance_tolerance_pct: float = float(
            rules.get("variance_tolerance_pct", 0)
        )
        self._empty_terms = {
            (t or "").strip().lower() for t in rules.get("empty_or_generic", [])
        }
        self._not_acceptable = self._compile(
            rules.get("not_acceptable", {}).get("keywords", [])
        )
        self._acceptable = self._compile(
            rules.get("acceptable", {}).get("keywords", [])
        )
        self._potentially = self._compile(
            rules.get("potentially_acceptable", {}).get("keywords", [])
        )

    @staticmethod
    def _compile(terms: Iterable[str]) -> list[tuple[str, re.Pattern]]:
        compiled = []
        for term in terms:
            term_clean = term.strip()
            if not term_clean:
                continue
            # Word-boundary match for single tokens; substring for phrases with spaces/hyphens.
            if re.fullmatch(r"[A-Za-z']+", term_clean):
                pattern = re.compile(rf"\b{re.escape(term_clean)}\b", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(term_clean), re.IGNORECASE)
            compiled.append((term_clean, pattern))
        return compiled

    def _match(
        self, text: str, patterns: list[tuple[str, re.Pattern]]
    ) -> str | None:
        for term, pat in patterns:
            if pat.search(text):
                return term
        return None

    def classify(
        self,
        reason: str | None,
        claimed_miles: float | None = None,
        expected_miles: float | None = None,
    ) -> ClassificationResult:
        # 1. Within tolerance -> Acceptable automatically.
        variance_pct = _variance_pct(claimed_miles, expected_miles)
        if variance_pct is not None and variance_pct <= self.variance_tolerance_pct:
            return ClassificationResult(
                category=ACCEPTABLE,
                matched_term=None,
                rationale=(
                    f"Variance {variance_pct:.1f}% within tolerance "
                    f"({self.variance_tolerance_pct:.0f}%)"
                ),
            )

        if reason is None or (isinstance(reason, float) and reason != reason):
            reason = ""
        normalised = str(reason).strip().lower()

        # 2. Empty or generic -> Not Acceptable.
        if normalised in self._empty_terms:
            return ClassificationResult(
                category=NOT_ACCEPTABLE,
                matched_term=normalised or "(blank)",
                rationale="No meaningful reason provided",
            )

        # 3. Priority match: Not Acceptable > Acceptable > Potentially Acceptable.
        if (m := self._match(normalised, self._not_acceptable)):
            return ClassificationResult(
                NOT_ACCEPTABLE, m, f"Matched not-acceptable term: '{m}'"
            )
        if (m := self._match(normalised, self._acceptable)):
            return ClassificationResult(
                ACCEPTABLE, m, f"Matched acceptable term: '{m}'"
            )
        if (m := self._match(normalised, self._potentially)):
            return ClassificationResult(
                POTENTIALLY_ACCEPTABLE,
                m,
                f"Matched potentially-acceptable term: '{m}'",
            )

        # 4. No match -> default to Potentially Acceptable (flag for review).
        return ClassificationResult(
            POTENTIALLY_ACCEPTABLE,
            None,
            "No keyword match — manual review required",
        )


def _variance_pct(claimed: float | None, expected: float | None) -> float | None:
    if claimed is None or expected is None:
        return None
    try:
        claimed_f = float(claimed)
        expected_f = float(expected)
    except (TypeError, ValueError):
        return None
    if expected_f <= 0:
        return None
    return ((claimed_f - expected_f) / expected_f) * 100.0
