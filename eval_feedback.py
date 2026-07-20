"""Replay Amy's misclassification feedback against the current rubric.

Amy's feedback workbook ("MV Classification Feedback.xlsx") has two sheets of
rows the old rubric got wrong:

  - "Incorrectly Classified - Accept": classified Acceptable, but the reason
    reveals an unlogged stop / return leg / special case. The fix is right when
    the new rubric returns "Acceptable - Driver Guidance" or
    "Manual Review Required".
  - "Incorrectly Classified - Not Ac": classified Not Acceptable, but they are
    route/measurement explanations Amy would wave through. The fix is right
    when the new rubric returns anything except "Not Acceptable".

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python eval_feedback.py "~/Downloads/MV Classification Feedback.xlsx"
    python eval_feedback.py feedback.xlsx --model anthropic/claude-sonnet-4.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import llm_classifier as llm
from classify_report import find_column, normalise_reason, REASON_CANDIDATES
from llm_classifier import ACCEPTABLE, DRIVER_GUIDANCE, MANUAL_REVIEW, NOT_ACCEPTABLE

SHEETS = {
    # sheet-name prefix -> set of NEW categories that count as fixed
    "Incorrectly Classified - Accept": {DRIVER_GUIDANCE, MANUAL_REVIEW},
    "Incorrectly Classified - Not Ac": {ACCEPTABLE, DRIVER_GUIDANCE, MANUAL_REVIEW},
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("feedback", type=Path)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    args = p.parse_args()

    xl = pd.ExcelFile(args.feedback.expanduser())
    client = llm.make_client()

    for sheet, good in SHEETS.items():
        if sheet not in xl.sheet_names:
            print(f"(sheet {sheet!r} not found — skipping)")
            continue
        df = xl.parse(sheet)
        reason_col = find_column(df.columns, REASON_CANDIDATES)
        reasons = sorted(
            {normalise_reason(r) for r in df[reason_col] if normalise_reason(r)}
        )
        print(f"\n=== {sheet}: {len(reasons)} distinct reasons ===")
        results = llm.classify(client, reasons, model=args.model)

        fixed, still_wrong = 0, []
        for reason, res in zip(reasons, results):
            if res.category in good:
                fixed += 1
            else:
                still_wrong.append((reason, res))
        print(f"Fixed: {fixed}/{len(reasons)} ({fixed / len(reasons) * 100:.0f}%)")
        if still_wrong:
            print("Still wrong:")
            for reason, res in still_wrong:
                print(f"  [{res.category}] {reason[:80]!r} — {res.rationale}")


if __name__ == "__main__":
    main()
