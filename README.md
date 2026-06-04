# TMC Trip Reason Variance Automation

Classifies free-text trip reasons from Milcap (and the standard Trip Purpose report) into
**Acceptable / Potentially Acceptable / Not Acceptable** to reduce manual review effort
and improve HMRC-audit consistency.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Classify a Milcap report
python main.py path/to/milcap_report.xlsx

# Specify output and custom column names if auto-detection fails
python main.py input.xlsx --output reviewed.xlsx \
    --reason-col "Driver Reason" \
    --claimed-col "Actual Miles" \
    --expected-col "Google Miles"
```

The script auto-detects common column names (Trip Reason / Reason / Driver Reason /
Comments / Trip Purpose; Claimed Miles / Actual Miles; Expected Miles / Google Miles).

### Output columns added

| Column | Meaning |
|---|---|
| Classification | Acceptable / Potentially Acceptable / Not Acceptable |
| Variance % | (Claimed − Expected) / Expected × 100 |
| Match Term | Which keyword/phrase drove the classification |
| Rationale | Human-readable reason for the classification |

Non-Acceptable rows are highlighted red, Potentially Acceptable amber, Acceptable green.
Header is frozen with a filter applied so Amy can sort/filter quickly.

## LLM classifier (for the real Milcap export)

The rules engine above works on clean, English keyword data. The real Mileage
Variance export has messy, abbreviated, and multilingual free text that
keyword rules barely match, so reason classification uses an LLM (via
OpenRouter) instead.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...   # from openrouter.ai/keys

# 1. Estimate scale/cost first (no key needed):
python classify_report.py "Mileage Variance Last 30 Days v2.xlsx" --estimate

# 2. Validate accuracy on a cheap sample, eyeballing 30 classified rows:
python classify_report.py "Mileage Variance Last 30 Days v2.xlsx" --limit 200 --spot-check 30

# 3. Full run (Haiku is the default — cheapest, usually sufficient):
python classify_report.py "Mileage Variance Last 30 Days v2.xlsx"

# Higher-quality model once you've reviewed sample accuracy:
python classify_report.py "Mileage Variance Last 30 Days v2.xlsx" \
    --model anthropic/claude-sonnet-4.6
```

`classify_report.py` reads the real schema (`vcReason`, `BusinessMileage`,
`SystemCalculatedMileage`, the two tax-year sheets), de-duplicates reason
strings, packs ~40 reasons per request (amortising the rubric — see
`--batch-size`/`--workers`), writes an enriched colour-coded workbook, and
compares the 3-way classification against the existing `Column1` keep/remove
decision. The HMRC rubric lives in `llm_classifier.py`. Model slugs are
OpenRouter's (`anthropic/claude-haiku-4.5`, `anthropic/claude-sonnet-4.6`,
`anthropic/claude-opus-4.8`, or any OpenRouter chat model).

## Web dashboard

For interactive review, run the Streamlit dashboard: upload a report, watch a
progress bar while it classifies, then explore a category summary, a
by-company breakdown of Not-Acceptable trips (volume and rate), a filterable
flagged-trips table, and a one-click download of the enriched workbook.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...   # read server-side, never shown in the browser
streamlit run app.py
```

It opens at http://localhost:8501. Pick the model in the sidebar, or tick
"Quick test" to classify a ~300-row sample first. Results are cached in the
session, so filtering the table doesn't re-run (or re-bill) the classification.

## Classification logic

Precedence (highest wins):

1. **Variance within tolerance** (default 10%) → `Acceptable` automatically
2. **Empty / generic reason** (blank, "n/a", "none", "?") → `Not Acceptable`
3. **Not-acceptable keyword match** (personal, school run, gym, etc.) → `Not Acceptable`
4. **Acceptable keyword match** (diversion, roadworks, multi-drop, etc.) → `Acceptable`
5. **Potentially-acceptable keyword match** (meeting, visit, detour) → `Potentially Acceptable`
6. **No match** → `Potentially Acceptable` (flagged for manual review)

Rules live in `rules.yaml` and can be edited by Employee Services without touching code.

## Tests

```bash
pip install pytest
pytest tests/
```

## Try it with sample data

```bash
python generate_sample.py             # creates sample_milcap_report.xlsx
python main.py sample_milcap_report.xlsx
open sample_milcap_report_classified.xlsx
```

## Files

- `main.py` — CLI entry point, Excel I/O, colour formatting
- `classifier.py` — rules-based classifier
- `rules.yaml` — keywords + variance tolerance (editable by business users)
- `generate_sample.py` — produces a representative test workbook
- `tests/test_classifier.py` — unit tests
