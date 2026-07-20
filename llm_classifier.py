"""LLM-based trip reason classifier (OpenRouter / OpenAI-compatible API).

Classifies free-text trip reasons — including messy abbreviations, route
descriptions, and non-English text (French / German / Italian / Bulgarian) —
into Acceptable / Acceptable - Driver Guidance / Manual Review Required /
Not Acceptable, with a short rationale. Two assessments drive the call: does
the reason explain the variance, and does it reveal the journey was logged
incorrectly (an unlogged extra stop or return leg → Driver Guidance).

Cost-efficient design:
  - The HMRC rubric is the (stable) system prompt.
  - Reasons are PACKED ~40 per request so the rubric is sent once per chunk
    instead of once per reason — the rubric, not the reason, dominates token
    cost, so this is ~40-50% cheaper than one-reason-per-call.
  - Chunks run concurrently across a small thread pool.
  - The classifier judges the reason text in isolation, so identical reason
    strings are de-duplicated by the caller before they reach the API.

Runs against OpenRouter (an OpenAI-compatible endpoint). Set OPENROUTER_API_KEY.
Model slugs are OpenRouter's, e.g. "anthropic/claude-haiku-4.5" (default),
"anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.8". Any OpenRouter
chat model works — pass it via --model.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import OpenAI

ACCEPTABLE = "Acceptable"
DRIVER_GUIDANCE = "Acceptable - Driver Guidance"
MANUAL_REVIEW = "Manual Review Required"
NOT_ACCEPTABLE = "Not Acceptable"
# Legacy name (pre-July-2026 taxonomy); old saved reports may still contain it.
POTENTIALLY_ACCEPTABLE = "Potentially Acceptable"
_VALID = {ACCEPTABLE, DRIVER_GUIDANCE, MANUAL_REVIEW, NOT_ACCEPTABLE}

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_BATCH_SIZE = 40
DEFAULT_WORKERS = 8
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- The rubric (system prompt) --------------------------------------------
RUBRIC = """\
You are an HMRC compliance assistant for a UK fleet/mileage team (TMC).

Drivers claim business mileage. A system (Google/Waze-based) calculates an
expected distance for each trip. When the claimed distance materially exceeds
the calculated distance, the driver enters a free-text reason explaining the
excess. Make TWO separate assessments of each reason:

  Assessment 1 — Does the reason explain the mileage variance?
  Assessment 2 — Does the reason reveal that the journey was LOGGED
                 INCORRECTLY (an additional destination was visited but not
                 recorded as its own trip leg)?

A reason can explain the variance perfectly well AND still reveal a logging
error — that combination is exactly what the review team needs surfaced.

Classify into exactly one of four categories:

ACCEPTABLE — variance explained, no logging issue. Route choice and honest
measurement are FINE — the system calculates at a different time of day, so a
different route or a tracked/odometer distance legitimately differs:
  - Route/navigation: "Google Maps", "sat nav", "followed satnav all day",
    "best/fastest/quickest route", "avoided tolls", "HOME VIA M25",
    "A3 is a quicker route", named roads/motorways, "motorway not A-roads".
  - Measurement: "auto track", "as per tracker/telematics/odometer/tripmeter",
    "app tracking", "that's what the car said", tracker/app left running or
    not stopped, "checked on Google and it's right".
  - Conditions: traffic, congestion, road closure, accident, diversion,
    roadworks, weather — where no extra destination is mentioned.
  - Operational micro-stops that are not a real destination: fuel stop,
    EV charge, toilet/comfort break, quick drink, parking further away.
  - Couldn't-log situations: aborted/cancelled visit (customer left, no
    access), called away mid-route — no waypoint existed to log.
Do NOT mark these down for being brief or vague ("Google maps", "Auto track",
"Best route" alone are all Acceptable). Brevity is not a defect.

ACCEPTABLE - DRIVER GUIDANCE — variance explained, BUT the wording reveals an
additional business location or journey leg that was not logged separately.
The driver should be advised to log each leg (Office → Colleague → Customer,
not Office → Customer). Trigger on any additional-destination language:
  - Picked up / dropped off a colleague, staff, escort, passenger
    ("Dropped and picked up colleague Jason", "Colleague collection:
    Bristol Airport", "picked up Simon from Andover").
  - Collected / delivered / dropped parts, tools, equipment, stock, keys,
    materials ("Pick up parts", "diversion to Screwfix for parts",
    "stock drop off", "via B&Q for work equipment").
  - Called into / went via another site, office, depot, branch, warehouse,
    customer, supplier, hotel, airport, station, storage unit.
  - Return journey folded into one logged trip: "12.2 miles there and 12.2
    back", "inc return journey", "round trip", "7 miles then returned = 14"
    — the return leg must be logged as its own trip.
  - Multiple postcodes/segments listed in the reason instead of logged as
    separate trips ("ran out of space on my grid" for extra postcodes).

MANUAL REVIEW REQUIRED — a human must look before any driver contact:
  - Garage road test, vehicle testing/repair mileage, demonstration drives —
    the customer contact must decide whether it counts as business.
  - Missing, unreadable, or genuinely ambiguous explanations that could be
    acceptable or not depending on account details.
  - Obscenities or abuse in the reason (needs escalation to the contact).
  - Employment-terms / contractual disputes (TUPE, "paid full mileage per my
    terms", missing-postcode system complaints) — payroll/HR territory.

NOT ACCEPTABLE — the reason itself shows non-business mileage or says nothing:
  - Personal trips: shopping, gym, pub, social, holiday, school run,
    dropping/collecting family, personal appointments, home for lunch.
  - Ordinary commuting (home to work and back) claimed as business.
  - Blank, "n/a", "none", "?", "don't know", or otherwise meaningless.

Guidance:
  - Judge the EXPLANATION, not the journey type or how polished the text is.
  - The reason may be in any language. Translate and judge it on its meaning.
  - Ignore artifacts like "_x000D_" (stray carriage returns) and typos
    ("fastist route", "follow sst bav" = followed sat nav).
  - The deciding question between ACCEPTABLE and DRIVER GUIDANCE: did the
    driver mention going SOMEWHERE ELSE that isn't logged? Different route or
    measurement source → Acceptable. Extra place, person, pickup, delivery,
    or an unlogged return leg → Driver Guidance.
  - You are an aid to a human reviewer (Amy), not the final decision. Each
    rationale must be ONE short sentence a reviewer can act on quickly; for
    Driver Guidance say what should have been logged.

You will be given a numbered list of trip reasons. Classify every one.
Respond with a single JSON object and nothing else, of the form:
{"results": [{"n": 1, "category": "Acceptable", "rationale": "..."}, ...]}
where "n" matches the reason's number, "category" is exactly one of
"Acceptable", "Acceptable - Driver Guidance", "Manual Review Required", or
"Not Acceptable", and there is one entry for every reason in the list."""


@dataclass
class LLMResult:
    category: str
    rationale: str


def make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY (an sk-or-v1-... key from openrouter.ai/keys)."
        )
    return OpenAI(
        api_key=key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        default_headers={"X-Title": "TMC Trip Reason Variance"},
        max_retries=4,
    )


def _parse_json(content: str) -> dict:
    """Parse model output, tolerating markdown ```json fences or surrounding prose."""
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        return json.loads(content[start : end + 1])
    raise ValueError("no JSON object found")


def _normalise(reason: str) -> str:
    reason = re.sub(r"_x000D_|\r", " ", str(reason or "")).strip()
    return reason


def _numbered(reasons: list[str]) -> str:
    lines = []
    for i, r in enumerate(reasons, 1):
        r = _normalise(r)
        lines.append(f"{i}. {r if r else '(blank — no reason entered)'}")
    return "\n".join(lines)


def _classify_chunk(
    client: OpenAI, reasons: list[str], model: str
) -> tuple[list[LLMResult], str | None]:
    """Classify one packed chunk.

    Returns (results aligned to `reasons`, error message or None). On an API or
    parse error the results are the fallback and the error string is returned so
    the caller can surface it instead of silently substituting fallbacks.
    """
    fallback = [
        LLMResult(MANUAL_REVIEW, "Classification failed — manual review required")
        for _ in reasons
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=100 * len(reasons) + 300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": _numbered(reasons)},
            ],
        )
        data = _parse_json(resp.choices[0].message.content)
    except Exception as exc:
        return fallback, f"{type(exc).__name__}: {exc}"

    by_n: dict[int, LLMResult] = {}
    for item in data.get("results", []):
        try:
            n = int(item["n"])
            cat = item["category"].strip()
            if cat == POTENTIALLY_ACCEPTABLE:  # model slipped into the old taxonomy
                cat = MANUAL_REVIEW
            if cat not in _VALID:
                cat = MANUAL_REVIEW
            by_n[n] = LLMResult(cat, str(item.get("rationale", "")).strip())
        except (KeyError, ValueError, TypeError, AttributeError):
            continue

    return [by_n.get(i, fallback[i - 1]) for i in range(1, len(reasons) + 1)], None


def classify(
    client: OpenAI,
    reasons: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_WORKERS,
    progress=print,
    on_progress=None,
) -> list[LLMResult]:
    """Classify many distinct reasons. Packs `batch_size` per request and runs
    chunks concurrently. Returns results aligned to the input order.

    `on_progress(done_chunks, total_chunks)`, if given, is called after each
    chunk completes — used to drive a UI progress bar.
    """
    chunks = [reasons[i : i + batch_size] for i in range(0, len(reasons), batch_size)]
    progress(
        f"Classifying {len(reasons)} distinct reasons in {len(chunks)} chunks "
        f"of <= {batch_size} ({model}, {max_workers} workers)..."
    )
    results: list[LLMResult] = []
    done = 0
    failed = 0
    first_error: str | None = None
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for chunk_result, error in pool.map(
            lambda c: _classify_chunk(client, c, model), chunks
        ):
            results.extend(chunk_result)
            if error is not None:
                failed += 1
                if first_error is None:
                    first_error = error
            done += 1
            if on_progress is not None:
                on_progress(done, len(chunks))
            if done % 10 == 0 or done == len(chunks):
                progress(f"  {done}/{len(chunks)} chunks ({time.monotonic()-start:.0f}s)")

    # If every batch failed, the run is misconfigured (bad key, no credits,
    # unknown model, …) — surface the real error instead of returning a sheet
    # full of identical "Classification failed" fallbacks.
    if failed == len(chunks) and first_error is not None:
        raise RuntimeError(
            f"All {len(chunks)} batches failed calling the model "
            f"({model}). First error — {first_error}"
        )
    if failed:
        progress(f"  WARNING: {failed}/{len(chunks)} batches failed — {first_error}")
    return results
