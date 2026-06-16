"""LLM-based trip reason classifier (OpenRouter / OpenAI-compatible API).

Classifies free-text trip reasons — including messy abbreviations, route
descriptions, and non-English text (French / German / Italian / Bulgarian) —
into Acceptable / Potentially Acceptable / Not Acceptable, with a short
rationale.

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
POTENTIALLY_ACCEPTABLE = "Potentially Acceptable"
NOT_ACCEPTABLE = "Not Acceptable"
_VALID = {ACCEPTABLE, POTENTIALLY_ACCEPTABLE, NOT_ACCEPTABLE}

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
excess. Your job is to classify how acceptable that *reason* is as a
justification for the extra mileage, from an HMRC business-mileage standpoint.

Classify into exactly one of three categories:

ACCEPTABLE — A clear, specific, business-legitimate explanation for the excess
that aligns with HMRC rules. For example:
  - Diversions: roadworks, road/bridge/motorway closure, accident, police
    diversion, flooding, traffic congestion forcing a longer route.
  - Multiple business stops: multi-drop, several deliveries/collections,
    visiting multiple sites/customers on one trip.
  - Return journey explicitly accounted for (system calculated one-way only),
    where the maths is stated and consistent.
  - Genuine business detours: customer/client/site/depot visits where the
    longer route is explained, vehicle breakdown, emergency call-out.

POTENTIALLY ACCEPTABLE — Plausibly valid but lacking detail, or requiring
human/HR verification before approval. For example:
  - Vague route justifications: "used main roads", "quickest route not the
    shortest", "followed satnav/GPS", "took the A23", "longer route".
  - Bare assertions that don't explain the excess: "as per vehicle odometer",
    "this is the distance shown on Google", figures restated without cause.
  - Employment-terms / contractual claims: TUPE transfers, "paid full mileage
    per my terms", missing-postcode or missing-trip complaints — real issues
    that need payroll/HR to verify, not a driving-distance reason.
  - Generic business words with no detail: "meeting", "business", "work",
    "visit", "appointment", "fieldwork".

NOT ACCEPTABLE — Clearly invalid, personal, or no meaningful reason:
  - Personal trips: shopping, gym, pub, social, holiday, school run, dropping
    or collecting family, personal appointments, going home for lunch.
  - Ordinary commuting (home to work and back) — not allowable business
    mileage under HMRC rules, even if phrased neutrally.
  - Blank, "n/a", "none", "?", "don't know", or otherwise empty/meaningless.

Guidance:
  - The reason may be in any language. Translate and judge it on its meaning.
  - Ignore artifacts like "_x000D_" (stray carriage returns) and typos.
  - Be conservative: if a reason could be legitimate but you cannot confirm a
    concrete business cause for the extra distance, use POTENTIALLY ACCEPTABLE
    rather than ACCEPTABLE.
  - You are an aid to a human reviewer (Amy), not the final decision. Each
    rationale must be ONE short sentence a reviewer can act on quickly.

You will be given a numbered list of trip reasons. Classify every one.
Respond with a single JSON object and nothing else, of the form:
{"results": [{"n": 1, "category": "Acceptable", "rationale": "..."}, ...]}
where "n" matches the reason's number, "category" is exactly one of
"Acceptable", "Potentially Acceptable", or "Not Acceptable", and there is one
entry for every reason in the list."""


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


def _classify_chunk(client: OpenAI, reasons: list[str], model: str) -> list[LLMResult]:
    """Classify one packed chunk. Returns results aligned to `reasons`."""
    fallback = [
        LLMResult(POTENTIALLY_ACCEPTABLE, "Classification failed — manual review required")
        for _ in reasons
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=80 * len(reasons) + 200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": _numbered(reasons)},
            ],
        )
        data = _parse_json(resp.choices[0].message.content)
    except Exception:
        return fallback

    by_n: dict[int, LLMResult] = {}
    for item in data.get("results", []):
        try:
            n = int(item["n"])
            cat = item["category"].strip()
            if cat not in _VALID:
                cat = POTENTIALLY_ACCEPTABLE
            by_n[n] = LLMResult(cat, str(item.get("rationale", "")).strip())
        except (KeyError, ValueError, TypeError, AttributeError):
            continue

    return [by_n.get(i, fallback[i - 1]) for i in range(1, len(reasons) + 1)]


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
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for chunk_result in pool.map(
            lambda c: _classify_chunk(client, c, model), chunks
        ):
            results.extend(chunk_result)
            done += 1
            if on_progress is not None:
                on_progress(done, len(chunks))
            if done % 10 == 0 or done == len(chunks):
                progress(f"  {done}/{len(chunks)} chunks ({time.monotonic()-start:.0f}s)")
    return results
