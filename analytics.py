# analytics.py — usage instrumentation (PostHog). Shared across TMC Streamlit apps.
# Drop in next to app.py; set analytics.APP then call analytics.page_open() once.
# The default key below is PostHog's write-only client key (safe to embed; send-only).
# Override any of these via env: POSTHOG_KEY, POSTHOG_HOST, APP_ID.
from __future__ import annotations

import os
import uuid

import streamlit as st

try:
    from posthog import Posthog
except Exception:  # posthog not installed yet -> tracking silently disabled
    Posthog = None

_DEFAULT_KEY = "phc_snvmegJQwrSSGtDNjQjjopUbbfuzhb4MnZuacx8F2hBb"  # public client key
APP = os.environ.get("APP_ID", "CHANGE_ME")
HOST = os.environ.get("POSTHOG_HOST", "https://eu.i.posthog.com")


def _key() -> str | None:
    try:
        if "POSTHOG_KEY" in st.secrets:
            return st.secrets["POSTHOG_KEY"]
    except Exception:
        pass
    return os.environ.get("POSTHOG_KEY", _DEFAULT_KEY)


@st.cache_resource
def _client():
    key = _key()
    if not key or Posthog is None:
        return None
    return Posthog(key, host=HOST)


def _pid() -> str:
    if "pid" not in st.session_state:
        st.session_state.pid = str(uuid.uuid4())
    return st.session_state.pid


def track(event: str, **props) -> None:
    """posthog>=7 API: capture(event, distinct_id=, properties=)."""
    client = _client()
    if client is None:
        return
    client.capture(event, distinct_id=_pid(),
                   properties={"app": APP, "platform": "streamlit", **props})


def page_open() -> None:
    if not st.session_state.get("_opened"):
        st.session_state["_opened"] = True
        track("app_opened")
