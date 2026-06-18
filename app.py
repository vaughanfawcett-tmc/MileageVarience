"""Web dashboard for the TMC Trip Reason Variance classifier.

Upload a Mileage Variance / Milcap export, classify the free-text reasons with
the LLM, then explore the results: category summary, by-company breakdowns, a
filterable flagged-trips table, and a download of the enriched workbook.

Run it:
    export OPENROUTER_API_KEY=sk-or-v1-...
    streamlit run app.py

The API key is read from the environment server-side — it is never shown in
or entered through the browser.
"""

from __future__ import annotations

import io
import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

import classify_report as cr
import llm_classifier as llm
from llm_classifier import ACCEPTABLE, NOT_ACCEPTABLE, POTENTIALLY_ACCEPTABLE

CATEGORIES = [ACCEPTABLE, POTENTIALLY_ACCEPTABLE, NOT_ACCEPTABLE]
COLOR = {ACCEPTABLE: "#16A34A", POTENTIALLY_ACCEPTABLE: "#D97706", NOT_ACCEPTABLE: "#DC2626"}
INK, MUTED, LINE, ACCENT = "#111827", "#6B7280", "#E5E7EB", "#2563EB"

DISPLAY_COLS = [
    "Parent Name", "FirstName", "LastName", "vcReason", "Classification",
    "Rationale", "BusinessMileage", "SystemCalculatedMileage", "% Difference",
    "Column1",
]
COLUMN_CONFIG = {
    "Parent Name": st.column_config.TextColumn("Company"),
    "FirstName": st.column_config.TextColumn("First"),
    "LastName": st.column_config.TextColumn("Last"),
    "vcReason": st.column_config.TextColumn("Reason", width="large"),
    "Classification": st.column_config.TextColumn("Class", width="small"),
    "Rationale": st.column_config.TextColumn("Rationale", width="large"),
    "BusinessMileage": st.column_config.NumberColumn("Claimed", format="%.0f"),
    "SystemCalculatedMileage": st.column_config.NumberColumn("System", format="%.0f"),
    "% Difference": st.column_config.NumberColumn("Variance", format="%.2f"),
    "Column1": st.column_config.TextColumn("Existing", width="small"),
}

st.set_page_config(page_title="Trip Reason Variance", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"], button, input, textarea, select { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif; }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
      header[data-testid="stHeader"] { background: transparent; height: 0; }
      .block-container { padding: 2.2rem 3rem 3rem; max-width: 1240px; }
      h1, h2, h3 { color: #111827; letter-spacing: -0.01em; }
      h2 { font-size: 1.15rem !important; font-weight: 600 !important; margin: 1.6rem 0 0.6rem; }
      h3 { font-size: 1rem !important; font-weight: 600 !important; }
      .app-title { font-size: 1.7rem; font-weight: 700; color: #111827; letter-spacing: -0.02em; margin: 0; }
      .app-sub { color: #6B7280; font-size: 0.95rem; margin: 0.25rem 0 0; }
      .rule { height: 3px; width: 44px; background: #2563EB; border-radius: 3px; margin: 0.9rem 0 1.6rem; }
      .card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
              padding: 1.1rem 1.25rem; box-shadow: 0 1px 2px rgba(16,24,40,0.04); height: 100%; }
      .card .val { font-size: 2rem; font-weight: 700; line-height: 1.1; color: #111827; }
      .card .lab { font-size: 0.82rem; font-weight: 500; color: #6B7280; margin-top: 0.25rem;
                   text-transform: uppercase; letter-spacing: 0.04em; }
      .card .pct { font-size: 0.82rem; color: #9CA3AF; margin-top: 0.15rem; }
      .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }
      .empty { border: 1.5px dashed #D1D5DB; border-radius: 16px; padding: 2.5rem; text-align: center; color: #6B7280; }
      [data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
      [data-testid="stFileUploaderDropzone"] { border-radius: 12px; }
      .stButton button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _check_password() -> bool:
    """Gate the app behind a password when APP_PASSWORD is configured.

    Reads APP_PASSWORD from the environment (a Render secret) or st.secrets.
    If neither is set (e.g. local dev), the app is open.
    """
    try:
        secret_pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        secret_pw = None
    expected = os.environ.get("APP_PASSWORD") or secret_pw
    if not expected or st.session_state.get("authed"):
        return True
    st.markdown('<p class="app-title">Trip Reason Variance</p>'
                '<p class="app-sub">Enter the access password to continue.</p>'
                '<div class="rule"></div>', unsafe_allow_html=True)
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter") :
            if pw == expected:
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

st.markdown(
    '<p class="app-title">Trip Reason Variance</p>'
    '<p class="app-sub">HMRC review assistant — classify driver mileage reasons as '
    'Acceptable, Potentially Acceptable, or Not Acceptable.</p>'
    '<div class="rule"></div>',
    unsafe_allow_html=True,
)


def metric_card(col, label, value, color, pct=None):
    sub = f'<div class="pct">{pct}</div>' if pct else ""
    col.markdown(
        f'<div class="card"><div class="val">'
        f'<span class="dot" style="background:{color}"></span>{value}</div>'
        f'<div class="lab">{label}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def _classify_df(df: pd.DataFrame, model: str, limit: int | None):
    reason_col = cr.find_column(df.columns, cr.REASON_CANDIDATES)
    if reason_col is None:
        st.error(f"No reason column found. Columns: {list(df.columns)}")
        st.stop()
    exclude_col = cr.find_column(df.columns, cr.EXCLUDE_CANDIDATES)

    df = df.copy()
    df["_reason"] = df[reason_col].map(cr.normalise_reason)
    work = df
    excluded = 0
    if exclude_col is not None:
        mask = df[exclude_col].astype(str).str.strip().str.upper() == "Y"
        excluded = int(mask.sum())
        work = df[~mask]
    if limit:
        work = work.iloc[:: max(1, len(work) // limit)].head(limit)

    distinct = sorted({r for r in work["_reason"]})
    st.caption(
        f"{len(work):,} rows · {excluded:,} excluded-company rows skipped · "
        f"{len(distinct):,} distinct reasons sent to the model"
    )

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        st.error(
            "No **OPENROUTER_API_KEY** is set on the server, so classification "
            "can't run. Add it in Render → your service → **Environment** "
            "(an `sk-or-v1-…` key from openrouter.ai/keys), save, and redeploy."
        )
        st.stop()

    n_batches = (len(distinct) + llm.DEFAULT_BATCH_SIZE - 1) // llm.DEFAULT_BATCH_SIZE
    if len(distinct) > 2000:
        st.warning(
            f"This will send **{len(distinct):,} distinct reasons** to the model "
            f"(~{n_batches:,} batches) and may take a few minutes. To try it quickly "
            "first, tick **Quick test** in the sidebar and re-run."
        )

    try:
        with st.status(
            f"Classifying {len(distinct):,} reasons in {n_batches:,} batches…",
            expanded=True,
        ) as status:
            client = llm.make_client()
            bar = st.progress(0.0, text="Starting…")
            results = llm.classify(
                client, distinct, model=model, max_workers=12,
                on_progress=lambda d, t: bar.progress(
                    d / t, text=f"{d}/{t} batches done"),
            )
            status.update(label=f"Classified {len(distinct):,} reasons.", state="complete")
    except Exception as exc:  # surface API/network errors instead of dying silently
        st.error(f"Classification failed: {exc}")
        st.stop()

    by_reason = dict(zip(distinct, results))
    df["Classification"] = df["_reason"].map(lambda r: by_reason[r].category if r in by_reason else "")
    df["Rationale"] = df["_reason"].map(lambda r: by_reason[r].rationale if r in by_reason else "")
    return df


@st.cache_data(show_spinner=False)
def _load_rows_cached(data: bytes) -> pd.DataFrame:
    """Parse an uploaded workbook once and cache it, keyed on the file bytes.

    Streamlit re-runs the whole script on every interaction; without caching,
    the (slow) openpyxl parse of a large export would repeat each time.
    """
    return cr.load_rows(io.BytesIO(data))


def _to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=True) as tmp:
        cr.write_workbook(df, tmp.name)
        with open(tmp.name, "rb") as fh:
            return fh.read()


def bar_h(data: pd.DataFrame, cat_col: str, val_col: str, color_range, sort=None):
    enc_color = alt.Color(
        f"{cat_col}:N",
        scale=alt.Scale(domain=sort, range=color_range) if isinstance(color_range, list)
        else alt.Scale(scheme="blues"),
        legend=None,
    )
    base = alt.Chart(data).encode(
        y=alt.Y(f"{cat_col}:N", sort=sort, title=None,
                axis=alt.Axis(labelFontSize=13, labelColor=INK, ticks=False)),
        x=alt.X(f"{val_col}:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=5, height=24).encode(color=enc_color)
    labels = base.mark_text(align="left", dx=7, fontSize=12, color=MUTED).encode(
        text=alt.Text(f"{val_col}:Q", format=",")
    )
    return (bars + labels).properties(height=max(120, 34 * len(data))).configure_view(
        strokeWidth=0
    ).configure_axis(domain=False)


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    model = st.selectbox(
        "Model",
        ["anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.8"],
        help="Haiku is cheapest and usually sufficient. Sonnet / Opus for tougher cases.",
    )
    quick = st.checkbox("Quick test (~300-row sample)", value=False)
    st.divider()
    st.caption("API key is read from the server environment and never shown in the browser.")

# --- Upload / classify ------------------------------------------------------
uploaded = st.file_uploader("Mileage Variance / Milcap export (.xlsx)", type="xlsx")

if uploaded is not None and st.session_state.get("file_name") != uploaded.name:
    st.session_state.pop("classified", None)
    st.session_state["file_name"] = uploaded.name

if uploaded is None and "classified" not in st.session_state:
    st.markdown(
        '<div class="empty">Upload a report to begin.<br>'
        'You will get a category summary, a by-company breakdown, a filterable '
        'flagged-trips table, and a downloadable enriched workbook.</div>',
        unsafe_allow_html=True,
    )

if uploaded is not None and "classified" not in st.session_state:
    try:
        with st.spinner(f"Reading {uploaded.name}… large exports can take a few seconds."):
            df_raw = _load_rows_cached(uploaded.getvalue())
    except Exception as exc:  # surface bad/unexpected workbooks instead of halting silently
        st.error(f"Couldn't read **{uploaded.name}**: {exc}")
        st.stop()
    st.caption(f"Loaded {len(df_raw):,} rows from {uploaded.name}")
    if st.button("Classify reasons", type="primary"):
        classified = _classify_df(df_raw, model, limit=300 if quick else None)
        with st.spinner("Building the downloadable workbook…"):
            st.session_state["xlsx"] = _to_xlsx_bytes(classified)
        st.session_state["classified"] = classified
        st.rerun()

# --- Results ----------------------------------------------------------------
def _render_results():
    df = st.session_state["classified"]
    done = df[df["Classification"] != ""]
    total = len(done)
    if total == 0:
        st.warning("No rows were classified. Try re-running, or untick Quick test.")
        return

    top = st.columns([3, 1])
    top[0].markdown("## Summary")
    xlsx = st.session_state.get("xlsx") or _to_xlsx_bytes(df)
    top[1].download_button(
        "Download workbook",
        data=xlsx,
        file_name=st.session_state.get("file_name", "report").rsplit(".", 1)[0] + "_classified.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

    cols = st.columns(3)
    for col, cat in zip(cols, CATEGORIES):
        n = int((done["Classification"] == cat).sum())
        pct = f"{n / total * 100:.0f}% of {total:,} trips" if total else ""
        metric_card(col, cat, f"{n:,}", COLOR[cat], pct)

    counts = (done["Classification"].value_counts().reindex(CATEGORIES).fillna(0)
              .rename_axis("Category").reset_index(name="Trips"))
    st.altair_chart(
        bar_h(counts, "Category", "Trips", [COLOR[c] for c in CATEGORIES], sort=CATEGORIES),
        use_container_width=True,
    )

    pc = cr.find_column(df.columns, ["Parent Name"])
    if pc:
        st.markdown("## Not-acceptable trips by company")
        na = done[done["Classification"] == NOT_ACCEPTABLE]
        tot = done.groupby(pc).size().rename("Total")
        nac = na.groupby(pc).size().rename("Not Acceptable")
        tbl = pd.concat([tot, nac], axis=1).fillna(0)
        tbl["Not Acceptable"] = tbl["Not Acceptable"].astype(int)
        tbl["% Not Acceptable"] = (tbl["Not Acceptable"] / tbl["Total"] * 100).round(1)

        t1, t2 = st.tabs(["By volume", "By rate (min 50 trips)"])
        with t1:
            vol = (tbl.sort_values("Not Acceptable", ascending=False).head(12)
                   .reset_index().rename(columns={pc: "Company"}))
            st.altair_chart(
                bar_h(vol, "Company", "Not Acceptable", "blues",
                      sort=alt.SortField("Not Acceptable", order="descending")),
                use_container_width=True,
            )
        with t2:
            rate = (tbl[tbl["Total"] >= 50].sort_values("% Not Acceptable", ascending=False)
                    .head(15).reset_index().rename(columns={pc: "Company"}))
            st.dataframe(
                rate, width="stretch", hide_index=True,
                column_config={
                    "% Not Acceptable": st.column_config.ProgressColumn(
                        "% Not Acceptable", format="%.1f%%", min_value=0, max_value=100),
                },
            )

    st.markdown("## Flagged trips")
    f1, f2 = st.columns([1, 2])
    pick = f1.multiselect("Classification", CATEGORIES, default=[NOT_ACCEPTABLE])
    companies = sorted(done[pc].dropna().astype(str).unique()) if pc else []
    chosen = f2.multiselect("Company (blank = all)", companies)
    search = st.text_input("Search reason text", placeholder="e.g. school, satnav, tupe…")

    view = done[done["Classification"].isin(pick)] if pick else done
    if pc and chosen:
        view = view[view[pc].astype(str).isin(chosen)]
    if search:
        view = view[view["_reason"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(view):,} rows")
    show = [c for c in DISPLAY_COLS if c in view.columns]
    st.dataframe(
        view[show], width="stretch", hide_index=True, height=440,
        column_config={k: v for k, v in COLUMN_CONFIG.items() if k in show},
    )


if "classified" in st.session_state:
    try:
        _render_results()
    except Exception as exc:  # never leave a blank page after a successful run
        st.error(f"Couldn't render the report: {exc}")
        st.exception(exc)
