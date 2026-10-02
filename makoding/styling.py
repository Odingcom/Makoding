"""Visual design system for DataLab Pro.

A restrained editorial interface for a serious analytical workspace.
No gradients, decorative emoji, or visual effects that compete with the data.

Besides the theme and layout primitives, this module renders the audit
objects produced by ``makoding.cleaning`` (cleaning audit, quality report,
contract validation report, run manifest). Those renderers are duck-typed:
they read attributes with safe defaults and never import ``cleaning``, so
styling stays independent of the analytical layer and keeps working if an
audit object gains or loses fields.
"""

from __future__ import annotations

import dataclasses
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from .config import APP

__all__ = [
    "INK",
    "NAVY",
    "TEAL",
    "TEAL_DARK",
    "TEAL_LIGHT",
    "PAPER",
    "WHITE",
    "MUTED",
    "BORDER",
    "inject_theme",
    "render_sidebar_brand",
    "render_hero",
    "render_section",
    "render_card",
    "render_callout",
    "render_lake_card",
    "render_footer",
    "render_audit_summary",
    "render_quality_report",
    "render_validation_report",
    "render_manifest",
]

INK = "#17212B"
NAVY = "#173B4D"
TEAL = "#176B87"
TEAL_DARK = "#0F5268"
TEAL_LIGHT = "#E8F2F5"
PAPER = "#F7F7F4"
WHITE = "#FFFFFF"
MUTED = "#68737D"
BORDER = "#D9E0E3"

_DEFAULT_HERO_TITLE = "Work with data clearly."
_DEFAULT_HERO_SUBTITLE = (
    "Prepare, explore, transform and model your data in one focused "
    "analytical workspace. The interface stays quiet so the evidence "
    "stays visible."
)


def _html(content: str) -> str:
    """Flatten HTML so Streamlit's markdown parser does not treat it as code.

    Leading indentation is stripped (4+ spaces would become a code block),
    and blank lines are dropped: in markdown a blank line ends an HTML
    block, so a stray empty line -- for example from a multi-line value
    interpolated into a template -- would split the block and print the
    remainder as literal text.
    """
    return "\n".join(
        line.lstrip() for line in content.strip("\n").splitlines() if line.strip()
    )


def _count(obj: Any, name: str) -> int:
    """Read an integer attribute defensively (missing/None -> 0)."""
    value = getattr(obj, name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    """Inject the DataLab Pro visual design system."""
    st.markdown(
        _html(
            """
            <style>
            .stApp {
                background: #F7F7F4;
                color: #17212B;
            }

            .main .block-container {
                max-width: 1500px;
                padding-top: 1.25rem;
                padding-bottom: 3rem;
            }

            h1, h2, h3, h4 {
                color: #173B4D;
                letter-spacing: -0.02em;
            }

            p, label, .stCaption {
                color: #68737D;
            }

            section[data-testid="stSidebar"] {
                background: #FFFFFF;
                border-right: 1px solid #D9E0E3;
            }

            section[data-testid="stSidebar"] * {
                color: #17212B;
            }

            section[data-testid="stSidebar"] .stTextInput input,
            section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
                background: #F7F7F4;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 8px;
                border: 1px solid #C9D4D9;
                min-height: 2.5rem;
                font-weight: 650;
                color: #173B4D;
                background: #FFFFFF;
                transition: border-color 0.15s ease, background 0.15s ease;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                border-color: #176B87;
                background: #F1F7F8;
            }

            .stButton > button[kind="primary"],
            .stButton > button[data-testid="stBaseButton-primary"] {
                background: #176B87;
                color: #FFFFFF;
                border-color: #176B87;
            }

            .stButton > button[kind="primary"]:hover,
            .stButton > button[data-testid="stBaseButton-primary"]:hover {
                background: #0F5268;
                border-color: #0F5268;
            }

            .stButton > button[kind="primary"] *,
            .stButton > button[data-testid="stBaseButton-primary"] * {
                color: #FFFFFF;
            }

            button[data-baseweb="tab"] {
                font-weight: 650;
                color: #68737D;
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                color: #176B87;
            }

            div[data-testid="stMetric"] {
                background: #FFFFFF;
                border: 1px solid #D9E0E3;
                border-radius: 10px;
                padding: 0.95rem 1rem;
                box-shadow: none;
            }

            div[data-testid="stMetricLabel"] {
                color: #68737D;
            }

            div[data-testid="stMetricValue"] {
                color: #173B4D;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid #D9E0E3;
            }

            div[data-testid="stAlert"] {
                border-radius: 8px;
            }

            .dl-card {
                background: #FFFFFF;
                border: 1px solid #D9E0E3;
                border-radius: 10px;
                padding: 1.15rem 1.25rem;
                margin-bottom: 1rem;
            }

            .dl-card-title {
                color: #173B4D;
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .dl-card-text {
                color: #68737D;
                font-size: 0.9rem;
                line-height: 1.55;
            }

            .dl-hero {
                background: #173B4D;
                border: 1px solid #173B4D;
                border-radius: 12px;
                margin: 0 0 1.4rem 0;
                padding: 2.2rem 2.4rem;
            }

            .dl-eyebrow {
                color: #9EC8D4;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.7rem;
                font-weight: 750;
                margin-bottom: 0.55rem;
            }

            .dl-hero-title {
                color: #FFFFFF;
                font-size: clamp(2rem, 4vw, 3.35rem);
                font-weight: 750;
                line-height: 1.03;
                margin: 0;
            }

            .dl-hero-title span {
                color: #9EC8D4;
            }

            .dl-hero-subtitle {
                color: #DDE9ED;
                font-size: 1rem;
                line-height: 1.55;
                margin-top: 0.9rem;
                max-width: 760px;
            }

            .dl-process {
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
                margin-top: 1.35rem;
            }

            .dl-process-item {
                padding: 0.5rem 0.7rem;
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 7px;
                color: #FFFFFF;
                font-size: 0.78rem;
                font-weight: 650;
            }

            .dl-process-number {
                color: #9EC8D4;
                margin-right: 0.35rem;
                font-variant-numeric: tabular-nums;
            }

            .dl-brand {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                padding: 0.35rem 0 1.25rem 0;
            }

            .dl-brand-mark {
                width: 38px;
                height: 38px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #173B4D;
                color: #FFFFFF;
                font-weight: 800;
                font-size: 0.95rem;
            }

            .dl-brand-name {
                color: #173B4D;
                font-size: 1.05rem;
                font-weight: 750;
            }

            .dl-brand-subtitle {
                color: #68737D;
                font-size: 0.72rem;
            }

            .dl-section {
                display: flex;
                align-items: baseline;
                gap: 0.55rem;
                margin: 1.45rem 0 0.75rem 0;
                padding-bottom: 0.45rem;
                border-bottom: 1px solid #D9E0E3;
            }

            .dl-section-icon {
                color: #176B87;
                font-size: 0.72rem;
                font-weight: 750;
                font-variant-numeric: tabular-nums;
            }

            .dl-section-title {
                color: #173B4D;
                font-size: 1.08rem;
                font-weight: 750;
            }

            .dl-lake-card {
                background: #E8F2F5;
                border-left: 3px solid #176B87;
                border-radius: 8px;
                padding: 1.15rem 1.25rem;
                color: #173B4D;
                margin: 1rem 0;
            }

            .dl-lake-title {
                font-size: 1rem;
                font-weight: 750;
                margin-bottom: 0.3rem;
            }

            .dl-lake-text {
                color: #68737D;
                font-size: 0.86rem;
                line-height: 1.55;
                max-width: 760px;
            }

            .dl-footer {
                margin-top: 3rem;
                padding: 1rem 0 0.5rem 0;
                border-top: 1px solid #D9E0E3;
                color: #68737D;
                font-size: 0.72rem;
                text-align: center;
            }

            .dl-download-card {
                background: #FFFFFF;
                border: 1px solid #D9E0E3;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.75rem 0;
            }

            .dl-download-title {
                color: #173B4D;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }

            .dl-download-text {
                color: #68737D;
                font-size: 0.82rem;
                line-height: 1.45;
                margin-bottom: 0.7rem;
            }

            @media (max-width: 800px) {
                .dl-hero {
                    padding: 1.7rem 1.4rem;
                }

                .dl-hero-title {
                    font-size: 2.2rem;
                }
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------

def render_sidebar_brand() -> None:
    """Render restrained Makoding DataLab Pro branding."""
    st.markdown(
        _html(
            """
            <div class="dl-brand">
                <div class="dl-brand-mark">M</div>
                <div>
                    <div class="dl-brand-name">Makoding</div>
                    <div class="dl-brand-subtitle">DataLab Pro</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _accent_last_word(title: str) -> str:
    """Escape ``title`` and wrap its final word in the accent span."""
    words = str(title).split()
    if not words:
        return ""
    head = escape(" ".join(words[:-1]))
    tail = escape(words[-1])
    return f"{head} <span>{tail}</span>" if head else f"<span>{tail}</span>"


def render_hero(
    title: str | None = None,
    subtitle: str | None = None,
    *,
    eyebrow: str = "Makoding DataLab Pro",
    show_process: bool = True,
) -> None:
    """Render the main product introduction.

    Called with no arguments it renders the original default hero. When a
    ``title`` is given, its last word is set in the accent colour. All text
    is HTML-escaped.
    """
    title_html = _accent_last_word(title if title else _DEFAULT_HERO_TITLE)
    subtitle_html = escape(str(subtitle if subtitle else _DEFAULT_HERO_SUBTITLE))
    eyebrow_html = escape(str(eyebrow))

    process_html = ""
    if show_process:
        process_html = _html(
            """
            <div class="dl-process">
                <div class="dl-process-item"><span class="dl-process-number">01</span>Clean</div>
                <div class="dl-process-item"><span class="dl-process-number">02</span>Explore</div>
                <div class="dl-process-item"><span class="dl-process-number">03</span>Engineer</div>
                <div class="dl-process-item"><span class="dl-process-number">04</span>Model</div>
            </div>
            """
        )

    st.markdown(
        _html(
            f"""
            <div class="dl-hero">
                <div class="dl-eyebrow">{eyebrow_html}</div>
                <div class="dl-hero-title">{title_html}</div>
                <div class="dl-hero-subtitle">{subtitle_html}</div>
                {process_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_section(title: str, icon: str = "—") -> None:
    """Render a compact section heading."""
    safe_title = escape(str(title))
    safe_icon = escape(str(icon))
    st.markdown(
        _html(
            f"""
            <div class="dl-section">
                <div class="dl-section-icon">{safe_icon}</div>
                <div class="dl-section-title">{safe_title}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_card(title: str, text: str) -> None:
    """Render a white bordered card with a title and body text."""
    st.markdown(
        _html(
            f"""
            <div class="dl-card">
                <div class="dl-card-title">{escape(str(title))}</div>
                <div class="dl-card-text">{escape(str(text))}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_callout(title: str, text: str) -> None:
    """Render a tinted callout with an accent rule, for notes and guidance."""
    st.markdown(
        _html(
            f"""
            <div class="dl-lake-card">
                <div class="dl-lake-title">{escape(str(title))}</div>
                <div class="dl-lake-text">{escape(str(text))}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_lake_card() -> None:
    """Render a subtle Lake Victoria product identity note."""
    render_callout(
        "Built with a sense of place",
        "Makoding takes its visual restraint from the calm, practical "
        "character of the Lake Victoria region: useful tools, clear "
        "evidence, and decisions grounded in the data.",
    )


def render_footer() -> None:
    """Render the application footer."""
    version = str(getattr(APP, "version", "") or "").strip()
    suffix = f" · v{escape(version)}" if version else ""
    st.markdown(
        _html(
            f"""
            <div class="dl-footer">
                Makoding · DataLab Pro{suffix}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Renderers for makoding.cleaning audit objects
# ---------------------------------------------------------------------------

def render_audit_summary(audit: Any) -> None:
    """Render a ``CleaningAudit`` (or a compatible object).

    Shows what cleaning changed as two rows of metrics, flags dtype drift,
    and lists the human-readable decisions in an expander. Do not call this
    from inside another expander: Streamlit does not allow nested ones.
    """
    if audit is None:
        return

    rows_before, rows_after = _count(audit, "rows_before"), _count(audit, "rows_after")
    missing_before, missing_after = (
        _count(audit, "missing_before"),
        _count(audit, "missing_after"),
    )

    top = st.columns(4)
    top[0].metric(
        "Rows",
        f"{rows_after:,}",
        delta=f"{rows_after - rows_before:+,}" if rows_after != rows_before else None,
        delta_color="off",
    )
    top[1].metric(
        "Missing cells",
        f"{missing_after:,}",
        delta=(
            f"{missing_after - missing_before:+,}"
            if missing_after != missing_before
            else None
        ),
        delta_color="off",
    )
    top[2].metric("Values filled", f"{_count(audit, 'values_filled'):,}")
    top[3].metric("Outliers capped", f"{_count(audit, 'values_capped'):,}")

    bottom = st.columns(4)
    bottom[0].metric("Duplicates removed", f"{_count(audit, 'duplicates_removed'):,}")
    bottom[1].metric("Infinite values replaced", f"{_count(audit, 'infinite_replaced'):,}")
    bottom[2].metric("Values whitespace-trimmed", f"{_count(audit, 'whitespace_trimmed'):,}")
    bottom[3].metric("Columns dropped", f"{len(getattr(audit, 'columns_dropped', ()) or ()):,}")

    drift = tuple(getattr(audit, "dtype_drift_columns", ()) or ())
    if drift:
        st.warning(
            "Column type changed since the cleaner was fitted: "
            + ", ".join(str(column) for column in drift)
        )

    decisions = tuple(getattr(audit, "decisions", ()) or ())
    if decisions:
        with st.expander(f"What cleaning did ({len(decisions)} step(s))"):
            for decision in decisions:
                st.text(str(decision))
    else:
        st.caption("Cleaning made no changes to this dataset.")


def _as_record(item: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return dataclasses.asdict(item)
    return dict(vars(item))


def render_quality_report(report: Any) -> None:
    """Render a ``QualityReport`` from ``cleaning.profile_frame``."""
    if report is None:
        return

    cells = st.columns(4)
    cells[0].metric("Rows", f"{_count(report, 'rows'):,}")
    cells[1].metric("Columns", f"{_count(report, 'columns'):,}")
    cells[2].metric("Duplicate rows", f"{_count(report, 'duplicate_rows'):,}")
    cells[3].metric("Missing rate", f"{float(getattr(report, 'missing_rate', 0.0) or 0.0):.1%}")

    profiles = tuple(getattr(report, "column_profiles", ()) or ())
    if not profiles:
        return

    frame = pd.DataFrame([_as_record(item) for item in profiles])
    preferred = [
        "name",
        "dtype",
        "missing",
        "missing_rate",
        "unique",
        "finite",
        "constant",
        "likely_identifier",
    ]
    frame = frame[[column for column in preferred if column in frame.columns]]
    if "missing_rate" in frame.columns:
        frame["missing_rate"] = frame["missing_rate"].astype(float).round(4)
    st.dataframe(frame, hide_index=True)


def render_validation_report(report: Any) -> None:
    """Render a ``ValidationReport`` from ``cleaning.validate_contract``."""
    if report is None:
        return

    if getattr(report, "valid", False):
        st.success("Dataset contract satisfied.")
        return

    issues = tuple(getattr(report, "issues", ()) or ())
    st.error(f"Dataset contract failed with {len(issues)} issue(s).")
    if issues:
        frame = pd.DataFrame(
            [
                {
                    "column": getattr(issue, "column", None),
                    "check": getattr(issue, "code", ""),
                    "detail": getattr(issue, "message", ""),
                }
                for issue in issues
            ]
        )
        st.dataframe(frame, hide_index=True)


def render_manifest(manifest: Any) -> None:
    """Render a ``RunManifest`` provenance record."""
    if manifest is None:
        return

    created = getattr(manifest, "created_at_utc", "")
    if created:
        st.caption(f"Created {created} (UTC)")

    fingerprint = getattr(manifest, "dataset_fingerprint", "")
    if fingerprint:
        st.caption("Dataset fingerprint (SHA-256)")
        st.code(str(fingerprint), language=None)

    notes = getattr(manifest, "notes", "")
    if notes:
        st.caption(f"Notes: {notes}")

    config = getattr(manifest, "cleaner_config", None)
    if config:
        with st.expander("Cleaner configuration"):
            st.json(dict(config))