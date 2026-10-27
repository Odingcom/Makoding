"""
Visual design system for Makoding DataLab Pro.

Fabric-inspired analytical workspace.

Design goals
------------
- Professional enterprise analytics interface
- Compact workspace navigation
- Strong information hierarchy
- Minimal visual noise
- Dense but readable analytical layouts
- Consistent cards, metrics, tables and controls
- Preserve Makoding identity rather than copying Microsoft Fabric
- Backward compatible with existing DataLab Pro rendering functions
"""

from __future__ import annotations

import dataclasses
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from .config import APP


# =============================================================================
# Public API
# =============================================================================

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
    "SURFACE",
    "SURFACE_ALT",
    "inject_theme",
    "render_sidebar_brand",
    "render_workspace_header",
    "render_breadcrumb",
    "render_hero",
    "render_section",
    "render_card",
    "render_callout",
    "render_lake_card",
    "render_status_badge",
    "render_workspace_intro",
    "render_footer",
    "render_audit_summary",
    "render_quality_report",
    "render_validation_report",
    "render_manifest",
]


# =============================================================================
# Makoding design tokens
# =============================================================================

INK = "#17212B"
NAVY = "#173B4D"

TEAL = "#176B87"
TEAL_DARK = "#0F5268"
TEAL_LIGHT = "#E8F2F5"

PAPER = "#F7F8FA"
WHITE = "#FFFFFF"

MUTED = "#68737D"
BORDER = "#D9E0E3"

SURFACE = "#FFFFFF"
SURFACE_ALT = "#F3F5F7"

SIDEBAR = "#F5F6F8"
SIDEBAR_HOVER = "#E9EDF1"

SUCCESS = "#107C10"
WARNING = "#986F0B"
DANGER = "#C42B1C"

SHADOW = "0 1px 2px rgba(0, 0, 0, 0.04)"


_DEFAULT_HERO_TITLE = "Work with data clearly."

_DEFAULT_HERO_SUBTITLE = (
    "Prepare, explore, analyse, transform and model your data "
    "in one professional analytical workspace."
)


# =============================================================================
# Internal helpers
# =============================================================================

def _html(content: str) -> str:
    """
    Flatten HTML so Streamlit's markdown parser does not accidentally
    interpret indented HTML as a code block.
    """

    return "\n".join(
        line.lstrip()
        for line in content.strip("\n").splitlines()
        if line.strip()
    )


def _count(obj: Any, name: str) -> int:
    """Safely read an integer attribute."""

    value = getattr(obj, name, 0)

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _accent_last_word(title: str) -> str:
    """Accent the last word of a title."""

    words = str(title).split()

    if not words:
        return ""

    head = escape(" ".join(words[:-1]))
    tail = escape(words[-1])

    if head:
        return f"{head} <span>{tail}</span>"

    return f"<span>{tail}</span>"


# =============================================================================
# Global theme
# =============================================================================

def inject_theme() -> None:
    """
    Inject the complete Makoding DataLab Pro visual system.

    The design intentionally borrows the workspace philosophy of enterprise
    analytics products such as Microsoft Fabric without reproducing Microsoft's
    branding.
    """

    st.markdown(
        _html(
            """
            <style>

            /* =============================================================
               GLOBAL APPLICATION
               ============================================================= */

            :root {
                --dl-ink: #17212B;
                --dl-navy: #173B4D;
                --dl-teal: #176B87;
                --dl-teal-dark: #0F5268;
                --dl-teal-light: #E8F2F5;
                --dl-paper: #F7F8FA;
                --dl-white: #FFFFFF;
                --dl-muted: #68737D;
                --dl-border: #D9E0E3;
                --dl-surface-alt: #F3F5F7;
            }

            html,
            body,
            [class*="css"] {
                font-family:
                    Inter,
                    "Segoe UI",
                    Arial,
                    sans-serif;
            }

            .stApp {
                background: var(--dl-paper);
                color: var(--dl-ink);
            }

            .main .block-container,
            [data-testid="stMainBlockContainer"] {
                max-width: 1600px;
                padding-top: 1.05rem;
                padding-left: 1.6rem;
                padding-right: 1.6rem;
                padding-bottom: 3rem;
            }

            h1,
            h2,
            h3,
            h4,
            h5 {
                color: var(--dl-navy);
                letter-spacing: -0.015em;
            }

            p,
            label,
            .stCaption {
                color: var(--dl-muted);
            }


            /* =============================================================
               STREAMLIT HEADER
               ============================================================= */

            header[data-testid="stHeader"] {
                background: rgba(255,255,255,0.96);
                border-bottom: 1px solid var(--dl-border);
            }


            /* =============================================================
               SIDEBAR / WORKSPACE NAVIGATION
               ============================================================= */

            section[data-testid="stSidebar"] {
                background: #F5F6F8;
                border-right: 1px solid var(--dl-border);
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 0.8rem;
            }

            section[data-testid="stSidebar"] * {
                color: var(--dl-ink);
            }

            section[data-testid="stSidebar"] hr {
                border-color: var(--dl-border);
                margin-top: 1rem;
                margin-bottom: 1rem;
            }

            section[data-testid="stSidebar"] label {
                font-size: 0.82rem;
                font-weight: 600;
            }


            /* =============================================================
               SIDEBAR BRAND
               ============================================================= */

            .dl-brand {
                display: flex;
                align-items: center;
                gap: 0.72rem;
                padding: 0.25rem 0 1rem 0;
                margin-bottom: 0.25rem;
            }

            .dl-brand-mark {
                width: 34px;
                height: 34px;
                min-width: 34px;
                border-radius: 7px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--dl-teal);
                color: white;
                font-weight: 800;
                font-size: 0.92rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.08);
            }

            .dl-brand-name {
                color: var(--dl-navy);
                font-size: 1rem;
                font-weight: 750;
                line-height: 1.15;
            }

            .dl-brand-subtitle {
                color: var(--dl-muted);
                font-size: 0.72rem;
                margin-top: 0.12rem;
            }


            /* =============================================================
               NAVIGATION LABELS
               ============================================================= */

            .dl-nav-group {
                color: #7A848D;
                font-size: 0.64rem;
                font-weight: 750;
                letter-spacing: 0.11em;
                text-transform: uppercase;
                margin-top: 1.25rem;
                margin-bottom: 0.45rem;
            }

            .dl-nav-item {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                min-height: 36px;
                padding: 0.42rem 0.55rem;
                margin: 0.12rem 0;
                border-radius: 6px;
                color: #36454F;
                font-size: 0.84rem;
                font-weight: 550;
            }

            .dl-nav-item:hover {
                background: #E9EDF1;
            }

            .dl-nav-item-active {
                background: #E8F2F5;
                color: var(--dl-teal-dark);
                font-weight: 700;
                border-left: 3px solid var(--dl-teal);
                padding-left: calc(0.55rem - 3px);
            }

            .dl-nav-icon {
                width: 18px;
                text-align: center;
                color: inherit;
            }


            /* =============================================================
               TEXT INPUTS / SELECTS
               ============================================================= */

            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                background: var(--dl-white);
                border-color: #C9D2D8;
                border-radius: 6px;
            }

            div[data-baseweb="select"] > div {
                background: var(--dl-white);
                border-color: #C9D2D8;
                border-radius: 6px;
            }

            .stTextInput input:focus,
            .stNumberInput input:focus,
            .stTextArea textarea:focus {
                border-color: var(--dl-teal);
                box-shadow: 0 0 0 1px var(--dl-teal);
            }


            /* =============================================================
               BUTTONS
               ============================================================= */

            .stButton > button,
            .stDownloadButton > button {
                min-height: 2.35rem;
                border-radius: 6px;
                border: 1px solid #C7D0D5;
                background: var(--dl-white);
                color: var(--dl-navy);
                font-weight: 650;
                box-shadow: none;
                transition:
                    background 0.12s ease,
                    border-color 0.12s ease;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background: #F0F6F7;
                border-color: var(--dl-teal);
                color: var(--dl-teal-dark);
            }

            .stButton > button[kind="primary"],
            .stButton > button[data-testid="stBaseButton-primary"] {
                background: var(--dl-teal);
                border-color: var(--dl-teal);
                color: white;
            }

            .stButton > button[kind="primary"]:hover,
            .stButton > button[data-testid="stBaseButton-primary"]:hover {
                background: var(--dl-teal-dark);
                border-color: var(--dl-teal-dark);
            }

            .stButton > button[kind="primary"] *,
            .stButton > button[data-testid="stBaseButton-primary"] * {
                color: white;
            }


            /* =============================================================
               TABS
               ============================================================= */

            div[data-testid="stTabs"] {
                margin-top: 0.25rem;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 0;
                background: transparent;
                border-bottom: 1px solid var(--dl-border);
            }

            button[data-baseweb="tab"] {
                height: 43px;
                padding-left: 1rem;
                padding-right: 1rem;
                font-size: 0.83rem;
                font-weight: 650;
                color: #66717A;
            }

            button[data-baseweb="tab"]:hover {
                background: #F0F3F5;
                color: var(--dl-navy);
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                color: var(--dl-teal-dark);
                font-weight: 750;
            }

            div[data-testid="stTabs"]
            [data-baseweb="tab-highlight"] {
                background-color: var(--dl-teal);
                height: 3px;
            }


            /* =============================================================
               WORKSPACE HEADER
               ============================================================= */

            .dl-workspace-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1.5rem;
                padding: 0.35rem 0 1.05rem 0;
                margin-bottom: 0.75rem;
                border-bottom: 1px solid var(--dl-border);
            }

            .dl-workspace-eyebrow {
                color: var(--dl-teal);
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-size: 0.66rem;
                font-weight: 750;
                margin-bottom: 0.3rem;
            }

            .dl-workspace-title {
                color: var(--dl-navy);
                font-size: 1.65rem;
                font-weight: 750;
                line-height: 1.15;
                letter-spacing: -0.025em;
            }

            .dl-workspace-description {
                color: var(--dl-muted);
                font-size: 0.88rem;
                line-height: 1.5;
                margin-top: 0.35rem;
                max-width: 800px;
            }

            .dl-workspace-status {
                white-space: nowrap;
                padding-top: 0.3rem;
            }


            /* =============================================================
               BREADCRUMBS
               ============================================================= */

            .dl-breadcrumb {
                color: #7A858E;
                font-size: 0.73rem;
                margin-bottom: 0.45rem;
            }

            .dl-breadcrumb-current {
                color: var(--dl-navy);
                font-weight: 650;
            }


            /* =============================================================
               HERO
               ============================================================= */

            .dl-hero {
                background: var(--dl-white);
                border: 1px solid var(--dl-border);
                border-radius: 8px;
                padding: 1.65rem 1.8rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.025);
            }

            .dl-eyebrow {
                color: var(--dl-teal);
                text-transform: uppercase;
                letter-spacing: 0.11em;
                font-size: 0.65rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }

            .dl-hero-title {
                color: var(--dl-navy);
                font-size: clamp(1.9rem, 3.5vw, 2.85rem);
                font-weight: 760;
                line-height: 1.05;
                letter-spacing: -0.035em;
                margin: 0;
            }

            .dl-hero-title span {
                color: var(--dl-teal);
            }

            .dl-hero-subtitle {
                color: var(--dl-muted);
                font-size: 0.95rem;
                line-height: 1.6;
                max-width: 820px;
                margin-top: 0.8rem;
            }


            /* =============================================================
               PROCESS / WORKFLOW
               ============================================================= */

            .dl-process {
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                margin-top: 1.25rem;
            }

            .dl-process-item {
                padding: 0.42rem 0.65rem;
                border: 1px solid var(--dl-border);
                border-radius: 5px;
                background: #FAFBFC;
                color: #485761;
                font-size: 0.76rem;
                font-weight: 650;
            }

            .dl-process-number {
                color: var(--dl-teal);
                margin-right: 0.3rem;
                font-weight: 750;
                font-variant-numeric: tabular-nums;
            }


            /* =============================================================
               SECTION HEADINGS
               ============================================================= */

            .dl-section {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                margin: 1.45rem 0 0.8rem 0;
                padding-bottom: 0.45rem;
                border-bottom: 1px solid var(--dl-border);
            }

            .dl-section-icon {
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 24px;
                height: 24px;
                padding: 0 0.35rem;
                border-radius: 4px;
                background: var(--dl-teal-light);
                color: var(--dl-teal-dark);
                font-size: 0.67rem;
                font-weight: 800;
                font-variant-numeric: tabular-nums;
            }

            .dl-section-title {
                color: var(--dl-navy);
                font-size: 1rem;
                font-weight: 750;
            }


            /* =============================================================
               CARDS
               ============================================================= */

            .dl-card {
                background: var(--dl-white);
                border: 1px solid var(--dl-border);
                border-radius: 7px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.85rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.025);
            }

            .dl-card-title {
                color: var(--dl-navy);
                font-size: 0.92rem;
                font-weight: 720;
                margin-bottom: 0.3rem;
            }

            .dl-card-text {
                color: var(--dl-muted);
                font-size: 0.84rem;
                line-height: 1.5;
            }


            /* =============================================================
               WORKSPACE INTRO CARD
               ============================================================= */

            .dl-workspace-intro {
                background: #FFFFFF;
                border: 1px solid var(--dl-border);
                border-left: 4px solid var(--dl-teal);
                border-radius: 7px;
                padding: 1rem 1.15rem;
                margin: 0.8rem 0 1rem 0;
            }

            .dl-workspace-intro-title {
                color: var(--dl-navy);
                font-weight: 750;
                font-size: 0.92rem;
                margin-bottom: 0.25rem;
            }

            .dl-workspace-intro-text {
                color: var(--dl-muted);
                font-size: 0.83rem;
                line-height: 1.5;
            }


            /* =============================================================
               CALLOUT
               ============================================================= */

            .dl-lake-card {
                background: var(--dl-teal-light);
                border: 1px solid #C9DEE5;
                border-left: 3px solid var(--dl-teal);
                border-radius: 6px;
                padding: 1rem 1.1rem;
                color: var(--dl-navy);
                margin: 0.9rem 0;
            }

            .dl-lake-title {
                font-size: 0.9rem;
                font-weight: 750;
                margin-bottom: 0.25rem;
            }

            .dl-lake-text {
                color: #596A73;
                font-size: 0.82rem;
                line-height: 1.5;
                max-width: 850px;
            }


            /* =============================================================
               STATUS BADGES
               ============================================================= */

            .dl-status {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.28rem 0.5rem;
                border-radius: 999px;
                border: 1px solid var(--dl-border);
                background: #FFFFFF;
                color: #59656E;
                font-size: 0.7rem;
                font-weight: 650;
            }

            .dl-status-dot {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: var(--dl-teal);
            }


            /* =============================================================
               METRICS
               ============================================================= */

            div[data-testid="stMetric"] {
                background: var(--dl-white);
                border: 1px solid var(--dl-border);
                border-radius: 7px;
                padding: 0.85rem 0.95rem;
                box-shadow: none;
            }

            div[data-testid="stMetricLabel"] {
                color: var(--dl-muted);
                font-size: 0.76rem;
                font-weight: 600;
            }

            div[data-testid="stMetricValue"] {
                color: var(--dl-navy);
                font-size: 1.45rem;
                font-weight: 720;
            }

            div[data-testid="stMetricDelta"] {
                font-size: 0.72rem;
            }


            /* =============================================================
               DATAFRAMES
               ============================================================= */

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--dl-border);
                border-radius: 6px;
                overflow: hidden;
                background: var(--dl-white);
            }


            /* =============================================================
               ALERTS
               ============================================================= */

            div[data-testid="stAlert"] {
                border-radius: 6px;
                border-width: 1px;
            }


            /* =============================================================
               EXPANDERS
               ============================================================= */

            details[data-testid="stExpander"] {
                background: var(--dl-white);
                border: 1px solid var(--dl-border);
                border-radius: 6px;
            }


            /* =============================================================
               FILE UPLOADER
               ============================================================= */

            section[data-testid="stFileUploaderDropzone"] {
                background: #FFFFFF;
                border: 1px dashed #BFCBD1;
                border-radius: 7px;
            }


            /* =============================================================
               DOWNLOAD CARD
               ============================================================= */

            .dl-download-card {
                background: var(--dl-white);
                border: 1px solid var(--dl-border);
                border-radius: 7px;
                padding: 0.95rem;
                margin: 0.7rem 0;
            }

            .dl-download-title {
                color: var(--dl-navy);
                font-weight: 700;
                margin-bottom: 0.2rem;
            }

            .dl-download-text {
                color: var(--dl-muted);
                font-size: 0.8rem;
                line-height: 1.45;
                margin-bottom: 0.65rem;
            }


            /* =============================================================
               FOOTER
               ============================================================= */

            .dl-footer {
                margin-top: 3rem;
                padding: 1rem 0 0.4rem 0;
                border-top: 1px solid var(--dl-border);
                color: #7A858E;
                font-size: 0.69rem;
                text-align: center;
            }


            /* =============================================================
               RESPONSIVE
               ============================================================= */

            @media (max-width: 900px) {

                .main .block-container,
                [data-testid="stMainBlockContainer"] {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .dl-workspace-header {
                    flex-direction: column;
                }

                .dl-hero {
                    padding: 1.3rem 1.2rem;
                }

                .dl-hero-title {
                    font-size: 2rem;
                }

                .dl-workspace-title {
                    font-size: 1.4rem;
                }
            }

            </style>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar branding
# =============================================================================

def render_sidebar_brand() -> None:
    """Render Makoding DataLab Pro workspace branding."""

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


# =============================================================================
# Workspace navigation helpers
# =============================================================================

def render_workspace_header(
    title: str,
    description: str = "",
    *,
    eyebrow: str = "DataLab Pro workspace",
    status: str | None = None,
) -> None:
    """Render a compact Fabric-style workspace page header."""

    status_html = ""

    if status:
        status_html = f"""
        <div class="dl-workspace-status">
            <span class="dl-status">
                <span class="dl-status-dot"></span>
                {escape(str(status))}
            </span>
        </div>
        """

    st.markdown(
        _html(
            f"""
            <div class="dl-workspace-header">
                <div>
                    <div class="dl-workspace-eyebrow">
                        {escape(str(eyebrow))}
                    </div>

                    <div class="dl-workspace-title">
                        {escape(str(title))}
                    </div>

                    <div class="dl-workspace-description">
                        {escape(str(description))}
                    </div>
                </div>

                {status_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_breadcrumb(*items: str) -> None:
    """Render a small workspace breadcrumb."""

    cleaned = [
        escape(str(item))
        for item in items
        if str(item).strip()
    ]

    if not cleaned:
        return

    if len(cleaned) == 1:
        content = (
            f'<span class="dl-breadcrumb-current">{cleaned[0]}</span>'
        )
    else:
        content = " / ".join(cleaned[:-1])

        content += (
            ' / '
            f'<span class="dl-breadcrumb-current">{cleaned[-1]}</span>'
        )

    st.markdown(
        _html(
            f"""
            <div class="dl-breadcrumb">
                {content}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_workspace_intro(title: str, text: str) -> None:
    """Render explanatory text at the start of a workspace."""

    st.markdown(
        _html(
            f"""
            <div class="dl-workspace-intro">
                <div class="dl-workspace-intro-title">
                    {escape(str(title))}
                </div>

                <div class="dl-workspace-intro-text">
                    {escape(str(text))}
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_status_badge(text: str) -> None:
    """Render a small status badge."""

    st.markdown(
        _html(
            f"""
            <span class="dl-status">
                <span class="dl-status-dot"></span>
                {escape(str(text))}
            </span>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Hero
# =============================================================================

def render_hero(
    title: str | None = None,
    subtitle: str | None = None,
    *,
    eyebrow: str = "Makoding DataLab Pro",
    show_process: bool = True,
) -> None:
    """Render the main DataLab Pro introduction."""

    actual_title = title if title else _DEFAULT_HERO_TITLE
    actual_subtitle = subtitle if subtitle else _DEFAULT_HERO_SUBTITLE

    title_html = _accent_last_word(actual_title)

    process_html = ""

    if show_process:
        process_html = """
        <div class="dl-process">

            <div class="dl-process-item">
                <span class="dl-process-number">01</span>
                Data
            </div>

            <div class="dl-process-item">
                <span class="dl-process-number">02</span>
                Clean
            </div>

            <div class="dl-process-item">
                <span class="dl-process-number">03</span>
                Explore
            </div>

            <div class="dl-process-item">
                <span class="dl-process-number">04</span>
                Statistics
            </div>

            <div class="dl-process-item">
                <span class="dl-process-number">05</span>
                Engineer
            </div>

            <div class="dl-process-item">
                <span class="dl-process-number">06</span>
                Model
            </div>

        </div>
        """

    st.markdown(
        _html(
            f"""
            <div class="dl-hero">

                <div class="dl-eyebrow">
                    {escape(str(eyebrow))}
                </div>

                <div class="dl-hero-title">
                    {title_html}
                </div>

                <div class="dl-hero-subtitle">
                    {escape(str(actual_subtitle))}
                </div>

                {process_html}

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Sections
# =============================================================================

def render_section(
    title: str,
    icon: str = "—",
) -> None:
    """Render compact enterprise section heading."""

    st.markdown(
        _html(
            f"""
            <div class="dl-section">

                <div class="dl-section-icon">
                    {escape(str(icon))}
                </div>

                <div class="dl-section-title">
                    {escape(str(title))}
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Cards
# =============================================================================

def render_card(
    title: str,
    text: str,
) -> None:
    """Render a compact analytical card."""

    st.markdown(
        _html(
            f"""
            <div class="dl-card">

                <div class="dl-card-title">
                    {escape(str(title))}
                </div>

                <div class="dl-card-text">
                    {escape(str(text))}
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Callouts
# =============================================================================

def render_callout(
    title: str,
    text: str,
) -> None:
    """Render analytical guidance callout."""

    st.markdown(
        _html(
            f"""
            <div class="dl-lake-card">

                <div class="dl-lake-title">
                    {escape(str(title))}
                </div>

                <div class="dl-lake-text">
                    {escape(str(text))}
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_lake_card() -> None:
    """Render Makoding identity note."""

    render_callout(
        "Built with a sense of place",
        (
            "Makoding combines practical analytical tools, clear evidence "
            "and decision-focused workflows inspired by the Lake Victoria "
            "region and built for organisations working with real data."
        ),
    )


# =============================================================================
# Footer
# =============================================================================

def render_footer() -> None:
    """Render DataLab Pro footer."""

    version = str(
        getattr(APP, "version", "") or ""
    ).strip()

    suffix = (
        f" · v{escape(version)}"
        if version
        else ""
    )

    st.markdown(
        _html(
            f"""
            <div class="dl-footer">
                Makoding · DataLab Pro{suffix}
                &nbsp; · &nbsp;
                Analytics Workspace
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Cleaning audit renderers
# =============================================================================

def render_audit_summary(audit: Any) -> None:
    """Render CleaningAudit or compatible audit object."""

    if audit is None:
        return

    rows_before = _count(
        audit,
        "rows_before",
    )

    rows_after = _count(
        audit,
        "rows_after",
    )

    missing_before = _count(
        audit,
        "missing_before",
    )

    missing_after = _count(
        audit,
        "missing_after",
    )

    top = st.columns(4)

    top[0].metric(
        "Rows",
        f"{rows_after:,}",
        delta=(
            f"{rows_after - rows_before:+,}"
            if rows_after != rows_before
            else None
        ),
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

    top[2].metric(
        "Values filled",
        f"{_count(audit, 'values_filled'):,}",
    )

    top[3].metric(
        "Outliers capped",
        f"{_count(audit, 'values_capped'):,}",
    )

    bottom = st.columns(4)

    bottom[0].metric(
        "Duplicates removed",
        f"{_count(audit, 'duplicates_removed'):,}",
    )

    bottom[1].metric(
        "Infinite values replaced",
        f"{_count(audit, 'infinite_replaced'):,}",
    )

    bottom[2].metric(
        "Whitespace trimmed",
        f"{_count(audit, 'whitespace_trimmed'):,}",
    )

    columns_dropped = tuple(
        getattr(
            audit,
            "columns_dropped",
            (),
        )
        or ()
    )

    bottom[3].metric(
        "Columns dropped",
        f"{len(columns_dropped):,}",
    )

    drift = tuple(
        getattr(
            audit,
            "dtype_drift_columns",
            (),
        )
        or ()
    )

    if drift:
        st.warning(
            "Column type changed since the cleaner was fitted: "
            + ", ".join(
                str(column)
                for column in drift
            )
        )

    decisions = tuple(
        getattr(
            audit,
            "decisions",
            (),
        )
        or ()
    )

    if decisions:

        with st.expander(
            f"Cleaning operations ({len(decisions)} step(s))"
        ):

            for decision in decisions:
                st.text(str(decision))

    else:
        st.caption(
            "Cleaning made no changes to this dataset."
        )


# =============================================================================
# Quality report
# =============================================================================

def _as_record(item: Any) -> dict[str, Any]:

    if (
        dataclasses.is_dataclass(item)
        and not isinstance(item, type)
    ):
        return dataclasses.asdict(item)

    return dict(vars(item))


def render_quality_report(report: Any) -> None:
    """Render dataset quality report."""

    if report is None:
        return

    cells = st.columns(4)

    cells[0].metric(
        "Rows",
        f"{_count(report, 'rows'):,}",
    )

    cells[1].metric(
        "Columns",
        f"{_count(report, 'columns'):,}",
    )

    cells[2].metric(
        "Duplicate rows",
        f"{_count(report, 'duplicate_rows'):,}",
    )

    missing_rate = float(
        getattr(
            report,
            "missing_rate",
            0.0,
        )
        or 0.0
    )

    cells[3].metric(
        "Missing rate",
        f"{missing_rate:.1%}",
    )

    profiles = tuple(
        getattr(
            report,
            "column_profiles",
            (),
        )
        or ()
    )

    if not profiles:
        return

    frame = pd.DataFrame(
        [
            _as_record(item)
            for item in profiles
        ]
    )

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

    frame = frame[
        [
            column
            for column in preferred
            if column in frame.columns
        ]
    ]

    if "missing_rate" in frame.columns:
        frame["missing_rate"] = (
            frame["missing_rate"]
            .astype(float)
            .round(4)
        )

    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Validation report
# =============================================================================

def render_validation_report(report: Any) -> None:
    """Render dataset contract validation results."""

    if report is None:
        return

    if getattr(
        report,
        "valid",
        False,
    ):

        st.success(
            "Dataset contract satisfied."
        )

        return

    issues = tuple(
        getattr(
            report,
            "issues",
            (),
        )
        or ()
    )

    st.error(
        f"Dataset contract failed with "
        f"{len(issues)} issue(s)."
    )

    if not issues:
        return

    frame = pd.DataFrame(
        [
            {
                "column": getattr(
                    issue,
                    "column",
                    None,
                ),
                "check": getattr(
                    issue,
                    "code",
                    "",
                ),
                "detail": getattr(
                    issue,
                    "message",
                    "",
                ),
            }
            for issue in issues
        ]
    )

    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Run manifest
# =============================================================================

def render_manifest(manifest: Any) -> None:
    """Render DataLab Pro provenance manifest."""

    if manifest is None:
        return

    created = getattr(
        manifest,
        "created_at_utc",
        "",
    )

    if created:
        st.caption(
            f"Created {created} (UTC)"
        )

    fingerprint = getattr(
        manifest,
        "dataset_fingerprint",
        "",
    )

    if fingerprint:

        st.caption(
            "Dataset fingerprint (SHA-256)"
        )

        st.code(
            str(fingerprint),
            language=None,
        )

    notes = getattr(
        manifest,
        "notes",
        "",
    )

    if notes:
        st.caption(
            f"Notes: {notes}"
        )

    config = getattr(
        manifest,
        "cleaner_config",
        None,
    )

    if config:

        with st.expander(
            "Cleaner configuration"
        ):

            st.json(
                dict(config)
            )