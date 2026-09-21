"""Visual design system for DataLab Pro.

A restrained editorial interface for a serious analytical workspace.
No gradients, decorative emoji, or visual effects that compete with the data.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from .config import APP

INK = "#17212B"
NAVY = "#173B4D"
TEAL = "#176B87"
TEAL_DARK = "#0F5268"
TEAL_LIGHT = "#E8F2F5"
PAPER = "#F7F7F4"
WHITE = "#FFFFFF"
MUTED = "#68737D"
BORDER = "#D9E0E3"


def _html(content: str) -> str:
    """Flatten HTML indentation so Streamlit does not interpret it as code."""
    return "\n".join(line.lstrip() for line in content.strip("\n").splitlines())


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

            .stButton > button[kind="primary"] {
                background: #176B87;
                color: #FFFFFF;
                border-color: #176B87;
            }

            .stButton > button[kind="primary"]:hover {
                background: #0F5268;
                border-color: #0F5268;
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


def render_hero() -> None:
    """Render the main product introduction."""
    st.markdown(
        _html(
            """
            <div class="dl-hero">
                <div class="dl-eyebrow">Makoding DataLab Pro</div>
                <div class="dl-hero-title">
                    Work with data <span>clearly.</span>
                </div>
                <div class="dl-hero-subtitle">
                    Prepare, explore, transform and model your data in one
                    focused analytical workspace. The interface stays quiet
                    so the evidence stays visible.
                </div>
                <div class="dl-process">
                    <div class="dl-process-item"><span class="dl-process-number">01</span>Clean</div>
                    <div class="dl-process-item"><span class="dl-process-number">02</span>Explore</div>
                    <div class="dl-process-item"><span class="dl-process-number">03</span>Engineer</div>
                    <div class="dl-process-item"><span class="dl-process-number">04</span>Model</div>
                </div>
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


def render_lake_card() -> None:
    """Render a subtle Lake Victoria product identity note."""
    st.markdown(
        _html(
            """
            <div class="dl-lake-card">
                <div class="dl-lake-title">Built with a sense of place</div>
                <div class="dl-lake-text">
                    Makoding takes its visual restraint from the calm,
                    practical character of the Lake Victoria region:
                    useful tools, clear evidence, and decisions grounded
                    in the data.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the application footer."""
    version = escape(str(APP.version))
    st.markdown(
        _html(
            f"""
            <div class="dl-footer">
                Makoding · DataLab Pro · v{version}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
