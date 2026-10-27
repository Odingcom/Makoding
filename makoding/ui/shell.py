"""Shared DataLab Pro application shell.

The shell provides product-level navigation and workspace headers while
keeping analytical functionality in the existing Makoding modules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from makoding.config import APP
from makoding.ui.navigation import (
    HOME,
    NAVIGATION_GROUPS,
    NavigationItem,
    navigation_item,
)


def initialize_navigation_state() -> None:
    """Initialize UI navigation state."""

    if "active_workspace" not in st.session_state:
        st.session_state["active_workspace"] = HOME.key


def set_workspace(key: str) -> None:
    """Change the active DataLab workspace."""

    navigation_item(key)  # validation
    st.session_state["active_workspace"] = key


def get_active_workspace() -> NavigationItem:
    """Return metadata for the active workspace."""

    initialize_navigation_state()

    key = st.session_state.get("active_workspace", HOME.key)

    try:
        return navigation_item(key)
    except KeyError:
        st.session_state["active_workspace"] = HOME.key
        return HOME


def _navigation_button(item: NavigationItem) -> None:
    """Render one sidebar navigation button."""

    active = (
        st.session_state.get("active_workspace", HOME.key)
        == item.key
    )

    label = f"{item.icon}  {item.label}"

    if st.button(
        label,
        key=f"nav_{item.key}",
        use_container_width=True,
        type="primary" if active else "secondary",
    ):
        set_workspace(item.key)
        st.rerun()


def render_sidebar_navigation() -> None:
    """Render Fabric-inspired product navigation."""

    initialize_navigation_state()

    st.markdown(
        f"""
        <div class="datalab-brand">
            <div>
                <div class="datalab-brand-name">{APP.name}</div>
                <div class="datalab-brand-product">Makoding</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="datalab-nav-home"></div>',
        unsafe_allow_html=True,
    )

    _navigation_button(HOME)

    for group in NAVIGATION_GROUPS:
        st.markdown(
            f'<div class="datalab-nav-group">{group.label}</div>',
            unsafe_allow_html=True,
        )

        for item in group.items:
            _navigation_button(item)


def render_workspace_header(
    *,
    source_name: str | None = None,
    frame: pd.DataFrame | None = None,
) -> NavigationItem:
    """Render the current workspace breadcrumb and title."""

    item = get_active_workspace()

    source_label = source_name or "Workspace"

    st.markdown(
        f"""
        <div class="datalab-breadcrumb">
            Workspace&nbsp;&nbsp;/&nbsp;&nbsp;{source_label}
        </div>

        <div class="datalab-workspace-heading">
            <div>
                <h1>{item.label}</h1>
                <p>{item.description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if frame is not None:
        render_dataset_status(frame)

    return item


def render_dataset_status(frame: pd.DataFrame) -> None:
    """Render compact dataset metadata."""

    rows = len(frame)
    columns = len(frame.columns)
    missing = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", f"{columns:,}")
    col3.metric("Missing", f"{missing:,}")
    col4.metric("Duplicates", f"{duplicates:,}")


def render_empty_workspace(
    title: str,
    message: str,
    *,
    icon: str = "◇",
) -> None:
    """Render a professional placeholder for an unfinished workspace."""

    st.markdown(
        f"""
        <div class="datalab-empty-workspace">
            <div class="datalab-empty-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_card(
    title: str,
    description: str,
    *,
    value: Any | None = None,
) -> None:
    """Render a compact workspace information card."""

    value_html = ""

    if value is not None:
        value_html = (
            f'<div class="datalab-card-value">{value}</div>'
        )

    st.markdown(
        f"""
        <div class="datalab-workspace-card">
            <div class="datalab-card-title">{title}</div>
            {value_html}
            <div class="datalab-card-description">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )