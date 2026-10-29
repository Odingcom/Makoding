"""Home dashboard for Makoding DataLab Pro."""

import pandas as pd
import streamlit as st

from makoding.dashboard_metrics import (
    calculate_dashboard_metrics,
    missingness_by_column,
)
from makoding.ui.shell import set_workspace


def record_activity(message: str) -> None:
    """Record recent activity for the current Streamlit session."""

    history = st.session_state.setdefault(
        "dashboard_activity", []
    )

    if not history or history[-1] != message:
        history.append(message)
        del history[:-12]


def _go_to(workspace: str) -> None:
    set_workspace(workspace)
    st.rerun()


def render_home_dashboard(
    frame: pd.DataFrame | None,
    source_name: str | None = None,
) -> None:
    """Display the Home dashboard."""

    st.markdown("### Workspace overview")

    if frame is None:
        st.info("Load a dataset to begin your analysis.")
        return

    metrics = calculate_dashboard_metrics(frame)

    st.caption(
        f"Current working dataset: {source_name or 'Untitled dataset'}"
    )

    a, b, c, d = st.columns(4)

    a.metric("Rows", f"{metrics.rows:,}")
    b.metric("Columns", f"{metrics.columns:,}")
    c.metric(
        "Completeness",
        f"{metrics.completeness_pct:.2f}%",
    )
    d.metric(
        "Duplicate rows",
        f"{metrics.duplicate_rows:,}",
    )

    st.markdown("#### Data quality")

    left, right = st.columns([2, 1])

    with left:
        st.caption(
            f"{metrics.missing_cells:,} missing cells across "
            f"{metrics.columns_with_missing:,} columns"
        )

        st.dataframe(
            missingness_by_column(frame).head(15),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.metric("Numeric features", metrics.numeric_columns)
        st.metric("Categorical features", metrics.categorical_columns)
        st.metric("Other feature types", metrics.other_columns)

    st.markdown("#### Continue your workflow")

    destinations = [
        ("Clean", "clean"),
        ("Explore", "explore"),
        ("Statistical Analysis", "statistics"),
        ("Model Builder", "models"),
    ]

    cols = st.columns(4)

    for col, (label, workspace) in zip(cols, destinations):
        with col:
            if st.button(
                label,
                key=f"home_go_{workspace}",
                use_container_width=True,
            ):
                _go_to(workspace)

    st.markdown("#### Recent activity")

    activity = st.session_state.get(
        "dashboard_activity", []
    )

    if activity:
        for message in reversed(activity[-5:]):
            st.caption(f"• {message}")
    else:
        st.caption("No session activity recorded yet.")
