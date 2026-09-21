"""
DataLab Pro
-----------
Interactive data-science workspace built on the Makoding package.

Application responsibilities:
- Streamlit presentation and interaction
- Dataset loading
- Cleaning configuration
- EDA exploration
- Feature engineering
- Supervised modelling
- Unsupervised learning

The reusable analytical logic remains inside the Makoding package.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from makoding import (
    cleaning,
    data_io,
    eda,
    feature_engineering,
    modeling,
    styling,
    visualization,
    unsupervised,
)

from makoding.config import (
    APP,
    LIMITS,
    MODEL_DEFAULTS,
    SUPERVISED_MODELS,
)


# ============================================================================
# Application configuration
# ============================================================================

st.set_page_config(
    page_title=APP.name,
    layout="wide",
    initial_sidebar_state="expanded",
)

styling.inject_theme()

logger = logging.getLogger("datalab")


# ============================================================================
# Session state
# ============================================================================

DEFAULT_STATE = {
    "dataframe": None,
    "source_name": None,
    "source_key": None,
    "cleaning_report": None,
    "model_result": None,
    "model_features": None,
    "model_target": None,
    "model_task": None,
    "model_y_test": None,
    "model_predictions": None,
    "model_probabilities": None,
}


for key, default in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default


# ============================================================================
# Helper functions
# ============================================================================


def reset_analysis_state() -> None:
    """
    Clear analytical outputs that depend on the current dataset.
    """

    st.session_state["cleaning_report"] = None
    st.session_state["model_result"] = None
    st.session_state["model_features"] = None
    st.session_state["model_target"] = None
    st.session_state["model_task"] = None
    st.session_state["model_y_test"] = None
    st.session_state["model_predictions"] = None
    st.session_state["model_probabilities"] = None


def display_dataframe(
    frame: pd.DataFrame,
    *,
    height: int = 420,
) -> None:
    """
    Display a DataFrame consistently throughout the application.
    """

    st.dataframe(
        frame,
        use_container_width=True,
        height=height,
    )


def numeric_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """
    Return numeric columns.
    """

    return frame.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """
    Return categorical columns.
    """

    return frame.select_dtypes(
        include=[
            "object",
            "category",
            "string",
            "bool",
        ]
    ).columns.tolist()


def prepare_model_frame(
    frame: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare a modelling frame.

    The current Makoding modelling utilities require:

    - numeric features
    - non-missing features
    - non-missing target values

    Categorical variables are one-hot encoded before modelling.
    """

    if target not in frame.columns:

        raise KeyError(
            f"Target column {target!r} was not found."
        )

    working = frame.copy(
        deep=True
    )

    # ------------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------------

    y = working[target].copy()

    X = working.drop(
        columns=[target]
    )

    # ------------------------------------------------------------------------
    # Remove observations with missing target
    # ------------------------------------------------------------------------

    valid_target = ~y.isna()

    X = (
        X.loc[valid_target]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_target]
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------------
    # Encode categorical features
    # ------------------------------------------------------------------------

    categorical = categorical_columns(X)

    if categorical:

        X = (
            feature_engineering.encode_categorical_features(
                X,
                columns=categorical,
                method="onehot",
            )
        )

    # ------------------------------------------------------------------------
    # Keep numeric features
    # ------------------------------------------------------------------------

    X = X.select_dtypes(
        include=np.number
    )

    if X.empty:

        raise ValueError(
            "No numeric model features remain after preprocessing."
        )

    # ------------------------------------------------------------------------
    # Missing feature validation
    # ------------------------------------------------------------------------

    missing = X.isna().sum()

    missing = missing[
        missing > 0
    ]

    if not missing.empty:

        missing_columns = (
            missing.index.tolist()
        )

        raise ValueError(
            "Model features contain missing values in: "
            f"{missing_columns}. "
            "Use the cleaning controls to remove or "
            "impute missing values before modelling."
        )

    return X, y


def render_metric_row(
    metrics: dict[str, float],
) -> None:
    """
    Render model metrics as Streamlit metric cards.
    """

    if not metrics:

        return

    columns = st.columns(
        min(
            len(metrics),
            4,
        )
    )

    for index, (
        name,
        value,
    ) in enumerate(metrics.items()):

        column = columns[
            index % len(columns)
        ]

        if isinstance(
            value,
            (
                int,
                float,
                np.number,
            ),
        ):

            column.metric(
                name.replace(
                    "_",
                    " ",
                ).title(),
                f"{float(value):.4f}",
            )

        else:

            column.metric(
                name.replace(
                    "_",
                    " ",
                ).title(),
                str(value),
            )


def dataset_signature(
    uploaded_file,
) -> tuple:
    """
    Create a stable signature for an uploaded file.

    This prevents Streamlit reruns from repeatedly reloading
    the same uploaded file.
    """

    return (
        "upload",
        uploaded_file.name,
        uploaded_file.size,
    )


def url_signature(
    url: str,
) -> tuple:
    """
    Create a stable signature for a remote dataset URL.
    """

    return (
        "url",
        url.strip(),
    )



# ============================================================================
# Downloads and visualization helpers
# ============================================================================


def figure_png_bytes(fig) -> bytes:
    """Serialize a matplotlib Figure as a PNG download."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return buffer.getvalue()


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame as UTF-8 CSV bytes."""
    return frame.to_csv(index=False).encode("utf-8")


def json_bytes(payload: dict) -> bytes:
    """Serialize a JSON-compatible payload as UTF-8 bytes."""
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def build_html_report(
    frame: pd.DataFrame,
    source_name: str,
    cleaning_report=None,
) -> bytes:
    """Build a self-contained HTML analytical report with embedded charts."""
    overview = eda.dataset_overview(frame)
    missing = eda.missing_values_summary(frame)
    numeric = eda.numeric_summary(frame)
    categorical = eda.categorical_summary(frame)
    correlation = eda.correlation_matrix(frame)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_source = escape(str(source_name))

    sections: list[str] = []

    sections.append(
        f"""
        <section class="summary">
            <div><strong>{overview["rows"]:,}</strong><span>Rows</span></div>
            <div><strong>{overview["columns"]:,}</strong><span>Columns</span></div>
            <div><strong>{overview["duplicate_rows"]:,}</strong><span>Duplicate rows</span></div>
            <div><strong>{overview["total_missing_cells"]:,}</strong><span>Missing cells</span></div>
        </section>
        """
    )

    def table(title: str, data: pd.DataFrame) -> str:
        if data.empty:
            return f"<h2>{escape(title)}</h2><p>No results available.</p>"
        return f"<h2>{escape(title)}</h2>{data.to_html(index=False, border=0, classes='table')}"

    sections.append(table("Data types and quality", missing))
    sections.append(table("Numeric summary", numeric))
    sections.append(table("Categorical summary", categorical))
    sections.append(table("Correlation matrix", correlation))

    numeric_cols = numeric_columns(frame)[:6]
    categorical_cols = categorical_columns(frame)[:4]

    chart_blocks: list[str] = []

    for column in numeric_cols:
        try:
            fig = visualization.plot_numeric_distribution(frame, column)
            image = base64.b64encode(figure_png_bytes(fig)).decode("ascii")
            chart_blocks.append(
                f'<div class="chart"><h3>{escape(column)}</h3>'
                f'<img src="data:image/png;base64,{image}" alt="Distribution of {escape(column)}"></div>'
            )
        except Exception:
            continue

    for column in categorical_cols:
        try:
            fig = visualization.plot_categorical_counts(frame, column, top_n=12)
            image = base64.b64encode(figure_png_bytes(fig)).decode("ascii")
            chart_blocks.append(
                f'<div class="chart"><h3>{escape(column)}</h3>'
                f'<img src="data:image/png;base64,{image}" alt="Counts of {escape(column)}"></div>'
            )
        except Exception:
            continue

    sections.append(
        "<h2>Visual exploration</h2>"
        + "<div class='charts'>"
        + "".join(chart_blocks)
        + "</div>"
    )

    cleaning_html = ""
    if cleaning_report is not None:
        cleaning_html = (
            "<h2>Preparation notes</h2>"
            f"<pre>{escape(str(cleaning_report))}</pre>"
        )

    html_report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DataLab Pro report — {safe_source}</title>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 0;
    background: #F7F7F4;
    color: #17212B;
    line-height: 1.5;
}}
main {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 42px 28px 70px;
}}
header {{
    background: #173B4D;
    color: white;
    padding: 32px;
    border-radius: 12px;
    margin-bottom: 24px;
}}
header .eyebrow {{
    color: #9EC8D4;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: 12px;
    font-weight: 700;
}}
header h1 {{
    margin: 8px 0;
    font-size: 34px;
}}
header p {{
    color: #DDE9ED;
    margin: 0;
}}
.summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 20px 0 30px;
}}
.summary div {{
    background: white;
    border: 1px solid #D9E0E3;
    border-radius: 10px;
    padding: 18px;
}}
.summary strong {{
    display: block;
    font-size: 26px;
    color: #173B4D;
}}
.summary span {{
    color: #68737D;
    font-size: 13px;
}}
h2 {{
    color: #173B4D;
    border-bottom: 1px solid #D9E0E3;
    padding-bottom: 7px;
    margin-top: 34px;
}}
.table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    font-size: 13px;
}}
.table th, .table td {{
    border: 1px solid #D9E0E3;
    padding: 8px;
    text-align: left;
}}
.table th {{
    background: #E8F2F5;
    color: #173B4D;
}}
.charts {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
}}
.chart {{
    background: white;
    border: 1px solid #D9E0E3;
    border-radius: 10px;
    padding: 14px;
}}
.chart h3 {{
    color: #173B4D;
    font-size: 15px;
}}
.chart img {{
    width: 100%;
    height: auto;
}}
pre {{
    background: white;
    border: 1px solid #D9E0E3;
    padding: 14px;
    overflow-x: auto;
}}
footer {{
    margin-top: 50px;
    color: #68737D;
    font-size: 12px;
    border-top: 1px solid #D9E0E3;
    padding-top: 14px;
}}
@media(max-width: 800px) {{
    .summary, .charts {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<header>
<div class="eyebrow">Makoding / DataLab Pro</div>
<h1>Dataset analysis report</h1>
<p>{safe_source}</p>
<p>Generated {generated}</p>
</header>
{''.join(sections)}
{cleaning_html}
<footer>Makoding · DataLab Pro · Analytical report</footer>
</main>
</body>
</html>"""
    return html_report.encode("utf-8")


def render_downloads(
    frame: pd.DataFrame,
    source_name: str,
    cleaning_report=None,
    key_prefix: str = "downloads",
) -> None:
    """Render working downloads for the current dataset.

    The download payloads are built before the buttons are rendered so that
    every button receives a stable, concrete byte payload. A key prefix is
    used because this component is rendered in more than one location in the
    application.
    """
    styling.render_section("Downloads", "05")
    st.caption(
        "Export the current cleaned dataset or a self-contained analytical report."
    )

    safe_stem = Path(str(source_name).split("?", 1)[0]).stem or "dataset"
    csv_name = f"{safe_stem}_cleaned.csv"
    report_name = f"{safe_stem}_report.html"
    profile_name = f"{safe_stem}_profile.json"

    overview = eda.dataset_overview(frame)
    profile = {
        "source": source_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overview": overview,
        "columns": list(frame.columns),
        "numeric_columns": numeric_columns(frame),
        "categorical_columns": categorical_columns(frame),
    }

    # Build each payload once per render. This keeps the download buttons
    # deterministic and avoids regenerating the HTML report for the button.
    csv_data = dataframe_csv_bytes(frame)
    report_data = build_html_report(
        frame,
        source_name,
        cleaning_report,
    )
    profile_data = json_bytes(profile)

    left, middle, right = st.columns(3)

    with left:
        st.markdown(
            '<div class="dl-download-card">'
            '<div class="dl-download-title">Cleaned dataset</div>'
            '<div class="dl-download-text">'
            "CSV containing the data currently shown in DataLab Pro."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=csv_name,
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv",
        )

    with middle:
        st.markdown(
            '<div class="dl-download-card">'
            '<div class="dl-download-title">Analysis report</div>'
            '<div class="dl-download-text">'
            "Self-contained HTML report with quality tables and embedded charts."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download HTML report",
            data=report_data,
            file_name=report_name,
            mime="text/html",
            use_container_width=True,
            key=f"{key_prefix}_html",
        )

    with right:
        st.markdown(
            '<div class="dl-download-card">'
            '<div class="dl-download-title">Dataset profile</div>'
            '<div class="dl-download-text">'
            "Machine-readable JSON summary for downstream workflows."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download JSON",
            data=profile_data,
            file_name=profile_name,
            mime="application/json",
            use_container_width=True,
            key=f"{key_prefix}_json",
        )



def render_cluster_scatter(
    frame: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    """Render a two-feature cluster view using the first two selected features."""
    if frame.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        frame.iloc[:, 0],
        frame.iloc[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.75,
        s=28,
    )
    ax.set_xlabel(str(frame.columns[0]))
    ax.set_ylabel(str(frame.columns[1]))
    ax.set_title("Cluster structure")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_visual_overview(frame: pd.DataFrame) -> None:
    """Render a useful first-pass visual profile of the dataset."""
    numeric = numeric_columns(frame)
    categorical = categorical_columns(frame)

    styling.render_section("Visual exploration", "06")

    if not numeric and not categorical:
        st.info("No plottable columns are available.")
        return

    if numeric:
        st.markdown("#### Numeric distributions")
        selected_numeric = st.multiselect(
            "Numeric columns",
            numeric,
            default=numeric[: min(4, len(numeric))],
            key="visual_numeric_columns",
        )
        if selected_numeric:
            cols = st.columns(2)
            for index, column in enumerate(selected_numeric):
                with cols[index % 2]:
                    try:
                        fig = visualization.plot_numeric_distribution(
                            frame,
                            column,
                            bins=30,
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"Could not plot {column}: {exc}")

    if categorical:
        st.markdown("#### Categorical distributions")
        selected_categorical = st.multiselect(
            "Categorical columns",
            categorical,
            default=categorical[: min(2, len(categorical))],
            key="visual_categorical_columns",
        )
        if selected_categorical:
            cols = st.columns(2)
            for index, column in enumerate(selected_categorical):
                with cols[index % 2]:
                    try:
                        fig = visualization.plot_categorical_counts(
                            frame,
                            column,
                            top_n=12,
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"Could not plot {column}: {exc}")

    if len(numeric) >= 2:
        st.markdown("#### Relationship explorer")
        x_col, y_col = st.columns(2)
        with x_col:
            x = st.selectbox(
                "X variable",
                numeric,
                key="visual_x_column",
            )
        with y_col:
            y = st.selectbox(
                "Y variable",
                [column for column in numeric if column != x] or numeric,
                key="visual_y_column",
            )

        if x and y:
            try:
                fig = visualization.plot_scatter(frame, x, y)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Could not create the relationship plot: {exc}")



# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:

    styling.render_sidebar_brand()

    st.markdown("### Data source")

    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=[
            "csv",
            "tsv",
            "xlsx",
            "xls",
        ],
        help=(
            f"Maximum upload size: "
            f"{LIMITS.max_upload_mb} MB. "
            "Supported formats: CSV, TSV and Excel."
        ),
    )

    st.markdown("**or**")

    remote_url = st.text_input(
        "Load from URL",
        placeholder="https://example.com/data.csv",
    )

    load_button = st.button(
        "Load dataset",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown("### Cleaning")

    missing_strategy = st.selectbox(
        "Missing values",
        options=cleaning.MISSING_STRATEGIES,
        index=0,
        help=(
            "Choose how missing values should be handled."
        ),
    )

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True,
    )

    trim_whitespace = st.checkbox(
        "Trim text whitespace",
        value=True,
    )


# ============================================================================
# Dataset loading
# ============================================================================

# ----------------------------------------------------------------------------
# Uploaded file
# ----------------------------------------------------------------------------

if uploaded_file is not None:

    current_upload_key = dataset_signature(
        uploaded_file
    )

    previous_source_key = (
        st.session_state["source_key"]
    )

    if previous_source_key != current_upload_key:

        reset_analysis_state()

        try:

            logger.info(
                "Loading uploaded file: %s",
                uploaded_file.name,
            )

            dataframe = data_io.load_dataframe(
                uploaded_file,
                uploaded_file.name,
            )

            if not isinstance(
                dataframe,
                pd.DataFrame,
            ):

                raise TypeError(
                    "The data loader returned an unexpected "
                    f"object: {type(dataframe).__name__}."
                )

            if dataframe.empty:

                raise ValueError(
                    "The loaded dataset contains no rows."
                )

            st.session_state["dataframe"] = (
                dataframe
            )

            st.session_state["source_name"] = (
                uploaded_file.name
            )

            st.session_state["source_key"] = (
                current_upload_key
            )

            logger.info(
                "Dataset ready: %d rows × %d columns.",
                len(dataframe),
                len(dataframe.columns),
            )

            st.success(
                f"Loaded {uploaded_file.name} — "
                f"{len(dataframe):,} rows × "
                f"{len(dataframe.columns):,} columns."
            )

        except Exception as exc:

            logger.exception(
                "The dataset could not be loaded."
            )

            st.error(
                "The dataset could not be loaded."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    f"{type(exc).__name__}: {exc}"
                )

            st.stop()


# ----------------------------------------------------------------------------
# Remote URL
# ----------------------------------------------------------------------------

elif load_button and remote_url.strip():

    current_url_key = url_signature(
        remote_url
    )

    previous_source_key = (
        st.session_state["source_key"]
    )

    if previous_source_key != current_url_key:

        reset_analysis_state()

        try:

            logger.info(
                "Loading remote dataset: %s",
                remote_url.strip(),
            )

            dataframe = data_io.load_csv_url(
                remote_url.strip()
            )

            if not isinstance(
                dataframe,
                pd.DataFrame,
            ):

                raise TypeError(
                    "The data loader returned an unexpected "
                    f"object: {type(dataframe).__name__}."
                )

            if dataframe.empty:

                raise ValueError(
                    "The loaded dataset contains no rows."
                )

            st.session_state["dataframe"] = (
                dataframe
            )

            st.session_state["source_name"] = (
                remote_url.strip()
            )

            st.session_state["source_key"] = (
                current_url_key
            )

            logger.info(
                "Dataset ready: %d rows × %d columns.",
                len(dataframe),
                len(dataframe.columns),
            )

            st.success(
                "Remote dataset loaded — "
                f"{len(dataframe):,} rows × "
                f"{len(dataframe.columns):,} columns."
            )

        except Exception as exc:

            logger.exception(
                "The remote dataset could not be loaded."
            )

            st.error(
                "The dataset could not be loaded."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    f"{type(exc).__name__}: {exc}"
                )

            st.stop()


# ----------------------------------------------------------------------------
# Load button without a URL
# ----------------------------------------------------------------------------

elif load_button:

    st.warning(
        "Choose a file or enter a dataset URL before loading."
    )

    st.stop()


# ============================================================================
# Empty state
# ============================================================================

if st.session_state["dataframe"] is None:

    styling.render_hero()

    st.markdown("")

    left, right = st.columns(
        [2, 1]
    )

    with left:

        styling.render_section(
            "Start with your data",
            "01",
        )

        st.markdown(
            """
            Upload a CSV, TSV, or Excel file, or provide a
            remote dataset URL. DataLab Pro keeps the
            analytical workflow in one place: preparation,
            exploration, feature engineering, modelling and
            unsupervised learning.
            """
        )

    with right:

        styling.render_lake_card()
    styling.render_footer()

    st.stop()


# ============================================================================
# Current dataset
# ============================================================================

current_dataset = (
    st.session_state["dataframe"]
)

source_name = (
    st.session_state["source_name"]
)


# ============================================================================
# Cleaning
# ============================================================================

try:

    cleaned_frame, cleaning_report = (
        cleaning.clean_frame(
            current_dataset,
            missing=missing_strategy,
            remove_duplicates=remove_duplicates,
            trim_whitespace=trim_whitespace,
        )
    )

    current_dataset = cleaned_frame

    st.session_state["cleaning_report"] = (
        cleaning_report
    )

except Exception as exc:

    logger.exception(
        "Unexpected cleaning failure."
    )

    st.error(
        "The dataset loaded successfully, "
        "but cleaning could not be completed."
    )

    with st.expander(
        "Technical details"
    ):

        st.code(
            f"{type(exc).__name__}: {exc}"
        )

    st.stop()


# ============================================================================
# Dataset status
# ============================================================================

st.markdown(
    f"""
    <div class="dl-card">
        <div class="dl-card-title">{escape(str(source_name))}</div>
        <div class="dl-card-text">
            Current working dataset · {len(current_dataset):,} rows ·
            {len(current_dataset.columns):,} columns
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Navigation
# ============================================================================

(
    overview_tab,
    eda_tab,
    feature_tab,
    model_tab,
    unsupervised_tab,
    downloads_tab,
) = st.tabs(
    [
        "Overview",
        "EDA",
        "Feature Engineering",
        "Model Builder",
        "Unsupervised",
        "Downloads",
    ]
)


# ============================================================================
# Overview
# ============================================================================

with overview_tab:

    styling.render_section(
        "Dataset overview",
        "01",
    )

    overview = eda.dataset_overview(
        current_dataset
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Rows",
        f"{overview['rows']:,}",
    )

    metric_columns[1].metric(
        "Columns",
        f"{overview['columns']:,}",
    )

    metric_columns[2].metric(
        "Duplicate rows",
        f"{overview['duplicate_rows']:,}",
    )

    metric_columns[3].metric(
        "Missing cells",
        f"{overview['total_missing_cells']:,}",
    )

    st.markdown("")

    left, right = st.columns(2)

    with left:

        styling.render_section(
            "Data types",
            "02",
        )

        display_dataframe(
            eda.data_types_summary(
                current_dataset
            ),
            height=360,
        )

    with right:

        styling.render_section(
            "Missing values",
            "03",
        )

        missing = (
            eda.missing_values_summary(
                current_dataset
            )
        )

        display_dataframe(
            missing,
            height=360,
        )

    styling.render_section(
        "Preview",
        "04",
    )

    display_dataframe(
        current_dataset.head(
            LIMITS.max_preview_rows
        ),
        height=440,
    )

    render_downloads(
        current_dataset,
        source_name,
        st.session_state["cleaning_report"],
        key_prefix="overview_downloads",
    )


# ============================================================================
# EDA
# ============================================================================

with eda_tab:

    styling.render_section(
        "Exploratory data analysis",
        "01",
    )

    eda_section = st.selectbox(
        "Analysis",
        [
            "Visual exploration",
            "Numeric summary",
            "Categorical summary",
            "Correlation matrix",
            "Outliers",
            "Duplicates",
            "Unique values",
            "Memory usage",
            "Complete EDA report",
        ],
    )

    if eda_section == "Visual exploration":
        render_visual_overview(current_dataset)

    elif eda_section == "Numeric summary":

        result = eda.numeric_summary(current_dataset)

        if result.empty:
            st.info("No numeric columns are available.")
        else:
            display_dataframe(result)

    elif eda_section == "Categorical summary":

        result = eda.categorical_summary(current_dataset)

        if result.empty:
            st.info("No categorical columns are available.")
        else:
            display_dataframe(result)

    elif eda_section == "Correlation matrix":

        method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
        )

        result = eda.correlation_matrix(
            current_dataset,
            method=method,
        )

        if result.empty:
            st.info(
                "At least two numeric variables are required "
                "for a correlation matrix."
            )
        else:
            display_dataframe(result)

            if len(result.columns) >= 2:
                import matplotlib.pyplot as _plt

                fig, ax = _plt.subplots(
                    figsize=(8, 6),
                )
                image = ax.imshow(
                    result.to_numpy(dtype=float),
                    aspect="auto",
                    cmap="Blues",
                    vmin=-1,
                    vmax=1,
                )
                ax.set_xticks(range(len(result.columns)))
                ax.set_xticklabels(
                    result.columns,
                    rotation=45,
                    ha="right",
                )
                ax.set_yticks(range(len(result.index)))
                ax.set_yticklabels(result.index)
                ax.set_title(f"{method.title()} correlation")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                _plt.close(fig)

    elif eda_section == "Outliers":

        result = eda.outlier_summary(current_dataset)

        if result.empty:
            st.info(
                "No numeric columns are available for outlier analysis."
            )
        else:
            display_dataframe(result)

    elif eda_section == "Duplicates":

        result = eda.duplicate_rows_summary(current_dataset)
        st.json(result)

    elif eda_section == "Unique values":

        result = eda.unique_values_summary(current_dataset)
        display_dataframe(result)

    elif eda_section == "Memory usage":

        result = eda.memory_usage_summary(current_dataset)
        display_dataframe(result)

    elif eda_section == "Complete EDA report":

        report = eda.generate_eda_report(current_dataset)

        for name, result in report.items():

            st.markdown(
                f"#### {name.replace('_', ' ').title()}"
            )

            if isinstance(result, pd.DataFrame):

                if result.empty:
                    st.info(f"No results available for {name}.")
                else:
                    display_dataframe(result, height=300)

            else:
                st.json(result)

        st.markdown("")
        render_visual_overview(current_dataset)


# ============================================================================
# Feature engineering
# ============================================================================

with feature_tab:

    styling.render_section(
        "Feature engineering",
        "01",
    )

    st.caption(
        "Create transformed features without modifying "
        "the loaded dataset."
    )

    feature_frame = (
        current_dataset.copy(
            deep=True
        )
    )

    operation = st.selectbox(
        "Transformation",
        [
            "Scale numeric features",
            "Encode categorical features",
            "Log transform",
            "Square-root transform",
            "Power transform",
            "Add missing indicators",
            "Add datetime features",
            "Add cyclical features",
            "Add frequency features",
        ],
    )

    numeric = numeric_columns(
        feature_frame
    )

    categorical = categorical_columns(
        feature_frame
    )


    # ------------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------------

    if operation == "Scale numeric features":

        if not numeric:

            st.info(
                "No numeric columns are available."
            )

        else:

            selected = st.multiselect(
                "Numeric columns",
                numeric,
                default=numeric,
            )

            method = st.selectbox(
                "Scaling method",
                [
                    "standard",
                    "minmax",
                    "robust",
                ],
            )

            if selected:

                try:

                    result = (
                        feature_engineering.scale_numeric_features(
                            feature_frame,
                            columns=selected,
                            method=method,
                        )
                    )

                    display_dataframe(
                        result
                    )

                except Exception as exc:

                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )


    # ------------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------------

    elif operation == "Encode categorical features":

        if not categorical:

            st.info(
                "No categorical columns are available."
            )

        else:

            selected = st.multiselect(
                "Categorical columns",
                categorical,
                default=categorical,
            )

            method = st.selectbox(
                "Encoding method",
                [
                    "onehot",
                    "ordinal",
                ],
            )

            if selected:

                try:

                    result = (
                        feature_engineering.encode_categorical_features(
                            feature_frame,
                            columns=selected,
                            method=method,
                        )
                    )

                    display_dataframe(
                        result
                    )

                except Exception as exc:

                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )


    # ------------------------------------------------------------------------
    # Log transform
    # ------------------------------------------------------------------------

    elif operation == "Log transform":

        if not numeric:

            st.info(
                "No numeric columns are available."
            )

        else:

            selected = st.multiselect(
                "Columns",
                numeric,
            )

            offset = st.number_input(
                "Offset",
                value=0.0,
                step=1.0,
            )

            if selected:

                try:

                    result = (
                        feature_engineering.log_transform(
                            feature_frame,
                            selected,
                            offset=offset,
                        )
                    )

                    display_dataframe(
                        result
                    )

                except Exception as exc:

                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )


    # ------------------------------------------------------------------------
    # Square-root transform
    # ------------------------------------------------------------------------

    elif operation == "Square-root transform":

        if not numeric:

            st.info(
                "No numeric columns are available."
            )

        else:

            selected = st.multiselect(
                "Columns",
                numeric,
            )

            if selected:

                try:

                    result = (
                        feature_engineering.sqrt_transform(
                            feature_frame,
                            selected,
                        )
                    )

                    display_dataframe(
                        result
                    )

                except Exception as exc:

                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )


    # ------------------------------------------------------------------------
    # Power transform
    # ------------------------------------------------------------------------

    elif operation == "Power transform":

        if not numeric:

            st.info(
                "No numeric columns are available."
            )

        else:

            selected = st.multiselect(
                "Columns",
                numeric,
            )

            method = st.selectbox(
                "Power method",
                [
                    "yeo-johnson",
                    "box-cox",
                ],
            )

            if selected:

                try:

                    result = (
                        feature_engineering.power_transform(
                            feature_frame,
                            selected,
                            method=method,
                        )
                    )

                    display_dataframe(
                        result
                    )

                except Exception as exc:

                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )


    # ------------------------------------------------------------------------
    # Missing indicators
    # ------------------------------------------------------------------------

    elif operation == "Add missing indicators":

        selected = st.multiselect(
            "Columns",
            list(feature_frame.columns),
        )

        if selected:

            try:

                result = (
                    feature_engineering.add_missing_indicators(
                        feature_frame,
                        columns=selected,
                    )
                )

                display_dataframe(
                    result
                )

            except Exception as exc:

                st.error(
                    f"{type(exc).__name__}: {exc}"
                )


    # ------------------------------------------------------------------------
    # Datetime features
    # ------------------------------------------------------------------------

    elif operation == "Add datetime features":

        selected = st.selectbox(
            "Datetime column",
            list(feature_frame.columns),
        )

        errors = st.selectbox(
            "Invalid datetime handling",
            [
                "raise",
                "coerce",
            ],
        )

        try:

            result = (
                feature_engineering.add_datetime_features(
                    feature_frame,
                    selected,
                    errors=errors,
                )
            )

            display_dataframe(
                result
            )

        except Exception as exc:

            st.error(
                f"{type(exc).__name__}: {exc}"
            )


    # ------------------------------------------------------------------------
    # Cyclical features
    # ------------------------------------------------------------------------

    elif operation == "Add cyclical features":

        if not numeric:

            st.info(
                "No numeric columns are available."
            )

        else:

            selected = st.selectbox(
                "Periodic column",
                numeric,
            )

            period = st.number_input(
                "Period",
                min_value=0.0001,
                value=12.0,
                step=1.0,
            )

            try:

                result = (
                    feature_engineering.add_cyclical_features(
                        feature_frame,
                        selected,
                        period=period,
                    )
                )

                display_dataframe(
                    result
                )

            except Exception as exc:

                st.error(
                    f"{type(exc).__name__}: {exc}"
                )


    # ------------------------------------------------------------------------
    # Frequency features
    # ------------------------------------------------------------------------

    elif operation == "Add frequency features":

        if not feature_frame.columns.tolist():

            st.info(
                "No columns are available."
            )

        else:

            selected = st.multiselect(
                "Columns",
                list(feature_frame.columns),
            )

            if selected:

                try:

                    result = (
                        feature_engineering.add_frequency_features(
                            feature_frame,
                            selected,
                        )
                    )

                    display_dataframe(
                        result
                    )

                except Exception as exc:

                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )


# ============================================================================
# Model Builder
# ============================================================================

with model_tab:

    styling.render_section(
        "Model builder",
        "01",
    )

    st.caption(
        "Build a supervised model from the cleaned dataset."
    )

    target_candidates = list(
        current_dataset.columns
    )

    target = st.selectbox(
        "Target variable",
        target_candidates,
    )

    task = st.selectbox(
        "Task",
        [
            "classification",
            "regression",
        ],
    )

    if task == "classification":

        model_options = [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

    else:

        model_options = [
            "linear_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

    model_name = st.selectbox(
        "Model",
        model_options,
    )

    test_size = st.slider(
        "Test size",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )

    stratify = False

    if task == "classification":

        stratify = st.checkbox(
            "Stratify train/test split",
            value=True,
        )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        value=int(
            MODEL_DEFAULTS.random_state
        ),
        step=1,
    )

    train_button = st.button(
        "Train model",
        type="primary",
        use_container_width=True,
    )

    if train_button:

        try:

            X, y = prepare_model_frame(
                current_dataset,
                target,
            )

            if task == "regression":

                y = pd.to_numeric(
                    y,
                    errors="raise",
                )

            if len(X) < 10:

                raise ValueError(
                    "At least 10 usable observations "
                    "are recommended for model training."
                )

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = modeling.train_test_split_data(
                X,
                y,
                test_size=test_size,
                random_state=int(
                    random_state
                ),
                stratify=stratify,
            )

            with st.spinner(
                f"Training {model_name}..."
            ):

                result = modeling.fit_model(
                    X_train,
                    y_train,
                    task=task,
                    model_name=model_name,
                    random_state=int(
                        random_state
                    ),
                )

            predictions = modeling.predict(
                result.model,
                X_test,
            )

            # ----------------------------------------------------------------
            # Classification evaluation
            # ----------------------------------------------------------------

            if task == "classification":

                probabilities = None

                if hasattr(
                    result.model,
                    "predict_proba",
                ):

                    probability_matrix = (
                        result.model.predict_proba(
                            X_test
                        )
                    )

                    if (
                        probability_matrix.ndim == 2
                        and probability_matrix.shape[1] == 2
                    ):

                        probabilities = (
                            probability_matrix[:, 1]
                        )

                metrics = (
                    modeling.evaluate_classification(
                        y_test,
                        predictions,
                        probabilities,
                    )
                )

            # ----------------------------------------------------------------
            # Regression evaluation
            # ----------------------------------------------------------------

            else:

                metrics = (
                    modeling.evaluate_regression(
                        y_test,
                        predictions,
                    )
                )

            # ----------------------------------------------------------------
            # Save model state
            # ----------------------------------------------------------------

            st.session_state["model_result"] = (
                result
            )

            st.session_state["model_features"] = (
                X_train.columns.tolist()
            )

            st.session_state["model_target"] = (
                target
            )
            st.session_state["model_task"] = task
            st.session_state["model_y_test"] = (
                y_test.reset_index(drop=True)
            )
            st.session_state["model_predictions"] = (
                pd.Series(predictions).reset_index(drop=True)
            )
            st.session_state["model_probabilities"] = (
                None if probabilities is None else np.asarray(probabilities)
            )

            # ----------------------------------------------------------------
            # Results
            # ----------------------------------------------------------------

            st.success(
                f"{model_name} trained successfully."
            )

            styling.render_section(
                "Model performance",
                "02",
            )

            render_metric_row(
                metrics
            )

            # ----------------------------------------------------------------
            # Feature importance
            # ----------------------------------------------------------------

            styling.render_section(
                "Feature importance",
                "03",
            )

            try:

                importance = (
                    modeling.get_feature_importance(
                        result.model,
                        X_train.columns.tolist(),
                    )
                )

                display_dataframe(
                    importance,
                    height=420,
                )

            except Exception as exc:

                st.info(
                    "Feature importance is not available "
                    f"for this model: {exc}"
                )

            # ----------------------------------------------------------------
            # Predictions
            # ----------------------------------------------------------------

            styling.render_section(
                "Predictions",
                "04",
            )

            prediction_frame = pd.DataFrame(
                {
                    "actual": (
                        y_test
                        .reset_index(drop=True)
                    ),
                    "prediction": (
                        pd.Series(
                            predictions
                        ).reset_index(drop=True)
                    ),
                }
            )

            display_dataframe(
                prediction_frame.head(200),
                height=420,
            )

            styling.render_section(
                "Model diagnostics",
                "05",
            )

            if task == "classification":
                diagnostic_left, diagnostic_right = st.columns(2)

                with diagnostic_left:
                    try:
                        fig = visualization.plot_confusion_matrix(
                            y_test,
                            predictions,
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.info(f"Confusion matrix unavailable: {exc}")

                with diagnostic_right:
                    if probabilities is not None:
                        try:
                            fig = visualization.plot_roc_curve(
                                y_test,
                                probabilities,
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        except Exception as exc:
                            st.info(f"ROC curve unavailable: {exc}")

                if probabilities is not None:
                    try:
                        fig = visualization.plot_precision_recall_curve(
                            y_test,
                            probabilities,
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.info(
                            f"Precision-recall curve unavailable: {exc}"
                        )

            else:
                diagnostic_left, diagnostic_right = st.columns(2)

                with diagnostic_left:
                    try:
                        fig = visualization.plot_predicted_vs_actual(
                            y_test,
                            predictions,
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.info(
                            f"Predicted-vs-actual plot unavailable: {exc}"
                        )

                with diagnostic_right:
                    try:
                        fig = visualization.plot_residuals(
                            y_test,
                            predictions,
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.info(f"Residual plot unavailable: {exc}")

            try:
                importance_for_plot = modeling.get_feature_importance(
                    result.model,
                    X_train.columns.tolist(),
                )
                fig = visualization.plot_feature_importance(
                    importance_for_plot,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.info(f"Feature-importance chart unavailable: {exc}")


        except Exception as exc:

            logger.exception(
                "Model training failed."
            )

            st.error(
                "Model training could not be completed."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    f"{type(exc).__name__}: {exc}"
                )


# ============================================================================
# Unsupervised learning
# ============================================================================

with unsupervised_tab:

    styling.render_section(
        "Unsupervised learning",
        "01",
    )

    st.caption(
        "Discover structure, groups and lower-dimensional "
        "representations without a target variable."
    )

    unsupervised_numeric = numeric_columns(
        current_dataset
    )

    if len(unsupervised_numeric) < 2:

        st.warning(
            "At least two numeric columns are required "
            "for unsupervised analysis."
        )

    else:

        selected_columns = st.multiselect(
            "Features",
            unsupervised_numeric,
            default=unsupervised_numeric[
                :min(
                    5,
                    len(unsupervised_numeric),
                )
            ],
        )

        if selected_columns:

            working = (
                current_dataset[
                    selected_columns
                ].copy()
            )

            missing_count = int(
                working.isna()
                .sum()
                .sum()
            )

            if missing_count:

                st.warning(
                    f"The selected features contain "
                    f"{missing_count:,} missing cells. "
                    "Clean the dataset before running "
                    "unsupervised analysis."
                )

            else:

                algorithm = st.selectbox(
                    "Method",
                    [
                        "K-Means",
                        "Agglomerative clustering",
                        "DBSCAN",
                        "PCA",
                    ],
                )

                # ============================================================
                # K-Means
                # ============================================================

                if algorithm == "K-Means":

                    max_clusters = min(
                        12,
                        len(working) - 1,
                    )

                    n_clusters = st.slider(
                        "Number of clusters",
                        min_value=2,
                        max_value=max_clusters,
                        value=min(
                            3,
                            max_clusters,
                        ),
                    )

                    if st.button(
                        "Run K-Means",
                        type="primary",
                    ):

                        try:

                            scaled = (
                                unsupervised.standardize_features(
                                    working
                                )
                            )

                            result = (
                                unsupervised.fit_kmeans(
                                    scaled,
                                    n_clusters=n_clusters,
                                    random_state=int(
                                        MODEL_DEFAULTS.random_state
                                    ),
                                )
                            )

                            labels = np.asarray(
                                result.labels
                            )

                            display_frame = (
                                working.copy()
                            )

                            display_frame[
                                "cluster"
                            ] = labels

                            st.success(
                                "K-Means completed."
                            )

                            render_cluster_scatter(working, labels)


                            styling.render_section(
                                "Cluster assignments",
                                "02",
                            )

                            display_dataframe(
                                display_frame.head(
                                    300
                                ),
                                height=420,
                            )

                            try:

                                score = (
                                    unsupervised.silhouette_score(
                                        scaled,
                                        labels,
                                    )
                                )

                                st.metric(
                                    "Silhouette score",
                                    f"{score:.4f}",
                                )

                            except Exception:

                                pass

                            styling.render_section(
                                "Cluster summary",
                                "03",
                            )

                            try:

                                summary = (
                                    unsupervised.cluster_summary(
                                        working,
                                        labels,
                                    )
                                )

                                display_dataframe(
                                    summary,
                                    height=360,
                                )

                            except Exception as exc:

                                st.info(
                                    "Cluster summary unavailable: "
                                    f"{exc}"
                                )

                        except Exception as exc:

                            st.error(
                                f"{type(exc).__name__}: {exc}"
                            )

                # ============================================================
                # Agglomerative clustering
                # ============================================================

                elif algorithm == "Agglomerative clustering":

                    max_clusters = min(
                        12,
                        len(working) - 1,
                    )

                    n_clusters = st.slider(
                        "Number of clusters",
                        min_value=2,
                        max_value=max_clusters,
                        value=min(
                            3,
                            max_clusters,
                        ),
                    )

                    if st.button(
                        "Run agglomerative clustering",
                        type="primary",
                    ):

                        try:

                            scaled = (
                                unsupervised.standardize_features(
                                    working
                                )
                            )

                            result = (
                                unsupervised.fit_agglomerative(
                                    scaled,
                                    n_clusters=n_clusters,
                                )
                            )

                            labels = np.asarray(
                                result.labels
                            )

                            display_frame = (
                                working.copy()
                            )

                            display_frame[
                                "cluster"
                            ] = labels

                            st.success(
                                "Agglomerative clustering completed."
                            )

                            render_cluster_scatter(working, labels)


                            display_dataframe(
                                display_frame.head(
                                    300
                                ),
                                height=420,
                            )

                            try:

                                score = (
                                    unsupervised.silhouette_score(
                                        scaled,
                                        labels,
                                    )
                                )

                                st.metric(
                                    "Silhouette score",
                                    f"{score:.4f}",
                                )

                            except Exception:

                                pass

                        except Exception as exc:

                            st.error(
                                f"{type(exc).__name__}: {exc}"
                            )

                # ============================================================
                # DBSCAN
                # ============================================================

                elif algorithm == "DBSCAN":

                    eps = st.number_input(
                        "Epsilon",
                        min_value=0.01,
                        value=0.5,
                        step=0.05,
                    )

                    min_samples = st.number_input(
                        "Minimum samples",
                        min_value=2,
                        value=5,
                        step=1,
                    )

                    if st.button(
                        "Run DBSCAN",
                        type="primary",
                    ):

                        try:

                            scaled = (
                                unsupervised.standardize_features(
                                    working
                                )
                            )

                            result = (
                                unsupervised.fit_dbscan(
                                    scaled,
                                    eps=float(eps),
                                    min_samples=int(
                                        min_samples
                                    ),
                                )
                            )

                            labels = np.asarray(
                                result.labels
                            )

                            display_frame = (
                                working.copy()
                            )

                            display_frame[
                                "cluster"
                            ] = labels

                            st.success(
                                "DBSCAN completed."
                            )

                            render_cluster_scatter(working, labels)


                            display_dataframe(
                                display_frame.head(
                                    300
                                ),
                                height=420,
                            )

                            unique_labels = sorted(
                                set(
                                    labels.tolist()
                                )
                            )

                            st.write(
                                "Detected labels: "
                                f"{unique_labels}"
                            )

                        except Exception as exc:

                            st.error(
                                f"{type(exc).__name__}: {exc}"
                            )

                # ============================================================
                # PCA
                # ============================================================

                elif algorithm == "PCA":

                    max_components = min(
                        len(selected_columns),
                        len(working),
                    )

                    n_components = st.slider(
                        "Components",
                        min_value=2,
                        max_value=max_components,
                        value=min(
                            2,
                            max_components,
                        ),
                    )

                    if st.button(
                        "Run PCA",
                        type="primary",
                    ):

                        try:

                            pca_result = (
                                unsupervised.apply_pca(
                                    working,
                                    n_components=n_components,
                                )
                            )

                            st.success(
                                "PCA completed."
                            )

                            styling.render_section(
                                "Explained variance",
                                "02",
                            )

                            # ------------------------------------------------
                            # Support PCAResult-style objects
                            # ------------------------------------------------

                            if hasattr(
                                pca_result,
                                "components",
                            ):

                                components = (
                                    pca_result.components
                                )

                                variance = (
                                    pca_result.explained_variance_ratio
                                )

                            else:

                                components, variance = (
                                    pca_result
                                )

                            if isinstance(
                                variance,
                                pd.Series,
                            ):

                                variance_values = (
                                    variance.values
                                )

                            else:

                                variance_values = (
                                    np.asarray(
                                        variance
                                    )
                                )

                            variance_frame = (
                                pd.DataFrame(
                                    {
                                        "component": [
                                            f"PC{i + 1}"
                                            for i in range(
                                                len(
                                                    variance_values
                                                )
                                            )
                                        ],
                                        "explained_variance_ratio": (
                                            variance_values
                                        ),
                                    }
                                )
                            )

                            variance_frame[
                                "cumulative_variance"
                            ] = (
                                variance_frame[
                                    "explained_variance_ratio"
                                ].cumsum()
                            )

                            display_dataframe(
                                variance_frame,
                                height=300,
                            )

                            fig, ax = plt.subplots(figsize=(8, 4.5))
                            ax.plot(
                                variance_frame["component"],
                                variance_frame["cumulative_variance"],
                                marker="o",
                            )
                            ax.set_ylim(0, 1.05)
                            ax.set_ylabel("Cumulative explained variance")
                            ax.set_xlabel("Component")
                            ax.set_title("PCA explained variance")
                            ax.grid(alpha=0.2)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)


                            styling.render_section(
                                "Principal components",
                                "03",
                            )

                            display_dataframe(
                                components.head(
                                    300
                                ),
                                height=420,
                            )

                        except Exception as exc:

                            st.error(
                                f"{type(exc).__name__}: {exc}"
                            )



# ============================================================================
# Downloads
# ============================================================================

with downloads_tab:
    styling.render_section(
        "Export your work",
        "01",
    )
    st.caption(
        "Save the cleaned dataset, analytical report, or machine-readable profile."
    )
    render_downloads(
        current_dataset,
        source_name,
        st.session_state["cleaning_report"],
        key_prefix="downloads_tab",
    )


# ============================================================================
# Footer
# ============================================================================

styling.render_lake_card()

styling.render_footer()