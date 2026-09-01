"""DataLab Pro — an interactive data science workspace.

Run with:  streamlit run app.py
Install:   pip install -r requirements.txt

See README.md for configuration (env vars), deployment, and licensing notes.
"""
from __future__ import annotations

import streamlit as st

from makoding import cleaning, data_io, eda, features, modeling, styling
from makoding.config import APP, LIMITS, MODEL_DEFAULTS, SUPERVISED_MODELS
from makoding.logging_config import setup_logging

logger = setup_logging()

st.set_page_config(page_title=APP.name, page_icon=APP.icon, layout="wide")
styling.inject_theme()
styling.render_header()

# ---------------------------------------------------------------------------
# Sidebar: data source + cleaning controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data source")
    uploaded = st.file_uploader(
        "Upload CSV, TSV, or Excel",
        type=["csv", "tsv", "xlsx", "xls"],
        help=f"Max size: {LIMITS.max_upload_mb} MB",
    )
    url = st.text_input("...or a CSV URL", placeholder="https://example.com/data.csv")

    st.header("2. Cleaning")
    missing_strategy = st.selectbox("Missing values", cleaning.MISSING_STRATEGIES)
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    trim_whitespace = st.checkbox("Trim text whitespace", value=True)

    st.divider()
    st.caption(f"{APP.name} v{APP.version}")

# ---------------------------------------------------------------------------
# Load + clean data
# ---------------------------------------------------------------------------
if uploaded is None and not url.strip():
    st.info("👋 Upload a dataset or paste a CSV URL in the sidebar to get started.")
    styling.render_footer()
    st.stop()

try:
    with st.spinner("Loading data..."):
        dataset = data_io.load_data(uploaded, url)
except data_io.DataLoadError as exc:
    st.error(f"**Could not load data.** {exc}")
    logger.warning("Data load failed: %s", exc)
    st.stop()
except Exception as exc:  # noqa: BLE001 - last line of defense for a commercial app
    st.error("An unexpected error occurred while loading your data. Please try again.")
    logger.exception("Unexpected error loading data")
    st.stop()

try:
    df, report = cleaning.clean_frame(dataset.frame, missing_strategy, remove_duplicates, trim_whitespace)
except Exception as exc:  # noqa: BLE001
    st.error("An unexpected error occurred while cleaning your data.")
    logger.exception("Unexpected error cleaning data")
    st.stop()

if df.empty:
    st.warning("Cleaning removed every row (check your missing-value strategy). Nothing left to analyze.")
    st.stop()

st.success(f"Loaded **{dataset.source_name}** — {len(df):,} rows × {len(df.columns):,} columns.")

tab_overview, tab_eda, tab_features, tab_model = st.tabs(
    ["📋 Overview", "🔍 EDA", "🛠️ Feature engineering", "🤖 Model builder"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows removed", f"{report.duplicates_removed:,}")

    with st.expander("What changed during cleaning?"):
        st.markdown(report.as_markdown())

    st.dataframe(df.head(LIMITS.max_preview_rows), use_container_width=True)
    st.download_button(
        "⬇ Download cleaned CSV",
        df.to_csv(index=False),
        "cleaned_data.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
with tab_eda:
    numeric = eda.numeric_columns(df)
    categorical = eda.categorical_columns(df)

    missing_summary = eda.missing_value_summary(df)
    if not missing_summary.empty:
        st.subheader("Missing values by column")
        st.dataframe(missing_summary, use_container_width=True)

    if numeric:
        st.subheader("Numeric distribution")
        selected = st.selectbox("Column", numeric, key="dist_col")
        st.bar_chart(df[selected].value_counts().sort_index().head(100))
        st.dataframe(df[numeric].describe().T, use_container_width=True)

        bounds = eda.detect_outliers_iqr(df, selected)
        if bounds.count:
            st.caption(
                f"⚠️ {bounds.count} potential outlier(s) in **{selected}** "
                f"outside [{bounds.lower:.2f}, {bounds.upper:.2f}] (IQR rule)."
            )

    if len(numeric) >= 2:
        st.subheader("Bivariate relationship")
        x_col, y_col = st.columns(2)
        xcol = x_col.selectbox("X variable", numeric, index=0, key="x_var")
        ycol = y_col.selectbox("Y variable", numeric, index=min(1, len(numeric) - 1), key="y_var")
        st.scatter_chart(df[[xcol, ycol]].dropna().set_index(xcol))

        st.subheader("Correlation matrix")
        corr = eda.correlation_matrix(df)
        if not corr.empty:
            st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)

    if categorical:
        st.subheader("Categorical counts")
        cat = st.selectbox("Column", categorical, key="cat_col")
        st.bar_chart(df[cat].astype(str).value_counts().head(LIMITS.max_categorical_levels))

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
with tab_features:
    st.subheader("Feature engineering")
    feature_df = df.copy()

    date_col = st.selectbox("Extract date parts from (optional)", ["None"] + list(df.columns))
    if date_col != "None":
        try:
            feature_df = features.add_date_parts(feature_df, date_col)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not parse '{date_col}' as a date: {exc}")

    numeric_cols = eda.numeric_columns(df)
    if numeric_cols:
        st.caption("Numeric transforms")
        cols = st.columns(3)
        for i, col in enumerate(numeric_cols):
            with cols[i % 3]:
                if st.checkbox(f"log1p({col})", key=f"log_{col}"):
                    feature_df = features.add_log1p(feature_df, col)
                if st.checkbox(f"z-score({col})", key=f"z_{col}"):
                    feature_df = features.add_zscore(feature_df, col)

    st.dataframe(feature_df.head(LIMITS.max_preview_rows), use_container_width=True)
    st.download_button(
        "⬇ Download engineered CSV",
        feature_df.to_csv(index=False),
        "engineered_data.csv",
        "text/csv",
        key="download_features",
    )

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Model builder")
    numeric_cols = eda.numeric_columns(feature_df)

    if not numeric_cols:
        st.warning("At least one numeric feature is required for modeling.")
    else:
        target = st.selectbox("Target column", feature_df.columns, key="target_col")
        default_features = [c for c in numeric_cols if c != target][: MODEL_DEFAULTS.max_features_default]
        selected_features = st.multiselect(
            "Feature columns",
            [c for c in feature_df.columns if c != target],
            default=default_features,
        )
        model_name = st.selectbox("Algorithm", list(SUPERVISED_MODELS.keys()))

        if st.button("Train model", type="primary"):
            try:
                with st.spinner("Training and cross-validating..."):
                    result = modeling.train_and_evaluate(
                        feature_df, target, selected_features, model_name
                    )
            except modeling.ModelingError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error("Training failed unexpectedly. Try different features or a different model.")
                logger.exception("Unexpected error training model")
            else:
                st.success(result.summary_markdown())

                if result.feature_importance is not None:
                    st.subheader("Feature importance")
                    st.dataframe(result.feature_importance, use_container_width=True)
                    st.bar_chart(result.feature_importance.set_index("feature")["importance"].head(15))

                st.download_button(
                    "⬇ Download trained model (.pkl)",
                    modeling.serialize_model(result),
                    f"{model_name.lower().replace(' ', '_')}_model.pkl",
                    "application/octet-stream",
                )
                st.caption(
                    "Load later with: `pickle.load(open('model.pkl', 'rb'))` → "
                    "dict with keys `pipeline`, `feature_names`, `task_type`, `model_name`."
                )

styling.render_footer()