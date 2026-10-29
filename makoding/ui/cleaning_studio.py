"""Data Cleaning Studio: inspect, preview, apply, undo, and export changes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from makoding.data_processing import apply_clean_step, profile_columns, summarize_quality

MAX_UNDO = 5


def _invalidate_model_outputs() -> None:
    """Prevent displaying models trained on an older working dataset."""
    for key in (
        "model_result", "model_features", "model_target", "model_task",
        "model_y_test", "model_predictions", "model_probabilities",
    ):
        if key in st.session_state:
            st.session_state[key] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _report() -> None:
    """Provide a lightweight summary to the existing HTML report exporter."""
    events = st.session_state.get("processing_events", [])
    st.session_state["cleaning_report"] = (
        json.dumps(events, indent=2, default=str) if events else None
    )


def ensure_processing_state(source_key, original_frame: pd.DataFrame) -> None:
    """Keep pristine upload separate from applied transformations."""
    if (st.session_state.get("processing_source_key") != source_key or
            "working_dataframe" not in st.session_state):
        st.session_state["processing_source_key"] = source_key
        st.session_state["working_dataframe"] = original_frame.copy(deep=True)
        st.session_state["processing_undo"] = []
        st.session_state["processing_events"] = []
        st.session_state["processing_preview"] = None
        _report()


def _event(action: str, detail: dict | None = None) -> None:
    events = st.session_state["processing_events"]
    events.append({"at_utc": _now(), "action": action, **(detail or {})})
    _report()


def render_cleaning_studio(original_frame: pd.DataFrame) -> None:
    """Render an explicit cleaning workspace with non-destructive previews."""
    working = st.session_state["working_dataframe"]
    baseline = summarize_quality(original_frame)
    current = summarize_quality(working)

    st.subheader("Data Cleaning Studio")
    st.caption(
        "The uploaded dataset stays unchanged. Preview an operation, then Apply "
        "it to the working dataset used by analysis, models and exports."
    )
    st.markdown("#### Original versus working data")
    left, right = st.columns(2)
    for column, name, values in (
        (left, "Original", baseline), (right, "Working", current)
    ):
        with column:
            st.markdown(f"**{name}**")
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{values['rows']:,}")
            c2.metric("Columns", f"{values['columns']:,}")
            c1.metric("Missing cells", f"{values['missing_cells']:,}")
            dup = values["duplicate_rows"]
            c2.metric("Duplicates", "n/a" if dup is None else f"{dup:,}")

    with st.expander("Column quality profile", expanded=True):
        st.dataframe(profile_columns(working), use_container_width=True, height=270)

    operation_labels = {
        "Remove duplicate rows": "remove_duplicates",
        "Trim whitespace in text": "trim_whitespace",
        "Convert missing-value text tokens to null": "standardize_missing",
        "Fill missing values": "fill_missing",
        "Drop rows with missing values": "drop_missing_rows",
        "Remove columns": "drop_columns",
    }
    st.markdown("#### Prepare a change")
    label = st.selectbox("Operation", list(operation_labels), key="studio_operation")
    operation = operation_labels[label]
    text_columns = [
        str(column) for column in working.columns
        if (pd.api.types.is_object_dtype(working[column]) or
            pd.api.types.is_string_dtype(working[column]))
    ]
    numeric_columns = [
        str(column) for column in working.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(working[column])
    ]
    columns = []
    strategy = None
    constant = None

    if operation in {"trim_whitespace", "standardize_missing"}:
        columns = st.multiselect("Text columns", text_columns, key="studio_text_columns")
        if operation == "standardize_missing":
            st.caption('Recognized tokens: blank, NA, N/A, null, none, missing, nan, ?')
    elif operation == "fill_missing":
        strategy_label = st.selectbox(
            "Replacement strategy",
            ["Median (numeric)", "Mean (numeric)", "Mode (any column)", "Constant (text)"],
            key="studio_fill_strategy",
        )
        strategy = {
            "Median (numeric)": "median",
            "Mean (numeric)": "mean",
            "Mode (any column)": "mode",
            "Constant (text)": "constant",
        }[strategy_label]
        eligible = (numeric_columns if strategy in {"median", "mean"}
                    else text_columns if strategy == "constant"
                    else [str(col) for col in working.columns])
        columns = st.multiselect("Columns to fill", eligible, key=f"studio_fill_cols_{strategy}")
        if strategy == "constant":
            constant = st.text_input("Replacement text", key="studio_fill_constant")
        st.caption(
            "For unbiased predictive-model evaluation, learn imputation values "
            "from the training split only. This studio applies values to the whole dataset."
        )
    elif operation == "drop_missing_rows":
        eligible = [str(col) for col in working.columns if working[col].isna().any()]
        columns = st.multiselect("Check these columns", eligible, key="studio_drop_missing_cols")
    elif operation == "drop_columns":
        columns = st.multiselect(
            "Columns to remove", [str(col) for col in working.columns], key="studio_drop_cols"
        )

    signature = (
        len(st.session_state["processing_events"]), label,
        tuple(columns), strategy, constant,
    )
    if st.button("Preview changes", type="secondary", key="studio_preview_button"):
        try:
            proposal, detail = apply_clean_step(
                working, operation, columns=columns, strategy=strategy, constant=constant
            )
            st.session_state["processing_preview"] = (signature, proposal, detail)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state["processing_preview"] = None
            st.error(str(exc))

    pending = st.session_state.get("processing_preview")
    if pending and pending[0] == signature:
        _, proposal, detail = pending
        st.markdown("#### Proposed result — not yet applied")
        before = detail["before"]
        after = detail["after"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{after['rows']:,}", delta=after["rows"] - before["rows"])
        c2.metric("Columns", f"{after['columns']:,}", delta=after["columns"] - before["columns"])
        c3.metric("Missing cells", f"{after['missing_cells']:,}",
                  delta=after["missing_cells"] - before["missing_cells"])
        st.dataframe(proposal.head(25), use_container_width=True)
        if before["rows"] and detail["removed_rows"] / before["rows"] > 0.1:
            st.warning("This will remove more than 10% of the current rows. Review carefully.")
        if st.button("Apply to working dataset", type="primary", key="studio_apply_button"):
            undo = st.session_state["processing_undo"]
            undo.append(working.copy(deep=True))
            if len(undo) > MAX_UNDO:
                undo.pop(0)
            st.session_state["working_dataframe"] = proposal
            st.session_state["processing_preview"] = None
            _invalidate_model_outputs()
            _event("apply", detail)
            st.rerun()
    elif pending:
        st.info("Settings changed. Preview again before applying.")

    st.markdown("#### Recovery and audit")
    undo_col, reset_col = st.columns(2)
    with undo_col:
        if st.button("Undo last applied change", disabled=not st.session_state["processing_undo"]):
            st.session_state["working_dataframe"] = st.session_state["processing_undo"].pop()
            st.session_state["processing_preview"] = None
            _invalidate_model_outputs()
            _event("undo")
            st.rerun()
    with reset_col:
        if st.button("Reset to original data", disabled=not st.session_state["processing_events"]):
            st.session_state["working_dataframe"] = original_frame.copy(deep=True)
            st.session_state["processing_undo"] = []
            st.session_state["processing_preview"] = None
            _invalidate_model_outputs()
            _event("reset_to_original")
            st.rerun()

    if st.session_state["processing_events"]:
        st.dataframe(pd.DataFrame(st.session_state["processing_events"])[["at_utc", "action"]],
                     use_container_width=True)
    csv_col, log_col = st.columns(2)
    with csv_col:
        st.download_button(
            "Download working data (CSV)",
            data=working.to_csv(index=False).encode("utf-8-sig"),
            file_name="makoding_working_data.csv", mime="text/csv",
            use_container_width=True, key="studio_download_csv",
        )
    with log_col:
        st.download_button(
            "Download audit trail (JSON)",
            data=json.dumps(st.session_state["processing_events"], indent=2, default=str),
            file_name="makoding_cleaning_audit.json", mime="application/json",
            use_container_width=True, key="studio_download_audit",
        )
