"""Streamlit interface for non-destructive Data Quality Validation Studio."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from makoding.data_validation import (
    DATE_FORMATS, Rule, export_rule_config, import_rule_config, validate_dataframe,
)

# Only a small, stable set of Streamlit APIs: compatible with the existing app.
RULE_OPTIONS = {
    "Required column (schema)": "required_column",
    "Expected column data type": "dtype",
    "Required field": "required",
    "Unique identifier": "unique",
    "Numeric values": "numeric",
    "Date values": "date",
    "Numeric range": "range",
    "Approved categories": "allowed_values",
    "IQR outliers": "iqr_outlier",
    "Z-score outliers": "zscore_outlier",
}


def _get_config_key() -> tuple:
    # incremented whenever data changes, including undo/reset; avoids stale scans
    return (
        str(st.session_state.get("processing_source_key")),
        len(st.session_state.get("processing_events", [])),
    )


def _active_rules() -> list[Rule]:
    if "validation_specs" not in st.session_state:
        st.session_state["validation_specs"] = []
    return st.session_state["validation_specs"]


def render_validation_studio(working: pd.DataFrame) -> None:
    """Render rule builder and a dataset-specific quality report.

    This page intentionally never edits working_dataframe. Validation is separate
    from the Cleaning Studio's preview/apply/undo operations.
    """
    st.subheader("Data Quality Validation")
    st.caption(
        "Check your working dataset before statistical analysis or modelling. "
        "Validation flags potential issues; it never changes, drops, or imputes observations."
    )
    st.info(
        "Detected issues are not automatically errors. For example, an outlier may be "
        "a genuine observation, and a category may be a new valid value."
    )

    # New uploaded datasets do not inherit stale rules or previous scan results.
    source_key = str(st.session_state.get("processing_source_key"))
    if st.session_state.get("validation_bound_source") != source_key:
        st.session_state["validation_bound_source"] = source_key
        st.session_state["validation_specs"] = []
        st.session_state["validation_result"] = None

    # Streamlit widget keys may only be changed *before* widget creation.
    if "validation_auto_override" in st.session_state:
        st.session_state["validation_auto_checks"] = st.session_state.pop("validation_auto_override")

    specs = _active_rules()
    automatic = st.checkbox(
        "Run automatic baseline checks (missing, duplicates, whitespace, text markers and non-finite values)",
        value=True, key="validation_auto_checks",
    )

    with st.expander("Add a validation rule", expanded=True):
        if not len(working.columns):
            st.warning("No columns are available for validation.")
            return
        names = [str(col) for col in working.columns]
        c1, c2 = st.columns(2)
        with c1:
            selected_label = st.selectbox("Check", list(RULE_OPTIONS), key="validation_rule_kind")
            if RULE_OPTIONS[selected_label] == "required_column":
                column = st.text_input("Required column name", key="validation_required_column", placeholder="member_id").strip()
            else:
                column = st.selectbox("Column", names, key="validation_column")
        with c2:
            severity = st.selectbox("Severity", ["warning", "critical", "info"], key="validation_severity")
        kind = RULE_OPTIONS[selected_label]
        params: dict = {}
        error = None
        if kind == "dtype":
            params["expected"] = st.selectbox(
                "Expected column type", ["numeric", "text", "datetime", "boolean"],
                key="validation_expected_dtype",
            )
        elif kind == "required":
            params["include_text_tokens"] = st.checkbox(
                "Also treat common text markers as missing", value=True, key="validation_required_tokens")
        elif kind == "date":
            date_format = st.selectbox(
                "Expected date format",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"],
                key="validation_date_format",
            )
            params["format"] = date_format
        elif kind == "range":
            lower, upper = st.columns(2)
            with lower:
                use_min = st.checkbox("Set minimum", value=True, key="validation_use_min")
                minimum = st.number_input("Minimum", value=0.0, key="validation_min")
            with upper:
                use_max = st.checkbox("Set maximum", value=False, key="validation_use_max")
                maximum = st.number_input("Maximum", value=100.0, key="validation_max")
            params = {"min": float(minimum) if use_min else None,
                      "max": float(maximum) if use_max else None}
        elif kind == "allowed_values":
            # Separate lines instead of CSV to support category values containing commas.
            categories = st.text_area(
                "Approved values (one value per line)", key="validation_categories",
                placeholder="Nairobi\nKisumu\nMombasa", height=120,
            )
            params["values"] = [v.strip() for v in categories.splitlines() if v.strip()]
            params["case_sensitive"] = st.checkbox("Case-sensitive", value=False, key="validation_case")
            params["strip_whitespace"] = st.checkbox(
                "Ignore surrounding whitespace", value=True, key="validation_strip")
        elif kind == "iqr_outlier":
            params["multiplier"] = st.number_input(
                "IQR fence multiplier", min_value=0.1, value=1.5, step=0.1,
                key="validation_iqr_multiplier")
        elif kind == "zscore_outlier":
            params["threshold"] = st.number_input(
                "Absolute Z-score threshold", min_value=0.1, value=3.0, step=0.1,
                key="validation_zscore_threshold")
        if st.button("Add rule", key="validation_add_rule"):
            candidate = Rule(kind=kind, column=column, severity=severity, params=params)
            try:
                # Verify the new rule against the current working dataset.
                validate_dataframe(working, [candidate], automatic=False, max_examples=0)
                if len(specs) >= 100:
                    raise ValueError("A maximum of 100 rules is supported.")
                specs.append(candidate)
                st.session_state["validation_result"] = None
                st.success(f"Added: {selected_label} / {column}.")
            except (ValueError, TypeError, OverflowError) as exc:
                error = str(exc)
        if error:
            st.error(error)

    st.markdown("#### Active rules")
    if specs:
        st.dataframe(pd.DataFrame([
            {"#": i+1, "Check": r.kind.replace("_", " ").title(),
             "Column": r.column, "Severity": r.severity,
             "Parameters": json.dumps(r.params, ensure_ascii=False)}
            for i, r in enumerate(specs)
        ]), use_container_width=True, height=220)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Remove last rule", key="validation_remove_rule"):
                specs.pop()
                st.session_state["validation_result"] = None
                st.rerun()
        with c2:
            if st.button("Clear all rules", key="validation_clear_rules"):
                specs.clear()
                st.session_state["validation_result"] = None
                st.rerun()
    else:
        st.caption("No custom rules yet. Automatic checks can be run without custom rules.")

    with st.expander("Save or load validation rules"):
        st.download_button(
            "Export rule configuration (JSON)",
            data=export_rule_config(specs, automatic=automatic),
            file_name="makoding_validation_rules.json", mime="application/json",
            key="validation_export_rules", use_container_width=True,
        )
        uploaded = st.file_uploader(
            "Import an exported rule configuration", type=["json"],
            key="validation_import_upload",
        )
        if uploaded is not None and st.button("Import rule configuration", key="validation_import_button"):
            try:
                loaded, auto_value = import_rule_config(uploaded.getvalue())
                # Validate all rules in the new dataset *before* changing app state.
                validate_dataframe(working, loaded, automatic=False, max_examples=0)
                st.session_state["validation_specs"] = loaded
                st.session_state["validation_auto_override"] = auto_value
                st.session_state["validation_result"] = None
                st.rerun()
            except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
                st.error(f"Configuration could not be imported: {exc}")

    st.markdown("#### Run validation")
    st.caption("The report uses the **currently applied working dataset**, not a pending cleaning preview.")
    if st.button("Scan working dataset", type="primary", key="validation_run_scan"):
        try:
            report = validate_dataframe(working, specs, automatic=automatic, max_examples=5)
            signature = (_get_config_key(), export_rule_config(specs, automatic=automatic))
            st.session_state["validation_result"] = (signature, report)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state["validation_result"] = None
            st.error(str(exc))

    result = st.session_state.get("validation_result")
    signature = (_get_config_key(), export_rule_config(specs, automatic=automatic))
    if result and result[0] == signature:
        report = result[1]
        st.markdown("#### Validation findings")
        metrics = st.columns(4)
        metrics[0].metric("Rows scanned", f"{report.dataset['rows']:,}")
        metrics[1].metric("Checks run", f"{report.rules_run:,}")
        metrics[2].metric("Flagged checks", f"{report.dataset['flagged_checks']:,}")
        metrics[3].metric("Critical checks", f"{report.dataset['critical_checks']:,}")
        if report.skipped_checks:
            st.warning(
                f"{len(report.skipped_checks)} check(s) could not be evaluated. "
                "Review the reasons below before interpreting the results."
            )
            with st.expander("Skipped checks", expanded=True):
                for note in report.skipped_checks:
                    st.write("• " + note)
        if report.findings:
            st.dataframe(report.table(), use_container_width=True, height=330)
            st.caption("Affected-row counts can overlap between checks; do not add them as unique-row totals.")
            labels = [f"{i+1}. [{f.severity}] {f.column} — {f.check}" for i, f in enumerate(report.findings)]
            selected = st.selectbox("Inspect sample evidence", range(len(labels)),
                                     format_func=lambda i: labels[i], key="validation_finding_choice")
            finding = report.findings[selected]
            st.write(finding.message)
            st.caption(finding.recommendation)
            if finding.examples:
                st.dataframe(pd.DataFrame(finding.examples), use_container_width=True)
            else:
                st.caption("No sample rows are attached to this type of suggestion.")
        else:
            if report.skipped_checks:
                st.info("No issues detected among evaluated checks; skipped checks require review.")
            else:
                st.success("No issues detected under the checks that were run. This is not a guarantee of correctness.")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download validation report (JSON)",
                data=json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                file_name="makoding_validation_report.json", mime="application/json",
                use_container_width=True, key="validation_download_report",
            )
        with c2:
            st.download_button(
                "Download findings (CSV)",
                data=report.table().to_csv(index=False).encode("utf-8-sig"),
                file_name="makoding_validation_findings.csv", mime="text/csv",
                use_container_width=True, key="validation_download_findings",
            )
        st.caption("Report downloads include up to five original values per flagged check; treat them as potentially sensitive.")
        st.warning(
            "Statistical outlier thresholds are exploratory and not proof of invalid data. "
            "Rule configurations are reusable, but their suitability must be rechecked for each dataset."
        )
    elif result:
        st.info("The working data or validation configuration changed. Run a fresh scan.")
    else:
        st.caption("Set up rules above, then run a scan to see and export findings.")


def render_validation_notice() -> None:
    """Brief quality context for modelling/statistics without blocking research."""
    result = st.session_state.get("validation_result")
    if not result:
        st.caption("Data quality: no validation scan has been completed for this working dataset. Use Clean → Data Validation.")
        return
    try:
        current = (_get_config_key(), export_rule_config(
            st.session_state.get("validation_specs", []),
            automatic=st.session_state.get("validation_auto_checks", True),
        ))
    except (TypeError, ValueError):
        st.caption("Data quality: validation configuration needs attention. Use Clean → Data Validation.")
        return
    if result[0] != current:
        st.caption("Data quality: the previous scan is out of date. Re-scan in Clean → Data Validation.")
        return
    report = result[1]
    critical = report.dataset["critical_checks"]
    warnings = report.dataset["warning_checks"]
    if critical:
        st.warning(
            f"Data quality: {critical} critical check(s) flagged. "
            "Review Clean → Data Validation before using results for decisions."
        )
    elif warnings:
        st.info(
            f"Data quality: {warnings} warning check(s) flagged. "
            "Review Clean → Data Validation to inspect evidence."
        )
    else:
        st.caption("Data quality: no critical/warning findings under the most recent configured checks.")
