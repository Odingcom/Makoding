"""Phase 3: declarative recipe builder, preview and explicit transactional apply."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd
import streamlit as st

from makoding.data_processing import summarize_quality
from makoding.pipeline_pdf import export_recipe_pdf, import_recipe_pdf, export_audit_pdf
from makoding.data_validation import export_rule_config, import_rule_config
from makoding.pipelines import (
    MAX_STEPS, PipelineRecipe, PipelineStep, execute_recipe, export_recipe, import_recipe,
)
from makoding.ui.cleaning_studio import MAX_UNDO, _event, _invalidate_model_outputs


OPERATIONS = {
    "Remove duplicate rows": "remove_duplicates",
    "Trim text whitespace": "trim_whitespace",
    "Standardize missing-value text tokens": "standardize_missing",
    "Fill missing values": "fill_missing",
    "Drop rows with missing values": "drop_missing_rows",
    "Drop selected columns": "drop_columns",
}
GATE_CHOICES = {
    "None (report only)": "none",
    "Block on critical findings": "critical",
    "Block on warnings or critical findings": "warning",
}
DEFAULT_RULES = export_rule_config([], automatic=True)


def _initialize_state() -> None:
    defaults = {
        "pipeline_steps": [],
        "pipeline_name": "Makoding cleaning recipe",
        "pipeline_required_text": "",
        "pipeline_rule_json": DEFAULT_RULES,
        "pipeline_pre_gate": "none",
        "pipeline_post_gate": "critical",
        "pipeline_pre_label": "None (report only)",
        "pipeline_post_label": "Block on critical findings",
        "pipeline_preview": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    source = st.session_state.get("processing_source_key")
    if st.session_state.get("pipeline_bound_source") != source:
        st.session_state["pipeline_bound_source"] = source
        st.session_state["pipeline_preview"] = None
        # Clear stale *widget selections*, not the portable recipe, when a new
        # source has different columns. Do this before widgets are rendered.
        for key in list(st.session_state):
            if key.startswith("pipeline_cols_"):
                del st.session_state[key]
        # Recipe is deliberately retained for reuse on another dataset.
        # Required columns and rule compatibility are rechecked at preview.


def _recipe() -> PipelineRecipe:
    required = [s.strip() for s in st.session_state["pipeline_required_text"].splitlines() if s.strip()]
    return PipelineRecipe(
        name=st.session_state["pipeline_name"],
        required_columns=tuple(required),
        steps=tuple(st.session_state["pipeline_steps"]),
        validation_rules=st.session_state["pipeline_rule_json"],
        pre_gate=st.session_state["pipeline_pre_gate"],
        post_gate=st.session_state["pipeline_post_gate"],
    )


def _signature(payload: str, input_mode: str) -> tuple:
    return (payload, input_mode, str(st.session_state.get("processing_source_key")),
            len(st.session_state.get("processing_events", [])))


def _render_step_builder(working: pd.DataFrame) -> None:
    st.markdown("#### Build processing steps")
    label = st.selectbox("Transformation", list(OPERATIONS), key="pipeline_operation")
    op = OPERATIONS[label]
    all_cols = [str(c) for c in working.columns]
    text_cols = [str(c) for c in working.columns
                 if pd.api.types.is_object_dtype(working[c]) or
                 pd.api.types.is_string_dtype(working[c])]
    numeric_cols = [str(c) for c in working.select_dtypes(include="number").columns
                    if not pd.api.types.is_bool_dtype(working[c])]
    strategy = None
    constant = None
    selected: list[str] = []
    if op == "fill_missing":
        strategy = st.selectbox("Fill strategy", ["median", "mean", "mode", "constant"],
                                key="pipeline_fill_strategy")
        eligible = numeric_cols if strategy in {"median", "mean"} else (text_cols if strategy == "constant" else all_cols)
        selected = st.multiselect("Columns", eligible, key=f"pipeline_cols_fill_{strategy}")
        if strategy == "constant":
            constant = st.text_input("Replacement text", key="pipeline_fill_constant")
        st.caption("Each execution recalculates imputation values on its input. "
                   "Do not use whole-dataset imputation as a training-fold preprocessing pipeline.")
    elif op != "remove_duplicates":
        eligible = text_cols if op in {"trim_whitespace", "standardize_missing"} else all_cols
        selected = st.multiselect("Columns", eligible, key=f"pipeline_cols_{op}")
    if st.button("Add processing step", key="pipeline_add_step"):
        try:
            steps = st.session_state["pipeline_steps"]
            if len(steps) >= MAX_STEPS:
                raise ValueError(f"A recipe supports at most {MAX_STEPS} steps.")
            candidate = PipelineStep(op, tuple(selected), strategy, constant)
            # Validate proposed recipe before accepting it.
            candidate_recipe = PipelineRecipe(
                name="Step validation", steps=(candidate,),
                validation_rules=DEFAULT_RULES,
            )
            export_recipe(candidate_recipe)
            steps.append(candidate)
            st.session_state["pipeline_preview"] = None
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(f"Step not added: {exc}")

    steps = st.session_state["pipeline_steps"]
    if steps:
        st.dataframe(pd.DataFrame([
            {"Step": i + 1, "Action": step.operation,
             "Columns": ", ".join(step.columns) or "(entire dataset)",
             "Strategy": step.strategy or "—"}
            for i, step in enumerate(steps)
        ]), use_container_width=True, height=min(330, 90 + 35 * len(steps)))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Remove last step", key="pipeline_remove_last"):
                steps.pop()
                st.session_state["pipeline_preview"] = None
                st.rerun()
        with c2:
            if st.button("Clear recipe steps", key="pipeline_clear_steps"):
                st.session_state["pipeline_steps"] = []
                st.session_state["pipeline_preview"] = None
                st.rerun()
    else:
        st.info("Add at least one processing step. All changes remain a preview until applied.")


def _show_report(report, label: str) -> None:
    if report is None:
        return
    data = report.dataset
    st.caption(f"{label}: {report.rules_run} checks; "
               f"{data['critical_checks']} critical, {data['warning_checks']} warnings; "
               f"{len(report.skipped_checks)} skipped checks.")
    if report.findings:
        with st.expander(f"{label} flagged checks ({len(report.findings)})"):
            st.dataframe(pd.DataFrame([
                {"Severity": f.severity, "Check": f.check, "Column": f.column,
                 "Affected rows": f.affected_rows}
                for f in report.findings
            ]), use_container_width=True, height=220)
    if report.skipped_checks:
        with st.expander(f"{label} skipped checks"):
            for item in report.skipped_checks:
                st.write("• " + item)


def render_pipeline_studio(original_frame: pd.DataFrame) -> None:
    """Render reusable recipe editor for the existing Data Cleaning workspace."""
    _initialize_state()
    working = st.session_state["working_dataframe"]
    st.subheader("Reproducible Processing Pipelines")
    st.caption("Save reusable JSON or PDF recipes, preview the whole workflow, and apply it only after "
               "validation checks pass. Neither preview nor a blocked run modifies your data.")

    with st.expander("Import an existing recipe", expanded=False):
        uploaded = st.file_uploader("Recipe JSON or Makoding recipe PDF", type=["json", "pdf"],
                                    key="pipeline_upload")
        st.caption("PDF import works for PDFs exported by Makoding. The app reads the embedded JSON recipe; "
                   "a scanned or ordinary PDF cannot be imported as a recipe.")
        if uploaded is not None and st.button("Load recipe", key="pipeline_load"):
            try:
                if uploaded.name.lower().endswith(".pdf"):
                    recipe = import_recipe_pdf(uploaded.getvalue())
                else:
                    recipe = import_recipe(uploaded.getvalue())
                st.session_state["pipeline_steps"] = list(recipe.steps)
                st.session_state["pipeline_name"] = recipe.name
                st.session_state["pipeline_required_text"] = "\n".join(recipe.required_columns)
                st.session_state["pipeline_rule_json"] = recipe.validation_rules
                st.session_state["pipeline_pre_gate"] = recipe.pre_gate
                st.session_state["pipeline_post_gate"] = recipe.post_gate
                st.session_state["pipeline_pre_label"] = next(k for k, v in GATE_CHOICES.items() if v == recipe.pre_gate)
                st.session_state["pipeline_post_label"] = next(k for k, v in GATE_CHOICES.items() if v == recipe.post_gate)
                st.session_state["pipeline_preview"] = None
                st.rerun()
            except (ValueError, TypeError, UnicodeError) as exc:
                st.error(f"Recipe could not be imported: {exc}")

    st.text_input("Recipe name", key="pipeline_name")
    st.text_area("Required input columns (one name per line, optional)",
                 key="pipeline_required_text", height=95,
                 help="Missing required columns stop a run before any transformation.")
    _render_step_builder(working)

    st.markdown("#### Validation gates")
    st.caption("Use Phase 2 validation rules to check both input and output. "
               "A gate can block all changes when configured findings remain. "
               "Skipped checks also block a non-None gate.")

    with st.expander("Select validation rules"):
        try:
            current_rules, auto = import_rule_config(st.session_state["pipeline_rule_json"])
            st.caption(f"Recipe contains {len(current_rules)} custom rule(s); "
                       f"automatic baseline checks: {'on' if auto else 'off'}.")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
        config_upload = st.file_uploader("Phase 2 validation-rule JSON", type=["json"],
                                         key="pipeline_rule_upload")
        if config_upload is not None and st.button("Use uploaded validation rules", key="pipeline_import_rules"):
            try:
                specs, enabled = import_rule_config(config_upload.getvalue())
                st.session_state["pipeline_rule_json"] = export_rule_config(specs, automatic=enabled)
                st.session_state["pipeline_preview"] = None
                st.rerun()
            except (ValueError, TypeError, UnicodeError) as exc:
                st.error(str(exc))
        if st.button("Capture current Data Validation tab rules", key="pipeline_capture_rules"):
            bound = st.session_state.get("validation_bound_source")
            source = str(st.session_state.get("processing_source_key"))
            if bound != source:
                st.warning("Open the Data Validation tab for this dataset first, "
                           "or import its exported rule configuration here.")
            else:
                st.session_state["pipeline_rule_json"] = export_rule_config(
                    st.session_state.get("validation_specs", []),
                    automatic=st.session_state.get("validation_auto_checks", True),
                )
                st.session_state["pipeline_preview"] = None
                st.rerun()

    c1, c2 = st.columns(2)
    gate_names = list(GATE_CHOICES)
    with c1:
        st.selectbox("Before cleaning", gate_names, key="pipeline_pre_label")
    with c2:
        st.selectbox("After cleaning", gate_names, key="pipeline_post_label")
    # Keep actual gate values synchronized with visible controls, without
    # writing into an already-instantiated widget key.
    st.session_state["pipeline_pre_gate"] = GATE_CHOICES[st.session_state["pipeline_pre_label"]]
    st.session_state["pipeline_post_gate"] = GATE_CHOICES[st.session_state["pipeline_post_label"]]

    with st.expander("Save recipe / inspect source options"):
        st.caption("Recipe JSON contains transformation parameters and validation rules, never dataset records. "
                   "Replacement constants and category names may themselves be sensitive.")
        try:
            recipe_json = export_recipe(_recipe())
        except (ValueError, TypeError) as exc:
            st.info(f"Add/complete steps to export a valid recipe: {exc}")
        else:
            col_json, col_pdf = st.columns(2)
            with col_json:
                st.download_button("Download recipe (JSON)", data=recipe_json,
                                   file_name="makoding_pipeline_recipe.json", mime="application/json",
                                   key="pipeline_recipe_download", use_container_width=True)
            with col_pdf:
                try:
                    recipe_pdf = export_recipe_pdf(_recipe())
                except (ValueError, RuntimeError) as exc:
                    st.warning(f"Recipe PDF unavailable: {exc}")
                else:
                    st.download_button("Download recipe (PDF - importable)",
                                       data=recipe_pdf,
                                       file_name="makoding_pipeline_recipe.pdf", mime="application/pdf",
                                       key="pipeline_recipe_pdf_download", use_container_width=True)
            st.caption("The PDF is readable and carries the canonical recipe JSON as an attachment. "
                       "Do not edit it in a PDF editor if you intend to re-import changed settings; "
                       "change the recipe here and export again.")

    mode = st.radio("Pipeline input", ["Current working dataset", "Original uploaded dataset"],
                    key="pipeline_input_mode", horizontal=True)
    if mode == "Original uploaded dataset":
        st.warning("Applying this preview will replace the current working dataset with "
                   "a result based on the original upload. Undo can restore the previous working dataset.")
    source_frame = original_frame if mode == "Original uploaded dataset" else working

    st.markdown("#### Preview complete pipeline")
    if st.button("Run pipeline preview", type="primary", key="pipeline_preview_run"):
        try:
            recipe = _recipe()
            payload = export_recipe(recipe)
            result = execute_recipe(source_frame, recipe)
            st.session_state["pipeline_preview"] = (_signature(payload, mode), result)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state["pipeline_preview"] = None
            st.error(f"Recipe is invalid: {exc}")

    pending = st.session_state.get("pipeline_preview")
    try:
        current_json = export_recipe(_recipe())
        expected = _signature(current_json, mode)
    except (ValueError, TypeError):
        expected = None
    if pending is not None and expected == pending[0]:
        result = pending[1]
        if result.status != "completed":
            st.error(f"Pipeline {result.status}: {result.reason}")
            st.info("No dataset changes were applied. Fix the recipe/rules and preview again.")
        else:
            before, after = result.input_quality, result.output_quality
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", after["rows"], delta=after["rows"] - before["rows"])
            c2.metric("Columns", after["columns"], delta=after["columns"] - before["columns"])
            c3.metric("Missing cells", after["missing_cells"],
                      delta=after["missing_cells"] - before["missing_cells"])
            st.dataframe(result.output.head(25), use_container_width=True)
            if before["rows"] and (before["rows"] - after["rows"]) / before["rows"] > 0.10:
                st.warning("More than 10% of the input rows were removed. Review carefully.")
            st.success("All steps and required validation gates passed. This is still only a preview.")
            st.download_button(
                "Download preview result (CSV)",
                data=result.output.to_csv(index=False).encode("utf-8-sig"),
                file_name="makoding_pipeline_preview.csv", mime="text/csv",
                use_container_width=True, key="pipeline_preview_csv",
            )
        _show_report(result.pre_validation, "Before")
        _show_report(result.post_validation, "After")
        with st.expander("Pipeline execution audit"):
            st.json(result.audit())
        cjson, cpdf = st.columns(2)
        with cjson:
            st.download_button("Download execution audit (JSON)",
                               data=json.dumps(result.audit(), indent=2, ensure_ascii=False),
                               file_name="makoding_pipeline_audit.json", mime="application/json",
                               key="pipeline_audit_download", use_container_width=True)
        with cpdf:
            try:
                audit_pdf = export_audit_pdf(result.audit())
            except (ValueError, RuntimeError) as exc:
                st.warning(f"Audit PDF unavailable: {exc}")
            else:
                st.download_button("Download execution audit (PDF)",
                                   data=audit_pdf,
                                   file_name="makoding_pipeline_audit.pdf", mime="application/pdf",
                                   key="pipeline_audit_pdf_download", use_container_width=True)
        if result.status == "completed":
            if st.button("Apply entire pipeline to working dataset", type="primary", key="pipeline_apply"):
                undo = st.session_state.setdefault("processing_undo", [])
                undo.append(working.copy(deep=True))
                if len(undo) > MAX_UNDO:
                    undo.pop(0)
                st.session_state["working_dataframe"] = result.output.copy(deep=True)
                st.session_state["processing_preview"] = None
                st.session_state["pipeline_preview"] = None
                st.session_state["validation_result"] = None
                _invalidate_model_outputs()
                _event("apply_pipeline", {
                    "recipe_name": result.recipe_name,
                    "recipe_sha256": result.recipe_sha256,
                    "steps": len(result.steps),
                    "before": summarize_quality(working),
                    "after": result.output_quality,
                    "input_source": mode,
                })
                st.rerun()
    elif pending is not None:
        st.info("The recipe, dataset, or input selection has changed. Run a new preview before applying.")
    else:
        st.caption("Run a preview to inspect proposed changes and validation results before applying them.")
