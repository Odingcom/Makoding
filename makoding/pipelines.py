"""Portable, auditable Makoding data-preparation pipelines (no Streamlit dependency).

A pipeline is declarative JSON, not Python code. Every run is transactional:
failed cleaning steps or validation gates return *no* output dataframe and never
edit the input. Fitting mean/median/mode is computed afresh on each input; for
predictive evaluation use training-fold-only preprocessing instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

import pandas as pd

from makoding.data_processing import apply_clean_step, summarize_quality
from makoding.data_validation import (
    ValidationReport, export_rule_config, import_rule_config, validate_dataframe,
)

SCHEMA_VERSION = 1
MAX_RECIPE_BYTES = 120_000
MAX_STEPS = 30
MAX_REQUIRED_COLUMNS = 500
SUPPORTED_OPERATIONS = frozenset({
    "remove_duplicates", "trim_whitespace", "standardize_missing",
    "fill_missing", "drop_missing_rows", "drop_columns",
})
GATES = frozenset({"none", "critical", "warning"})
STEP_FIELDS = frozenset({"operation", "columns", "strategy", "constant"})
RECIPE_FIELDS = frozenset({
    "schema_version", "name", "required_columns", "steps",
    "validation_rules", "pre_gate", "post_gate",
})


@dataclass(frozen=True)
class PipelineStep:
    operation: str
    columns: tuple[str, ...] = ()
    strategy: str | None = None
    constant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "columns": list(self.columns),
                "strategy": self.strategy, "constant": self.constant}


@dataclass(frozen=True)
class PipelineRecipe:
    name: str
    steps: tuple[PipelineStep, ...]
    required_columns: tuple[str, ...] = ()
    validation_rules: str = ""  # normalized Phase 2 validation configuration JSON
    pre_gate: str = "none"
    post_gate: str = "critical"

    def to_dict(self) -> dict[str, Any]:
        config = json.loads(self.validation_rules or export_rule_config([]))
        return {"schema_version": SCHEMA_VERSION, "name": self.name,
                "required_columns": list(self.required_columns),
                "steps": [step.to_dict() for step in self.steps],
                "validation_rules": config, "pre_gate": self.pre_gate,
                "post_gate": self.post_gate}


@dataclass(frozen=True)
class PipelineRun:
    status: str  # completed, blocked, failed
    output: pd.DataFrame | None
    recipe_name: str
    recipe_sha256: str
    started_at_utc: str
    input_quality: dict[str, Any]
    output_quality: dict[str, Any] | None
    steps: tuple[dict[str, Any], ...]
    pre_validation: ValidationReport | None
    post_validation: ValidationReport | None
    reason: str | None = None

    def audit(self) -> dict[str, Any]:
        """Metadata-only audit; never exports dataset rows or evidence values."""
        def summarized(report: ValidationReport | None) -> dict[str, Any] | None:
            if report is None:
                return None
            return {
                "rules_run": report.rules_run,
                "flagged_checks": report.dataset["flagged_checks"],
                "critical_checks": report.dataset["critical_checks"],
                "warning_checks": report.dataset["warning_checks"],
                "skipped_checks": list(report.skipped_checks),
                "findings": [{"check": f.check, "column": f.column,
                              "severity": f.severity, "affected_rows": f.affected_rows}
                             for f in report.findings],
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "recipe_name": self.recipe_name,
            "recipe_sha256": self.recipe_sha256,
            "started_at_utc": self.started_at_utc,
            "pandas_version": pd.__version__,
            "input_quality": self.input_quality,
            "output_quality": self.output_quality,
            "steps": list(self.steps),
            "pre_validation": summarized(self.pre_validation),
            "post_validation": summarized(self.post_validation),
            "reason": self.reason,
            "note": "Counts are diagnostic; reuse on new datasets recomputes imputation values.",
        }


def _unique_columns(value: Any, *, max_count: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_count:
        raise ValueError(f"Columns must be a list of at most {max_count} names.")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in value):
        raise ValueError("Column names must be nonblank strings of at most 200 characters.")
    if len(set(value)) != len(value):
        raise ValueError("Column names must be unique.")
    return tuple(value)


def _parse_step(item: Any) -> PipelineStep:
    if not isinstance(item, dict) or set(item) != STEP_FIELDS:
        raise ValueError("Each step requires operation, columns, strategy and constant fields.")
    operation = item["operation"]
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unknown pipeline operation: {operation!r}.")
    columns = _unique_columns(item["columns"], max_count=MAX_REQUIRED_COLUMNS)
    strategy = item["strategy"]
    constant = item["constant"]
    if operation == "remove_duplicates":
        if columns or strategy is not None or constant is not None:
            raise ValueError("Duplicate removal cannot have extra parameters.")
    elif operation == "fill_missing":
        if not columns or strategy not in {"mean", "median", "mode", "constant"}:
            raise ValueError("Fill missing requires columns and a supported strategy.")
        if strategy == "constant":
            if not isinstance(constant, str) or not constant or len(constant) > 500:
                raise ValueError("Constant replacement must be 1–500 characters of text.")
        elif constant is not None:
            raise ValueError("Only constant filling accepts a constant value.")
    else:
        if not columns or strategy is not None or constant is not None:
            raise ValueError(f"{operation} requires columns but no filling parameters.")
    return PipelineStep(operation=operation, columns=columns,
                        strategy=strategy, constant=constant)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number {value!r} is not supported.")


def import_recipe(payload: str | bytes) -> PipelineRecipe:
    """Strictly parse bounded, versioned JSON; never execute imported content."""
    if isinstance(payload, bytes):
        if len(payload) > MAX_RECIPE_BYTES:
            raise ValueError("Recipe exceeds 120 KB.")
        try:
            payload = payload.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("Recipe must be valid UTF-8.") from exc
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_RECIPE_BYTES:
        raise ValueError("Recipe must be UTF-8 JSON of at most 120 KB.")
    try:
        data = json.loads(payload, object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=_reject_nonfinite)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid recipe JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != RECIPE_FIELDS:
        raise ValueError("Recipe fields do not match the supported schema.")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported recipe schema version.")
    name = data["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("Recipe name must be 1–100 characters.")
    required = _unique_columns(data["required_columns"], max_count=MAX_REQUIRED_COLUMNS)
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_STEPS:
        raise ValueError(f"Recipe must contain 1–{MAX_STEPS} steps.")
    steps = tuple(_parse_step(s) for s in raw_steps)
    pre_gate, post_gate = data["pre_gate"], data["post_gate"]
    if type(pre_gate) is not str or pre_gate not in GATES or type(post_gate) is not str or post_gate not in GATES:
        raise ValueError("Validation gate must be none, critical or warning.")
    config = data["validation_rules"]
    if not isinstance(config, dict):
        raise ValueError("validation_rules must be a Phase 2 rule configuration object.")
    config_json = json.dumps(config, ensure_ascii=False, allow_nan=False)
    rules, automatic = import_rule_config(config_json)
    if not automatic and not rules:
        raise ValueError("Enable baseline checks or provide at least one validation rule.")
    normalized = export_rule_config(rules, automatic=automatic)
    return PipelineRecipe(name=name, steps=steps, required_columns=required,
                          validation_rules=normalized, pre_gate=pre_gate, post_gate=post_gate)


def export_recipe(recipe: PipelineRecipe) -> str:
    """Validate before exporting so even programmatically built recipes are safe."""
    if not isinstance(recipe, PipelineRecipe):
        raise TypeError("Expected PipelineRecipe.")
    raw = json.dumps(recipe.to_dict(), ensure_ascii=False, allow_nan=False)
    parsed = import_recipe(raw)
    return json.dumps(parsed.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)


def _gate_problem(report: ValidationReport, gate: str) -> str | None:
    if gate == "none":
        return None
    if report.skipped_checks:
        return f"Validation gate cannot pass: {len(report.skipped_checks)} check(s) could not be evaluated."
    critical = report.dataset["critical_checks"]
    warning = report.dataset["warning_checks"]
    if critical or (gate == "warning" and warning):
        return (f"Validation gate blocked: {critical} critical and {warning} warning "
                f"check(s) flagged (gate={gate}).")
    return None


def execute_recipe(frame: pd.DataFrame, recipe: PipelineRecipe) -> PipelineRun:
    """Execute transactionally: only returns output after all checks and steps pass.

    No external code, filesystem writes, network access, or Streamlit state changes.
    All processing uses the existing `apply_clean_step` implementation.
    """
    normalized = import_recipe(export_recipe(recipe))
    canonical = json.dumps(normalized.to_dict(), sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False)
    recipe_hash = sha256(canonical.encode("utf-8")).hexdigest()
    before = summarize_quality(frame)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    pre: ValidationReport | None = None
    post: ValidationReport | None = None
    rules, automatic = import_rule_config(normalized.validation_rules)

    def result(status: str, reason: str | None = None, output: pd.DataFrame | None = None) -> PipelineRun:
        return PipelineRun(status=status, output=output, recipe_name=normalized.name,
                           recipe_sha256=recipe_hash, started_at_utc=started,
                           input_quality=before,
                           output_quality=summarize_quality(output) if output is not None else None,
                           steps=tuple(records), pre_validation=pre, post_validation=post,
                           reason=reason)

    absent = [c for c in normalized.required_columns if c not in frame.columns]
    if absent:
        return result("blocked", f"Required input columns are missing: {absent}.")
    try:
        pre = validate_dataframe(frame, rules, automatic=automatic, max_examples=0)
    except (ValueError, TypeError, OverflowError) as exc:
        return result("blocked", f"Pre-validation could not run: {exc}")
    pre_problem = _gate_problem(pre, normalized.pre_gate)
    if pre_problem:
        return result("blocked", "Pre-validation: " + pre_problem)

    working = frame.copy(deep=True)
    for index, step in enumerate(normalized.steps, start=1):
        try:
            proposed, detail = apply_clean_step(
                working, step.operation, columns=step.columns,
                strategy=step.strategy, constant=step.constant,
            )
        except (ValueError, TypeError, OverflowError, KeyError) as exc:
            records.append({"number": index, "operation": step.operation,
                            "status": "failed", "error": str(exc)})
            return result("failed", f"Step {index} ({step.operation}) failed: {exc}")
        records.append({"number": index, "operation": step.operation,
                        "columns": list(step.columns), "status": "applied",
                        "before": detail["before"], "after": detail["after"],
                        "removed_rows": detail["removed_rows"],
                        "removed_columns": detail["removed_columns"],
                        "missing_reduced": detail["missing_reduced"]})
        working = proposed

    try:
        post = validate_dataframe(working, rules, automatic=automatic, max_examples=0)
    except (ValueError, TypeError, OverflowError) as exc:
        return result("blocked", f"Post-validation could not run: {exc}")
    post_problem = _gate_problem(post, normalized.post_gate)
    if post_problem:
        return result("blocked", "Post-validation: " + post_problem)
    return result("completed", output=working)
