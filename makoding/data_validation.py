"""Non-destructive, auditable, rule-based data-quality validation.

No Streamlit dependency. Checks never modify their input DataFrame. A failed rule
returns counts and bounded evidence, rather than deleting or fixing records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
RULE_KINDS = frozenset({
    "required_column", "dtype", "required", "unique", "numeric", "date", "range", "allowed_values",
    "iqr_outlier", "zscore_outlier",
})
SEVERITIES = frozenset({"critical", "warning", "info"})
DATE_FORMATS = frozenset({"%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"})
TEXT_MISSING = frozenset({"", "na", "n/a", "null", "none", "missing", "nan", "?"})
MAX_RULES = 100
MAX_ALLOWED = 500
MAX_EXAMPLES = 20
MAX_CONFIG_BYTES = 100_000


@dataclass(frozen=True)
class Rule:
    kind: str
    column: str
    severity: str = "warning"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finding:
    check: str
    column: str
    severity: str
    affected_rows: int
    message: str
    examples: tuple[dict[str, Any], ...] = ()
    recommendation: str = "Inspect flagged records before making changes."

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check, "column": self.column,
            "severity": self.severity, "affected_rows": self.affected_rows,
            "message": self.message, "examples": list(self.examples),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class ValidationReport:
    dataset: dict[str, Any]
    findings: tuple[Finding, ...]
    rules_run: int
    created_at_utc: str
    skipped_checks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": self.created_at_utc,
            "dataset": self.dataset,
            "rules_run": self.rules_run,
            "skipped_checks": list(self.skipped_checks),
            "findings": [f.to_dict() for f in self.findings],
            "note": "Flags are diagnostic, not proof of incorrect data. No rows were altered.",
        }

    def table(self) -> pd.DataFrame:
        rows = [{
            "Severity": f.severity.title(), "Check": f.check,
            "Column": f.column, "Affected rows": f.affected_rows,
            "Explanation": f.message, "Recommendation": f.recommendation,
        } for f in self.findings]
        return pd.DataFrame(rows, columns=[
            "Severity", "Check", "Column", "Affected rows", "Explanation", "Recommendation",
        ])


def _check_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate column names must be resolved before validation.")
    if any(not isinstance(c, str) for c in frame.columns):
        raise ValueError("Validation requires string column names; rename non-string columns first.")


def _text_series(series: pd.Series) -> pd.Series:
    """Safe display/parse representation, preserving missingness without modifying source."""
    return series.astype("string")


def _missing_mask(series: pd.Series, *, missing_tokens: bool = False) -> np.ndarray:
    mask = series.isna().to_numpy(dtype=bool, copy=True)
    if missing_tokens:
        s = _text_series(series).str.strip().str.casefold()
        mask |= s.isin(TEXT_MISSING).fillna(False).to_numpy(dtype=bool)
    return mask


def _numeric_series(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    if pd.api.types.is_bool_dtype(series):
        raise ValueError("Boolean columns are not treated as numeric measurements.")
    # Disallow silent comma stripping: different locales use different separators.
    converted = pd.to_numeric(series, errors="coerce")
    raw_present = ~_missing_mask(series)
    numeric_values = converted.to_numpy(dtype="float64", na_value=np.nan)
    invalid = raw_present & ~np.isfinite(numeric_values)
    return converted, invalid


def _safe_value(value: Any) -> str:
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return "<missing>"
    except (TypeError, ValueError):
        pass
    result = str(value)
    return result[:117] + "..." if len(result) > 120 else result


def _examples(frame: pd.DataFrame, mask: np.ndarray, column: str, maximum: int) -> tuple[dict[str, Any], ...]:
    positions = np.flatnonzero(mask)[:maximum]
    result = []
    for pos in positions:
        value = "<whole row>" if column == "(dataset)" else _safe_value(frame[column].iloc[int(pos)])
        result.append({
            "row_position": int(pos) + 1,  # human-readable, independent of pandas index
            "index_label": _safe_value(frame.index[int(pos)]),
            "value": value,
        })
    return tuple(result)


def _make(frame: pd.DataFrame, mask: np.ndarray, *, check: str, column: str,
          severity: str, message: str, recommendation: str,
          max_examples: int) -> Finding | None:
    count = int(np.count_nonzero(mask))
    if not count:
        return None
    return Finding(check, column, severity, count, message,
                   _examples(frame, mask, column, max_examples), recommendation)


def _validated_rule(rule: Rule, frame: pd.DataFrame) -> None:
    if rule.kind not in RULE_KINDS:
        raise ValueError(f"Unknown validation rule kind: {rule.kind!r}.")
    if not isinstance(rule.column, str) or not rule.column.strip():
        raise ValueError("Validation rules require a nonblank column name.")
    if rule.kind != "required_column" and rule.column not in frame.columns:
        raise ValueError(f"Rule {rule.kind!r} refers to unknown column {rule.column!r}.")
    if rule.severity not in SEVERITIES:
        raise ValueError("Severity must be critical, warning, or info.")
    p = rule.params
    if not isinstance(p, dict):
        raise ValueError("Rule parameters must be a JSON object.")
    allowed_keys = {
        "required_column": set(),
        "dtype": {"expected"},
        "required": {"include_text_tokens"},
        "unique": set(),
        "numeric": set(),
        "date": {"format"},
        "range": {"min", "max"},
        "allowed_values": {"values", "case_sensitive", "strip_whitespace"},
        "iqr_outlier": {"multiplier"},
        "zscore_outlier": {"threshold"},
    }[rule.kind]
    if set(p) - allowed_keys:
        raise ValueError(f"Unsupported parameters for {rule.kind}: {sorted(set(p) - allowed_keys)}")
    if rule.kind == "dtype" and p.get("expected") not in {"numeric", "text", "datetime", "boolean"}:
        raise ValueError("Expected data type must be numeric, text, datetime or boolean.")
    if rule.kind == "date" and p.get("format", "%Y-%m-%d") not in DATE_FORMATS:
        raise ValueError("Select an explicit supported date format.")
    if rule.kind == "range":
        if p.get("min") is None and p.get("max") is None:
            raise ValueError("Range rule needs a minimum, maximum, or both.")
        for key in ("min", "max"):
            if p.get(key) is not None and (isinstance(p[key], bool) or
                not isinstance(p[key], (int, float)) or not math.isfinite(p[key])):
                raise ValueError(f"{key} must be a finite number.")
        if p.get("min") is not None and p.get("max") is not None and p["min"] > p["max"]:
            raise ValueError("Minimum cannot exceed maximum.")
    if rule.kind == "allowed_values":
        values = p.get("values")
        if not isinstance(values, list) or not values or len(values) > MAX_ALLOWED or not all(
            isinstance(v, str) and v.strip() for v in values
        ):
            raise ValueError("Allowed values must be a non-empty list of at most 500 nonblank strings.")
        for key in ("case_sensitive", "strip_whitespace"):
            if key in p and not isinstance(p[key], bool):
                raise ValueError(f"{key} must be true or false.")
    for kind, key, default in (("iqr_outlier", "multiplier", 1.5), ("zscore_outlier", "threshold", 3.0)):
        if rule.kind == kind:
            number = p.get(key, default)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"{key} must be a positive finite number.")
    if rule.kind == "required" and "include_text_tokens" in p and not isinstance(p["include_text_tokens"], bool):
        raise ValueError("include_text_tokens must be true or false.")


def _evaluate_rule(frame: pd.DataFrame, rule: Rule, maximum: int) -> Finding | None:
    col, kind, p = rule.column, rule.kind, rule.params
    if kind == "required_column":
        if col in frame.columns:
            return None
        return Finding(
            "Required column", col, rule.severity, len(frame),
            "Expected column is missing from this dataset (schema-level failure).",
            (), "Check incoming schema, column naming and source extract changes.",
        )
    s = frame[col]
    if kind == "dtype":
        expected = p["expected"]
        dtype = s.dtype
        actual = (
            "boolean" if pd.api.types.is_bool_dtype(dtype) else
            "datetime" if pd.api.types.is_datetime64_any_dtype(dtype) else
            "numeric" if pd.api.types.is_numeric_dtype(dtype) else
            "text" if (pd.api.types.is_string_dtype(dtype) or
                       pd.api.types.is_object_dtype(dtype) or
                       isinstance(dtype, pd.CategoricalDtype)) else
            "other"
        )
        if actual == expected:
            return None
        return Finding("Column data type", col, rule.severity, len(frame),
            f"Expected {expected} type, received {dtype} ({actual}); this is a schema-level mismatch.",
            (), "Check import options and source metadata before converting column types.")
    present = ~_missing_mask(s)
    if kind == "required":
        bad = _missing_mask(s, missing_tokens=p.get("include_text_tokens", True))
        return _make(frame, bad, check="Required values", column=col, severity=rule.severity,
            message="Null or designated missing-value tokens in a required field.",
            recommendation="Investigate missing records; only impute using justified, documented rules.", max_examples=maximum)
    if kind == "unique":
        try:
            bad = s.duplicated(keep=False).fillna(False).to_numpy(dtype=bool) & present
        except TypeError as exc:
            raise ValueError(f"Cannot assess uniqueness for {col!r} (unhashable values).") from exc
        return _make(frame, bad, check="Unique identifier", column=col, severity=rule.severity,
            message="Repeated non-missing identifier values; every colliding row is flagged.",
            recommendation="Check whether duplicates are legitimate repeated observations before deduplication.", max_examples=maximum)
    if kind == "numeric":
        _, bad = _numeric_series(s)
        return _make(frame, bad, check="Numeric parse", column=col, severity=rule.severity,
            message="Non-missing values cannot be parsed as finite numeric values.",
            recommendation="Check units, decimal separators, text codes, and non-finite values.", max_examples=maximum)
    if kind == "date":
        fmt = p.get("format", "%Y-%m-%d")
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        bad = present & parsed.isna().to_numpy(dtype=bool)
        return _make(frame, bad, check="Date parse", column=col, severity=rule.severity,
            message=f"Non-missing values do not conform to the explicit {fmt} date format.",
            recommendation="Verify source date conventions and ambiguous day/month order.", max_examples=maximum)
    if kind == "range":
        converted, invalid = _numeric_series(s)
        values = converted.to_numpy(dtype="float64", na_value=np.nan)
        bad = invalid.copy()
        if p.get("min") is not None:
            bad |= values < p["min"]
        if p.get("max") is not None:
            bad |= values > p["max"]
        return _make(frame, bad, check="Numeric range", column=col, severity=rule.severity,
            message=f"Values are outside [{p.get('min', '-∞')}, {p.get('max', '+∞')}] or not finite numeric values.",
            recommendation="Verify units and business/domain limits; do not automatically clip extremes.", max_examples=maximum)
    if kind == "allowed_values":
        entries = _text_series(s)
        allowed = pd.Series(p["values"], dtype="string")
        if p.get("strip_whitespace", True):
            entries, allowed = entries.str.strip(), allowed.str.strip()
        if not p.get("case_sensitive", False):
            entries, allowed = entries.str.casefold(), allowed.str.casefold()
        bad = present & ~entries.isin(allowed).fillna(False).to_numpy(dtype=bool)
        return _make(frame, bad, check="Allowed categories", column=col, severity=rule.severity,
            message="Non-missing values fall outside the configured category vocabulary.",
            recommendation="Review spelling, missing labels, and legitimately new categories.", max_examples=maximum)
    if kind in {"iqr_outlier", "zscore_outlier"}:
        converted, _ = _numeric_series(s)
        data = converted.to_numpy(dtype="float64", na_value=np.nan)
        valid = np.isfinite(data)
        clean = data[valid]
        if len(clean) < 4:
            return None  # Insufficient observations, not evidence of no outliers.
        if kind == "iqr_outlier":
            q1, q3 = np.quantile(clean, [.25, .75])
            delta = q3 - q1
            if delta <= 0 or not np.isfinite(delta):
                return None
            mult = p.get("multiplier", 1.5)
            bad = valid & ((data < q1 - mult * delta) | (data > q3 + mult * delta))
            check, msg = "IQR outliers", f"Values outside the {mult} × IQR fences (Q1={q1:.4g}, Q3={q3:.4g})."
        else:
            sd = float(np.std(clean, ddof=0))
            if sd <= 0 or not np.isfinite(sd):
                return None
            threshold = p.get("threshold", 3.0)
            bad = valid & (np.abs((data - float(np.mean(clean))) / sd) > threshold)
            check, msg = "Z-score outliers", f"Absolute population Z-score exceeds {threshold}."
        return _make(frame, bad, check=check, column=col, severity=rule.severity,
            message=msg, recommendation="Statistical outliers may be genuine; assess context before treatment.",
            max_examples=maximum)
    raise AssertionError("Rule dispatch out of sync with rule validation.")


def _automatic_findings(frame: pd.DataFrame, maximum: int) -> tuple[list[Finding], int, list[str]]:
    findings: list[Finding] = []
    checks = 0
    skipped: list[str] = []
    try:
        duplicated = frame.duplicated(keep=False).to_numpy(dtype=bool)
        checks += 1
        result = _make(frame, duplicated, check="Duplicate rows", column="(dataset)",
            severity="warning", message="Entire rows match (all members of each duplicate group counted).",
            recommendation="Decide whether repeated records are meaningful before removing duplicates.",
            max_examples=maximum)
        if result:
            findings.append(result)
    except TypeError:
        skipped.append("Duplicate rows: cells contain unhashable objects; this check could not run.")

    for column in frame.columns:
        series = frame[column]
        checks += 1
        result = _make(frame, series.isna().to_numpy(dtype=bool), check="Missing values",
            column=column, severity="warning", message="Null entries (NaN, None, pandas NA).",
            recommendation="Check missingness patterns before deletion or imputation.", max_examples=maximum)
        if result:
            findings.append(result)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            checks += 1
            texts = _text_series(series)
            trim = texts.str.strip()
            bad = (texts.notna() & texts.ne(trim)).fillna(False).to_numpy(dtype=bool)
            result = _make(frame, bad, check="Surrounding whitespace", column=column,
                severity="info", message="Text values have leading or trailing whitespace.",
                recommendation="Preview text trimming and check whether whitespace is meaningful.",
                max_examples=maximum)
            if result:
                findings.append(result)
            checks += 1
            tokens = trim.str.casefold().isin(TEXT_MISSING).fillna(False).to_numpy(dtype=bool)
            result = _make(frame, tokens, check="Text missing markers", column=column,
                severity="info", message="Text looks like a missing-value marker but is not null.",
                recommendation="Confirm whether tokens such as NA or ? mean missing before converting.",
                max_examples=maximum)
            if result:
                findings.append(result)
            # Conservative suggestions: don't infer numeric type for ID/code columns.
            name = column.casefold()
            if not any(token in name for token in ("id", "code", "phone", "zip", "postal", "account", "serial", "reference")):
                raw_nonnull = (~_missing_mask(series, missing_tokens=True))
                n = int(raw_nonnull.sum())
                if n >= 5:
                    try:
                        values = pd.to_numeric(series, errors="coerce")
                        numeric = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
                        ratio = int((numeric & raw_nonnull).sum()) / n
                        if ratio >= 0.9:
                            checks += 1
                            findings.append(Finding(
                                "Numeric type suggestion", column, "info", n,
                                f"{ratio:.0%} of present values parse as numbers, but the column is text. This is a suggestion, not an error count.",
                                (), "Check identifiers and leading zeros before converting to numeric.",
                            ))
                    except (TypeError, ValueError):
                        pass
            if any(token in name for token in ("date", "time", "dob", "joined", "birth", "created", "updated")):
                raw_present = ~_missing_mask(series, missing_tokens=True)
                n = int(raw_present.sum())
                if n >= 5:
                    iso_looking = trim.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
                    parsed = pd.to_datetime(trim.where(iso_looking), format="%Y-%m-%d", errors="coerce")
                    ratio = int((parsed.notna().to_numpy(dtype=bool) & raw_present).sum()) / n
                    if ratio >= .9:
                        checks += 1
                        findings.append(Finding(
                            "Date type suggestion", column, "info", n,
                            f"{ratio:.0%} of present values parse as strict ISO dates, but the column is text. This is a suggestion, not an error count.",
                            (), "Confirm date semantics and timezone needs before converting.",
                        ))
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            checks += 1
            try:
                _, nonfinite = _numeric_series(series)
                result = _make(frame, nonfinite, check="Non-finite numeric", column=column,
                    severity="warning", message="Numeric values include ±infinity or unparseable non-null entries.",
                    recommendation="Find source of non-finite values before statistical analysis.", max_examples=maximum)
                if result:
                    findings.append(result)
            except (ValueError, TypeError):
                pass
    return findings, checks, skipped


def validate_dataframe(frame: pd.DataFrame, rules: Sequence[Rule] | None = None,
                       *, automatic: bool = True, max_examples: int = 5) -> ValidationReport:
    """Scan a dataset without modification; counts use all rows, evidence is capped.

    Configuration errors are raised explicitly rather than silently skipped.
    Statistical outlier checks with insufficient data are omitted (not reported as passed).
    """
    _check_frame(frame)
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or not 0 <= max_examples <= MAX_EXAMPLES:
        raise ValueError(f"max_examples must be an integer between 0 and {MAX_EXAMPLES}.")
    specs = list(rules or [])
    if not automatic and not specs:
        raise ValueError("No checks selected. Enable automatic checks or add a validation rule.")
    if len(specs) > MAX_RULES:
        raise ValueError(f"A maximum of {MAX_RULES} validation rules is supported.")
    for spec in specs:
        if not isinstance(spec, Rule):
            raise TypeError("Rules must be Rule objects.")
        _validated_rule(spec, frame)
    findings: list[Finding] = []
    skipped: list[str] = []
    checks = 0
    if automatic:
        findings, checks, skipped = _automatic_findings(frame, max_examples)
    for rule in specs:
        checks += 1
        # Be explicit when outlier methods cannot be evaluated. A skipped test
        # must not be interpreted as evidence that a variable has no outliers.
        if rule.kind in {"iqr_outlier", "zscore_outlier"}:
            try:
                numeric, _ = _numeric_series(frame[rule.column])
            except (TypeError, ValueError) as exc:
                skipped.append(f"{rule.kind} / {rule.column}: {exc}")
                continue
            values = numeric.to_numpy(dtype="float64", na_value=np.nan)
            finite = values[np.isfinite(values)]
            if len(finite) < 4:
                skipped.append(f"{rule.kind} / {rule.column}: fewer than four finite numeric observations.")
                continue
            if rule.kind == "iqr_outlier":
                q1, q3 = np.quantile(finite, [.25, .75])
                if not np.isfinite(q3-q1) or q3 <= q1:
                    skipped.append(f"{rule.kind} / {rule.column}: zero or undefined IQR.")
                    continue
            if rule.kind == "zscore_outlier":
                sd = float(np.std(finite, ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    skipped.append(f"{rule.kind} / {rule.column}: zero or undefined standard deviation.")
                    continue
        result = _evaluate_rule(frame, rule, max_examples)
        if result:
            findings.append(result)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (severity_order[f.severity], -f.affected_rows, f.column, f.check))
    rows, cols = frame.shape
    return ValidationReport(
        dataset={"rows": rows, "columns": cols, "cells": rows * cols,
                 "null_cells": int(frame.isna().to_numpy().sum()),
                 "flagged_checks": len(findings),
                 "critical_checks": sum(f.severity == "critical" for f in findings),
                 "warning_checks": sum(f.severity == "warning" for f in findings)},
        findings=tuple(findings), rules_run=checks,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        skipped_checks=tuple(skipped),
    )


def export_rule_config(rules: Sequence[Rule], *, automatic: bool = True) -> str:
    """Export portable JSON validation configuration, not executable Python."""
    if len(rules) > MAX_RULES:
        raise ValueError("Too many validation rules.")
    for r in rules:
        if not isinstance(r, Rule) or r.kind not in RULE_KINDS or r.severity not in SEVERITIES:
            raise ValueError("Invalid rule definition.")
    return json.dumps({
        "schema_version": SCHEMA_VERSION, "automatic_checks": bool(automatic),
        "rules": [{"kind": r.kind, "column": r.column, "severity": r.severity,
                   "params": r.params} for r in rules],
    }, indent=2, ensure_ascii=False, allow_nan=False)


def import_rule_config(payload: str | bytes) -> tuple[list[Rule], bool]:
    """Deserialize only known rule types; data columns validated at scan time."""
    if isinstance(payload, bytes):
        if len(payload) > MAX_CONFIG_BYTES:
            raise ValueError("Rule configuration exceeds 100 KB.")
        try:
            payload = payload.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("Rule configuration is not valid UTF-8.") from exc
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("Rule configuration exceeds 100 KB.")
    try:
        data = json.loads(payload, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON number: {x}")))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid JSON rule configuration.") from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported validation-rule schema version.")
    if set(data) != {"schema_version", "automatic_checks", "rules"}:
        raise ValueError("Unexpected configuration fields.")
    if not isinstance(data["automatic_checks"], bool):
        raise ValueError("automatic_checks must be true or false.")
    raw_rules = data["rules"]
    if not isinstance(raw_rules, list) or len(raw_rules) > MAX_RULES:
        raise ValueError("rules must be a list of up to 100 entries.")
    specs: list[Rule] = []
    for item in raw_rules:
        if not isinstance(item, dict) or set(item) != {"kind", "column", "severity", "params"}:
            raise ValueError("Invalid rule configuration entry.")
        if not isinstance(item["kind"], str) or item["kind"] not in RULE_KINDS:
            raise ValueError("Unknown rule kind.")
        if not isinstance(item["column"], str) or not item["column"]:
            raise ValueError("Rule column must be a nonempty string.")
        if not isinstance(item["severity"], str) or item["severity"] not in SEVERITIES:
            raise ValueError("Invalid severity.")
        if not isinstance(item["params"], dict):
            raise ValueError("Rule params must be a JSON object.")
        specs.append(Rule(**item))
    return specs, data["automatic_checks"]
