"""
Leakage-safe, auditable data cleaning for Makoding.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal, Mapping, get_args

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MissingStrategy = Literal[
    "keep",
    "drop_rows",
    "median",
    "mean",
    "mode",
    "auto",
]

SUPPORTED_MISSING_STRATEGIES: tuple[str, ...] = get_args(MissingStrategy)

DtypeDriftMode = Literal["error", "warn"]
SUPPORTED_DTYPE_DRIFT_MODES: tuple[str, ...] = get_args(DtypeDriftMode)

MISSING_STRATEGIES = (
    "Keep",
    "Drop rows",
    "Fill numeric median",
    "Fill numeric mean",
    "Fill categorical mode",
    "Fill all with mode",
    "Fill missing values",
)

_LEGACY_TO_INTERNAL = {
    "Keep": "keep",
    "Drop rows": "drop_rows",
    "Fill numeric median": "median",
    "Fill numeric mean": "mean",
    "Fill categorical mode": "mode",
    "Fill all with mode": "mode",
    "Fill missing values": "auto",
}


__all__ = [
    "MissingStrategy",
    "SUPPORTED_MISSING_STRATEGIES",
    "DtypeDriftMode",
    "SUPPORTED_DTYPE_DRIFT_MODES",
    "CleaningPolicy",
    "LearnedColumnState",
    "CleaningAudit",
    "Cleaner",
    "NotFittedError",
    "SchemaMismatchError",
    "clean_frame",
    "CleaningReport",
    "MISSING_STRATEGIES",
    "stats_readiness",
    "ColumnQuality",
    "QualityReport",
    "profile_frame",
    "ColumnContract",
    "DatasetContract",
    "ValidationIssue",
    "ValidationReport",
    "validate_contract",
    "dataset_fingerprint",
    "RunManifest",
    "PreparedData",
    "AnalyticalPreprocessor",
]


class NotFittedError(RuntimeError):
    pass


class SchemaMismatchError(ValueError):
    pass


@dataclass(frozen=True)
class CleaningPolicy:
    missing: MissingStrategy = "keep"
    trim_strings: bool = True
    trim_column_names: bool = True
    replace_infinite: bool = True
    remove_duplicates: bool = False
    drop_all_missing_columns: bool = False
    drop_constant_columns: bool = False
    add_missing_indicators: bool = False
    cap_outliers: bool = False
    outlier_multiplier: float = 1.5
    dtype_drift: DtypeDriftMode = "error"
    strict_column_order: bool = False
    outlier_columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.missing not in SUPPORTED_MISSING_STRATEGIES:
            raise ValueError(
                f"Unsupported missing strategy {self.missing!r}. "
                f"Expected one of {SUPPORTED_MISSING_STRATEGIES}."
            )
        if self.dtype_drift not in SUPPORTED_DTYPE_DRIFT_MODES:
            raise ValueError(
                f"Unsupported dtype_drift mode {self.dtype_drift!r}. "
                f"Expected one of {SUPPORTED_DTYPE_DRIFT_MODES}."
            )
        if self.outlier_columns is not None:
            if len(set(self.outlier_columns)) != len(self.outlier_columns):
                raise ValueError("outlier_columns must not contain duplicates.")
            if any(not isinstance(column, str) or not column.strip() for column in self.outlier_columns):
                raise ValueError("outlier_columns must contain non-empty strings.")
        if not np.isfinite(self.outlier_multiplier):
            raise ValueError("outlier_multiplier must be finite.")
        if self.outlier_multiplier <= 0:
            raise ValueError("outlier_multiplier must be greater than zero.")
        if self.add_missing_indicators and self.missing == "drop_rows":
            raise ValueError(
                "add_missing_indicators=True has no effect when "
                "missing='drop_rows'."
            )


@dataclass(frozen=True)
class LearnedColumnState:
    name: str
    dtype: str
    dtype_family: str = "unknown"
    is_numeric: bool = False
    missing_before_fit: int = 0
    fill_value: object | None = None
    lower_fence: float | None = None
    upper_fence: float | None = None


@dataclass(frozen=True)
class CleaningAudit:
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    missing_before: int
    missing_after: int
    infinite_replaced: int
    duplicates_removed: int
    values_filled: int
    whitespace_trimmed: int = 0
    infinite_replaced_by_column: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    values_filled_by_column: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    values_capped: int = 0
    values_capped_by_column: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    columns_dropped: tuple[str, ...] = field(default_factory=tuple)
    dtype_drift_columns: tuple[str, ...] = field(default_factory=tuple)
    missing_indicator_columns: tuple[str, ...] = field(default_factory=tuple)
    operations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    decisions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def missing_resolved(self) -> int:
        return self.missing_before - self.missing_after

    @property
    def missing_removed(self) -> int:
        return self.missing_resolved

    @property
    def missing_values_before(self) -> int:
        return self.missing_before

    @property
    def missing_values_after(self) -> int:
        return self.missing_after

    def as_markdown(self) -> str:
        return self.summary()

    def summary(self) -> str:
        parts = [
            f"rows {self.rows_before:,}->{self.rows_after:,}",
            f"missing {self.missing_before:,}->{self.missing_after:,}",
        ]
        if self.whitespace_trimmed:
            parts.append(f"{self.whitespace_trimmed:,} values whitespace-trimmed")
        if self.duplicates_removed:
            parts.append(f"{self.duplicates_removed:,} duplicates removed")
        if self.infinite_replaced:
            parts.append(f"{self.infinite_replaced:,} infinite values replaced")
        if self.values_filled:
            parts.append(f"{self.values_filled:,} values filled")
        if self.values_capped:
            parts.append(f"{self.values_capped:,} outliers capped")
        if self.columns_dropped:
            parts.append(f"{len(self.columns_dropped)} columns dropped")
        return "; ".join(parts)


CleaningReport = CleaningAudit


class Cleaner:
    def __init__(self, policy: CleaningPolicy | None = None) -> None:
        self.policy = policy or CleaningPolicy()
        self._fitted = False
        self._input_columns: tuple[str, ...] = ()
        self._output_columns: tuple[str, ...] = ()
        self._states: dict[str, LearnedColumnState] = {}
        self._dropped_columns: tuple[str, ...] = ()
        self._fit_rows: int | None = None
        self._fit_fingerprint: str | None = None

    def __repr__(self) -> str:
        if not self._fitted:
            return f"Cleaner(policy={self.policy!r}, fitted=False)"
        return (
            f"Cleaner(policy={self.policy!r}, fitted=True, "
            f"fit_rows={self._fit_rows}, "
            f"input_columns={len(self._input_columns)}, "
            f"dropped_columns={len(self._dropped_columns)}, "
            f"output_columns={len(self._output_columns)})"
        )

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def fit_rows(self) -> int | None:
        return self._fit_rows

    @property
    def fit_fingerprint(self) -> str | None:
        """Content fingerprint of the normalized training frame."""
        return self._fit_fingerprint

    @property
    def input_columns(self) -> tuple[str, ...]:
        return self._input_columns

    @property
    def output_columns(self) -> tuple[str, ...]:
        return self._output_columns

    @property
    def dropped_columns(self) -> tuple[str, ...]:
        return self._dropped_columns

    @property
    def learned_state(self) -> Mapping[str, LearnedColumnState]:
        return dict(self._states)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(
                "Cleaner must be fitted before this operation."
            )

    def fit(self, frame: pd.DataFrame) -> "Cleaner":
        _validate_frame(frame)
        work, _ = _normalize_structure(frame, self.policy)
        _validate_unique_columns(work)

        logger.info(
            "Fitting Cleaner on %d rows, %d columns (missing=%r, cap_outliers=%s).",
            len(work), work.shape[1], self.policy.missing, self.policy.cap_outliers,
        )

        self._input_columns = tuple(str(column) for column in work.columns)
        self._fit_rows = len(work)
        self._fit_fingerprint = dataset_fingerprint(work)
        self._states = {}

        dropped: list[str] = []

        if self.policy.drop_all_missing_columns:
            for column in work.columns:
                if work[column].isna().all():
                    dropped.append(str(column))

        if self.policy.drop_constant_columns:
            for column in work.columns:
                name = str(column)
                if name in dropped:
                    continue
                non_missing = work[column].dropna()
                if not non_missing.empty and non_missing.nunique() <= 1:
                    dropped.append(name)

        self._dropped_columns = tuple(dict.fromkeys(dropped))

        undroppable_all_missing = [
            str(column) for column in work.columns
            if str(column) not in self._dropped_columns and work[column].isna().all()
        ]
        if undroppable_all_missing:
            warnings.warn(
                f"The following column(s) are entirely missing but were not dropped: "
                f"{undroppable_all_missing}.",
                UserWarning, stacklevel=2,
            )
            logger.warning("Entirely-missing columns retained: %s", undroppable_all_missing)

        for column in work.columns:
            name = str(column)
            if name in self._dropped_columns:
                continue

            series = work[column]
            is_numeric = _is_numeric(series)
            dtype_family = _dtype_family(series)
            stats_series = series.replace([np.inf, -np.inf], np.nan)

            fill_value: object | None = None
            lower_fence: float | None = None
            upper_fence: float | None = None

            if self.policy.missing in {"median", "mean"} and is_numeric:
                value = (
                    stats_series.median() if self.policy.missing == "median" else stats_series.mean()
                )
                if pd.notna(value):
                    fill_value = float(value)
            elif self.policy.missing == "mode":
                modes = stats_series.mode(dropna=True)
                if not modes.empty:
                    fill_value = modes.iloc[0]
            elif self.policy.missing == "auto":
                if is_numeric:
                    value = stats_series.median()
                    if pd.notna(value):
                        fill_value = float(value)
                else:
                    modes = stats_series.mode(dropna=True)
                    if not modes.empty:
                        fill_value = modes.iloc[0]

            should_cap = (
                self.policy.cap_outliers
                and is_numeric
                and (self.policy.outlier_columns is None or name in self.policy.outlier_columns)
            )
            if should_cap:
                values = stats_series.dropna()
                if not values.empty:
                    q1 = float(values.quantile(0.25))
                    q3 = float(values.quantile(0.75))
                    iqr = q3 - q1
                    if np.isfinite(iqr) and iqr > 0:
                        lower_fence = q1 - self.policy.outlier_multiplier * iqr
                        upper_fence = q3 + self.policy.outlier_multiplier * iqr

            self._states[name] = LearnedColumnState(
                name=name, dtype=str(series.dtype), dtype_family=dtype_family, is_numeric=is_numeric,
                missing_before_fit=int(series.isna().sum()),
                fill_value=fill_value, lower_fence=lower_fence, upper_fence=upper_fence,
            )

        preview, _ = self._transform_internal(work)
        self._output_columns = tuple(str(column) for column in preview.columns)
        self._fitted = True

        logger.info(
            "Cleaner fit complete: %d column(s) dropped, %d column(s) retained.",
            len(self._dropped_columns), len(self._output_columns),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
        self._require_fitted()
        _validate_frame(frame)

        work, whitespace_trimmed = _normalize_structure(frame, self.policy)
        _validate_unique_columns(work)

        expected = set(self._input_columns)
        actual = set(str(column) for column in work.columns)

        missing_columns = expected - actual
        extra_columns = actual - expected

        if missing_columns or extra_columns:
            raise SchemaMismatchError(
                f"Input schema differs from the schema observed during fit. "
                f"Missing={sorted(missing_columns)}; extra={sorted(extra_columns)}."
            )

        actual_order = tuple(str(column) for column in work.columns)
        if self.policy.strict_column_order and actual_order != self._input_columns:
            raise SchemaMismatchError(
                "Input column order differs from the order observed during fit. "
                f"Expected={self._input_columns}; received={actual_order}."
            )

        work = work.loc[:, list(self._input_columns)]

        drift_columns = [
            column for column in self._input_columns
            if (state := self._states.get(column)) is not None
            and _dtype_family(work[column]) != state.dtype_family
        ]
        if drift_columns:
            message = (
                "The following column(s) changed semantic dtype family since "
                f"fit(): {drift_columns}. Expected families={[self._states[c].dtype_family for c in drift_columns]}; "
                f"received={[ _dtype_family(work[c]) for c in drift_columns ]}."
            )
            if self.policy.dtype_drift == "error":
                raise SchemaMismatchError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            logger.warning(message)

        cleaned, audit = self._transform_internal(work, whitespace_trimmed=whitespace_trimmed)
        if drift_columns:
            audit = replace(audit, dtype_drift_columns=tuple(drift_columns))
        return cleaned, audit

    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
        self.fit(frame)
        return self.transform(frame)

    def _transform_internal(
        self, frame: pd.DataFrame, *, whitespace_trimmed: int = 0
    ) -> tuple[pd.DataFrame, CleaningAudit]:
        work = frame.copy(deep=True)
        rows_before = len(work)
        columns_before = len(work.columns)

        infinite_replaced = 0
        infinite_by_column: dict[str, int] = {}
        duplicates_removed = 0
        values_filled = 0
        values_capped = 0
        filled_by_column: dict[str, int] = {}
        capped_by_column: dict[str, int] = {}
        decisions: list[str] = []

        if whitespace_trimmed:
            decisions.append(f"Trimmed whitespace from {whitespace_trimmed:,} value(s).")

        if self.policy.replace_infinite:
            for column in work.columns:
                if not _is_numeric(work[column]):
                    continue
                values = work[column].to_numpy(dtype=float, na_value=np.nan)
                mask = np.isinf(values)
                count = int(mask.sum())
                if count:
                    work[column] = work[column].replace([np.inf, -np.inf], np.nan)
                    infinite_replaced += count
                    infinite_by_column[str(column)] = count
            if infinite_replaced:
                detail = ", ".join(f"{c} ({n})" for c, n in sorted(infinite_by_column.items()))
                decisions.append(f"Replaced {infinite_replaced:,} non-finite numeric value(s): {detail}.")

        missing_before = int(work.isna().sum().sum())

        if self.policy.remove_duplicates:
            before = len(work)
            work = work.drop_duplicates(keep="first").reset_index(drop=True)
            duplicates_removed = before - len(work)
            if duplicates_removed:
                decisions.append(f"Removed {duplicates_removed:,} exact duplicate row(s).")

        if self._dropped_columns:
            present = [c for c in self._dropped_columns if c in work.columns]
            work = work.drop(columns=list(self._dropped_columns), errors="ignore")
            if present:
                decisions.append(f"Dropped column(s): {list(present)}.")

        indicator_columns: list[str] = []
        if self.policy.add_missing_indicators:
            for column in list(self._states):
                if column not in work.columns:
                    continue
                indicator_name = f"{column}__missing"
                work[indicator_name] = work[column].isna().astype("int8")
                indicator_columns.append(indicator_name)
            if indicator_columns:
                decisions.append(f"Added {len(indicator_columns)} missing-value indicator column(s): {indicator_columns}.")

        if self.policy.missing == "drop_rows":
            before = len(work)
            work = work.dropna().reset_index(drop=True)
            if before != len(work):
                decisions.append(f"Dropped {before - len(work):,} row(s) containing missing values.")
        else:
            for column, state in self._states.items():
                if column not in work.columns or state.fill_value is None:
                    continue
                if isinstance(state.fill_value, float) and not _is_numeric(work[column]):
                    continue
                missing_count = int(work[column].isna().sum())
                if missing_count:
                    work[column] = _safe_fill(work[column], state.fill_value)
                    values_filled += missing_count
                    filled_by_column[column] = missing_count
            if values_filled:
                detail = ", ".join(f"{c} ({n})" for c, n in sorted(filled_by_column.items()))
                decisions.append(
                    f"Filled {values_filled:,} missing value(s) using learned '{self.policy.missing}' "
                    f"values across {len(filled_by_column)} column(s): {detail}."
                )

        if self.policy.cap_outliers:
            for column, state in self._states.items():
                if column not in work.columns:
                    continue
                if state.lower_fence is None or state.upper_fence is None:
                    continue
                if self.policy.outlier_columns is not None and column not in self.policy.outlier_columns:
                    continue
                if not _is_numeric(work[column]):
                    continue
                values = work[column]
                mask = (values < state.lower_fence) | (values > state.upper_fence)
                count = int(mask.fillna(False).sum())
                if count:
                    work[column] = values.clip(lower=state.lower_fence, upper=state.upper_fence)
                    values_capped += count
                    capped_by_column[column] = count
            if values_capped:
                detail = ", ".join(f"{c} ({n})" for c, n in sorted(capped_by_column.items()))
                decisions.append(
                    f"Capped {values_capped:,} outlier value(s) using IQR fences "
                    f"(x{self.policy.outlier_multiplier}) across {len(capped_by_column)} column(s): {detail}."
                )

        missing_after = int(work.isna().sum().sum())

        audit = CleaningAudit(
            rows_before=rows_before, rows_after=len(work),
            columns_before=columns_before, columns_after=len(work.columns),
            missing_before=missing_before, missing_after=missing_after,
            infinite_replaced=infinite_replaced,
            whitespace_trimmed=whitespace_trimmed,
            infinite_replaced_by_column=tuple(sorted(infinite_by_column.items())),
            duplicates_removed=duplicates_removed,
            values_filled=values_filled, values_filled_by_column=tuple(sorted(filled_by_column.items())),
            values_capped=values_capped, values_capped_by_column=tuple(sorted(capped_by_column.items())),
            columns_dropped=self._dropped_columns,
            missing_indicator_columns=tuple(indicator_columns),
            operations=tuple(_build_operations(
                infinite_by_column=infinite_by_column,
                duplicates_removed=duplicates_removed,
                columns_dropped=self._dropped_columns,
                filled_by_column=filled_by_column,
                capped_by_column=capped_by_column,
                indicator_columns=indicator_columns,
                policy=self.policy,
            )),
            decisions=tuple(decisions),
        )
        return work, audit

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()

        def _safe(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, np.generic):
                return value.item()
            return str(value)

        return {
            "policy": {
                "missing": self.policy.missing,
                "trim_strings": self.policy.trim_strings,
                "trim_column_names": self.policy.trim_column_names,
                "replace_infinite": self.policy.replace_infinite,
                "remove_duplicates": self.policy.remove_duplicates,
                "drop_all_missing_columns": self.policy.drop_all_missing_columns,
                "drop_constant_columns": self.policy.drop_constant_columns,
                "add_missing_indicators": self.policy.add_missing_indicators,
                "cap_outliers": self.policy.cap_outliers,
                "outlier_multiplier": self.policy.outlier_multiplier,
                "dtype_drift": self.policy.dtype_drift,
                "strict_column_order": self.policy.strict_column_order,
                "outlier_columns": list(self.policy.outlier_columns) if self.policy.outlier_columns is not None else None,
            },
            "fit_rows": self._fit_rows,
            "fit_fingerprint": self._fit_fingerprint,
            "input_columns": list(self._input_columns),
            "output_columns": list(self._output_columns),
            "dropped_columns": list(self._dropped_columns),
            "learned_state": {
                name: {
                    "dtype": state.dtype, "dtype_family": state.dtype_family, "is_numeric": state.is_numeric,
                    "missing_before_fit": state.missing_before_fit,
                    "fill_value": _safe(state.fill_value),
                    "lower_fence": state.lower_fence, "upper_fence": state.upper_fence,
                }
                for name, state in self._states.items()
            },
        }

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        logger.info("Saved fitted Cleaner to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Cleaner":
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a Cleaner.")
        return obj


def clean_frame(
    frame: pd.DataFrame,
    *,
    missing: str = "Keep",
    remove_duplicates: bool = True,
    trim_whitespace: bool = True,
    treat_infinite_as_missing: bool = True,
    cap_outliers: bool = False,
    outlier_multiplier: float = 1.5,
    drop_all_missing_columns: bool = False,
    drop_constant_columns: bool = False,
    add_missing_indicators: bool = False,
    dtype_drift: DtypeDriftMode = "error",
    strict_column_order: bool = False,
    outlier_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    _validate_frame(frame)
    if missing not in MISSING_STRATEGIES and missing not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError(
            f"Unknown missing-value strategy: {missing!r}. Supported strategies are: {MISSING_STRATEGIES}"
        )

    internal = _LEGACY_TO_INTERNAL.get(missing, missing)
    policy = CleaningPolicy(
        missing=internal, trim_strings=trim_whitespace, trim_column_names=trim_whitespace,
        replace_infinite=treat_infinite_as_missing, remove_duplicates=remove_duplicates,
        cap_outliers=cap_outliers, outlier_multiplier=outlier_multiplier,
        drop_all_missing_columns=drop_all_missing_columns, drop_constant_columns=drop_constant_columns,
        add_missing_indicators=add_missing_indicators,
        dtype_drift=dtype_drift, strict_column_order=strict_column_order, outlier_columns=outlier_columns,
    )
    cleaner = Cleaner(policy)
    return cleaner.fit_transform(frame)


def stats_readiness(frame: pd.DataFrame, *, min_observations: int = 3) -> dict[str, dict[str, Any]]:
    _validate_frame(frame)
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")

    result: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        raw_numeric = pd.to_numeric(frame[column], errors="coerce")
        non_missing = raw_numeric.dropna()

        # "finite" must be checked on the non-missing values BEFORE any
        # infinities are stripped -- checking it after stripping (as a
        # prior version of this function did) is tautologically always
        # True, since nothing non-finite is left by that point.
        if non_missing.empty:
            is_finite = True  # vacuously true: nothing present to be non-finite
            finite_values = non_missing
        else:
            finite_mask = np.isfinite(non_missing.to_numpy(dtype=float))
            is_finite = bool(finite_mask.all())
            finite_values = non_missing[finite_mask]

        n = int(len(finite_values))
        unique = int(finite_values.nunique())

        result[str(column)] = {
            "valid_observations": n,
            "unique_values": unique,
            "enough_observations": n >= min_observations,
            "finite": is_finite,
            "constant": unique <= 1 if n else False,
            "column_ready": is_finite and n >= min_observations and unique > 1,
        }
    return result


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")


def _validate_unique_columns(frame: pd.DataFrame) -> None:
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].tolist()
        raise ValueError(f"Duplicate column names are not allowed after normalization: {duplicates}")


def _normalize_structure(frame: pd.DataFrame, policy: CleaningPolicy) -> tuple[pd.DataFrame, int]:
    """Trim column-name and string-value whitespace.

    Returns ``(normalized_frame, values_trimmed)`` -- ``values_trimmed``
    counts individual string *values* whose whitespace was stripped (not
    column-header renames), so callers can report it in a
    ``CleaningAudit`` instead of the historical hard-coded ``0``.
    """
    work = frame.copy(deep=True)
    trimmed_count = 0

    if policy.trim_column_names:
        rename_map: dict[object, object] = {
            column: column.strip()
            for column in work.columns
            if isinstance(column, str) and column.strip() != column
        }
        if rename_map:
            work = work.rename(columns=rename_map)

    if policy.trim_strings:
        for column in work.columns:
            series = work[column]
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                local_count = 0

                def _strip(value: object) -> object:
                    nonlocal local_count
                    if isinstance(value, str):
                        stripped = value.strip()
                        if stripped != value:
                            local_count += 1
                        return stripped
                    return value

                work[column] = series.map(_strip)
                trimmed_count += local_count

    _validate_unique_columns(work)
    return work, trimmed_count


def _dtype_family(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "timedelta"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical_or_string"
    if pd.api.types.is_string_dtype(dtype):
        return "categorical_or_string"
    if pd.api.types.is_object_dtype(dtype):
        return "categorical_or_string"
    return "other"


def _build_operations(
    *,
    infinite_by_column: Mapping[str, int],
    duplicates_removed: int,
    columns_dropped: tuple[str, ...],
    filled_by_column: Mapping[str, int],
    capped_by_column: Mapping[str, int],
    indicator_columns: list[str],
    policy: CleaningPolicy,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    if infinite_by_column:
        operations.append({"operation": "replace_infinite", "columns": dict(sorted(infinite_by_column.items())), "replacement": "missing"})
    if duplicates_removed:
        operations.append({"operation": "remove_duplicates", "rows_removed": duplicates_removed})
    if columns_dropped:
        operations.append({"operation": "drop_columns", "columns": list(columns_dropped)})
    if indicator_columns:
        operations.append({"operation": "add_missing_indicators", "columns": list(indicator_columns)})
    if filled_by_column:
        operations.append({"operation": "fill_missing", "strategy": policy.missing, "values_changed": dict(sorted(filled_by_column.items()))})
    if capped_by_column:
        operations.append({"operation": "cap_outliers", "method": "iqr", "multiplier": policy.outlier_multiplier, "values_changed": dict(sorted(capped_by_column.items()))})
    return operations


def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def _safe_fill(series: pd.Series, value: object) -> pd.Series:
    if pd.api.types.is_integer_dtype(series.dtype) and isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            value = int(numeric_value)
        else:
            return series.astype("float64").fillna(numeric_value)
    return series.fillna(value)


@dataclass(frozen=True)
class ColumnQuality:
    name: str
    dtype: str
    rows: int
    missing: int
    missing_rate: float
    unique: int
    unique_rate: float
    finite: bool | None
    constant: bool
    likely_identifier: bool

@dataclass(frozen=True)
class QualityReport:
    rows: int
    columns: int
    duplicate_rows: int
    total_missing: int
    columns_with_missing: int
    column_profiles: tuple[ColumnQuality, ...] = field(default_factory=tuple)

    @property
    def missing_rate(self) -> float:
        return self.total_missing / (self.rows * self.columns) if self.rows and self.columns else 0.0

    def by_name(self) -> dict[str, ColumnQuality]:
        return {item.name: item for item in self.column_profiles}

def profile_frame(frame: pd.DataFrame) -> QualityReport:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    items: list[ColumnQuality] = []
    for name in frame.columns:
        s = frame[name]
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        finite = None
        if numeric:
            values = s.to_numpy(dtype=float, na_value=np.nan)
            finite = bool(np.isfinite(values[~np.isnan(values)]).all())
        items.append(ColumnQuality(
            name=str(name), dtype=str(s.dtype), rows=rows,
            missing=missing, missing_rate=missing / rows if rows else 0.0,
            unique=unique, unique_rate=unique / rows if rows else 0.0,
            finite=finite, constant=(unique <= 1 if rows else False),
            likely_identifier=(unique == rows and rows > 0),
        ))
    return QualityReport(
        rows=rows, columns=len(frame.columns), duplicate_rows=duplicate_rows,
        total_missing=int(frame.isna().sum().sum()),
        columns_with_missing=sum(x.missing > 0 for x in items),
        column_profiles=tuple(items),
    )


@dataclass(frozen=True)
class ColumnContract:
    name: str
    required: bool = True
    dtype_family: str = "any"
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[Any, ...] | None = None

@dataclass(frozen=True)
class DatasetContract:
    columns: tuple[ColumnContract, ...]
    allow_extra_columns: bool = False

@dataclass(frozen=True)
class ValidationIssue:
    column: str | None
    code: str
    message: str

@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            details = "; ".join(issue.message for issue in self.issues)
            raise ValueError(f"Dataset contract failed: {details}")

def _family_ok(series: pd.Series, family: str) -> bool:
    if family == "any": return True
    if family == "numeric": return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))
    if family == "categorical": return bool(isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
    if family == "datetime": return bool(pd.api.types.is_datetime64_any_dtype(series))
    if family == "boolean": return bool(pd.api.types.is_bool_dtype(series))
    if family == "string": return bool(pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series))
    raise ValueError(f"Unknown dtype_family: {family!r}")

def validate_contract(
    frame: pd.DataFrame,
    contract: DatasetContract,
    *,
    structural_only: bool = False,
) -> ValidationReport:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    issues: list[ValidationIssue] = []
    expected = {c.name for c in contract.columns}
    actual = {str(c) for c in frame.columns}
    for spec in contract.columns:
        if spec.name not in actual:
            if spec.required:
                issues.append(ValidationIssue(spec.name, "missing_column", f"Required column '{spec.name}' is missing."))
            continue
        s = frame[spec.name]
        if not _family_ok(s, spec.dtype_family):
            issues.append(ValidationIssue(spec.name, "dtype", f"Column '{spec.name}' does not satisfy dtype family '{spec.dtype_family}'."))
        if structural_only:
            continue
        if not spec.nullable and bool(s.isna().any()):
            issues.append(ValidationIssue(spec.name, "nullability", f"Column '{spec.name}' contains missing values but is not nullable."))
        if spec.min_value is not None and pd.api.types.is_numeric_dtype(s):
            if bool((s.dropna() < spec.min_value).any()):
                issues.append(ValidationIssue(spec.name, "min_value", f"Column '{spec.name}' contains values below {spec.min_value}."))
        if spec.max_value is not None and pd.api.types.is_numeric_dtype(s):
            if bool((s.dropna() > spec.max_value).any()):
                issues.append(ValidationIssue(spec.name, "max_value", f"Column '{spec.name}' contains values above {spec.max_value}."))
        if spec.allowed_values is not None:
            allowed = set(spec.allowed_values)
            invalid = s.dropna().map(lambda x: x not in allowed)
            if bool(invalid.any()):
                issues.append(ValidationIssue(spec.name, "allowed_values", f"Column '{spec.name}' contains values outside the declared domain."))
    if not contract.allow_extra_columns:
        for extra in sorted(actual - expected):
            issues.append(ValidationIssue(extra, "extra_column", f"Unexpected column '{extra}' is present."))
    return ValidationReport(valid=not issues, issues=tuple(issues))


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in frame.columns], separators=(",", ":")).encode())
    h.update(json.dumps([str(t) for t in frame.dtypes], separators=(",", ":")).encode())
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64", copy=False)
    h.update(values.tobytes())
    return h.hexdigest()

@dataclass(frozen=True)
class RunManifest:
    created_at_utc: str
    dataset_fingerprint: str
    rows: int
    columns: int
    cleaner_config: Mapping[str, Any]
    notes: str = ""

    @classmethod
    def create(cls, frame: pd.DataFrame, *, cleaner_config: Mapping[str, Any], notes: str = "") -> "RunManifest":
        return cls(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            dataset_fingerprint=dataset_fingerprint(frame),
            rows=len(frame), columns=len(frame.columns),
            cleaner_config=dict(cleaner_config), notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    audit: CleaningAudit
    validation: ValidationReport | None
    manifest: RunManifest

class AnalyticalPreprocessor:
    def __init__(self, cleaner: Cleaner, contract: DatasetContract | None = None) -> None:
        self.cleaner = cleaner
        self.contract = contract
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "AnalyticalPreprocessor":
        if self.contract is not None:
            structural = validate_contract(train, self.contract, structural_only=True)
            structural.raise_if_invalid()
        self.cleaner.fit(train)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame, *, notes: str = "") -> PreparedData:
        if not self._fitted:
            raise NotFittedError(
                "AnalyticalPreprocessor must be fitted on training data first."
            )
        if self.contract is not None:
            structural = validate_contract(frame, self.contract, structural_only=True)
            structural.raise_if_invalid()

        cleaned, audit = self.cleaner.transform(frame)
        validation = validate_contract(cleaned, self.contract) if self.contract is not None else None
        if validation is not None:
            validation.raise_if_invalid()
        manifest = RunManifest.create(
            cleaned,
            cleaner_config=self.cleaner.to_dict(),
            notes=notes,
        )
        return PreparedData(cleaned, audit, validation, manifest)

    def fit_transform(self, train: pd.DataFrame, *, notes: str = "training") -> PreparedData:
        self.fit(train)
        return self.transform(train, notes=notes)