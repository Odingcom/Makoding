"""Regression/safety tests for the non-destructive validation engine."""
import json
import numpy as np
import pandas as pd
import pytest

from makoding.data_validation import (
    Rule, export_rule_config, import_rule_config, validate_dataframe,
)


def _find(report, name, column=None):
    return [f for f in report.findings if f.check == name and (column is None or f.column == column)]


def test_automatic_baseline_never_modifies_dataset():
    frame = pd.DataFrame({"age": [10, None, 10], "area": [" Nairobi ", "?", " Nairobi "]})
    original = frame.copy(deep=True)
    report = validate_dataframe(frame)
    assert _find(report, "Missing values", "age")[0].affected_rows == 1
    assert _find(report, "Surrounding whitespace", "area")[0].affected_rows == 2
    assert _find(report, "Text missing markers", "area")[0].affected_rows == 1
    pd.testing.assert_frame_equal(original, frame)
    assert report.dataset["rows"] == 3


def test_duplicate_rows_count_all_collisions():
    frame = pd.DataFrame({"code": ["X", "X", "Y", "Y", "Z"]})
    finding = _find(validate_dataframe(frame), "Duplicate rows")[0]
    assert finding.affected_rows == 4
    assert finding.examples[0]["row_position"] == 1


def test_duplicate_index_labels_do_not_corrupt_row_locations():
    frame = pd.DataFrame({"code": ["A", "A", "B"]}, index=[3, 3, 3])
    finding = _find(validate_dataframe(frame, [Rule("unique", "code")], automatic=False), "Unique identifier")[0]
    assert [ex["row_position"] for ex in finding.examples] == [1, 2]


def test_required_treats_configured_tokens_as_missing():
    frame = pd.DataFrame({"a": [None, " NA ", "", "value"]})
    rule = Rule("required", "a", "critical", {"include_text_tokens": True})
    assert _find(validate_dataframe(frame, [rule], automatic=False), "Required values")[0].affected_rows == 3


def test_required_can_ignore_text_tokens():
    frame = pd.DataFrame({"a": [None, "NA"]})
    rule = Rule("required", "a", params={"include_text_tokens": False})
    assert _find(validate_dataframe(frame, [rule], automatic=False), "Required values")[0].affected_rows == 1


def test_numeric_parse_flags_nonfinite_and_invalid_strings():
    frame = pd.DataFrame({"a": ["1", "2.3", "no", "inf", None]})
    finding = _find(validate_dataframe(frame, [Rule("numeric", "a")], automatic=False), "Numeric parse")[0]
    assert finding.affected_rows == 2


def test_date_parse_with_explicit_format():
    frame = pd.DataFrame({"when": ["2026-02-28", "2026-02-30", "28/02/2026", None]})
    rule = Rule("date", "when", params={"format": "%Y-%m-%d"})
    finding = _find(validate_dataframe(frame, [rule], automatic=False), "Date parse")[0]
    assert finding.affected_rows == 2


def test_range_flags_negative_and_unparseable():
    frame = pd.DataFrame({"age": [20, -1, 200, "bad", None]})
    finding = _find(validate_dataframe(frame, [Rule("range", "age", params={"min": 0, "max": 120})], automatic=False), "Numeric range")[0]
    assert finding.affected_rows == 3


def test_categories_insensitive_and_space_normalized():
    frame = pd.DataFrame({"county": [" Nairobi ", "KISUMU", "Nakuru", None]})
    rule = Rule("allowed_values", "county", params={"values": ["Nairobi", "Kisumu"]})
    finding = _find(validate_dataframe(frame, [rule], automatic=False), "Allowed categories")[0]
    assert finding.affected_rows == 1


def test_unique_excludes_missing_ids():
    frame = pd.DataFrame({"member_id": ["A", "A", None, None]})
    finding = _find(validate_dataframe(frame, [Rule("unique", "member_id")], automatic=False), "Unique identifier")[0]
    assert finding.affected_rows == 2


def test_iqr_outliers_and_zero_spread():
    frame = pd.DataFrame({"v": [1, 1, 2, 2, 2, 3, 100]})
    finding = _find(validate_dataframe(frame, [Rule("iqr_outlier", "v")], automatic=False), "IQR outliers")[0]
    assert finding.affected_rows == 1
    assert not validate_dataframe(pd.DataFrame({"v": [1, 1, 1, 1]}), [Rule("iqr_outlier", "v")], automatic=False).findings


def test_zscore_outliers():
    frame = pd.DataFrame({"v": [0] * 100 + [100]})
    finding = _find(validate_dataframe(frame, [Rule("zscore_outlier", "v")], automatic=False), "Z-score outliers")[0]
    assert finding.affected_rows == 1


def test_outlier_small_sample_is_not_claimed_passed():
    report = validate_dataframe(pd.DataFrame({"v": [1, 100, 2]}),
                                [Rule("iqr_outlier", "v")], automatic=False)
    assert report.findings == ()
    assert report.rules_run == 1


def test_numeric_type_suggestion_conservative_for_ids():
    frame = pd.DataFrame({"phone_code": ["001", "002", "003", "004", "005"],
                          "measurement": ["1", "2", "3", "4", "5"]})
    suggestions = _find(validate_dataframe(frame), "Numeric type suggestion")
    assert len(suggestions) == 1 and suggestions[0].column == "measurement"


def test_export_import_roundtrip_and_json_report():
    rules = [Rule("range", "x", "critical", {"min": 0, "max": 9}),
             Rule("allowed_values", "y", params={"values": ["a", "b"]})]
    encoded = export_rule_config(rules, automatic=False)
    decoded, auto = import_rule_config(encoded)
    assert decoded == rules and not auto
    report = validate_dataframe(pd.DataFrame({"x": [1, 10], "y": ["b", "c"]}), decoded, automatic=auto)
    assert json.loads(json.dumps(report.to_dict()))["dataset"]["flagged_checks"] == 2


@pytest.mark.parametrize("payload", [
    "not json", '{"schema_version": 999, "rules": [], "automatic_checks": true}',
    '{"schema_version": 1, "rules": [{"kind": "unknown", "column": "x", "severity": "warning", "params": {}}], "automatic_checks": true}',
    '{"schema_version": 1, "rules": [], "automatic_checks": true, "extra": 1}',
    '{"schema_version": 1, "rules": [], "automatic_checks": NaN}',
])
def test_import_rejects_invalid_rule_config(payload):
    with pytest.raises(ValueError):
        import_rule_config(payload)


@pytest.mark.parametrize("bad_rule", [
    Rule("range", "a", params={"min": 30, "max": 10}),
    Rule("range", "a", params={"min": float("nan")}),
    Rule("date", "a", params={"format": "freeform"}),
    Rule("allowed_values", "a", params={"values": []}),
    Rule("iqr_outlier", "a", params={"multiplier": -1}),
    Rule("required", "absent"),
    Rule("numeric", "a", "urgent"),
])
def test_invalid_rule_fails_explicitly(bad_rule):
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame({"a": [1, 2]}), [bad_rule])


def test_evidence_truncated_but_counts_complete():
    frame = pd.DataFrame({"x": [None] * 100})
    report = validate_dataframe(frame, [Rule("required", "x")], automatic=False, max_examples=3)
    f = report.findings[0]
    assert f.affected_rows == 100 and len(f.examples) == 3


def test_empty_frame_gracefully_scans():
    frame = pd.DataFrame({"x": pd.Series(dtype="float64")})
    report = validate_dataframe(frame)
    assert report.dataset["rows"] == 0 and report.findings == ()


def test_missing_column_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate column"):
        validate_dataframe(pd.DataFrame([[1, 2]], columns=["A", "A"]))


def test_no_automated_fix_and_clean_series():
    frame = pd.DataFrame({"x": [10, 20], "y": ["A", "B"]})
    assert validate_dataframe(frame).findings == ()


def test_required_column_schema_detects_missing_expected_column():
    frame = pd.DataFrame({"age": [10, 20]})
    report = validate_dataframe(frame, [Rule("required_column", "member_id", "critical")], automatic=False)
    match = _find(report, "Required column", "member_id")
    assert len(match) == 1 and match[0].severity == "critical"
    assert match[0].affected_rows == 2


def test_required_column_schema_passes_if_present():
    report = validate_dataframe(pd.DataFrame({"member_id": ["a"]}),
                                [Rule("required_column", "member_id")], automatic=False)
    assert report.findings == ()


def test_expected_dtype_schema_flags_mismatch():
    frame = pd.DataFrame({"a": ["1", "2"]})
    rule = Rule("dtype", "a", params={"expected": "numeric"})
    assert _find(validate_dataframe(frame, [rule], automatic=False), "Column data type")[0].affected_rows == 2
    assert not validate_dataframe(frame, [Rule("dtype", "a", params={"expected": "text"})], automatic=False).findings


def test_date_type_suggestion_is_non_destructive():
    frame = pd.DataFrame({"signup_date": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"]})
    original = frame.copy(deep=True)
    findings = _find(validate_dataframe(frame), "Date type suggestion")
    assert len(findings) == 1 and findings[0].column == "signup_date"
    pd.testing.assert_frame_equal(frame, original)


def test_skipped_outlier_check_is_explicit_in_report():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    report = validate_dataframe(frame, [Rule("iqr_outlier", "a")], automatic=False)
    assert report.skipped_checks and "fewer than four" in report.skipped_checks[0]
    assert "skipped_checks" in report.to_dict()


def test_unhashable_duplicate_scan_is_marked_skipped():
    frame = pd.DataFrame({"nested": [[1], [1], [2]], "tag": ["a", "a", "b"]})
    report = validate_dataframe(frame)
    assert any("Duplicate rows" in text for text in report.skipped_checks)


def test_requires_at_least_one_active_check():
    with pytest.raises(ValueError, match="No checks selected"):
        validate_dataframe(pd.DataFrame({"a": [1]}), automatic=False)
