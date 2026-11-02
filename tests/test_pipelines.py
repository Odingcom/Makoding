"""Phase 3 regression tests: portable, transactional cleaning recipes."""
from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from makoding.data_validation import Rule, export_rule_config
from makoding.pipelines import (
    MAX_RECIPE_BYTES, PipelineRecipe, PipelineStep, execute_recipe,
    export_recipe, import_recipe,
)


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "member_id": ["M01", "M02", "M02", "M03"],
        "county": [" Nairobi ", "Kisumu", "Kisumu", "?"],
        "savings": [10.0, 20.0, 20.0, None],
        "age": [25, 40, 40, 33],
    })


def recipe(*steps, rules=(), pre="none", post="critical", required=()):
    return PipelineRecipe(
        name="Chama data quality", steps=tuple(steps),
        required_columns=tuple(required),
        validation_rules=export_rule_config(list(rules), automatic=True),
        pre_gate=pre, post_gate=post,
    )


def steps():
    return (
        PipelineStep("remove_duplicates"),
        PipelineStep("trim_whitespace", ("county",)),
        PipelineStep("standardize_missing", ("county",)),
        PipelineStep("fill_missing", ("savings",), "median"),
        PipelineStep("fill_missing", ("county",), "mode"),
    )


def test_recipe_roundtrip_and_portable_json():
    item = recipe(*steps(), required=("member_id", "savings"))
    got = import_recipe(export_recipe(item))
    assert got.to_dict() == item.to_dict()
    assert got.steps[3].strategy == "median"
    assert len(json.loads(export_recipe(item))["steps"]) == 5


def test_pipeline_output_and_original_unchanged():
    frame = sample()
    original = frame.copy(deep=True)
    run = execute_recipe(frame, recipe(*steps()))
    assert run.status == "completed", run.reason
    assert run.output is not None
    assert len(run.output) == 3
    assert run.output["county"].tolist() == ["Nairobi", "Kisumu", "Kisumu"]
    assert run.output["savings"].isna().sum() == 0
    pd.testing.assert_frame_equal(frame, original)
    assert run.output is not frame
    assert run.output_quality["missing_cells"] == 0
    assert len(run.steps) == 5


def test_replay_same_data_same_result():
    r = import_recipe(export_recipe(recipe(*steps())))
    first = execute_recipe(sample(), r)
    second = execute_recipe(sample(), r)
    assert first.status == second.status == "completed"
    pd.testing.assert_frame_equal(first.output, second.output)
    assert first.recipe_sha256 == second.recipe_sha256


def test_replay_on_second_dataset_recomputes_median():
    r = recipe(PipelineStep("fill_missing", ("savings",), "median"))
    first = execute_recipe(pd.DataFrame({"savings": [1., 3., None]}), r)
    second = execute_recipe(pd.DataFrame({"savings": [100., 300., None]}), r)
    assert first.output.iloc[2, 0] == 2
    assert second.output.iloc[2, 0] == 200


def test_missing_required_column_blocks_without_changes():
    frame = sample()
    orig = frame.copy(deep=True)
    run = execute_recipe(frame, recipe(PipelineStep("remove_duplicates"), required=("absent",)))
    assert run.status == "blocked" and run.output is None
    assert "absent" in run.reason
    pd.testing.assert_frame_equal(frame, orig)


def test_pre_gate_blocks_critical_missing_before_clean():
    frame = sample()
    rule = Rule("required", "savings", "critical", {"include_text_tokens": True})
    run = execute_recipe(frame, recipe(PipelineStep("fill_missing", ("savings",), "median"),
                                       rules=[rule], pre="critical"))
    assert run.status == "blocked"
    assert not run.steps
    assert run.output is None
    assert run.pre_validation is not None


def test_post_gate_passes_after_missing_fixed():
    rule = Rule("required", "savings", "critical", {"include_text_tokens": True})
    run = execute_recipe(sample(), recipe(PipelineStep("fill_missing", ("savings",), "median"),
                                          rules=[rule], pre="none", post="critical"))
    assert run.status == "completed", run.reason
    assert run.pre_validation.dataset["critical_checks"] == 1
    assert run.post_validation.dataset["critical_checks"] == 0


def test_post_gate_blocks_transactionally():
    frame = sample()
    orig = frame.copy(deep=True)
    rule = Rule("range", "age", "critical", {"min": 30, "max": 120})
    run = execute_recipe(frame, recipe(PipelineStep("fill_missing", ("savings",), "median"),
                                       rules=[rule], pre="none", post="critical"))
    assert run.status == "blocked" and run.output is None
    assert run.post_validation.dataset["critical_checks"] == 1
    pd.testing.assert_frame_equal(frame, orig)


def test_warning_gate_blocks_where_critical_gate_allows():
    rule = Rule("range", "age", "warning", {"min": 30})
    step = PipelineStep("remove_duplicates")
    assert execute_recipe(sample(), recipe(step, rules=[rule], post="critical")).status == "completed"
    assert execute_recipe(sample(), recipe(step, rules=[rule], post="warning")).status == "blocked"


def test_failed_middle_step_does_not_return_partial_frame():
    frame = sample()
    orig = frame.copy(deep=True)
    run = execute_recipe(frame, recipe(
        PipelineStep("remove_duplicates"),
        PipelineStep("fill_missing", ("nonexistent",), "median"),
    ))
    assert run.status == "failed" and run.output is None
    assert run.steps[-1]["status"] == "failed"
    pd.testing.assert_frame_equal(frame, orig)


def test_no_raw_values_in_default_audit():
    frame = pd.DataFrame({"name": ["SECRET_PERSON_123", "SECRET_PERSON_123"], "n": [1, 1]})
    run = execute_recipe(frame, recipe(PipelineStep("remove_duplicates")))
    audit = json.dumps(run.audit())
    assert "SECRET_PERSON_123" not in audit
    assert "recipe_sha256" in audit
    assert "pandas_version" in audit
    assert run.status == "completed"


def test_invalid_json_rejected():
    for value in ("not json", "[]", "{}", b"\xff", '{"a":NaN}', '{"a":1,"a":2}'):
        with pytest.raises((ValueError, UnicodeError)):
            import_recipe(value)


def test_limit_json_size():
    with pytest.raises(ValueError, match="120 KB"):
        import_recipe(b" " * (MAX_RECIPE_BYTES + 1))


def test_unrecognized_schema_version_rejected():
    obj = recipe(PipelineStep("remove_duplicates")).to_dict()
    obj["schema_version"] = 2
    with pytest.raises(ValueError, match="version"):
        import_recipe(json.dumps(obj))


def test_unknown_recipe_key_rejected():
    obj = recipe(PipelineStep("remove_duplicates")).to_dict()
    obj["unsafe_code"] = "__import__('os').system('do-not-run')"
    with pytest.raises(ValueError, match="fields"):
        import_recipe(json.dumps(obj))


def test_empty_steps_rejected():
    with pytest.raises(ValueError, match="1–30"):
        export_recipe(recipe())


def test_too_many_steps_rejected():
    with pytest.raises(ValueError, match="1–30"):
        export_recipe(recipe(*[PipelineStep("remove_duplicates") for _ in range(31)]))


@pytest.mark.parametrize("name", ["__magic__", "import os", "custom_transform"])
def test_only_whitelisted_operations(name):
    with pytest.raises(ValueError, match="Unknown pipeline operation"):
        export_recipe(recipe(PipelineStep(name)))


@pytest.mark.parametrize("operation", ["trim_whitespace", "standardize_missing", "drop_missing_rows", "drop_columns"])
def test_columns_mandatory_for_column_operations(operation):
    with pytest.raises(ValueError, match="requires columns"):
        export_recipe(recipe(PipelineStep(operation)))


def test_duplicates_reject_column_selection():
    with pytest.raises(ValueError, match="extra parameters"):
        export_recipe(recipe(PipelineStep("remove_duplicates", ("age",))))


def test_numeric_missing_fill_requires_strategy():
    with pytest.raises(ValueError, match="supported strategy"):
        export_recipe(recipe(PipelineStep("fill_missing", ("age",))))


def test_constant_replacement_requires_nonempty():
    with pytest.raises(ValueError, match="Constant replacement"):
        export_recipe(recipe(PipelineStep("fill_missing", ("county",), "constant", "")))


def test_nonconstant_replacement_rejects_constant():
    with pytest.raises(ValueError, match="Only constant"):
        export_recipe(recipe(PipelineStep("fill_missing", ("age",), "median", "0")))


def test_duplicate_required_columns_rejected():
    with pytest.raises(ValueError, match="unique"):
        export_recipe(recipe(PipelineStep("remove_duplicates"), required=("member_id", "member_id")))


def test_invalid_gates_rejected():
    with pytest.raises(ValueError, match="gate"):
        export_recipe(replace(recipe(PipelineStep("remove_duplicates")), post_gate="ignore_all"))


def test_malformed_rule_config_rejected():
    with pytest.raises(ValueError, match="validation"):
        export_recipe(replace(recipe(PipelineStep("remove_duplicates")), validation_rules='{"bad": 1}'))


def test_validation_with_invalid_existing_column_blocks():
    r = Rule("required", "missing_column", "critical")
    run = execute_recipe(sample(), recipe(PipelineStep("remove_duplicates"), rules=[r]))
    assert run.status == "blocked"
    assert "Pre-validation" in run.reason


def test_step_failure_wrong_dtype_leaves_original_intact():
    frame = sample()
    original = frame.copy(deep=True)
    run = execute_recipe(frame, recipe(PipelineStep("trim_whitespace", ("age",))))
    assert run.status == "failed" and run.output is None
    pd.testing.assert_frame_equal(frame, original)


def test_report_quality_metadata_only():
    run = execute_recipe(sample(), recipe(PipelineStep("remove_duplicates")))
    assert set(run.audit()["output_quality"]) == {"rows", "columns", "missing_cells", "missing_pct", "duplicate_rows"}
    assert run.audit()["steps"][0]["removed_rows"] == 1


def test_recipe_with_automatic_checks_disabled_and_rules():
    r = recipe(PipelineStep("remove_duplicates"))
    replacement = export_rule_config([Rule("required", "member_id", "critical")], automatic=False)
    r = replace(r, validation_rules=replacement)
    assert execute_recipe(sample(), r).status == "completed"


def test_empty_checks_disallowed():
    r = recipe(PipelineStep("remove_duplicates"))
    with pytest.raises(ValueError, match="Enable baseline"):
        export_recipe(replace(r, validation_rules=export_rule_config([], automatic=False)))


def test_strict_gate_blocks_when_checks_are_skipped():
    frame = pd.DataFrame({"nested": [[1], [1], [2]], "tag": [" a ", " b ", " c "]})
    r = recipe(PipelineStep("trim_whitespace", ("tag",)), post="critical")
    result = execute_recipe(frame, r)
    assert result.output is None and result.status == "blocked"
    assert "could not be evaluated" in result.reason
    assert result.post_validation.skipped_checks


def test_rule_after_drop_column_blocks_pipeline():
    frame = sample()
    check = Rule("required_column", "savings", "critical")
    result = execute_recipe(frame, recipe(PipelineStep("drop_columns", ("savings",)),
                                          rules=(check,), pre="none", post="critical"))
    assert result.status == "blocked" and result.output is None
    assert "Post-validation" in result.reason
