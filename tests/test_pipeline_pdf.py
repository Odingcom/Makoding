"""PDF import/export regression and safety checks for Makoding recipes."""
from __future__ import annotations

from io import BytesIO
import json

import pytest
from pypdf import PdfReader, PdfWriter

from makoding.data_validation import export_rule_config
from makoding.pipeline_pdf import (
    MAX_PDF_BYTES, RECIPE_ATTACHMENT, export_recipe_pdf, import_recipe_pdf,
    export_audit_pdf,
)
from makoding.pipelines import PipelineRecipe, PipelineStep, import_recipe, export_recipe


def recipe() -> PipelineRecipe:
    return PipelineRecipe(
        name='Example: savings quality checks',
        steps=(PipelineStep('trim_whitespace', ('county',)),
               PipelineStep('remove_duplicates'),
               PipelineStep('fill_missing', ('savings',), 'median')),
        required_columns=('county', 'savings'),
        validation_rules=export_rule_config([], automatic=True),
        pre_gate='none', post_gate='critical',
    )


def test_importable_pdf_recipe_roundtrip():
    expected = recipe()
    pdf = export_recipe_pdf(expected)
    assert pdf.startswith(b'%PDF-')
    imported = import_recipe_pdf(pdf)
    assert imported.to_dict() == import_recipe(export_recipe(expected)).to_dict()
    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) >= 1
    assert set(reader.attachments) == {RECIPE_ATTACHMENT}
    text = ''.join(page.extract_text() or '' for page in reader.pages)
    assert 'Reusable Processing Recipe' in text
    assert 'savings quality checks' in text
    assert 'trim_whitespace' in text


def test_pdf_import_rejects_random_pdf_without_attachment():
    pdf = export_audit_pdf({'status': 'completed', 'recipe_name': 'sample', 'steps': []})
    assert pdf.startswith(b'%PDF-')
    with pytest.raises(ValueError, match='exactly one Makoding'):
        import_recipe_pdf(pdf)


def test_rejects_large_nonpdf_and_corrupted_input():
    with pytest.raises(ValueError, match='valid PDF'):
        import_recipe_pdf(b'garbage')
    with pytest.raises(ValueError, match='2 MB'):
        import_recipe_pdf(b'%PDF-' + b'z' * MAX_PDF_BYTES)
    with pytest.raises(ValueError, match='cannot be read'):
        import_recipe_pdf(b'%PDF-1.5\nthis is not a real pdf')


def test_rejects_wrong_attachment():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_attachment(filename='other_data.json', data=export_recipe(recipe()).encode())
    buff = BytesIO()
    writer.write(buff)
    with pytest.raises(ValueError, match='exactly one Makoding'):
        import_recipe_pdf(buff.getvalue())


def test_rejects_unsafe_attachment_recipe_json():
    payload = json.loads(export_recipe(recipe()))
    payload['steps'][0]['operation'] = 'execute_shell'
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_attachment(filename=RECIPE_ATTACHMENT, data=json.dumps(payload).encode())
    buff = BytesIO()
    writer.write(buff)
    with pytest.raises(ValueError, match='Unknown pipeline operation'):
        import_recipe_pdf(buff.getvalue())


def test_audit_pdf_summarizes_metadata_and_not_rows():
    pdf = export_audit_pdf({'status': 'completed', 'recipe_name': 'chama',
                            'recipe_sha256': '012345', 'steps': [{'status': 'completed'}],
                            'input_quality': {'rows': 5}, 'output_quality': {'rows': 4}})
    reader = PdfReader(BytesIO(pdf))
    text = ''.join(page.extract_text() or '' for page in reader.pages)
    assert 'Pipeline Execution Audit' in text
    assert 'chama' in text
    assert 'rows: 4' in text
    assert reader.attachments == {}


def test_pdf_plain_text_editor_not_source_of_truth():
    pdf = export_recipe_pdf(recipe())
    assert import_recipe_pdf(pdf).name == recipe().name
    # Export is deterministic at the recipe-content level despite PDF metadata differences.
    assert import_recipe_pdf(export_recipe_pdf(recipe())).to_dict() == import_recipe_pdf(pdf).to_dict()
