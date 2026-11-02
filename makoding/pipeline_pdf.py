"""Portable, readable PDF recipes for Makoding Pipelines.

The human-readable PDF carries the canonical recipe JSON as a genuine PDF
attachment. Re-import reads ONLY that attachment and validates it using the
existing strict recipe parser. It never executes or interprets PDF text.

Dependencies: reportlab and pypdf (imported lazily to keep JSON features usable).
"""
from __future__ import annotations

from io import BytesIO
from hashlib import sha256
from html import escape
from datetime import datetime, timezone
from typing import Any

from makoding.pipelines import PipelineRecipe, export_recipe, import_recipe

RECIPE_ATTACHMENT = "makoding_pipeline_recipe.json"
MAX_PDF_BYTES = 2_000_000
MAX_PDF_PAGES = 20


def _libraries():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError("PDF support needs both 'reportlab' and 'pypdf'. Install: pip install reportlab pypdf") from exc
    return locals()


def _s(value: object, limit: int = 800) -> str:
    return escape(str(value)[:limit], quote=False)


def _pdf_body(*, heading: str, subtitle: str, sections: list[tuple[str, list[tuple[str, str]]]], note: str) -> bytes:
    lib = _libraries()
    colors, A4, mm = lib['colors'], lib['A4'], lib['mm']
    Paragraph, Spacer, Table, TableStyle = lib['Paragraph'], lib['Spacer'], lib['Table'], lib['TableStyle']
    getSampleStyleSheet, ParagraphStyle = lib['getSampleStyleSheet'], lib['ParagraphStyle']
    stream = BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MKHeader", parent=styles['Heading1'], fontName="Helvetica-Bold",
                              fontSize=19, leading=24, spaceAfter=9, textColor=colors.HexColor('#183B50')))
    styles.add(ParagraphStyle(name="MKSub", parent=styles['Normal'], fontSize=10,
                              leading=15, spaceAfter=14, textColor=colors.HexColor('#4E6370')))
    styles.add(ParagraphStyle(name="MKSection", parent=styles['Heading2'], fontName="Helvetica-Bold",
                              fontSize=12, leading=17, spaceBefore=14, spaceAfter=7,
                              textColor=colors.HexColor('#165D73')))
    styles.add(ParagraphStyle(name="MKCell", parent=styles['Normal'], fontSize=8.6,
                              leading=12, wordWrap='CJK'))
    styles.add(ParagraphStyle(name="MKNote", parent=styles['Normal'], fontSize=9,
                              leading=14, textColor=colors.HexColor('#546774')))
    doc = lib['SimpleDocTemplate'](stream, pagesize=A4, topMargin=20 * mm,
                                   bottomMargin=20 * mm, leftMargin=19 * mm, rightMargin=19 * mm,
                                   title=heading, author="Makoding DataLab Pro")
    story = [Paragraph(_s(heading), styles['MKHeader']),
             Paragraph(_s(subtitle), styles['MKSub'])]
    available = A4[0] - 38 * mm
    for section_name, records in sections:
        story.append(Paragraph(_s(section_name), styles['MKSection']))
        if not records:
            story.append(Paragraph('No records.', styles['MKNote']))
            continue
        rows = [[Paragraph('FIELD', styles['MKCell']), Paragraph('VALUE', styles['MKCell'])]]
        for key, value in records:
            rows.append([Paragraph(_s(key, 140), styles['MKCell']),
                         Paragraph(_s(value, 1800).replace('\n', '<br/>'), styles['MKCell'])])
        table = Table(rows, colWidths=[available * .28, available * .72], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DDECF0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F9FA')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0, 0), (-1, -1), .4, colors.HexColor('#CBD7DC')),
            ('INNERGRID', (0, 0), (-1, -1), .3, colors.HexColor('#D6E1E5')),
            ('LEFTPADDING', (0, 0), (-1, -1), 9),
            ('RIGHTPADDING', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(table)
        story.append(Spacer(1, 8))
    story += [Spacer(1, 13), Paragraph(_s(note), styles['MKNote'])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#61727D'))
        canvas.drawString(19 * mm, 13 * mm, 'MAKODING  /  DataLab Pro  /  Reproducibility')
        canvas.drawRightString(A4[0] - 19 * mm, 13 * mm, f'Page {doc.page}')
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def export_recipe_pdf(recipe: PipelineRecipe) -> bytes:
    """Create a readable PDF with embedded, importable, schema-checked JSON."""
    import json
    canonical = export_recipe(recipe).encode('utf-8')
    record = json.loads(canonical)
    recipe_sha = sha256(canonical).hexdigest()
    sections = [
        ('Recipe details', [
            ('Name', record['name']),
            ('Schema version', str(record['schema_version'])),
            ('Required columns', ', '.join(record['required_columns']) or 'None'),
            ('Before-cleaning gate', record['pre_gate']),
            ('After-cleaning gate', record['post_gate']),
            ('Recipe SHA-256', recipe_sha),
        ]),
        ('Processing operations', [
            (f"Step {i}: {step['operation']}",
             f"Columns: {', '.join(step['columns']) or 'all rows'}; strategy: {step.get('strategy') or 'none'}"
             + (f"; replacement: {step['constant']}" if step.get('constant') is not None else ''))
            for i, step in enumerate(record['steps'], 1)
        ]),
        ('Validation configuration', [
            ('Automatic checks', str(record['validation_rules'].get('automatic', 'configured in JSON'))),
            ('Rule configuration', 'A full copy of the versioned validation-rule configuration is embedded as JSON, not reduced to this summary.'),
        ]),
    ]
    plain = _pdf_body(heading='Reusable Processing Recipe',
                      subtitle='Makoding DataLab Pro  |  PDF recipe with machine-readable JSON attachment',
                      sections=sections,
                      note=('IMPORT: In Clean > Pipelines, choose this PDF and click Load recipe. '
                            'The app imports the embedded makoding_pipeline_recipe.json attachment. '
                            'Editing the visible PDF text does NOT edit the recipe. '
                            'No raw dataset rows are included, but constants and rule settings may be sensitive.'))
    lib = _libraries()
    PdfWriter = lib['PdfWriter']
    writer = PdfWriter()
    writer.append(BytesIO(plain))
    writer.add_attachment(filename=RECIPE_ATTACHMENT, data=canonical)
    writer.add_metadata({'/Title': f"Makoding Recipe: {record['name']}",
                         '/Subject': 'Portable Makoding pipeline recipe with attached JSON',
                         '/Creator': 'Makoding DataLab Pro'})
    result = BytesIO()
    writer.write(result)
    pdf = result.getvalue()
    if len(pdf) > MAX_PDF_BYTES:
        raise ValueError('Generated recipe PDF exceeds the 2 MB safety limit.')
    return pdf


def import_recipe_pdf(payload: bytes) -> PipelineRecipe:
    """Import only the canonical attachment, never scraped/untrusted PDF text."""
    if not isinstance(payload, (bytes, bytearray)) or not payload.startswith(b'%PDF-'):
        raise ValueError('Choose a valid PDF exported by Makoding Pipelines.')
    if len(payload) > MAX_PDF_BYTES:
        raise ValueError('PDF exceeds the maximum supported size of 2 MB.')
    try:
        lib = _libraries()
        reader = lib['PdfReader'](BytesIO(payload), strict=True)
        if reader.is_encrypted:
            raise ValueError('Password-protected PDFs are not supported for recipe import.')
        if not 1 <= len(reader.pages) <= MAX_PDF_PAGES:
            raise ValueError('PDF page count is outside the allowed range.')
        attachments = reader.attachments
        if set(attachments) != {RECIPE_ATTACHMENT}:
            raise ValueError('PDF must contain exactly one Makoding recipe attachment.')
        files = attachments[RECIPE_ATTACHMENT]
        if len(files) != 1:
            raise ValueError('PDF contains an ambiguous recipe attachment.')
        content = files[0]
        if not isinstance(content, bytes):
            raise ValueError('Recipe PDF attachment is not binary JSON.')
        return import_recipe(content)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('PDF cannot be read as a Makoding recipe.') from exc


def export_audit_pdf(audit: dict[str, Any]) -> bytes:
    """Human-readable execution audit PDF; intentionally NOT recipe-importable."""
    if not isinstance(audit, dict):
        raise TypeError('Expected a pipeline execution audit dictionary.')
    records = [
        ('Status', audit.get('status', 'unknown')),
        ('Recipe', audit.get('recipe_name', 'unknown')),
        ('Recipe hash', audit.get('recipe_sha256', 'unknown')),
        ('Execution UTC', audit.get('started_at_utc', 'unknown')),
        ('Failure / block reason', audit.get('reason') or 'None'),
    ]
    quality = []
    for name, key in [('Input', 'input_quality'), ('Output', 'output_quality')]:
        metrics = audit.get(key) or {}
        quality.append((name, ', '.join(f'{k}: {v}' for k, v in metrics.items()) if metrics else 'Unavailable'))
    steps = []
    for i, item in enumerate(audit.get('steps', [])[:100], 1):
        steps.append((f'Step {i}', ', '.join(f'{k}={v}' for k, v in item.items())[:1200]))
    validations = []
    for label, key in [('Before', 'pre_validation'), ('After', 'post_validation')]:
        info = audit.get(key) or {}
        validations.append((label, ', '.join(f'{k}: {v}' for k, v in info.items() if k != 'findings') if info else 'Not available'))
        for finding in info.get('findings', [])[:40]:
            validations.append((f'{label} finding',
                f"{finding.get('severity')} / {finding.get('check')} / {finding.get('column')} - "
                f"{finding.get('affected_rows')} row(s)"))
    return _pdf_body(heading='Pipeline Execution Audit',
                     subtitle='Makoding DataLab Pro  |  Human-readable execution and quality report',
                     sections=[('Execution summary', records), ('Quality metrics', quality),
                               ('Step history', steps), ('Validation findings', validations)],
                     note=('This PDF is for review and sharing. It is NOT a reusable recipe: '
                           'to re-import processing steps, export the recipe PDF or JSON instead. '
                           'Quality findings are diagnostic, not proof that all data are correct.'))
