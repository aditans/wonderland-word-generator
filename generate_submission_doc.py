"""
generate_submission_doc.py - Generates the official submission Word document (.docx)
and converts it to PDF (.pdf) as required by Assignment 2 Step 3 & 4.
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

STUDENT_NAME = "Aditanshu Sahu"
REGISTRATION_NUM = "230953444"
ASSIGNMENT_TITLE = "Foundations of Generative AI (ICT4403) - Assignment 2"
SYSTEM_TITLE = "Wonderland Word Generator: Comparative Autoregressive Text Generation using Simple RNN, Bidirectional RNN, LSTM, and GRU"
DATASET_NAME = "Alice's Adventures in Wonderland by Lewis Carroll"
DATASET_LINK = "https://www.gutenberg.org/ebooks/11"
GITHUB_REPO_LINK = "https://github.com/aditans/wonderland-word-generator"

DOCX_FILENAME = f"{STUDENT_NAME.replace(' ', '')}_{REGISTRATION_NUM}_FGAI_Assignment2.docx"
PDF_FILENAME = f"{STUDENT_NAME.replace(' ', '')}_{REGISTRATION_NUM}_FGAI_Assignment2.pdf"


def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def create_submission_document():
    """Builds a formatted Word document adhering strictly to assignment guidelines."""
    doc = Document()

    # Set page margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Header / Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(ASSIGNMENT_TITLE)
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(26, 54, 93)  # Deep Navy

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Department of Information & Communication Technology\nAssignment Submission & GitHub Repository Link")
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(74, 85, 104)

    doc.add_paragraph()  # Spacing

    # Student & Project Information Table
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    details = [
        ("Student Name", STUDENT_NAME),
        ("Registration Number", REGISTRATION_NUM),
        ("Assignment Title", ASSIGNMENT_TITLE),
        ("Title of Developed System", SYSTEM_TITLE),
        ("Name and Link of Dataset", f"{DATASET_NAME}\n{DATASET_LINK}"),
        ("GitHub Repository Link", GITHUB_REPO_LINK)
    ]

    col_widths = [Inches(2.2), Inches(4.3)]

    for row_idx, (label, val) in enumerate(details):
        row = table.rows[row_idx]

        cell_label = row.cells[0]
        cell_label.width = col_widths[0]
        set_cell_background(cell_label, "EDF2F7")  # Subtle Slate Gray
        p_label = cell_label.paragraphs[0]
        r_label = p_label.add_run(label)
        r_label.font.name = "Calibri"
        r_label.font.size = Pt(11)
        r_label.font.bold = True
        r_label.font.color.rgb = RGBColor(45, 55, 72)
        cell_label.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

        cell_val = row.cells[1]
        cell_val.width = col_widths[1]
        p_val = cell_val.paragraphs[0]
        r_val = p_val.add_run(val)
        r_val.font.name = "Calibri"
        r_val.font.size = Pt(11)
        if "http" in val:
            r_val.font.color.rgb = RGBColor(43, 108, 176)  # Blue Link color
            r_val.font.underline = True
        else:
            r_val.font.color.rgb = RGBColor(26, 32, 44)
        cell_val.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    doc.add_paragraph()  # Spacing

    # Executive Overview Section
    h2 = doc.add_paragraph()
    h2_run = h2.add_run("Executive Summary & Project Deliverables")
    h2_run.font.name = "Calibri"
    h2_run.font.size = Pt(14)
    h2_run.font.bold = True
    h2_run.font.color.rgb = RGBColor(26, 54, 93)

    summary_text = (
        "This project implements an autoregressive text-generation system trained on Lewis Carroll's "
        "'Alice's Adventures in Wonderland' obtained from Project Gutenberg. Four distinct recurrent "
        "architectures—Simple RNN, Bidirectional RNN, LSTM, and GRU—were systematically trained under "
        "strictly controlled experimental conditions (matching sequence length, embedding dimensions, "
        "recurrent units, optimization hyperparameters, and loss objectives).\n\n"
        "All code, Jupyter notebook, trained checkpoints, loss/accuracy curves, perplexity comparisons, "
        "runtime benchmarks, and qualitative sampling temperature generations (T = 0.5, 0.8, 1.2) are "
        "fully documented and open-source in the GitHub repository linked above."
    )
    p_sum = doc.add_paragraph(summary_text)
    p_sum.runs[0].font.name = "Calibri"
    p_sum.runs[0].font.size = Pt(11)

    # Key Components Checklist
    bullet_items = [
        "Jupyter Notebook (.ipynb) & Modular Python Pipeline (.py) in repository root and notebooks/",
        "Autoregressive rolling sequence dataset construction with Project Gutenberg text cleaning",
        "Comparable implementation of Simple RNN, Bidirectional RNN, LSTM, and GRU",
        "Quantitative Evaluation: Loss curves, validation accuracy, perplexity, and training runtime",
        "Qualitative Evaluation: Seed generation across three sampling temperatures (T=0.5, 0.8, 1.2) assessing coherence, diversity, and repetition",
        "Comprehensive README.md with full analysis, benchmark tables, and best model justification"
    ]

    for item in bullet_items:
        bp = doc.add_paragraph(style='List Bullet')
        r = bp.add_run(item)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)

    # Save Word document
    doc.save(DOCX_FILENAME)
    print(f"[Document] Generated Word document: {DOCX_FILENAME}")
    return DOCX_FILENAME


def convert_docx_to_pdf(docx_path):
    """Converts DOCX to PDF using docx2pdf or reportlab fallback."""
    print(f"[Document] Converting {docx_path} to PDF...")
    try:
        from docx2pdf import convert
        convert(docx_path, PDF_FILENAME)
        if os.path.exists(PDF_FILENAME):
            print(f"[Document] Successfully generated PDF via docx2pdf: {PDF_FILENAME}")
            return PDF_FILENAME
    except Exception as e:
        print(f"[Document] docx2pdf direct conversion notice: {e}")

    # Fallback to python-docx + ReportLab or Win32 COM
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        abs_docx = os.path.abspath(docx_path)
        abs_pdf = os.path.abspath(PDF_FILENAME)
        doc = word.Documents.Open(abs_docx)
        doc.SaveAs(abs_pdf, FileFormat=17)  # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        if os.path.exists(abs_pdf):
            print(f"[Document] Successfully generated PDF via Word COM: {PDF_FILENAME}")
            return PDF_FILENAME
    except Exception as e2:
        print(f"[Document] Word COM notice: {e2}")

    return None


if __name__ == "__main__":
    docx_file = create_submission_document()
    convert_docx_to_pdf(docx_file)
