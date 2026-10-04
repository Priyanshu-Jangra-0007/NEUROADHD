import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 752, "NEUROADHD & BENCHMARK SUITE — WORKSPACE ARCHITECTURE & TOF SPECIFICATION")
            self.setFont("Helvetica", 8)
            self.drawRightString(612 - 54, 752, "FINAL SYSTEM SPECIFICATION")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 744, 612 - 54, 744)
            
        # Footer (All Pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(54, 45, 612 - 54, 45)
        
        self.setFont("Helvetica", 8)
        self.drawString(54, 30, "CONFIDENTIAL & PROPRIETARY — NEUROADHD RESEARCH BENCHMARK PIPELINE")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.setFont("Helvetica-Bold", 8)
        self.drawRightString(612 - 54, 30, page_str)
        self.restoreState()

def create_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#0F172A")
    secondary_color = colors.HexColor("#0284C7")
    text_color = colors.HexColor("#334155")
    bg_light = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14.5,
        textColor=secondary_color,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=text_color,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    code_tree_style = ParagraphStyle(
        'CodeTreeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=text_color
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=primary_color
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.white
    )

    story = []

    # Document Header Title Block
    story.append(Paragraph("NeuroADHD Final System Architecture & Benchmark Specification", title_style))
    story.append(Paragraph("Complete Directory Hierarchy (Tree of Folders), Isolated Environments, Multi-Model Pipeline & Results", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceAfter=10))

    # Section 1: Executive Summary & System Overview
    story.append(Paragraph("1. Executive Summary & System Identity", h1_style))
    summary_text = (
        "This specification documents the complete final architecture for the <b>NeuroADHD</b> research ecosystem and its "
        "reproducible multi-model benchmarking workspace <b>NeuroADHD_ModelBenchmark</b>. The system performs automated binary classification "
        "of <b>Attention Deficit Hyperactivity Disorder (ADHD=1)</b> vs. <b>Typically Developing Controls (Control=0)</b> using resting-state "
        "functional MRI (rs-fMRI) functional connectivity features derived from the <b>ADHD-200 Consortium</b>."
    )
    story.append(Paragraph(summary_text, body_style))

    key_facts = [
        "<b>Original Core Workspace (NeuroADHD/):</b> Treated as 100% READ-ONLY and preserved without modification.",
        "<b>Primary ML Feature Table:</b> 153 subjects × 19,900 Fisher-z transformed CC200 ROI connectivity features.",
        "<b>Partition Strategy:</b> Whole-site group holdout (114 Train, 19 NeuroIMAGE Validation, 20 Peking_2 Test).",
        "<b>Model Benchmark Workspace (NeuroADHD_ModelBenchmark/):</b> Isolated multi-model benchmarking environment.",
        "<b>Isolated Virtual Environments:</b> 7 independent Python environments (logistic_regression, linear_svm, rbf_svm, random_forest, extra_trees, xgboost, ensemble).",
        "<b>Evaluation & Selection:</b> Selected top model (Random Forest, Val ROC-AUC = 0.7333) and evaluated ONCE on locked Peking_2 Test set (Test Acc = 0.5500, Spec = 0.8000)."
    ]
    for kf in key_facts:
        story.append(Paragraph(f"• {kf}", bullet_style))

    story.append(Spacer(1, 10))

    # Section 2: Tree of Folders (TOF) Final Directory Architecture
    story.append(Paragraph("2. Final Workspace Tree of Folders (TOF)", h1_style))
    story.append(Paragraph(
        "The complete workspace root (<code>/Users/apple/Documents/Codex/2026-09-29/down</code>) contains two primary project directories: "
        "the core research project (<code>NeuroADHD/</code>) and the reproducible multi-model benchmark suite (<code>NeuroADHD_ModelBenchmark/</code>):",
        body_style
    ))

    tof_tree_lines = [
        "down/",
        "│── work_download_adhd200.py                  [Multithreaded concurrent S3 downloader script]",
        "│── generate_pdf_report.py                    [PDF documentation architecture generator script]",
        "│── NeuroADHD_Workspace_Architecture...pdf   [Generated publication-quality PDF report]",
        "│── work/                                     [Empty scratch directory for transient execution tasks]",
        "│── outputs/                                  [Raw BIDS dataset destination & S3 inventory]",
        "│   ├── ADHD200_RawDataBIDS/                  [BIDS formatted NIfTI anatomical and functional volumes]",
        "│   │   ├── Brown/                            [Brown University raw fMRI BIDS files]",
        "│   │   └── KKI/                              [Kennedy Krieger Institute raw fMRI BIDS files]",
        "│   ├── ADHD200_RawDataBIDS_manifest.tsv     [S3 object key & size inventory (2,426 rows)]",
        "│   └── ADHD200_RawDataBIDS_file_list.txt     [Plain text listing of S3 relative keys]",
        "│",
        "├── NeuroADHD/                                [Core Research & Data Lake Directory (READ-ONLY)]",
        "│   ├── .venv/                                [Local Python 3.11 Virtual Environment]",
        "│   ├── configs/splits/                       [cc200_site_holdout.json group holdout configuration]",
        "│   ├── data/adhd200/                         [Multi-stage Data Lake]",
        "│   │   ├── raw/                              [Raw ROI signals (cc200, aal, falff, motion, phenotypes)]",
        "│   │   ├── interim/                          [Validation reports, schemas, subject label maps]",
        "│   │   ├── features/cc200/                   [cc200_connectivity_features.csv (153 × 19,903 matrix)]",
        "│   │   └── processed/splits/                 [subject_split_manifest.csv, train/val/test CSVs]",
        "│   ├── src/                                  [Production Python Modules]",
        "│   │   ├── features/                         [extract_cc200_connectivity.py (Fisher-z transform)]",
        "│   │   └── preprocessing/                    [cc200_ml_preprocessing.py, split validators]",
        "│   ├── tests/                                [test_cc200_preprocessing_leakage.py unit test suite]",
        "│   ├── README.md & requirements.txt          [Documentation & dependency declarations]",
        "│   ├── models/, notebooks/, results/         [Reserved project output directories]",
        "│",
        "└── NeuroADHD_ModelBenchmark/                 [Isolated Multi-Model Benchmarking Suite]",
        "    ├── configs/                              [Master & Model Configuration JSONs]",
        "    │   ├── benchmark_config.json             [Master benchmark configuration file]",
        "    │   └── models/                           [logistic_regression, linear_svm, rbf_svm, rf, extra_trees, xgboost]",
        "    ├── model_envs/                           [7 Model-Specific Isolated Python Virtual Environments]",
        "    │   ├── logistic_regression/              [Isolated env + requirements + lockfile]",
        "    │   ├── linear_svm/                       [Isolated env + requirements + lockfile]",
        "    │   ├── rbf_svm/                          [Isolated env + requirements + lockfile]",
        "    │   ├── random_forest/                    [Isolated env + requirements + lockfile]",
        "    │   ├── extra_trees/                      [Isolated env + requirements + lockfile]",
        "    │   ├── xgboost/                          [Isolated env + xgboost 3.2.0 + lockfile]",
        "    │   └── ensemble/                         [Isolated voting & stacking ensemble environment]",
        "    ├── scripts/                              [Benchmarking Pipeline Execution Engine]",
        "    │   ├── preflight_checks.py               [Dataset, split, label & zero-leakage validator]",
        "    │   ├── setup_environments.py             [Virtual environment creator & lockfile generator]",
        "    │   ├── train_model.py                    [Model training, val grid search & artifact saver]",
        "    │   ├── evaluate_model.py                 [Metrics calculator (Acc, F1, ROC-AUC, Spec, Sens, CM)]",
        "    │   ├── run_all_models.py                 [Master benchmark orchestrator & test unleasher]",
        "    │   └── generate_report.py                [Self-contained HTML report dashboard generator]",
        "    ├── runs/                                 [Run-specific model logs, predictions & metrics]",
        "    ├── models/                               [Run-specific saved model & pipeline joblib artifacts]",
        "    ├── results/                              [Run-specific model comparison CSV/JSON & test predictions]",
        "    └── reports/                              [Interactive HTML Dashboard (model_comparison.html)]"
    ]

    tof_formatted_html = "<br/>".join(tof_tree_lines)
    tof_table = Table([[Paragraph(tof_formatted_html, code_tree_style)]], colWidths=[504])
    tof_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('BOX', (0,0), (-1,-1), 0.75, border_color),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(tof_table)

    story.append(Spacer(1, 10))

    # Section 3: TOF Functional Directory Breakdown
    story.append(Paragraph("3. TOF Component Directory Breakdown", h1_style))
    story.append(Paragraph(
        "The table below details every directory component in the final workspace architecture, summarizing contents, formats, and roles:",
        body_style
    ))

    tof_summary_data = [
        [Paragraph("Directory Path (TOF)", table_header), Paragraph("Subdirs & Files", table_header), Paragraph("Primary Formats", table_header), Paragraph("Operational Purpose in Architecture", table_header)],
        [
            Paragraph("NeuroADHD/", table_cell_bold),
            Paragraph("8 subdirs, 4 files", table_cell),
            Paragraph(".venv, data, src, tests", table_cell),
            Paragraph("Original project workspace containing data lake, raw ROI files, Fisher-z feature matrix, and preprocessing logic.", table_cell)
        ],
        [
            Paragraph("data/adhd200/features/cc200/", table_cell_bold),
            Paragraph("5 files (~42.2 MB)", table_cell),
            Paragraph("cc200_connectivity_features.csv", table_cell),
            Paragraph("Primary ML dataset storing 153 subjects × 19,900 Fisher-z connectivity edge features + diagnosis labels.", table_cell)
        ],
        [
            Paragraph("NeuroADHD_ModelBenchmark/", table_cell_bold),
            Paragraph("7 subdirs", table_cell),
            Paragraph("configs, model_envs, scripts, reports", table_cell),
            Paragraph("Isolated benchmark workspace executing multi-model training, hyperparameter tuning, ensembling, and HTML reporting.", table_cell)
        ],
        [
            Paragraph("model_envs/", table_cell_bold),
            Paragraph("7 isolated envs", table_cell),
            Paragraph("Python 3.11 virtualenvs, lockfiles", table_cell),
            Paragraph("Model-specific isolated virtual environments ensuring strict package isolation for every model family.", table_cell)
        ],
        [
            Paragraph("scripts/", table_cell_bold),
            Paragraph("6 Python scripts", table_cell),
            Paragraph("run_all_models.py, train_model.py...", table_cell),
            Paragraph("Execution engine handling preflight validation, isolated model fitting, out-of-fold stacking, and HTML generation.", table_cell)
        ],
        [
            Paragraph("reports/", table_cell_bold),
            Paragraph("1 HTML report", table_cell),
            Paragraph("model_comparison.html", table_cell),
            Paragraph("Self-contained interactive HTML dashboard containing leaderboard, best model callouts, confusion matrices, and test results.", table_cell)
        ]
    ]

    tof_summary_table = Table(tof_summary_data, colWidths=[120, 80, 144, 160])
    tof_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('PADDING', (0,0), (-1,-1), 4.5),
    ]))
    story.append(tof_summary_table)

    story.append(Spacer(1, 10))

    # Section 4: System Verification & Conclusions
    story.append(Paragraph("4. Benchmark Results Summary & Verification", h1_style))
    status_text = (
        "<b>Model Selection Result:</b><br/>"
        "• <b>Best Validation Model:</b> Random Forest Classifier (Validation ROC-AUC = 0.7333, F1 = 0.1538).<br/>"
        "• <b>Final Test Result (Peking_2 Site, 20 subjects):</b> Accuracy = 0.5500, Balanced Accuracy = 0.5500, Specificity = 0.8000, ROC-AUC = 0.6300.<br/>"
        "• <b>Ensemble Comparison:</b> Soft Voting Ensemble achieved Val ROC-AUC = 0.7111; Stacking Meta-Classifier achieved Val ROC-AUC = 0.6889.<br/>"
        "• <b>Data Safety Integrity:</b> Zero files deleted, overwritten, or modified in the original NeuroADHD repository.<br/>"
        "• <b>Interactive Dashboard:</b> Rendered cleanly at <code>NeuroADHD_ModelBenchmark/reports/model_comparison.html</code>."
    )
    story.append(Paragraph(status_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {filename}")

if __name__ == "__main__":
    out_pdf = "/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_Workspace_Architecture_and_Dataset_Analysis.pdf"
    create_pdf(out_pdf)
