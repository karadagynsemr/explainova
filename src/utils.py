"""
utils.py — Report generation utilities for Explainova.

Requirements (install if not present):
    pip install python-docx
"""

from io import BytesIO
from datetime import datetime
import pandas as pd


def _build_preprocessing_steps(report: dict) -> list:
    steps = []

    removed_dupes = report.get("removed_duplicates", 0)
    if removed_dupes > 0:
        steps.append(f"Removed {removed_dupes} duplicate rows.")

    dropped_empty = report.get("dropped_empty_columns", [])
    if dropped_empty:
        steps.append(f"Dropped {len(dropped_empty)} fully-empty columns: {', '.join(dropped_empty)}")

    dropped_missing = report.get("dropped_high_missing_columns", [])
    if dropped_missing:
        steps.append(f"Dropped {len(dropped_missing)} high-missing columns (>60% missing).")

    dropped_single = report.get("dropped_single_value_columns", [])
    if dropped_single:
        steps.append(f"Dropped {len(dropped_single)} single-value (constant) columns.")

    dropped_id = report.get("dropped_id_columns", [])
    if dropped_id:
        steps.append(f"Dropped ID-like columns: {', '.join(dropped_id)}")

    dropped_hc = report.get("dropped_high_cardinality_columns", [])
    if dropped_hc:
        steps.append(f"Dropped {len(dropped_hc)} high-cardinality categorical columns.")

    rows_removed = report.get("removed_rows_due_to_missing", 0)
    if rows_removed > 0:
        steps.append(f"Removed {rows_removed} rows that exceeded the missing-value threshold.")

    parsed_dt = report.get("parsed_datetime_columns", [])
    if parsed_dt:
        steps.append(
            f"Parsed {len(parsed_dt)} datetime column(s) into year/month/day/weekday features: "
            f"{', '.join(parsed_dt)}"
        )

    filled_num = report.get("filled_missing_numerical", [])
    if filled_num:
        steps.append(f"Filled missing values in {len(filled_num)} numeric column(s) using the column median.")

    filled_cat = report.get("filled_missing_categorical", [])
    if filled_cat:
        steps.append(f"Filled missing values in {len(filled_cat)} categorical column(s) using the mode.")

    capped = report.get("capped_outlier_columns", [])
    if capped:
        steps.append(f"Capped extreme outliers (3x IQR) in {len(capped)} column(s).")

    ordinal_enc = report.get("ordinal_encoded_columns", [])
    if ordinal_enc:
        steps.append(f"Ordinal-encoded columns: {', '.join(ordinal_enc)}")

    ohe = report.get("one_hot_encoded_columns", [])
    if ohe:
        steps.append(f"One-hot encoded {len(ohe)} remaining categorical column(s).")

    if report.get("target_encoded"):
        classes = report.get("target_classes", [])
        steps.append(f"Target variable label-encoded. Classes: {', '.join(str(c) for c in classes)}")

    if report.get("feature_reduction_applied"):
        low_var = len(report.get("removed_low_variance_columns", []))
        high_corr = len(report.get("removed_high_correlation_columns", []))
        low_imp = len(report.get("removed_low_importance_columns", []))
        steps.append(
            f"Feature reduction applied - removed {low_var} low-variance, "
            f"{high_corr} highly-correlated, and {low_imp} low-importance feature(s)."
        )

    return steps


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _get_primary_model_metric(problem_type: str):
    if problem_type == "classification":
        return "Accuracy", False
    return "R2 Score", False


def _get_best_model_summary(results_df: pd.DataFrame, problem_type: str):
    if results_df is None or results_df.empty:
        return None

    metric, ascending = _get_primary_model_metric(problem_type)
    if metric not in results_df.columns:
        return None

    best_row = results_df.sort_values(by=metric, ascending=ascending).iloc[0]
    return {
        "model_name": str(best_row["Model"]),
        "metric_name": metric,
        "metric_value": float(best_row[metric])
    }


def _add_table_explanation_lines(problem_type: str):
    if problem_type == "classification":
        return [
            "How to read this table:",
            "- Higher Accuracy, Precision, Recall, and F1 Score usually indicate better performance.",
            "- If ROC AUC is available, higher values indicate better class separation across thresholds.",
            "- The best model is not always the one with the highest accuracy; consider the metric that matters most for your use case."
        ]
    return [
        "How to read this table:",
        "- Higher R2 Score is better because it means the model explains more of the target variation.",
        "- Lower MAE and RMSE are better because they indicate smaller prediction errors.",
        "- RMSE penalizes larger mistakes more strongly than MAE."
    ]


def generate_word_report(
        target_column: str,
        report: dict,
        results_df: pd.DataFrame,
        problem_type: str,
        shap_outputs: dict = None,
        shap_model_name: str = None,
        shap_bar_fig=None,
        shap_summary_fig=None,
) -> BytesIO:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    INDIGO = RGBColor(0x4F, 0x46, 0xE5)
    SLATE = RGBColor(0x33, 0x41, 0x55)

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.font.color.rgb = SLATE

    title_para = doc.add_heading("", level=0)
    title_run = title_para.add_run("Explainova - Analysis Report")
    title_run.font.color.rgb = INDIGO
    title_run.font.size = Pt(22)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   "
        f"Target: {target_column}   |   "
        f"Problem type: {problem_type.capitalize()}"
    ).font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    doc.add_paragraph()

    doc.add_heading("Dataset Information", level=1)
    initial = report.get("initial_shape", ("N/A", "N/A"))
    final = report.get("final_shape", ("N/A", "N/A"))

    dataset_note = doc.add_paragraph()
    dataset_note.add_run(
        "This section compares the dataset before and after preprocessing so you can quickly see "
        "how much cleaning, filtering, and transformation changed the working data."
    )

    tbl = doc.add_table(rows=3, cols=2)
    tbl.style = "Light Grid Accent 1"
    data_rows = [
        ("Initial shape", f"{initial[0]} rows x {initial[1]} columns"),
        ("Final shape (after preprocessing)", f"{final[0]} rows x {final[1]} columns"),
        ("Duplicate rows removed", str(report.get("removed_duplicates", 0))),
    ]
    for i, (label, value) in enumerate(data_rows):
        tbl.rows[i].cells[0].text = label
        tbl.rows[i].cells[1].text = value

    doc.add_paragraph()

    doc.add_heading("Preprocessing Summary", level=1)
    intro_p = doc.add_paragraph()
    intro_p.add_run(
        "The following list summarizes the main data-preparation actions applied before model training. "
        "These steps are important because they directly affect both model quality and interpretability."
    )

    for step in _build_preprocessing_steps(report):
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(step)

    doc.add_paragraph()

    doc.add_heading("Model Training Results", level=1)

    summary = _get_best_model_summary(results_df, problem_type)
    if summary is not None:
        best_p = doc.add_paragraph()
        best_p.add_run("Best overall model: ").bold = True
        best_p.add_run(
            f"{summary['model_name']} based on {summary['metric_name']} = {summary['metric_value']:.4f}"
        )

    for line in _add_table_explanation_lines(problem_type):
        p = doc.add_paragraph(style="List Bullet" if line.startswith("-") else None)
        p.add_run(line[2:] if line.startswith("-") else line)

    if results_df is not None and not results_df.empty:
        cols = results_df.columns.tolist()
        tbl2 = doc.add_table(rows=1 + len(results_df), cols=len(cols))
        tbl2.style = "Light Grid Accent 1"

        for j, col in enumerate(cols):
            cell = tbl2.rows[0].cells[j]
            cell.text = col
            if cell.paragraphs and cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].bold = True

        for i, (_, row) in enumerate(results_df.iterrows()):
            for j, col in enumerate(cols):
                tbl2.rows[i + 1].cells[j].text = _fmt(row[col])

    doc.add_paragraph()

    if shap_outputs is not None:
        doc.add_heading(f"SHAP Explainability - {shap_model_name}", level=1)

        intro = doc.add_paragraph()
        intro.add_run(
            "SHAP (SHapley Additive exPlanations) reveals how each feature contributes "
            "to individual predictions. A positive SHAP value pushes the prediction "
            "upward; a negative value pushes it downward. Larger absolute values "
            "indicate stronger feature influence."
        )

        importance_df = shap_outputs.get("feature_importance_df")
        if importance_df is not None and not importance_df.empty:
            doc.add_heading("Top SHAP Features", level=2)

            shap_note = doc.add_paragraph()
            shap_note.add_run(
                "This table ranks features by average absolute contribution. Features near the top "
                "have a stronger and more consistent influence on predictions across the analyzed samples."
            )

            top_df = importance_df.head(12)
            tbl3 = doc.add_table(rows=1 + len(top_df), cols=2)
            tbl3.style = "Light Grid Accent 1"

            for j, h in enumerate(["Feature", "Mean |SHAP Value|"]):
                c = tbl3.rows[0].cells[j]
                c.text = h
                if c.paragraphs and c.paragraphs[0].runs:
                    c.paragraphs[0].runs[0].bold = True

            for i, (_, row) in enumerate(top_df.iterrows()):
                tbl3.rows[i + 1].cells[0].text = str(row["Feature"])
                tbl3.rows[i + 1].cells[1].text = f"{row['Mean |SHAP Value|']:.6f}"

        if shap_bar_fig is not None:
            doc.add_heading("SHAP Feature Importance Chart", level=2)
            chart_note = doc.add_paragraph()
            chart_note.add_run(
                "Longer bars indicate features with greater overall impact on the model output."
            )

            img_buf = BytesIO()
            shap_bar_fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=130)
            img_buf.seek(0)
            doc.add_picture(img_buf, width=Inches(5.8))

        if shap_summary_fig is not None:
            doc.add_heading("SHAP Summary Plot", level=2)
            summary_note = doc.add_paragraph()
            summary_note.add_run(
                "Each dot represents one sample. Position on the x-axis shows whether that feature pushed "
                "the prediction higher or lower, while color reflects the feature value."
            )

            img_buf2 = BytesIO()
            shap_summary_fig.savefig(img_buf2, format="png", bbox_inches="tight", dpi=130)
            img_buf2.seek(0)
            doc.add_picture(img_buf2, width=Inches(6.2))

    doc.add_paragraph()
    footer_p = doc.add_paragraph()
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer_p.add_run("Generated by Explainova | Powered by SHAP and scikit-learn")
    fr.font.size = Pt(8)
    fr.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf