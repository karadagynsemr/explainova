import streamlit as st
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data, suggest_ordinal_columns
from src.model_training import (
    detect_problem_type,
    get_available_models,
    train_and_evaluate_models
)
from src.visualization import (
    plot_confusion_matrix_figure,
    plot_metric_grid,
    plot_roc_curve_figure,
    build_outlier_dataframe,
    build_target_correlation_table,
    plot_correlation_heatmap_figure,
    get_confusion_matrix_interpretation,
    get_roc_interpretation,
    get_metric_commentary
)
from src.explainability import (
    compute_shap_outputs,
    plot_shap_importance_bar,
    plot_shap_summary_figure,
    get_shap_selection_guidance,
    get_shap_intro_text
)
from src.utils import generate_word_report

st.set_page_config(
    page_title="Explainova",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --bg-main: #F7F8FC;
        --bg-soft: #F1F5FB;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --text-main: #0F172A;
        --text-soft: #475569;
        --primary: #4F46E5;
        --primary-dark: #4338CA;
        --accent: #0EA5E9;
        --success: #10B981;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(99,102,241,0.10), transparent 24%),
            radial-gradient(circle at top left, rgba(14,165,233,0.08), transparent 22%),
            linear-gradient(180deg, #FAFBFE 0%, #F3F6FB 100%);
        color: var(--text-main);
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        padding-left: 1.6rem;
        padding-right: 1.6rem;
    }

    @media (max-width: 1200px) {
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    h1, h2, h3, h4 {
        color: var(--text-main) !important;
        letter-spacing: -0.02em;
    }

    p, li, label, span, div {
        color: var(--text-main);
    }

    .hero-wrap {
        text-align: center;
        margin-bottom: 1.15rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 58%, #0EA5E9 100%);
        border-radius: 30px;
        padding: 34px 28px 30px 28px;
        margin-bottom: 8px;
        box-shadow: 0 20px 42px rgba(79, 70, 229, 0.18);
        border: 1px solid rgba(255,255,255,0.18);
    }

    .hero-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin-bottom: 0.55rem;
        text-align: center;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.96);
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 860px;
        margin: 0 auto;
        text-align: center;
    }

    .status-strip {
        background: rgba(255,255,255,0.78);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        font-size: 0.95rem;
        color: #334155;
    }

    .stepper-wrap {
        background: rgba(255,255,255,0.92);
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 20px 24px 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.05);
        backdrop-filter: blur(8px);
    }

    .stepper {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 7px;
        flex: 1;
        max-width: 190px;
    }

    .step-connector {
        flex: 1;
        height: 4px;
        max-width: 96px;
        border-radius: 999px;
        background: #E2E8F0;
        margin-bottom: 24px;
        transition: background 0.3s ease;
    }

    .step-connector.done {
        background: linear-gradient(90deg, #4F46E5, #22C55E);
    }

    .step-circle {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.92rem;
        font-weight: 800;
        transition: all 0.3s ease;
    }

    .step-circle.pending {
        background: #F8FAFC;
        border: 2px solid #CBD5E1;
        color: #94A3B8;
    }

    .step-circle.active {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.32);
    }

    .step-circle.done {
        background: linear-gradient(135deg, #22C55E 0%, #10B981 100%);
        color: white;
        box-shadow: 0 8px 18px rgba(16, 185, 129, 0.22);
    }

    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-align: center;
        line-height: 1.35;
    }

    .step-label.active {
        color: #4F46E5;
        font-weight: 800;
    }

    .step-label.done {
        color: #0F766E;
    }

    .section-divider {
        border: none;
        border-top: 2px solid #EEF2FF;
        margin: 24px 0 16px 0;
    }

    .section-box {
        background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFF 100%);
        border: 1px solid var(--card-border);
        border-top: 4px solid #6366F1;
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--text-main);
        margin-bottom: 0.22rem;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-soft);
        margin-bottom: 0.25rem;
    }

    .insight-box {
        background: linear-gradient(180deg, #F8FAFF 0%, #F4F8FF 100%);
        border: 1px solid #DCE6F7;
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .insight-title {
        font-size: 1.03rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.45rem;
    }

    .insight-text {
        font-size: 1rem;
        line-height: 1.74;
        color: #334155;
    }

    .metric-comment, .chart-note {
        background: linear-gradient(180deg, #F8FAFF 0%, #FFFFFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 1rem;
        line-height: 1.72;
        color: #334155;
    }

    .order-box {
        background: #F8FAFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .download-card {
        background: linear-gradient(135deg, #F8FAFF 0%, #EEF2FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 18px;
        padding: 20px 22px;
        margin-top: 16px;
        margin-bottom: 8px;
        box-shadow: 0 10px 24px rgba(79, 70, 229, 0.08);
    }

    .download-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 4px;
    }

    .download-subtitle {
        font-size: 0.94rem;
        color: #475569;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FCFDFE 100%) !important;
        border: 1px solid #DCE6F2 !important;
        border-radius: 16px !important;
        padding: 10px 12px !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.62rem 1rem !important;
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.18);
        width: 100%;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: white !important;
    }

    div[data-baseweb="select"] > div,
    .stTextInput input {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    div[role="listbox"],
    ul[role="listbox"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12) !important;
    }

    div[role="option"],
    li[role="option"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
    }

    div[role="option"]:hover,
    li[role="option"]:hover {
        background: #EEF2FF !important;
        color: #0F172A !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background: #EEF2FF !important;
        color: #3730A3 !important;
        border-radius: 999px !important;
    }

    .stExpander {
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        background: #FCFDFE !important;
    }

    .stAlert {
        border-radius: 14px !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        background: white;
        box-shadow: 0 6px 16px rgba(15,23,42,0.04);
    }

    div[data-testid="stRadio"] label {
        color: #0F172A !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-card">
        <div class="hero-title">Explainova</div>
        <div class="hero-subtitle">
            Turn raw data into a clearer machine learning workflow with guided preprocessing,
            cleaner visuals, stronger reporting, and model explanations that are easier to understand.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def reset_training_state():
    for key in [
        "problem_type",
        "results_df",
        "detailed_results",
        "selected_training_mode",
        "selected_model_name",
        "shap_outputs",
        "shap_model_name",
        "shap_bar_fig",
        "shap_summary_fig",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def get_completed_steps() -> int:
    if "shap_outputs" in st.session_state:
        return 4
    if "results_df" in st.session_state:
        return 3
    if "X_processed" in st.session_state:
        return 2
    if st.session_state.get("data_uploaded", False):
        return 1
    return 0


def sync_uploaded_file_state(uploaded_file):
    if uploaded_file is None:
        st.session_state["data_uploaded"] = False
        st.session_state["last_uploaded_filename"] = None
        return

    current_name = uploaded_file.name
    if st.session_state.get("last_uploaded_filename") != current_name:
        st.session_state["last_uploaded_filename"] = current_name
        st.session_state["data_uploaded"] = True
        reset_training_state()
        for key in ["X_processed", "y_processed", "X_explain_reference", "preprocessing_report", "target_column", "large_dataset_flag"]:
            if key in st.session_state:
                del st.session_state[key]


def show_workflow_status(completed_steps: int):
    if completed_steps == 0:
        text = "Workflow status: waiting for dataset upload."
    elif completed_steps == 1:
        text = "Workflow status: data upload completed. Next step: preprocessing."
    elif completed_steps == 2:
        text = "Workflow status: preprocessing completed. Next step: model training."
    elif completed_steps == 3:
        text = "Workflow status: model training completed. Next step: SHAP analysis."
    else:
        text = "Workflow status: all workflow steps completed."

    st.markdown(f'<div class="status-strip">{text}</div>', unsafe_allow_html=True)


def show_step_progress(completed_steps: int):
    steps = ["Data Upload", "Preprocessing", "Model Training", "SHAP Analysis"]
    icons = ["📁", "⚙️", "🤖", "🔍"]

    html = '<div class="stepper-wrap"><div class="stepper">'

    for i, (label, icon) in enumerate(zip(steps, icons), 1):
        if i <= completed_steps:
            circle_class, label_class, circle_content = "done", "done", "✓"
        elif i == completed_steps + 1 and completed_steps < len(steps):
            circle_class, label_class, circle_content = "active", "active", str(i)
        else:
            circle_class, label_class, circle_content = "pending", "", str(i)

        html += (
            f'<div class="step-item">'
            f'  <div class="step-circle {circle_class}">{circle_content}</div>'
            f'  <div class="step-label {label_class}">{icon}<br>{label}</div>'
            f'</div>'
        )

        if i < len(steps):
            conn_class = "done" if i <= completed_steps else ""
            html += f'<div class="step-connector {conn_class}"></div>'

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def show_section_header(title, subtitle=None):
    st.markdown(
        f"""
        <div class="section-box">
            <div class="section-title">{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_section_divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def show_info_box(title, text):
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_metric_comment(text):
    st.markdown(f'<div class="metric-comment">{text}</div>', unsafe_allow_html=True)


def show_chart_note(text):
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


def show_list(title, items):
    st.markdown(f"**{title}**")
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.write("None")


def count_total_dropped_columns(report):
    return (
            len(report.get("dropped_empty_columns", []))
            + len(report.get("dropped_high_missing_columns", []))
            + len(report.get("dropped_single_value_columns", []))
            + len(report.get("dropped_id_columns", []))
            + len(report.get("dropped_high_cardinality_columns", []))
    )


def build_help_text(items, default_text="No items recorded."):
    if not items:
        return default_text
    joined = ", ".join(map(str, items))
    if len(joined) > 350:
        joined = joined[:347] + "..."
    return joined


def explain_preprocessing_step(title, explanation):
    st.markdown(f"**{title}**")
    st.write(explanation)


def show_preprocessing_explanations(report):
    st.subheader("Detailed Preprocessing Explanations")

    explain_preprocessing_step(
        "Duplicate rows removed",
        "Identical rows were removed so the model would not learn the same observation multiple times."
    )
    explain_preprocessing_step(
        "Empty and problematic columns removed",
        "Columns that were fully empty, had only one value, looked like IDs, or had too many unique categories were removed to reduce noise."
    )
    explain_preprocessing_step(
        "Missing values handled",
        "Numerical columns were filled with median values and categorical columns were filled with the most common value when needed."
    )
    explain_preprocessing_step(
        "Datetime columns transformed",
        "Detected date/time columns were converted into useful parts such as year, month, day, and day of week."
    )
    explain_preprocessing_step(
        "Outliers reviewed",
        "Outliers were detected using the IQR method. Extremely large or small values were capped when needed to reduce their impact."
    )
    explain_preprocessing_step(
        "Categorical columns encoded",
        "Ordinal columns were encoded with ordered numeric values when a known order was detected or when the user explicitly defined that order. Remaining categorical columns were one-hot encoded."
    )

    if report.get("feature_reduction_applied"):
        explain_preprocessing_step(
            "Feature reduction for explainability",
            "Because the dataset was large, optional feature reduction was used to simplify the model while preserving real feature names. Low-variance features, highly correlated features, and lower-importance features were reduced, except for columns explicitly protected by the user."
        )


def show_metric_explanations(problem_type, has_roc_auc=False):
    st.subheader("Metric Explanations")

    if problem_type == "classification":
        st.markdown("**Accuracy** — Shows the overall proportion of correct predictions.")
        st.markdown("**When it matters:** Useful when classes are balanced and all errors have similar importance.")
        st.markdown("**Precision** — Shows how many predicted positive cases were actually positive.")
        st.markdown("**When it matters:** Important when false positives are costly.")
        st.markdown("**Recall** — Shows how many real positive cases were successfully found.")
        st.markdown("**When it matters:** Important when missing a true positive is costly.")
        st.markdown("**F1 Score** — Balances precision and recall into a single value.")
        st.markdown("**When it matters:** Useful when both false positives and false negatives matter.")
        if has_roc_auc:
            st.markdown("**ROC AUC** — Measures how well the model separates two classes across different thresholds.")
            st.markdown("**When it matters:** Useful in binary classification when class separation matters.")
    else:
        st.markdown("**R2 Score** — Shows how well the model explains variation in the target.")
        st.markdown("**When it matters:** Good for understanding overall explanatory power.")
        st.markdown("**MAE** — The average absolute prediction error.")
        st.markdown("**When it matters:** Useful when you want an error measure in the original target units.")
        st.markdown("**RMSE** — Similar to MAE, but gives more weight to larger errors.")
        st.markdown("**When it matters:** Useful when large mistakes should be penalized more strongly.")


def get_best_model_info(results_df, problem_type):
    if results_df.empty:
        return None, None, None

    if problem_type == "classification":
        metric = "Accuracy"
        ascending = False
    else:
        metric = "R2 Score"
        ascending = False

    best_row = results_df.sort_values(by=metric, ascending=ascending).iloc[0]
    return best_row["Model"], metric, best_row[metric]


def is_large_dataset(df, row_threshold=50000, column_threshold=100, cell_threshold=2_000_000):
    rows, cols = df.shape
    return rows >= row_threshold or cols >= column_threshold or rows * cols >= cell_threshold


def should_offer_feature_reduction(df, row_threshold=10000, column_threshold=40, cell_threshold=300000):
    rows, cols = df.shape
    return rows >= row_threshold or cols >= column_threshold or rows * cols >= cell_threshold


def show_metric_plots(results_df, problem_type):
    if problem_type == "classification":
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        if "ROC AUC" in results_df.columns:
            metrics.append("ROC AUC")
    else:
        metrics = ["R2 Score", "MAE", "RMSE"]

    metric_figures = plot_metric_grid(results_df, metrics)
    available_metrics = [m for m in metrics if m in metric_figures]

    if not available_metrics:
        return

    st.subheader("Metric Comparisons")

    for i in range(0, len(available_metrics), 2):
        cols = st.columns(2)
        pair = available_metrics[i:i + 2]

        for col, metric in zip(cols, pair):
            with col:
                st.pyplot(metric_figures[metric], use_container_width=True)
                show_metric_comment(get_metric_commentary(results_df, metric, problem_type))


def parse_order_input(order_text):
    if not order_text:
        return []
    return [item.strip() for item in order_text.split(",") if item.strip()]


def show_download_section(target_column, report, results_df, problem_type,
                          shap_outputs, shap_model_name, shap_bar_fig, shap_summary_fig):
    st.markdown(
        """
        <div class="download-card">
            <div class="download-title"> Download Analysis Report</div>
            <div class="download-subtitle">
                Export a full summary of preprocessing steps, model results, and SHAP
                explanations as a Word document.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        word_buf = generate_word_report(
            target_column=target_column,
            report=report,
            results_df=results_df,
            problem_type=problem_type,
            shap_outputs=shap_outputs,
            shap_model_name=shap_model_name,
            shap_bar_fig=shap_bar_fig,
            shap_summary_fig=shap_summary_fig,
        )
        st.download_button(
            label="️ Download as Word (.docx)",
            data=word_buf,
            file_name="explainova_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except ImportError:
        st.warning("Word export requires python-docx. Install it with: pip install python-docx")
    except Exception as e:
        st.error(f"Word report could not be generated: {e}")


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "tsv"],
    help="Supported file types: CSV, XLSX, XLS, TSV."
)

sync_uploaded_file_state(uploaded_file)
show_workflow_status(get_completed_steps())
show_step_progress(get_completed_steps())

if uploaded_file is not None:
    try:
        show_section_header("Dataset Preview", "Review the uploaded data before selecting the target column.")
        df = load_dataset(uploaded_file)

        preview_rows = st.selectbox(
            "How many rows would you like to preview?",
            options=[5, 10, 15, 20],
            index=0
        )

        st.dataframe(df.head(preview_rows), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        show_section_divider()
        show_section_header("Feature and Target Selection", "Choose the target column and decide whether to use all features or only selected ones.")

        feature_mode = st.radio(
            "How would you like to use features?",
            options=["Use all available features", "Select features manually"],
            index=0,
            help="You can keep all features or manually select a subset before preprocessing."
        )

        target_column = st.selectbox(
            "Select the target column",
            df.columns,
            help="The target column is the output variable the model will try to predict."
        )

        selected_feature_columns = None
        available_feature_candidates = [col for col in df.columns if col != target_column]

        if feature_mode == "Select features manually":
            selected_feature_columns = st.multiselect(
                "Select the feature columns to include",
                options=available_feature_candidates,
                default=available_feature_candidates[: min(8, len(available_feature_candidates))],
                help="Only the selected feature columns will be used during preprocessing and model training."
            )

        ordinal_source_df = (
            df if selected_feature_columns is None
            else df[selected_feature_columns + [target_column]]
        )
        ordinal_info = suggest_ordinal_columns(ordinal_source_df, target_column)
        all_categorical_columns = ordinal_info["categorical_columns"]
        auto_detected_ordinal_columns = ordinal_info["auto_detected_ordinal_columns"]

        show_section_divider()
        show_section_header("Preprocessing Options", "Review the guidance and choose your preprocessing settings.")

        user_selected_ordinal_columns = []
        user_defined_ordinal_mappings = {}

        if all_categorical_columns:
            with st.expander("Ordinal data information"):
                st.write(
                    "Ordinal data contains categories with a meaningful order. "
                    "Examples include low < medium < high or mild < moderate < severe."
                )

                if auto_detected_ordinal_columns:
                    st.success("Automatically detected ordinal columns: " + ", ".join(auto_detected_ordinal_columns))
                else:
                    st.info("No ordinal columns were automatically detected from the known patterns.")

            selectable_manual_ordinal_columns = [
                col for col in all_categorical_columns if col not in auto_detected_ordinal_columns
            ]

            if feature_mode == "Select features manually" and selected_feature_columns:
                selectable_manual_ordinal_columns = [
                    col for col in selectable_manual_ordinal_columns
                    if col in selected_feature_columns
                ]

            if selectable_manual_ordinal_columns:
                user_selected_ordinal_columns = st.multiselect(
                    "Do you want to additionally mark any categorical columns as ordinal?",
                    options=selectable_manual_ordinal_columns,
                    help="Choose extra columns only if their categories have a true order."
                )

                if user_selected_ordinal_columns:
                    st.markdown("### Define category order for selected ordinal columns")
                    st.caption(
                        "For each selected column, enter the category order from lowest to highest, separated by commas."
                    )

                    for col in user_selected_ordinal_columns:
                        unique_values = (
                            ordinal_source_df[col].dropna().astype(str).str.strip().unique().tolist()
                            if col in ordinal_source_df.columns else []
                        )

                        st.markdown(
                            f"""
                            <div class="order-box">
                                <strong>{col}</strong><br>
                                Available values: {", ".join(unique_values) if unique_values else "No non-null values found"}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        order_text = st.text_input(
                            f"Order for {col}",
                            key=f"order_{col}",
                            placeholder="Example: low, medium, high",
                            help="Write the values in order from lowest to highest."
                        )

                        parsed_order = parse_order_input(order_text)
                        if parsed_order:
                            user_defined_ordinal_mappings[col] = parsed_order

        if feature_mode == "Select features manually" and selected_feature_columns:
            df_for_size = df[selected_feature_columns + [target_column]]
        else:
            df_for_size = df

        large_dataset_flag = is_large_dataset(df_for_size)
        feature_reduction_available = should_offer_feature_reduction(df_for_size)

        if large_dataset_flag:
            st.warning("This dataset is large. Preprocessing can still run, but model training may take longer.")

        apply_feature_reduction = "No"
        protected_original_features = []

        if feature_reduction_available:
            show_info_box(
                "Feature Reduction for Explainability",
                "Because this dataset is relatively large, you can optionally reduce the feature set without breaking feature-level interpretability. This keeps real feature names, unlike PCA."
            )

            apply_feature_reduction = st.radio(
                "Would you like to apply explainability-friendly feature reduction?",
                options=["No", "Yes"],
                index=0,
                help="This can simplify the model by removing low-variance, highly correlated, and lower-importance features while keeping original feature names."
            )

            if apply_feature_reduction == "Yes":
                protection_options = (
                    selected_feature_columns
                    if feature_mode == "Select features manually" and selected_feature_columns
                    else available_feature_candidates
                )

                protected_original_features = st.multiselect(
                    "Are there any original features that must definitely be kept?",
                    options=protection_options,
                    help="Selected features will be protected during feature reduction whenever possible."
                )

        if st.button("Run Preprocessing"):
            reset_training_state()

            if feature_mode == "Select features manually" and (selected_feature_columns is None or len(selected_feature_columns) == 0):
                st.error("Please select at least one feature column.")
            else:
                X, y, report, X_explain_reference = preprocess_data(
                    df=df,
                    target_column=target_column,
                    selected_feature_columns=selected_feature_columns,
                    user_selected_ordinal_columns=user_selected_ordinal_columns,
                    user_defined_ordinal_mappings=user_defined_ordinal_mappings,
                    apply_feature_reduction=(apply_feature_reduction == "Yes"),
                    protected_original_features=protected_original_features,
                    low_variance_threshold=0.0001,
                    high_correlation_threshold=0.95,
                    top_k_important_features=40
                )

                st.session_state["X_processed"] = X
                st.session_state["y_processed"] = y
                st.session_state["X_explain_reference"] = X_explain_reference
                st.session_state["preprocessing_report"] = report
                st.session_state["target_column"] = target_column
                st.session_state["large_dataset_flag"] = large_dataset_flag
                st.rerun()

        if "X_processed" in st.session_state and "y_processed" in st.session_state:
            X = st.session_state["X_processed"]
            y = st.session_state["y_processed"]
            X_explain_reference = st.session_state["X_explain_reference"]
            report = st.session_state["preprocessing_report"]

            show_section_divider()
            show_section_header("Preprocessing Review", "Inspect the processed dataset and the main preprocessing outcomes.")

            total_dropped_columns = count_total_dropped_columns(report)
            total_missing_filled = len(report.get("filled_missing_numerical", [])) + len(report.get("filled_missing_categorical", []))

            help_duplicates = f"Removed duplicate rows count: {report.get('removed_duplicates', 0)}"
            help_dropped = (
                    "Dropped columns include: " +
                    build_help_text(
                        report.get("dropped_empty_columns", [])
                        + report.get("dropped_high_missing_columns", [])
                        + report.get("dropped_single_value_columns", [])
                        + report.get("dropped_id_columns", [])
                        + report.get("dropped_high_cardinality_columns", [])
                    )
            )
            help_rows_removed = f"Rows removed due to missing-value threshold: {report.get('removed_rows_due_to_missing', 0)}"
            help_missing = (
                    "Missing-handled columns: " +
                    build_help_text(report.get("filled_missing_numerical", []) + report.get("filled_missing_categorical", []))
            )
            help_datetime = (
                    "Created datetime features: " +
                    build_help_text(report.get("created_datetime_features", []))
            )
            help_encoded = (
                    "Encoded columns include ordinal + one-hot encoded columns: " +
                    build_help_text(report.get("ordinal_encoded_columns", []) + report.get("one_hot_encoded_columns", []))
            )

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Duplicates Removed", report.get("removed_duplicates", 0), help=help_duplicates)
            summary_col2.metric("Columns Dropped", total_dropped_columns, help=help_dropped)
            summary_col3.metric("Rows Removed", report.get("removed_rows_due_to_missing", 0), help=help_rows_removed)

            summary_col4, summary_col5, summary_col6 = st.columns(3)
            summary_col4.metric("Missing Columns Handled", total_missing_filled, help=help_missing)
            summary_col5.metric("Datetime Features Created", len(report.get("created_datetime_features", [])), help=help_datetime)
            summary_col6.metric("Encoded Columns", len(report.get("ordinal_encoded_columns", [])) + len(report.get("one_hot_encoded_columns", [])), help=help_encoded)

            if X_explain_reference.empty:
                st.warning("Processed data is empty after preprocessing. Please review the preprocessing report.")
            else:
                st.subheader("Processed Data Review")
                st.dataframe(X_explain_reference.head(), use_container_width=True)

            with st.expander("Detailed preprocessing explanations"):
                show_preprocessing_explanations(report)

            with st.expander("Detailed preprocessing report"):
                st.subheader("Dataset Shape")
                st.write(f"Initial shape: {report.get('initial_shape')}")
                st.write(f"Final feature shape: {report.get('final_shape')}")

                st.subheader("Selected Features")
                if report.get("selected_feature_columns") is None:
                    st.write("All available features were used.")
                else:
                    st.write(report.get("selected_feature_columns"))

                st.subheader("Dropped Columns")
                show_list("Empty Columns", report.get("dropped_empty_columns", []))
                show_list("High-Missing Columns", report.get("dropped_high_missing_columns", []))
                show_list("Single-Value Columns", report.get("dropped_single_value_columns", []))
                show_list("ID Columns", report.get("dropped_id_columns", []))
                show_list("High-Cardinality Columns", report.get("dropped_high_cardinality_columns", []))

                st.subheader("Type Conversion")
                show_list("Converted to Numeric", report.get("converted_to_numeric", []))
                show_list("Parsed Datetime Columns", report.get("parsed_datetime_columns", []))
                show_list("Created Datetime Features", report.get("created_datetime_features", []))

                st.subheader("Missing Value Handling")
                show_list("Filled Numerical Columns", report.get("filled_missing_numerical", []))
                show_list("Filled Categorical Columns", report.get("filled_missing_categorical", []))

                st.subheader("Encoding")
                show_list("Auto-detected Ordinal Columns", report.get("auto_detected_ordinal_columns", []))
                show_list("User-selected Ordinal Columns", report.get("user_selected_ordinal_columns", []))
                show_list("User-defined Ordinal Columns", report.get("user_defined_ordinal_columns", []))
                show_list("Ordinal Columns That Could Not Be Safely Encoded", report.get("failed_user_ordinal_columns", []))
                show_list("Ordinal Encoded Columns", report.get("ordinal_encoded_columns", []))
                show_list("One-Hot Encoded Columns", report.get("one_hot_encoded_columns", []))

                st.subheader("Outlier Handling")
                outlier_df = build_outlier_dataframe(report)
                st.dataframe(outlier_df, use_container_width=True)
                show_list("Columns with Capped Extreme Outliers", report.get("capped_outlier_columns", []))

                st.subheader("Target Information")
                st.write(f"Target encoded: {report.get('target_encoded')}")
                if report.get("target_classes") is not None:
                    st.write("Target classes:")
                    for cls in report.get("target_classes", []):
                        st.write(f"- {cls}")

            show_section_divider()
            show_section_header("Feature Relationship Overview", "A direct view of linear relationships in the processed dataset before model-based explanations.")

            explain_X = X_explain_reference.copy()
            corr_table = build_target_correlation_table(
                explain_X,
                y,
                target_name=st.session_state["target_column"],
                top_n=10
            )

            if not corr_table.empty:
                show_info_box(
                    "What this shows",
                    "This section shows linear correlations between processed features and the target variable. Correlation values close to 1 or -1 indicate a stronger relationship, while values close to 0 indicate a weaker one."
                )

                corr_col1, corr_col2 = st.columns([1.0, 1.1])

                with corr_col1:
                    st.subheader("Top Feature–Target Correlations")
                    st.dataframe(corr_table[["Feature", "Correlation with Target"]], use_container_width=True)

                with corr_col2:
                    heatmap_fig = plot_correlation_heatmap_figure(
                        explain_X,
                        y,
                        target_name=st.session_state["target_column"],
                        top_n=10
                    )
                    if heatmap_fig is not None:
                        st.pyplot(heatmap_fig, use_container_width=True)

            else:
                st.info("A correlation-based overview could not be generated because no suitable numeric features were available after preprocessing.")

            show_section_divider()
            show_section_header("Model Training", "Choose the prediction type first, then decide whether to compare several models or focus on one.")

            detected_problem_type = detect_problem_type(y)

            show_info_box(
                "Choose the prediction type",
                "Use Classification when you want the model to predict categories or labels, such as yes/no, low/medium/high, or class names. Use Regression when you want the model to predict a numeric value, such as price, score, temperature, or quality value."
            )

            auto_label = f"Auto-detect (recommended: {detected_problem_type.capitalize()})"

            problem_type_display = st.radio(
                "What kind of prediction do you want?",
                options=[auto_label, "Classification", "Regression"],
                index=0,
                help="Choose Classification for category prediction and Regression for numeric value prediction."
            )

            if problem_type_display == auto_label:
                chosen_problem_type = detected_problem_type
            elif problem_type_display == "Classification":
                chosen_problem_type = "classification"
            else:
                chosen_problem_type = "regression"

            st.caption(
                f"Selected mode: {chosen_problem_type.capitalize()} — " +
                (
                    "the model will predict categories or classes."
                    if chosen_problem_type == "classification"
                    else "the model will predict a numeric value."
                )
            )

            available_models = get_available_models(chosen_problem_type)

            training_mode_display = st.radio(
                "How would you like to train models?",
                options=["Compare multiple models", "Train a single model"],
                help="Choose whether to compare several models or focus on a single one."
            )

            training_mode = "multiple" if training_mode_display == "Compare multiple models" else "single"

            selected_model_name = None
            if training_mode == "single":
                selected_model_name = st.selectbox(
                    "Select a model",
                    options=list(available_models.keys()),
                    help="Pick one model to train."
                )

            if st.button("Train Models"):
                if X is None or X.empty:
                    st.error("Model training cannot start because no usable feature columns are available after preprocessing.")
                else:
                    problem_type, results_df, detailed_results = train_and_evaluate_models(
                        X=X,
                        y=y,
                        training_mode=training_mode,
                        selected_model_name=selected_model_name,
                        class_labels=report.get("target_label_mapping") or report.get("target_classes"),
                        forced_problem_type=chosen_problem_type
                    )

                    st.session_state["problem_type"] = problem_type
                    st.session_state["results_df"] = results_df
                    st.session_state["detailed_results"] = detailed_results
                    st.session_state["selected_training_mode"] = training_mode
                    st.session_state["selected_model_name"] = selected_model_name
                    st.rerun()

        if (
                "results_df" in st.session_state
                and "problem_type" in st.session_state
                and "detailed_results" in st.session_state
                and "X_explain_reference" in st.session_state
        ):
            results_df = st.session_state["results_df"]
            problem_type = st.session_state["problem_type"]
            detailed_results = st.session_state["detailed_results"]
            X_explain_reference = st.session_state["X_explain_reference"]

            show_section_divider()
            show_section_header("Results Dashboard", "Review model results, metric comparisons, and classification visuals.")

            st.write(f"Detected problem type: **{problem_type.capitalize()}**")
            st.dataframe(results_df, use_container_width=True)

            best_model_name, best_metric_name, best_metric_value = get_best_model_info(results_df, problem_type)
            if best_model_name is not None:
                st.success(f"Best model based on {best_metric_name}: {best_model_name} ({best_metric_value:.4f})")

            has_roc_auc = "ROC AUC" in results_df.columns
            with st.expander("Metric explanations"):
                show_metric_explanations(problem_type, has_roc_auc=has_roc_auc)

            show_metric_plots(results_df, problem_type)

            with st.expander("Detailed model explanations"):
                if problem_type == "classification":
                    st.write("Classification models are evaluated by how accurately and consistently they predict the correct class.")
                    st.write("Confusion Matrix shows where the model is correct and where it mixes up classes.")
                    if has_roc_auc:
                        st.write("ROC Curve shows how well the model separates two classes across different decision thresholds.")
                else:
                    st.write("Regression models are evaluated by how close their predictions are to the real numeric values.")
                    st.write("R2 Score shows explanatory power, while MAE and RMSE show prediction error size.")

            if problem_type == "classification":
                st.subheader("Confusion Matrix")

                model_to_show = st.selectbox(
                    "Select a model for confusion matrix",
                    options=list(detailed_results.keys()),
                    help="Choose which model's confusion matrix to inspect."
                )

                selected_details = detailed_results[model_to_show]
                cm = selected_details.get("confusion_matrix")
                class_labels = selected_details.get("class_labels")

                if cm is not None and class_labels is not None:
                    st.pyplot(plot_confusion_matrix_figure(cm, class_labels), use_container_width=False)
                    show_info_box("Confusion Matrix Insight", get_confusion_matrix_interpretation(cm, class_labels))

                roc_fig = plot_roc_curve_figure(detailed_results)
                if roc_fig is not None:
                    st.subheader("ROC Curve")
                    st.pyplot(roc_fig, use_container_width=False)
                    show_info_box("ROC Curve Insight", get_roc_interpretation(detailed_results))

            show_section_divider()
            show_section_header("SHAP Explainability", "Select the model you want to interpret after reviewing the evaluation results.")

            show_info_box(
                "How to choose a model for SHAP",
                get_shap_selection_guidance(problem_type, has_roc_auc=has_roc_auc)
            )

            shap_model_name = st.selectbox(
                "Select the model to explain with SHAP",
                options=list(detailed_results.keys()),
                help="Choose the trained model whose predictions you want to interpret."
            )

            if st.button("Generate SHAP Analysis"):
                selected_shap_details = detailed_results[shap_model_name]
                trained_model = selected_shap_details["trained_model"]

                shap_outputs = compute_shap_outputs(
                    trained_model=trained_model,
                    X_reference=X_explain_reference,
                    problem_type=problem_type,
                    max_background_samples=100,
                    max_explain_samples=200
                )

                importance_df_cached = shap_outputs["feature_importance_df"]
                bar_fig_cached = plot_shap_importance_bar(importance_df_cached, top_n=12)
                summary_fig_cached = plot_shap_summary_figure(
                    shap_outputs["shap_values"],
                    shap_outputs["X_explain"],
                    max_display=12
                )

                st.session_state["shap_model_name"] = shap_model_name
                st.session_state["shap_outputs"] = shap_outputs
                st.session_state["shap_bar_fig"] = bar_fig_cached
                st.session_state["shap_summary_fig"] = summary_fig_cached
                st.rerun()

            if "shap_outputs" in st.session_state:
                shap_outputs = st.session_state["shap_outputs"]
                shap_model_name = st.session_state.get("shap_model_name", "Selected Model")
                shap_bar_fig = st.session_state.get("shap_bar_fig")
                shap_summary_fig = st.session_state.get("shap_summary_fig")

                st.subheader(f"SHAP Results for {shap_model_name}")
                show_info_box("What SHAP shows", get_shap_intro_text())

                importance_df = shap_outputs["feature_importance_df"]

                shap_col1, shap_col2 = st.columns([1.0, 1.15])

                with shap_col1:
                    st.subheader("Top SHAP Features")
                    st.dataframe(importance_df.head(12), use_container_width=True)
                    show_chart_note(
                        "This table ranks features by their average absolute SHAP contribution. A larger value means the feature has a stronger overall influence on the model across the analyzed samples."
                    )

                with shap_col2:
                    if shap_bar_fig is not None:
                        st.pyplot(shap_bar_fig, use_container_width=True)
                    show_chart_note(
                        "This bar chart presents the same feature importance ranking visually. Longer bars indicate stronger influence."
                    )

                st.subheader("SHAP Summary Plot")
                if shap_summary_fig is not None:
                    st.pyplot(shap_summary_fig, use_container_width=False)
                show_chart_note(
                    "Each dot represents one sample for one feature. Dots further to the right push the prediction upward, while dots further to the left push it downward."
                )

                show_section_divider()
                show_download_section(
                    target_column=st.session_state.get("target_column", "target"),
                    report=st.session_state.get("preprocessing_report", {}),
                    results_df=results_df,
                    problem_type=problem_type,
                    shap_outputs=shap_outputs,
                    shap_model_name=shap_model_name,
                    shap_bar_fig=shap_bar_fig,
                    shap_summary_fig=shap_summary_fig,
                )

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")