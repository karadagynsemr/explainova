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
    }

    .stApp {
        background: linear-gradient(180deg, #FAFBFE 0%, #F3F6FB 100%);
        color: var(--text-main);
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
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
        margin-bottom: 1.2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 58%, #0EA5E9 100%);
        border-radius: 28px;
        padding: 32px 28px;
        margin-bottom: 6px;
        box-shadow: 0 18px 40px rgba(79, 70, 229, 0.18);
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

    .section-box {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--text-main);
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-soft);
        margin-bottom: 0.8rem;
    }

    .insight-box {
        background: #F8FAFF;
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

    .metric-comment {
        background: #F8FAFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 1rem;
        line-height: 1.72;
        color: #334155;
    }

    .chart-note {
        background: #F8FAFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 13px 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 1.02rem;
        line-height: 1.75;
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

    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE6F2 !important;
        border-radius: 16px !important;
        padding: 10px 12px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.58rem 1rem !important;
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.18);
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: white !important;
    }

    div[data-baseweb="select"] > div {
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
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        background: white;
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
            Turn raw data into a clearer machine learning workflow — with guided preprocessing,
            model comparison, and results that are easier to understand.
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
        "shap_model_name"
    ]:
        if key in st.session_state:
            del st.session_state[key]


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
            len(report["dropped_empty_columns"])
            + len(report["dropped_high_missing_columns"])
            + len(report["dropped_single_value_columns"])
            + len(report["dropped_id_columns"])
            + len(report["dropped_high_cardinality_columns"])
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
    total_cells = rows * cols

    return (
            rows >= row_threshold
            or cols >= column_threshold
            or total_cells >= cell_threshold
    )


def should_offer_feature_reduction(df, row_threshold=10000, column_threshold=40, cell_threshold=300000):
    rows, cols = df.shape
    total_cells = rows * cols

    return (
            rows >= row_threshold
            or cols >= column_threshold
            or total_cells >= cell_threshold
    )


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
        pair = available_metrics[i:i+2]

        for col, metric in zip(cols, pair):
            with col:
                st.pyplot(metric_figures[metric], use_container_width=True)
                show_metric_comment(get_metric_commentary(results_df, metric, problem_type))


def parse_order_input(order_text):
    if not order_text:
        return []
    return [item.strip() for item in order_text.split(",") if item.strip()]


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "tsv"],
    help="Supported file types: CSV, XLSX, XLS, TSV."
)

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

        ordinal_source_df = df if selected_feature_columns is None else df[selected_feature_columns + [target_column]]
        ordinal_info = suggest_ordinal_columns(ordinal_source_df, target_column)
        all_categorical_columns = ordinal_info["categorical_columns"]
        auto_detected_ordinal_columns = ordinal_info["auto_detected_ordinal_columns"]

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

        large_dataset_flag = is_large_dataset(df)
        feature_reduction_available = should_offer_feature_reduction(df)

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
                protected_original_features = st.multiselect(
                    "Are there any original features that must definitely be kept?",
                    options=available_feature_candidates,
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

                st.success("Preprocessing completed successfully.")

        if "X_processed" in st.session_state and "y_processed" in st.session_state:
            X = st.session_state["X_processed"]
            y = st.session_state["y_processed"]
            X_explain_reference = st.session_state["X_explain_reference"]
            report = st.session_state["preprocessing_report"]

            show_section_header("Preprocessing Review", "Inspect the processed dataset and the main preprocessing outcomes.")

            total_dropped_columns = count_total_dropped_columns(report)
            total_missing_filled = len(report["filled_missing_numerical"]) + len(report["filled_missing_categorical"])

            help_duplicates = f"Removed duplicate rows count: {report['removed_duplicates']}"
            help_dropped = (
                    "Dropped columns include: "
                    + build_help_text(
                report["dropped_empty_columns"]
                + report["dropped_high_missing_columns"]
                + report["dropped_single_value_columns"]
                + report["dropped_id_columns"]
                + report["dropped_high_cardinality_columns"]
            )
            )
            help_rows_removed = f"Rows removed due to missing-value threshold: {report['removed_rows_due_to_missing']}"
            help_missing = (
                    "Missing-handled columns: "
                    + build_help_text(report["filled_missing_numerical"] + report["filled_missing_categorical"])
            )
            help_datetime = (
                    "Created datetime features: "
                    + build_help_text(report["created_datetime_features"])
            )
            help_encoded = (
                    "Encoded columns include ordinal + one-hot encoded columns: "
                    + build_help_text(report["ordinal_encoded_columns"] + report["one_hot_encoded_columns"])
            )

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Duplicates Removed", report["removed_duplicates"], help=help_duplicates)
            summary_col2.metric("Columns Dropped", total_dropped_columns, help=help_dropped)
            summary_col3.metric("Rows Removed", report["removed_rows_due_to_missing"], help=help_rows_removed)

            summary_col4, summary_col5, summary_col6 = st.columns(3)
            summary_col4.metric("Missing Columns Handled", total_missing_filled, help=help_missing)
            summary_col5.metric("Datetime Features Created", len(report["created_datetime_features"]), help=help_datetime)
            summary_col6.metric("Encoded Columns", len(report["ordinal_encoded_columns"]) + len(report["one_hot_encoded_columns"]), help=help_encoded)

            if report.get("feature_reduction_applied"):
                red_help_1 = "Low-variance columns removed: " + build_help_text(report["removed_low_variance_columns"])
                red_help_2 = "Highly correlated columns removed: " + build_help_text(report["removed_high_correlation_columns"])
                red_help_3 = "Lower-importance columns removed: " + build_help_text(report["removed_low_importance_columns"])

                red_col1, red_col2, red_col3 = st.columns(3)
                red_col1.metric("Low-Variance Features Removed", len(report["removed_low_variance_columns"]), help=red_help_1)
                red_col2.metric("Highly Correlated Features Removed", len(report["removed_high_correlation_columns"]), help=red_help_2)
                red_col3.metric("Lower-Importance Features Removed", len(report["removed_low_importance_columns"]), help=red_help_3)

            st.markdown("### What was done?")
            st.write("- Duplicate rows were removed when found.")
            st.write("- Problematic columns were detected and dropped when necessary.")
            st.write("- Missing values were handled automatically.")
            st.write("- Datetime columns were transformed into usable features.")
            st.write("- Ordinal and nominal categorical features were encoded.")
            st.write("- Extreme outliers were capped.")
            if report.get("feature_reduction_applied"):
                st.write("- Explainability-friendly feature reduction was applied on the final processed features.")

            st.subheader("Processed Data Review")
            st.dataframe(X_explain_reference.head(), use_container_width=True)

            with st.expander("Detailed preprocessing explanations"):
                show_preprocessing_explanations(report)

            with st.expander("Detailed preprocessing report"):
                st.subheader("Dataset Shape")
                st.write(f"Initial shape: {report['initial_shape']}")
                st.write(f"Final feature shape: {report['final_shape']}")

                st.subheader("Selected Features")
                if report["selected_feature_columns"] is None:
                    st.write("All available features were used.")
                else:
                    st.write(report["selected_feature_columns"])

                st.subheader("Dropped Columns")
                show_list("Empty Columns", report["dropped_empty_columns"])
                show_list("High-Missing Columns", report["dropped_high_missing_columns"])
                show_list("Single-Value Columns", report["dropped_single_value_columns"])
                show_list("ID Columns", report["dropped_id_columns"])
                show_list("High-Cardinality Columns", report["dropped_high_cardinality_columns"])

                st.subheader("Type Conversion")
                show_list("Converted to Numeric", report["converted_to_numeric"])
                show_list("Parsed Datetime Columns", report["parsed_datetime_columns"])
                show_list("Created Datetime Features", report["created_datetime_features"])

                st.subheader("Missing Value Handling")
                show_list("Filled Numerical Columns", report["filled_missing_numerical"])
                show_list("Filled Categorical Columns", report["filled_missing_categorical"])

                st.subheader("Encoding")
                show_list("Auto-detected Ordinal Columns", report["auto_detected_ordinal_columns"])
                show_list("User-selected Ordinal Columns", report["user_selected_ordinal_columns"])
                show_list("User-defined Ordinal Columns", report["user_defined_ordinal_columns"])
                show_list("Ordinal Columns That Could Not Be Safely Encoded", report["failed_user_ordinal_columns"])
                show_list("Ordinal Encoded Columns", report["ordinal_encoded_columns"])
                show_list("One-Hot Encoded Columns", report["one_hot_encoded_columns"])

                st.subheader("Outlier Handling")
                outlier_df = build_outlier_dataframe(report)
                st.dataframe(outlier_df, use_container_width=True)
                show_list("Columns with Capped Extreme Outliers", report["capped_outlier_columns"])

                if report.get("feature_reduction_applied"):
                    st.subheader("Feature Reduction for Explainability")
                    show_list("Protected Original Features", report["protected_original_features"])
                    show_list("Protected Transformed Features", report["protected_transformed_features"])
                    show_list("Removed Low-Variance Features", report["removed_low_variance_columns"])
                    show_list("Removed Highly Correlated Features", report["removed_high_correlation_columns"])
                    show_list("Removed Lower-Importance Features", report["removed_low_importance_columns"])

                    if report["feature_importance_ranking"] is not None:
                        st.write("Model-based feature importance ranking:")
                        st.dataframe(report["feature_importance_ranking"].head(20), use_container_width=True)

                st.subheader("Target Information")
                st.write(f"Target encoded: {report['target_encoded']}")
                if report["target_classes"] is not None:
                    st.write("Target classes:")
                    for cls in report["target_classes"]:
                        st.write(f"- {cls}")

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
                    "This section shows linear correlations between processed features and the target variable. Correlation values close to 1 or -1 indicate a stronger relationship, while values close to 0 indicate a weaker one. Positive values suggest that the feature tends to increase with the target, whereas negative values suggest an inverse relationship."
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

            show_section_header("Model Training", "Choose the prediction type first, then decide whether to compare several models or focus on one.")

            detected_problem_type = detect_problem_type(y)

            show_info_box(
                "Choose the prediction type",
                "Use Classification when you want the model to predict categories or labels, such as yes/no, low/medium/high, or class names. "
                "Use Regression when you want the model to predict a numeric value, such as price, score, temperature, or quality value."
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
                f"Selected mode: {chosen_problem_type.capitalize()} — "
                + (
                    "the model will predict categories or classes."
                    if chosen_problem_type == "classification"
                    else "the model will predict a numeric value."
                )
            )

            available_models = get_available_models(chosen_problem_type)

            show_info_box(
                "Training mode guidance",
                "Training a single model is faster and useful when you already have a preferred method. Comparing multiple models takes longer, but it helps you understand which algorithm fits your dataset better before moving into deeper explainability."
            )

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

            if st.session_state.get("large_dataset_flag", False):
                st.warning("Because the dataset is large, model training may take longer than usual.")

            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    class_labels = report["target_label_mapping"] if report["target_label_mapping"] is not None else report["target_classes"]

                    problem_type, results_df, detailed_results = train_and_evaluate_models(
                        X=X,
                        y=y,
                        training_mode=training_mode,
                        selected_model_name=selected_model_name,
                        class_labels=class_labels,
                        forced_problem_type=chosen_problem_type
                    )

                st.session_state["problem_type"] = problem_type
                st.session_state["results_df"] = results_df
                st.session_state["detailed_results"] = detailed_results
                st.session_state["selected_training_mode"] = training_mode
                st.session_state["selected_model_name"] = selected_model_name

                st.success("Model training completed successfully.")

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
                    show_info_box(
                        "Confusion Matrix Insight",
                        get_confusion_matrix_interpretation(cm, class_labels)
                    )

                roc_fig = plot_roc_curve_figure(detailed_results)
                if roc_fig is not None:
                    st.subheader("ROC Curve")
                    st.pyplot(roc_fig, use_container_width=False)
                    show_info_box(
                        "ROC Curve Insight",
                        get_roc_interpretation(detailed_results)
                    )

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
                with st.spinner("Generating SHAP explanations..."):
                    selected_shap_details = detailed_results[shap_model_name]
                    trained_model = selected_shap_details["trained_model"]

                    shap_outputs = compute_shap_outputs(
                        trained_model=trained_model,
                        X_reference=X_explain_reference,
                        problem_type=problem_type,
                        max_background_samples=100,
                        max_explain_samples=200
                    )

                    st.session_state["shap_model_name"] = shap_model_name
                    st.session_state["shap_outputs"] = shap_outputs

            if "shap_outputs" in st.session_state:
                shap_outputs = st.session_state["shap_outputs"]
                shap_model_name = st.session_state.get("shap_model_name", "Selected Model")

                st.subheader(f"SHAP Results for {shap_model_name}")
                show_info_box(
                    "What SHAP shows",
                    get_shap_intro_text()
                )

                importance_df = shap_outputs["feature_importance_df"]

                shap_col1, shap_col2 = st.columns([1.0, 1.15])

                with shap_col1:
                    st.subheader("Top SHAP Features")
                    st.dataframe(importance_df.head(12), use_container_width=True)
                    show_chart_note(
                        "This table ranks features by their average absolute SHAP contribution. A larger value means the feature has a stronger overall influence on the model across the analyzed samples. Features near the top are the ones the model relies on more consistently."
                    )

                with shap_col2:
                    shap_bar_fig = plot_shap_importance_bar(importance_df, top_n=12)
                    if shap_bar_fig is not None:
                        st.pyplot(shap_bar_fig, use_container_width=True)
                    show_chart_note(
                        "This bar chart presents the same feature importance ranking visually. Longer bars indicate stronger influence. It is useful when you want to quickly compare which variables matter most at the global model level."
                    )

                st.subheader("SHAP Summary Plot")
                shap_summary_fig = plot_shap_summary_figure(
                    shap_outputs["shap_values"],
                    shap_outputs["X_explain"],
                    max_display=12
                )
                if shap_summary_fig is not None:
                    st.pyplot(shap_summary_fig, use_container_width=False)
                show_chart_note(
                    "Each dot represents one sample for one feature. Dots further to the right push the prediction upward, while dots further to the left push it downward. Color represents the feature value itself, so you can also see whether high or low values tend to increase or decrease the model output."
                )

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")