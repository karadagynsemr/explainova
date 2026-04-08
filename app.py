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
    get_confusion_matrix_interpretation,
    get_roc_interpretation,
    get_metric_commentary
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
        --info-bg: #EEF2FF;
        --info-border: #C7D2FE;
    }

    .stApp {
        background: linear-gradient(180deg, #FAFBFE 0%, #F3F6FB 100%);
        color: var(--text-main);
    }

    .main .block-container {
        max-width: 1220px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
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
        margin-bottom: 1.4rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 58%, #0EA5E9 100%);
        border-radius: 28px;
        padding: 34px 30px;
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
        padding: 18px 18px 14px 18px;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--text-main);
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        font-size: 0.94rem;
        color: var(--text-soft);
        margin-bottom: 0.9rem;
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
        font-size: 1.02rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.45rem;
    }

    .insight-text {
        font-size: 1rem;
        line-height: 1.72;
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
        line-height: 1.7;
        color: #334155;
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
        "selected_model_name"
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
        "Ordinal columns were encoded with ordered numeric values when a known order was detected. Remaining categorical columns were one-hot encoded."
    )

    if report.get("pca_applied"):
        explain_preprocessing_step(
            "PCA applied",
            "PCA was used to reduce the number of features while preserving most of the information. This can speed up training, but it may reduce interpretability."
        )


def show_metric_explanations(problem_type):
    st.subheader("Metric Explanations")

    if problem_type == "classification":
        st.markdown("**Accuracy** — Overall proportion of correct predictions.")
        st.markdown("**Precision** — Among predicted positives, how many were actually positive.")
        st.markdown("**Recall** — Among actual positives, how many were correctly identified.")
        st.markdown("**F1 Score** — A balanced score combining precision and recall.")
        st.markdown("**ROC AUC** — Shows how well the model separates classes across thresholds. Higher is better.")
    else:
        st.markdown("**R2 Score** — Indicates how well the model explains the target variable. Higher is better.")
        st.markdown("**MAE** — Average absolute prediction error.")
        st.markdown("**RMSE** — Similar to MAE, but penalizes larger errors more strongly.")


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


def show_small_centered_plot(fig):
    col1, col2, col3 = st.columns([1.5, 2.1, 1.5])
    with col2:
        st.pyplot(fig, use_container_width=False)


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
                st.pyplot(metric_figures[metric], use_container_width=False)
                show_metric_comment(get_metric_commentary(results_df, metric, problem_type))


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "tsv"],
    help="Supported file types: CSV, XLSX, XLS, TSV."
)

if uploaded_file is not None:
    try:
        show_section_header("Dataset Preview", "Review the uploaded data before selecting the target column.")
        df = load_dataset(uploaded_file)

        st.dataframe(df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        show_section_header("Target Selection", "Choose the variable the model should predict.")
        target_column = st.selectbox(
            "Select the target column",
            df.columns,
            help="The target column is the output variable the model will try to predict."
        )

        ordinal_info = suggest_ordinal_columns(df, target_column)
        all_categorical_columns = ordinal_info["categorical_columns"]
        auto_detected_ordinal_columns = ordinal_info["auto_detected_ordinal_columns"]

        show_section_header("Preprocessing Options", "Review the guidance and choose your preprocessing settings.")

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

                with st.expander("How should I choose extra ordinal columns?"):
                    st.write(
                        "Select a column only if its values follow a clear sequence. "
                        "If a safe order cannot be detected automatically, the system will report that."
                    )
            else:
                user_selected_ordinal_columns = []
        else:
            user_selected_ordinal_columns = []

        large_dataset_flag = is_large_dataset(df)

        if large_dataset_flag:
            st.warning("This dataset is large. Preprocessing can still run, but model training may take longer.")

        with st.expander("PCA information"):
            st.write(
                "PCA reduces the number of features while trying to preserve most of the useful information. "
                "It can speed up training, but the new PCA components are less directly interpretable than the original columns."
            )

        use_pca = st.radio(
            "Would you like to use PCA before model training?",
            options=["No", "Yes"],
            index=0,
            help="PCA can reduce dimensionality and training time, but it may reduce interpretability."
        )

        if st.button("Run Preprocessing"):
            reset_training_state()

            X, y, report = preprocess_data(
                df=df,
                target_column=target_column,
                user_selected_ordinal_columns=user_selected_ordinal_columns,
                apply_pca=(use_pca == "Yes"),
                pca_variance_threshold=0.95
            )

            st.session_state["X_processed"] = X
            st.session_state["y_processed"] = y
            st.session_state["preprocessing_report"] = report
            st.session_state["target_column"] = target_column
            st.session_state["large_dataset_flag"] = large_dataset_flag

            st.success("Preprocessing completed successfully.")

        if "X_processed" in st.session_state and "y_processed" in st.session_state:
            X = st.session_state["X_processed"]
            y = st.session_state["y_processed"]
            report = st.session_state["preprocessing_report"]

            show_section_header("Preprocessing Review", "Inspect the processed dataset and the main preprocessing outcomes.")

            total_dropped_columns = count_total_dropped_columns(report)
            total_missing_filled = len(report["filled_missing_numerical"]) + len(report["filled_missing_categorical"])

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Duplicates Removed", report["removed_duplicates"])
            summary_col2.metric("Columns Dropped", total_dropped_columns)
            summary_col3.metric("Rows Removed", report["removed_rows_due_to_missing"])

            summary_col4, summary_col5, summary_col6 = st.columns(3)
            summary_col4.metric("Missing Columns Handled", total_missing_filled)
            summary_col5.metric("Datetime Features Created", len(report["created_datetime_features"]))
            summary_col6.metric("Encoded Columns", len(report["ordinal_encoded_columns"]) + len(report["one_hot_encoded_columns"]))

            if report.get("pca_applied") and report.get("pca_report") is not None:
                pca_report = report["pca_report"]

                pca_col1, pca_col2, pca_col3 = st.columns(3)
                pca_col1.metric("Original Feature Count", pca_report["original_feature_count"], help="The number of feature columns before PCA.")
                pca_col2.metric("Reduced Feature Count", pca_report["reduced_feature_count"], help="The number of PCA components kept after dimensionality reduction.")
                pca_col3.metric("Explained Variance", f"{pca_report['explained_variance_ratio_sum']:.2%}", help="This shows how much of the original information is preserved by the selected PCA components.")

            st.markdown("### What was done?")
            st.write("- Duplicate rows were removed when found.")
            st.write("- Problematic columns were detected and dropped when necessary.")
            st.write("- Missing values were handled automatically.")
            st.write("- Datetime columns were transformed into usable features.")
            st.write("- Ordinal and nominal categorical features were encoded.")
            st.write("- Extreme outliers were capped.")
            if report.get("pca_applied"):
                st.write("- PCA was applied to reduce feature count before model training.")

            st.subheader("Processed Data Review")
            if report.get("pca_applied"):
                st.caption("PCA is enabled. The table below shows the processed data before PCA so it remains easier to understand.")
                st.dataframe(report["preview_before_pca"], use_container_width=True)

                with st.expander("What changed after PCA?"):
                    pca_report = report["pca_report"]
                    st.write(
                        f"PCA reduced the feature space from **{pca_report['original_feature_count']}** columns "
                        f"to **{pca_report['reduced_feature_count']}** components."
                    )
                    st.write(
                        "The original processed columns were not individually deleted one by one. "
                        "Instead, PCA combined information from many columns into a smaller number of components."
                    )
                    st.write(
                        "This was done to preserve most of the useful variation in the data while making model training faster and lighter."
                    )
            else:
                st.dataframe(X.head(), use_container_width=True)

            with st.expander("Detailed preprocessing explanations"):
                show_preprocessing_explanations(report)

            with st.expander("Detailed preprocessing report"):
                st.subheader("Dataset Shape")
                st.write(f"Initial shape: {report['initial_shape']}")
                st.write(f"Final feature shape: {report['final_shape']}")

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
                show_list("Ordinal Columns That Could Not Be Safely Encoded", report["failed_user_ordinal_columns"])
                show_list("Ordinal Encoded Columns", report["ordinal_encoded_columns"])
                show_list("One-Hot Encoded Columns", report["one_hot_encoded_columns"])

                st.subheader("Outlier Handling")
                outlier_df = build_outlier_dataframe(report)
                st.dataframe(outlier_df, use_container_width=True)
                show_list("Columns with Capped Extreme Outliers", report["capped_outlier_columns"])

                st.subheader("Target Information")
                st.write(f"Target encoded: {report['target_encoded']}")
                if report["target_classes"] is not None:
                    st.write("Target classes:")
                    for cls in report["target_classes"]:
                        st.write(f"- {cls}")

                st.subheader("PCA Information")
                st.write(report["pca_report"])

            show_section_header("Model Training", "Choose whether to compare several models or focus on one.")
            detected_problem_type = detect_problem_type(y)
            available_models = get_available_models(detected_problem_type)

            with st.expander("Training mode information"):
                st.write("You can either train one model only, or train multiple baseline models and compare their performance side by side.")

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
                    class_labels = report["target_classes"] if report["target_encoded"] else None

                    problem_type, results_df, detailed_results = train_and_evaluate_models(
                        X=X,
                        y=y,
                        training_mode=training_mode,
                        selected_model_name=selected_model_name,
                        class_labels=class_labels
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
        ):
            results_df = st.session_state["results_df"]
            problem_type = st.session_state["problem_type"]
            detailed_results = st.session_state["detailed_results"]
            training_mode = st.session_state.get("selected_training_mode", "multiple")

            show_section_header("Results Dashboard", "Review model results, metric comparisons, and classification visuals.")

            st.write(f"Detected problem type: **{problem_type.capitalize()}**")
            st.dataframe(results_df, use_container_width=True)

            best_model_name, best_metric_name, best_metric_value = get_best_model_info(results_df, problem_type)
            if best_model_name is not None:
                st.success(f"Best model based on {best_metric_name}: {best_model_name} ({best_metric_value:.4f})")

            with st.expander("Metric explanations"):
                show_metric_explanations(problem_type)

            show_metric_plots(results_df, problem_type)

            with st.expander("Detailed model explanations"):
                if problem_type == "classification":
                    st.write("Classification models are evaluated by how accurately and consistently they predict the correct class.")
                    st.write("Confusion Matrix shows where the model is correct and where it mixes up classes.")
                    st.write("ROC Curve shows how well the model separates two classes across different decision thresholds.")
                else:
                    st.write("Regression models are evaluated by how close their predictions are to the real numeric values.")
                    st.write("R2 Score shows explanatory power, while MAE and RMSE show prediction error size.")

            if problem_type == "classification":
                st.subheader("Confusion Matrix")

                if training_mode == "single":
                    model_to_show = list(detailed_results.keys())[0]
                else:
                    model_to_show = st.selectbox(
                        "Select a model for confusion matrix",
                        options=list(detailed_results.keys()),
                        help="Choose which model's confusion matrix to inspect."
                    )

                selected_details = detailed_results[model_to_show]
                cm = selected_details.get("confusion_matrix")
                class_labels = selected_details.get("class_labels")

                if cm is not None and class_labels is not None:
                    show_small_centered_plot(plot_confusion_matrix_figure(cm, class_labels))
                    show_info_box(
                        "Confusion Matrix Insight",
                        get_confusion_matrix_interpretation(cm, class_labels)
                    )

                roc_fig = plot_roc_curve_figure(detailed_results)
                if roc_fig is not None:
                    st.subheader("ROC Curve")
                    show_small_centered_plot(roc_fig)

                    show_info_box(
                        "ROC Curve Insight",
                        get_roc_interpretation(detailed_results)
                    )
                else:
                    st.info("ROC curve is available only for compatible binary classification cases.")

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")