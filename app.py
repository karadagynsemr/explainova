import streamlit as st
import pandas as pd

from src.preprocessing import preprocess_data

st.set_page_config(
    page_title="Explainable ML Analysis Platform",
    layout="wide"
)

st.title("Explainable ML Analysis Platform")
st.write("Upload your dataset and review the automatic preprocessing report.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


def show_list_section(title, items):
    st.markdown(f"### {title}")
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.write("None")


def show_dict_section(title, data_dict):
    st.markdown(f"### {title}")
    if data_dict:
        df_report = pd.DataFrame(
            list(data_dict.items()),
            columns=["Column", "Value"]
        )
        st.dataframe(df_report, use_container_width=True)
    else:
        st.write("No information available.")


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique(dropna=True).values
        })
        st.dataframe(info_df, use_container_width=True)

        target_column = st.selectbox("Select the target column", df.columns)

        if st.button("Run Preprocessing"):
            X, y, report = preprocess_data(df, target_column)

            st.success("Preprocessing completed successfully.")

            # -----------------------------
            # Dataset shape summary
            # -----------------------------
            st.header("Preprocessing Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric(
                "Initial Shape",
                f"{report['initial_shape'][0]} x {report['initial_shape'][1]}"
            )
            summary_col2.metric(
                "Final Feature Shape",
                f"{report['final_shape'][0]} x {report['final_shape'][1]}"
            )
            summary_col3.metric(
                "Removed Duplicates",
                report["removed_duplicates"]
            )

            st.write(f"**Removed rows due to missing-value threshold:** {report['removed_rows_due_to_missing']}")

            # -----------------------------
            # Preprocessed data preview
            # -----------------------------
            st.header("Processed Data Preview")
            st.dataframe(X.head(), use_container_width=True)

            st.subheader("Target Preview")
            st.dataframe(y.head().to_frame(), use_container_width=True)

            # -----------------------------
            # Cleaning report
            # -----------------------------
            st.header("Cleaning Report")

            show_list_section("Dropped Empty Columns", report["dropped_empty_columns"])
            show_list_section("Dropped High-Missing Columns", report["dropped_high_missing_columns"])
            show_list_section("Dropped Single-Value Columns", report["dropped_single_value_columns"])
            show_list_section("Dropped ID Columns", report["dropped_id_columns"])
            show_list_section("Dropped High-Cardinality Columns", report["dropped_high_cardinality_columns"])

            # -----------------------------
            # Type conversion report
            # -----------------------------
            st.header("Type Conversion Report")

            show_list_section("Converted Object Columns to Numeric", report["converted_to_numeric"])
            show_list_section("Parsed Datetime Columns", report["parsed_datetime_columns"])
            show_list_section("Created Datetime Features", report["created_datetime_features"])

            # -----------------------------
            # Missing value handling
            # -----------------------------
            st.header("Missing Value Handling")

            show_list_section("Filled Numerical Columns", report["filled_missing_numerical"])
            show_list_section("Filled Categorical Columns", report["filled_missing_categorical"])

            # -----------------------------
            # Encoding report
            # -----------------------------
            st.header("Encoding Report")

            show_list_section("Categorical Columns Before Encoding", report["categorical_columns_before_encoding"])
            show_list_section("Numerical Columns Before Encoding", report["numerical_columns_before_encoding"])
            show_list_section("Ordinal Encoded Columns", report["ordinal_encoded_columns"])
            show_list_section("One-Hot Encoded Columns", report["one_hot_encoded_columns"])

            st.markdown("### Target Encoding")
            st.write(f"Target encoded: **{report['target_encoded']}**")

            if report["target_classes"] is not None:
                st.write("Target classes:")
                for cls in report["target_classes"]:
                    st.write(f"- {cls}")

            # -----------------------------
            # Outlier report
            # -----------------------------
            st.header("Outlier Report")

            show_dict_section("Detected Outliers (1.5 IQR Rule)", report["outlier_report"])
            show_dict_section("Detected Extreme Outliers (3.0 IQR Rule)", report["extreme_outlier_report"])
            show_list_section("Columns with Capped Extreme Outliers", report["capped_outlier_columns"])

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")