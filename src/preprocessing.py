import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


def is_id_column(column_name):
    col = column_name.strip().lower()

    exact_matches = {
        "id", "patient_id", "sample_id", "record_id",
        "user_id", "customer_id", "transaction_id"
    }

    if col in exact_matches:
        return True

    if col.endswith("_id"):
        return True

    return False


def try_convert_object_to_numeric(df, target_column, conversion_threshold=0.80):
    data = df.copy()
    converted_columns = []

    object_cols = data.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        if col == target_column:
            continue

        converted = pd.to_numeric(data[col], errors="coerce")

        original_non_null = data[col].notnull().sum()
        converted_non_null = converted.notnull().sum()

        if original_non_null > 0 and (converted_non_null / original_non_null) >= conversion_threshold:
            data[col] = converted
            converted_columns.append(col)

    return data, converted_columns


def is_datetime_candidate(column_name):
    col = column_name.strip().lower()

    datetime_keywords = [
        "date", "time", "timestamp", "created", "updated",
        "birth", "dob", "joined", "year", "month", "day"
    ]

    return any(keyword in col for keyword in datetime_keywords)


def try_parse_datetime_columns(df, target_column, parse_threshold=0.80):
    data = df.copy()

    parsed_datetime_columns = []
    created_datetime_features = []

    object_cols = data.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        if col == target_column:
            continue

        should_try = is_datetime_candidate(col)

        parsed = pd.to_datetime(data[col], errors="coerce")

        original_non_null = data[col].notnull().sum()
        parsed_non_null = parsed.notnull().sum()

        if original_non_null == 0:
            continue

        parse_ratio = parsed_non_null / original_non_null

        if should_try and parse_ratio >= parse_threshold:
            data[f"{col}_year"] = parsed.dt.year
            data[f"{col}_month"] = parsed.dt.month
            data[f"{col}_day"] = parsed.dt.day
            data[f"{col}_dayofweek"] = parsed.dt.dayofweek

            created_datetime_features.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek"
            ])

            if parsed.dt.hour.nunique(dropna=True) > 1:
                data[f"{col}_hour"] = parsed.dt.hour
                created_datetime_features.append(f"{col}_hour")

            data = data.drop(columns=[col])
            parsed_datetime_columns.append(col)

    return data, parsed_datetime_columns, created_datetime_features


def get_known_ordinal_mappings():
    return [
        {"low": 0, "medium": 1, "high": 2},
        {"poor": 0, "fair": 1, "good": 2, "excellent": 3},
        {"small": 0, "medium": 1, "large": 2},
        {"mild": 0, "moderate": 1, "severe": 2},
        {"beginner": 0, "intermediate": 1, "advanced": 2},
        {"stage1": 0, "stage2": 1, "stage3": 2, "stage4": 3},
        {"stage 1": 0, "stage 2": 1, "stage 3": 2, "stage 4": 3},
        {"i": 0, "ii": 1, "iii": 2, "iv": 3},
        {"1+": 0, "2+": 1, "3+": 2, "4+": 3},
    ]


def detect_ordinal_columns(df, categorical_cols):
    ordinal_columns = {}
    known_mappings = get_known_ordinal_mappings()

    for col in categorical_cols:
        unique_values = df[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        unique_set = set(unique_values)

        for mapping in known_mappings:
            if len(unique_set) > 0 and unique_set.issubset(set(mapping.keys())):
                ordinal_columns[col] = mapping
                break

    return ordinal_columns


def suggest_ordinal_columns(df, target_column):
    data = df.copy()

    if target_column not in data.columns:
        return {
            "auto_detected_ordinal_columns": [],
            "categorical_columns": [],
            "ordinal_mappings_found": {}
        }

    data, _, _ = try_parse_datetime_columns(data, target_column=target_column)
    data, _ = try_convert_object_to_numeric(data, target_column=target_column)

    if target_column in data.columns:
        X = data.drop(columns=[target_column]).copy()
    else:
        X = data.copy()

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    ordinal_mappings = detect_ordinal_columns(X, categorical_cols)

    return {
        "auto_detected_ordinal_columns": list(ordinal_mappings.keys()),
        "categorical_columns": categorical_cols,
        "ordinal_mappings_found": ordinal_mappings
    }


def encode_ordinal_columns(df, ordinal_columns):
    data = df.copy()

    for col, mapping in ordinal_columns.items():
        data[col] = (
            data[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    return data


def detect_outliers_iqr(df, numerical_cols, extreme_multiplier=3.0):
    outlier_report = {}
    extreme_outlier_report = {}

    for col in numerical_cols:
        clean_series = df[col].dropna()

        if clean_series.shape[0] == 0:
            outlier_report[col] = 0
            extreme_outlier_report[col] = 0
            continue

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_report[col] = 0
            extreme_outlier_report[col] = 0
            continue

        lower_15 = q1 - 1.5 * iqr
        upper_15 = q3 + 1.5 * iqr

        lower_extreme = q1 - extreme_multiplier * iqr
        upper_extreme = q3 + extreme_multiplier * iqr

        outlier_count = ((df[col] < lower_15) | (df[col] > upper_15)).sum()
        extreme_count = ((df[col] < lower_extreme) | (df[col] > upper_extreme)).sum()

        outlier_report[col] = int(outlier_count)
        extreme_outlier_report[col] = int(extreme_count)

    return outlier_report, extreme_outlier_report


def cap_extreme_outliers(df, numerical_cols, extreme_multiplier=3.0):
    data = df.copy()
    capped_columns = []

    for col in numerical_cols:
        clean_series = data[col].dropna()

        if clean_series.shape[0] == 0:
            continue

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_extreme = q1 - extreme_multiplier * iqr
        upper_extreme = q3 + extreme_multiplier * iqr

        before_values = data[col].copy()
        data[col] = data[col].clip(lower=lower_extreme, upper=upper_extreme)

        if not before_values.equals(data[col]):
            capped_columns.append(col)

    return data, capped_columns


def is_high_cardinality(series, unique_count_threshold=20, unique_ratio_threshold=0.10):
    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    unique_count = non_null.nunique()
    unique_ratio = unique_count / len(non_null)

    return (unique_count > unique_count_threshold) and (unique_ratio > unique_ratio_threshold)


def apply_pca_to_features(X, variance_threshold=0.95):
    if X.empty:
        return X.copy(), {
            "pca_applied": False,
            "original_feature_count": 0,
            "reduced_feature_count": 0,
            "selected_n_components": 0,
            "explained_variance_ratio_sum": 0.0,
            "scaled_before_pca": False
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)

    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

    pca_report = {
        "pca_applied": True,
        "original_feature_count": int(X.shape[1]),
        "reduced_feature_count": int(X_pca_df.shape[1]),
        "selected_n_components": int(X_pca_df.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "scaled_before_pca": True
    }

    return X_pca_df, pca_report


def preprocess_data(
        df,
        target_column,
        missing_column_threshold=0.60,
        missing_row_threshold=0.50,
        high_cardinality_unique_count=20,
        high_cardinality_unique_ratio=0.10,
        object_to_numeric_threshold=0.80,
        datetime_parse_threshold=0.80,
        extreme_outlier_multiplier=3.0,
        user_selected_ordinal_columns=None,
        apply_pca=False,
        pca_variance_threshold=0.95
):
    data = df.copy()
    user_selected_ordinal_columns = user_selected_ordinal_columns or []

    report = {
        "initial_shape": df.shape,
        "final_shape": None,
        "removed_duplicates": 0,
        "removed_rows_due_to_missing": 0,
        "dropped_empty_columns": [],
        "dropped_high_missing_columns": [],
        "dropped_single_value_columns": [],
        "dropped_id_columns": [],
        "dropped_high_cardinality_columns": [],
        "converted_to_numeric": [],
        "parsed_datetime_columns": [],
        "created_datetime_features": [],
        "ordinal_encoded_columns": [],
        "auto_detected_ordinal_columns": [],
        "user_selected_ordinal_columns": [],
        "failed_user_ordinal_columns": [],
        "one_hot_encoded_columns": [],
        "categorical_columns_before_encoding": [],
        "numerical_columns_before_encoding": [],
        "filled_missing_numerical": [],
        "filled_missing_categorical": [],
        "target_encoded": False,
        "target_classes": None,
        "outlier_report": {},
        "extreme_outlier_report": {},
        "capped_outlier_columns": [],
        "pca_applied": False,
        "pca_report": None,
        "preview_before_pca": None,
        "columns_before_pca": []
    }

    if target_column not in data.columns:
        raise ValueError(f"Target column cannot be found: {target_column}")

    before_rows = data.shape[0]
    data = data.drop_duplicates()
    after_rows = data.shape[0]
    report["removed_duplicates"] = before_rows - after_rows

    empty_columns = data.columns[data.isnull().all()].tolist()

    if target_column in empty_columns:
        raise ValueError("Target column cannot be empty.")

    data = data.drop(columns=empty_columns, errors="ignore")
    report["dropped_empty_columns"] = empty_columns

    data, parsed_datetime_columns, created_datetime_features = try_parse_datetime_columns(
        data,
        target_column=target_column,
        parse_threshold=datetime_parse_threshold
    )
    report["parsed_datetime_columns"] = parsed_datetime_columns
    report["created_datetime_features"] = created_datetime_features

    data, converted_columns = try_convert_object_to_numeric(
        data,
        target_column=target_column,
        conversion_threshold=object_to_numeric_threshold
    )
    report["converted_to_numeric"] = converted_columns

    high_missing_columns = []

    for col in data.columns:
        if col == target_column:
            continue

        if data[col].isnull().mean() > missing_column_threshold:
            high_missing_columns.append(col)

    data = data.drop(columns=high_missing_columns, errors="ignore")
    report["dropped_high_missing_columns"] = high_missing_columns

    row_missing_ratio = data.isnull().mean(axis=1)
    before_rows = data.shape[0]
    data = data.loc[row_missing_ratio <= missing_row_threshold].copy()
    after_rows = data.shape[0]
    report["removed_rows_due_to_missing"] = before_rows - after_rows

    single_value_columns = []

    for col in data.columns:
        if col == target_column:
            continue

        if data[col].nunique(dropna=False) <= 1:
            single_value_columns.append(col)

    data = data.drop(columns=single_value_columns, errors="ignore")
    report["dropped_single_value_columns"] = single_value_columns

    data = data.dropna(subset=[target_column]).copy()

    y = data[target_column].copy()
    X = data.drop(columns=[target_column]).copy()

    id_columns = [col for col in X.columns if is_id_column(col)]
    X = X.drop(columns=id_columns, errors="ignore")
    report["dropped_id_columns"] = id_columns

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    high_cardinality_columns = []

    for col in categorical_cols:
        if is_high_cardinality(
                X[col],
                unique_count_threshold=high_cardinality_unique_count,
                unique_ratio_threshold=high_cardinality_unique_ratio
        ):
            high_cardinality_columns.append(col)

    X = X.drop(columns=high_cardinality_columns, errors="ignore")
    report["dropped_high_cardinality_columns"] = high_cardinality_columns

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    report["categorical_columns_before_encoding"] = categorical_cols.copy()
    report["numerical_columns_before_encoding"] = numerical_cols.copy()

    for col in numerical_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
            report["filled_missing_numerical"].append(col)

    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            mode_value = X[col].mode(dropna=True)

            if len(mode_value) > 0:
                X[col] = X[col].fillna(mode_value[0])
            else:
                X[col] = X[col].fillna("Unknown")

            report["filled_missing_categorical"].append(col)

    outlier_report, extreme_outlier_report = detect_outliers_iqr(
        X,
        numerical_cols=numerical_cols,
        extreme_multiplier=extreme_outlier_multiplier
    )
    report["outlier_report"] = outlier_report
    report["extreme_outlier_report"] = extreme_outlier_report

    X, capped_columns = cap_extreme_outliers(
        X,
        numerical_cols=numerical_cols,
        extreme_multiplier=extreme_outlier_multiplier
    )
    report["capped_outlier_columns"] = capped_columns

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    auto_detected_ordinal_columns = detect_ordinal_columns(X, categorical_cols)
    report["auto_detected_ordinal_columns"] = list(auto_detected_ordinal_columns.keys())

    final_ordinal_columns = dict(auto_detected_ordinal_columns)

    valid_user_selected = []
    failed_user_selected = []

    for col in user_selected_ordinal_columns:
        if col not in X.columns:
            failed_user_selected.append(col)
            continue

        if col not in categorical_cols:
            failed_user_selected.append(col)
            continue

        if col in final_ordinal_columns:
            valid_user_selected.append(col)
            continue

        detected_mapping = detect_ordinal_columns(X[[col]], [col])

        if col in detected_mapping:
            final_ordinal_columns[col] = detected_mapping[col]
            valid_user_selected.append(col)
        else:
            failed_user_selected.append(col)

    if len(final_ordinal_columns) > 0:
        X = encode_ordinal_columns(X, final_ordinal_columns)
        report["ordinal_encoded_columns"] = list(final_ordinal_columns.keys())

    report["user_selected_ordinal_columns"] = valid_user_selected
    report["failed_user_ordinal_columns"] = failed_user_selected

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        report["one_hot_encoded_columns"] = categorical_cols

    if y.dtype == "object" or str(y.dtype) == "category" or str(y.dtype) == "bool":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_column, index=y.index)
        report["target_encoded"] = True
        report["target_classes"] = list(le.classes_)

    report["preview_before_pca"] = X.head().copy()
    report["columns_before_pca"] = list(X.columns)

    if apply_pca:
        X, pca_report = apply_pca_to_features(
            X,
            variance_threshold=pca_variance_threshold
        )
        report["pca_applied"] = True
        report["pca_report"] = pca_report
    else:
        report["pca_applied"] = False
        report["pca_report"] = {
            "pca_applied": False,
            "original_feature_count": int(X.shape[1]),
            "reduced_feature_count": int(X.shape[1]),
            "selected_n_components": int(X.shape[1]),
            "explained_variance_ratio_sum": 1.0,
            "scaled_before_pca": False
        }

    report["final_shape"] = (X.shape[0], X.shape[1])

    return X, y, report