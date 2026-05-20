import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def is_id_like_by_name(column_name):
    col = column_name.strip().lower()

    exact_matches = {
        "id", "patient_id", "sample_id", "record_id",
        "user_id", "customer_id", "transaction_id",
        "case_id", "subject_id", "uid"
    }

    if col in exact_matches:
        return True

    if col.endswith("_id") or col.startswith("id_"):
        return True

    return False


def _is_integer_like_series(series, tol=1e-9):
    non_null = pd.Series(series).dropna()
    if non_null.empty:
        return False

    if not pd.api.types.is_numeric_dtype(non_null):
        return False

    values = non_null.astype(float)
    return bool(np.all(np.abs(values - np.round(values)) < tol))


def is_probably_sequential_numeric_id(series):
    non_null = pd.Series(series).dropna()
    if len(non_null) < 30:
        return False

    if not pd.api.types.is_numeric_dtype(non_null):
        return False

    if not _is_integer_like_series(non_null):
        return False

    values = pd.Series(non_null).astype(float)
    unique_ratio = values.nunique() / len(values)
    if unique_ratio < 0.98:
        return False

    sorted_vals = np.sort(values.unique())
    if len(sorted_vals) < 3:
        return False

    diffs = np.diff(sorted_vals)
    if len(diffs) == 0:
        return False

    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return False

    dominant_step_share = (positive_diffs == positive_diffs[0]).mean()
    is_monotonic = values.is_monotonic_increasing or values.is_monotonic_decreasing

    return bool(is_monotonic or dominant_step_share >= 0.9)


def is_id_like_by_uniqueness(series, unique_ratio_threshold=0.98, min_rows=30):
    non_null = series.dropna()

    if len(non_null) < min_rows:
        return False

    unique_ratio = non_null.nunique() / len(non_null)
    if unique_ratio < unique_ratio_threshold:
        return False

    if pd.api.types.is_numeric_dtype(series):
        return is_probably_sequential_numeric_id(series)

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        avg_len = non_null.astype(str).str.len().mean()
        has_letters = non_null.astype(str).str.contains(r"[A-Za-z]", regex=True).mean() > 0.3
        return avg_len >= 6 or has_letters

    return False


def detect_id_like_columns(df):
    detected = []

    for col in df.columns:
        if is_id_like_by_name(col) or is_id_like_by_uniqueness(df[col]):
            detected.append(col)

    return detected


def try_convert_object_to_numeric(df, target_column, conversion_threshold=0.80):
    data = df.copy()
    converted_columns = []

    object_like_cols = [
        col for col in data.columns
        if (pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]))
    ]

    for col in object_like_cols:
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

    object_like_cols = [
        col for col in data.columns
        if (pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]))
    ]

    for col in object_like_cols:
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

    categorical_cols = [
        col for col in X.columns
        if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
        )
    ]
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


def normalize_user_defined_ordinal_mappings(user_defined_ordinal_mappings):
    normalized = {}

    if not user_defined_ordinal_mappings:
        return normalized

    for col, ordered_values in user_defined_ordinal_mappings.items():
        if not ordered_values:
            continue

        cleaned_values = []
        seen = set()

        for value in ordered_values:
            cleaned = str(value).strip().lower()
            if cleaned and cleaned not in seen:
                cleaned_values.append(cleaned)
                seen.add(cleaned)

        if cleaned_values:
            normalized[col] = {value: idx for idx, value in enumerate(cleaned_values)}

    return normalized


def resolve_protected_transformed_columns(protected_original_features, transformed_columns):
    protected_original_features = protected_original_features or []
    protected_transformed = set()

    for original_feature in protected_original_features:
        for col in transformed_columns:
            if col == original_feature or col.startswith(f"{original_feature}_"):
                protected_transformed.add(col)

    return protected_transformed


def replace_zero_values_with_missing(df, columns, target_column=None):
    data = df.copy()
    replaced_counts = {}

    for col in columns or []:
        if col not in data.columns or col == target_column:
            continue

        numeric_values = pd.to_numeric(data[col], errors="coerce")
        zero_mask = numeric_values.eq(0) & data[col].notna()
        zero_count = int(zero_mask.sum())

        if zero_count > 0:
            data.loc[zero_mask, col] = np.nan
            replaced_counts[col] = zero_count

    return data, replaced_counts


def remove_low_variance_features(X, protected_columns=None, variance_threshold=0.0001):
    protected_columns = protected_columns or set()
    variances = X.var(numeric_only=True)

    to_drop = []
    for col in X.columns:
        if col in protected_columns:
            continue
        if col in variances.index and variances[col] <= variance_threshold:
            to_drop.append(col)

    return X.drop(columns=to_drop, errors="ignore"), to_drop


def remove_highly_correlated_features(X, protected_columns=None, correlation_threshold=0.95):
    protected_columns = protected_columns or set()

    corr_matrix = X.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()

    for col in upper.columns:
        correlated_with = upper.index[upper[col] > correlation_threshold].tolist()

        for other_col in correlated_with:
            if col in protected_columns and other_col in protected_columns:
                continue
            elif col in protected_columns and other_col not in protected_columns:
                to_drop.add(other_col)
            elif col not in protected_columns and other_col in protected_columns:
                to_drop.add(col)
            else:
                to_drop.add(col)

    to_drop = list(sorted(set(to_drop)))
    return X.drop(columns=to_drop, errors="ignore"), to_drop


def keep_top_k_important_features(X, y, problem_type, protected_columns=None, top_k=40):
    protected_columns = protected_columns or set()

    if X.shape[1] <= top_k:
        return X.copy(), [], pd.DataFrame({
            "Feature": X.columns,
            "Importance": np.nan
        })

    if problem_type == "classification":
        selector_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    else:
        selector_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    selector_model.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": selector_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    protected_list = [col for col in importance_df["Feature"].tolist() if col in protected_columns]
    non_protected_ranked = [col for col in importance_df["Feature"].tolist() if col not in protected_columns]

    num_extra_needed = max(0, top_k - len(protected_list))
    selected_non_protected = non_protected_ranked[:num_extra_needed]

    keep_columns = list(dict.fromkeys(protected_list + selected_non_protected))
    drop_columns = [col for col in X.columns if col not in keep_columns]

    return X[keep_columns].copy(), drop_columns, importance_df


def target_should_be_encoded(y):
    return (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
            or pd.api.types.is_bool_dtype(y)
    )


def preprocess_data(
        df,
        target_column,
        selected_feature_columns=None,
        missing_column_threshold=0.60,
        missing_row_threshold=0.50,
        high_cardinality_unique_count=20,
        high_cardinality_unique_ratio=0.10,
        object_to_numeric_threshold=0.80,
        datetime_parse_threshold=0.80,
        extreme_outlier_multiplier=3.0,
        user_selected_ordinal_columns=None,
        user_defined_ordinal_mappings=None,
        apply_feature_reduction=False,
        feature_reduction_strategy="fast_interpretable",
        protected_original_features=None,
        zero_as_missing_columns=None,
        low_variance_threshold=0.0001,
        high_correlation_threshold=0.95,
        top_k_important_features=40
):
    if target_column not in df.columns:
        raise ValueError(f"Target column cannot be found: {target_column}")

    if selected_feature_columns is not None:
        selected_feature_columns = [col for col in selected_feature_columns if col in df.columns and col != target_column]
        if len(selected_feature_columns) == 0:
            raise ValueError("At least one feature column must be selected.")
        data = df[selected_feature_columns + [target_column]].copy()
    else:
        data = df.copy()

    user_selected_ordinal_columns = user_selected_ordinal_columns or []
    user_defined_ordinal_mappings = user_defined_ordinal_mappings or {}
    protected_original_features = protected_original_features or []
    zero_as_missing_columns = zero_as_missing_columns or []

    report = {
        "initial_shape": data.shape,
        "final_shape": None,
        "selected_feature_columns": selected_feature_columns,
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
        "user_defined_ordinal_columns": [],
        "failed_user_ordinal_columns": [],
        "one_hot_encoded_columns": [],
        "categorical_columns_before_encoding": [],
        "numerical_columns_before_encoding": [],
        "filled_missing_numerical": [],
        "filled_missing_categorical": [],
        "zero_as_missing_columns": [],
        "zero_as_missing_counts": {},
        "zero_missing_values_replaced": 0,
        "target_encoded": False,
        "target_classes": None,
        "target_label_mapping": None,
        "outlier_report": {},
        "extreme_outlier_report": {},
        "capped_outlier_columns": [],
        "feature_reduction_applied": False,
        "feature_reduction_strategy": None,
        "low_variance_threshold": low_variance_threshold,
        "high_correlation_threshold": high_correlation_threshold,
        "protected_original_features": protected_original_features,
        "protected_transformed_features": [],
        "removed_low_variance_columns": [],
        "removed_high_correlation_columns": [],
        "removed_low_importance_columns": [],
        "feature_importance_ranking": None
    }

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

    data, zero_as_missing_counts = replace_zero_values_with_missing(
        data,
        columns=zero_as_missing_columns,
        target_column=target_column
    )
    report["zero_as_missing_counts"] = zero_as_missing_counts
    report["zero_as_missing_columns"] = list(zero_as_missing_counts.keys())
    report["zero_missing_values_replaced"] = int(sum(zero_as_missing_counts.values()))

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

    id_columns = detect_id_like_columns(X)
    X = X.drop(columns=id_columns, errors="ignore")
    report["dropped_id_columns"] = id_columns

    categorical_cols = [
        col for col in X.columns
        if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
        )
    ]
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

    categorical_cols = [
        col for col in X.columns
        if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
        )
    ]
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

    categorical_cols = [
        col for col in X.columns
        if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
        )
    ]
    auto_detected_ordinal_columns = detect_ordinal_columns(X, categorical_cols)
    report["auto_detected_ordinal_columns"] = list(auto_detected_ordinal_columns.keys())

    final_ordinal_columns = dict(auto_detected_ordinal_columns)

    normalized_user_mappings = normalize_user_defined_ordinal_mappings(user_defined_ordinal_mappings)

    valid_user_selected = []
    valid_user_defined = []
    failed_user_selected = []

    for col in user_selected_ordinal_columns:
        if col not in X.columns or col not in categorical_cols:
            failed_user_selected.append(col)
            continue

        valid_user_selected.append(col)

        if col in normalized_user_mappings:
            col_mapping = normalized_user_mappings[col]

            existing_values = (
                X[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
            )
            missing_values = [v for v in existing_values if v not in col_mapping]

            if len(missing_values) == 0:
                final_ordinal_columns[col] = col_mapping
                valid_user_defined.append(col)
            else:
                failed_user_selected.append(col)

    if len(final_ordinal_columns) > 0:
        X = encode_ordinal_columns(X, final_ordinal_columns)
        report["ordinal_encoded_columns"] = list(final_ordinal_columns.keys())

    report["user_selected_ordinal_columns"] = valid_user_selected
    report["user_defined_ordinal_columns"] = valid_user_defined
    report["failed_user_ordinal_columns"] = list(sorted(set(failed_user_selected)))

    categorical_cols = [
        col for col in X.columns
        if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
        )
    ]

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        report["one_hot_encoded_columns"] = categorical_cols

    if target_should_be_encoded(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_column, index=y.index)
        original_classes = list(le.classes_)
        report["target_encoded"] = True
        report["target_classes"] = original_classes
        report["target_label_mapping"] = {int(i): str(cls) for i, cls in enumerate(original_classes)}

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.astype(float)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remained after preprocessing. "
            "This usually happens when columns are dropped as ID-like, empty, constant, or high-cardinality."
        )

    if apply_feature_reduction:
        protected_transformed = resolve_protected_transformed_columns(
            protected_original_features=protected_original_features,
            transformed_columns=X.columns.tolist()
        )

        report["feature_reduction_applied"] = True
        report["feature_reduction_strategy"] = feature_reduction_strategy
        report["protected_transformed_features"] = list(sorted(protected_transformed))

        X, dropped_low_variance = remove_low_variance_features(
            X,
            protected_columns=protected_transformed,
            variance_threshold=low_variance_threshold
        )
        report["removed_low_variance_columns"] = dropped_low_variance

        X, dropped_high_corr = remove_highly_correlated_features(
            X,
            protected_columns=protected_transformed,
            correlation_threshold=high_correlation_threshold
        )
        report["removed_high_correlation_columns"] = dropped_high_corr

        if X.shape[1] == 0:
            raise ValueError("Feature reduction removed all columns. Try disabling feature reduction or protecting important features.")

    report["final_shape"] = (X.shape[0], X.shape[1])

    return X, y, report, X.copy()
