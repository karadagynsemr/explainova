import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, roc_curve, roc_auc_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


def detect_problem_type(y, unique_threshold=10):
    if str(y.dtype) in ["object", "category", "bool"]:
        return "classification"

    unique_values = y.nunique()

    if unique_values <= unique_threshold:
        return "classification"

    return "regression"


def get_classification_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }


def get_regression_models():
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }


def get_available_models(problem_type):
    if problem_type == "classification":
        return get_classification_models()
    return get_regression_models()


def evaluate_classification_model(model, X_train, X_test, y_train, y_test, class_labels=None):
    trained_model = clone(model)
    trained_model.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    unique_classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

    roc_data = None

    if len(unique_classes) == 2:
        try:
            if hasattr(trained_model, "predict_proba"):
                y_score = trained_model.predict_proba(X_test)[:, 1]
            elif hasattr(trained_model, "decision_function"):
                y_score = trained_model.decision_function(X_test)
            else:
                y_score = None

            if y_score is not None:
                fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=unique_classes[1])
                auc_score = roc_auc_score(y_test, y_score)

                roc_data = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds,
                    "auc": float(auc_score)
                }
        except Exception:
            roc_data = None

    if class_labels is not None and len(class_labels) >= len(unique_classes):
        resolved_class_labels = [str(class_labels[int(cls)]) for cls in unique_classes]
    else:
        resolved_class_labels = [str(cls) for cls in unique_classes]

    return {
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "class_labels": resolved_class_labels,
        "roc_data": roc_data,
        "trained_model": trained_model
    }


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    trained_model = clone(model)
    trained_model.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse
    }

    return {
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "trained_model": trained_model
    }


def build_results_dataframe(problem_type, detailed_results):
    rows = []

    for model_name, details in detailed_results.items():
        row = {
            "Model": model_name,
            "Problem Type": problem_type
        }
        row.update(details["metrics"])

        if problem_type == "classification" and details.get("roc_data") is not None:
            row["ROC AUC"] = details["roc_data"]["auc"]

        rows.append(row)

    return pd.DataFrame(rows)


def train_single_model(
        X,
        y,
        model_name,
        test_size=0.2,
        random_state=42,
        class_labels=None
):
    problem_type = detect_problem_type(y)
    available_models = get_available_models(problem_type)

    if model_name not in available_models:
        raise ValueError(f"Selected model is not available for {problem_type}: {model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if problem_type == "classification" and y.nunique() > 1 else None
    )

    model = available_models[model_name]

    if problem_type == "classification":
        details = evaluate_classification_model(
            model, X_train, X_test, y_train, y_test, class_labels=class_labels
        )
    else:
        details = evaluate_regression_model(model, X_train, X_test, y_train, y_test)

    detailed_results = {model_name: details}
    results_df = build_results_dataframe(problem_type, detailed_results)

    return problem_type, results_df, detailed_results


def train_multiple_models(
        X,
        y,
        test_size=0.2,
        random_state=42,
        class_labels=None
):
    problem_type = detect_problem_type(y)
    models = get_available_models(problem_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if problem_type == "classification" and y.nunique() > 1 else None
    )

    detailed_results = {}

    for model_name, model in models.items():
        if problem_type == "classification":
            details = evaluate_classification_model(
                model, X_train, X_test, y_train, y_test, class_labels=class_labels
            )
        else:
            details = evaluate_regression_model(model, X_train, X_test, y_train, y_test)

        detailed_results[model_name] = details

    results_df = build_results_dataframe(problem_type, detailed_results)

    return problem_type, results_df, detailed_results


def train_and_evaluate_models(
        X,
        y,
        training_mode="multiple",
        selected_model_name=None,
        test_size=0.2,
        random_state=42,
        class_labels=None
):
    if training_mode == "single":
        if not selected_model_name:
            raise ValueError("Please select a model for single-model training.")

        return train_single_model(
            X=X,
            y=y,
            model_name=selected_model_name,
            test_size=test_size,
            random_state=random_state,
            class_labels=class_labels
        )

    return train_multiple_models(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        class_labels=class_labels
    )