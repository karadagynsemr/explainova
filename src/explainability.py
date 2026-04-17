import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def _sanitize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if X is None or X.empty:
        raise ValueError("No feature data is available for SHAP analysis.")

    clean = X.copy()

    for col in clean.columns:
        if pd.api.types.is_bool_dtype(clean[col]):
            clean[col] = clean[col].astype(int)

    for col in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.replace([np.inf, -np.inf], np.nan)

    for col in clean.columns:
        if clean[col].isnull().sum() > 0:
            median_val = clean[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            clean[col] = clean[col].fillna(median_val)

    clean = clean.astype(float)
    return clean


def _truncate_feature_name(name, max_len=26):
    text = str(name)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _build_display_frame(X: pd.DataFrame, max_len=26) -> pd.DataFrame:
    display_X = X.copy()
    display_X.columns = [_truncate_feature_name(col, max_len=max_len) for col in display_X.columns]

    seen = {}
    unique_cols = []
    for col in display_X.columns:
        if col not in seen:
            seen[col] = 1
            unique_cols.append(col)
        else:
            seen[col] += 1
            unique_cols.append(f"{col} ({seen[col]})")

    display_X.columns = unique_cols
    return display_X


def sample_background_data(X, max_samples=100):
    if X.shape[0] <= max_samples:
        return X.copy()
    return X.sample(n=max_samples, random_state=42)


def sample_explanation_data(X, max_samples=200):
    if X.shape[0] <= max_samples:
        return X.copy()
    return X.sample(n=max_samples, random_state=42)


def _resolve_prediction_function(trained_model, problem_type):
    if problem_type == "classification":
        if hasattr(trained_model, "predict_proba"):
            return lambda data: trained_model.predict_proba(data)[:, 1]
        return lambda data: trained_model.predict(data)

    return lambda data: trained_model.predict(data)


def compute_shap_outputs(
        trained_model,
        X_reference,
        problem_type,
        max_background_samples=100,
        max_explain_samples=200
):
    if X_reference is None or X_reference.empty:
        raise ValueError("No feature data is available for SHAP analysis.")

    X_reference = _sanitize_feature_frame(X_reference)

    X_background = sample_background_data(X_reference, max_samples=max_background_samples)
    X_explain = sample_explanation_data(X_reference, max_samples=max_explain_samples)

    predict_fn = _resolve_prediction_function(trained_model, problem_type)

    masker = shap.maskers.Independent(X_background)

    try:
        explainer = shap.Explainer(predict_fn, masker)
        shap_explanation = explainer(X_explain)
    except Exception:
        explainer = shap.Explainer(predict_fn, X_background.values)
        shap_explanation = explainer(X_explain.values)

    shap_values = shap_explanation.values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values, dtype=float)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError("SHAP values could not be converted into a 2D numeric array.")

    feature_importance = pd.DataFrame({
        "Feature": X_explain.columns,
        "Mean |SHAP Value|": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)

    return {
        "X_explain": X_explain,
        "shap_values": shap_values,
        "feature_importance_df": feature_importance
    }


def plot_shap_importance_bar(feature_importance_df, top_n=12):
    if feature_importance_df.empty:
        return None

    plot_df = feature_importance_df.head(top_n).copy()
    plot_df["Display Feature"] = plot_df["Feature"].apply(lambda x: _truncate_feature_name(x, max_len=28))

    fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")
    ax.set_facecolor("#FFFFFF")

    ax.barh(
        plot_df["Display Feature"][::-1],
        plot_df["Mean |SHAP Value|"][::-1],
        color="#6366F1",
        edgecolor="#4F46E5",
        linewidth=0.8,
        alpha=0.9
    )

    ax.set_title("SHAP Feature Importance", fontsize=10, pad=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#334155", labelsize=8)
    ax.tick_params(axis="y", colors="#334155", labelsize=8)
    ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.34, right=0.97, top=0.88, bottom=0.16)
    return fig


def plot_shap_summary_figure(shap_values, X_explain, max_display=12):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)

    display_X = _build_display_frame(X_explain, max_len=26)

    plt.close("all")
    fig = plt.figure(figsize=(7.8, 4.8), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    shap.summary_plot(
        shap_values,
        display_X,
        max_display=max_display,
        show=False,
        plot_size=None
    )

    current_fig = plt.gcf()
    current_fig.set_size_inches(7.8, 4.8)
    current_fig.patch.set_facecolor("#F6F8FC")

    for ax in current_fig.axes:
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

    current_fig.subplots_adjust(left=0.32, right=0.96, top=0.90, bottom=0.16)
    return current_fig


def get_shap_selection_guidance(problem_type, has_roc_auc=False):
    if problem_type == "classification":
        text = (
            "Choose the model you want to explain according to the kind of decision you care about most. "
            "If false positives would create unnecessary risk or cost, a model with stronger precision may be more suitable. "
            "If missing true positives would be more harmful, a model with stronger recall may be a better choice. "
            "If you want a more balanced trade-off between the two, F1 Score is usually a good reference."
        )
        if has_roc_auc:
            text += " In binary classification, ROC AUC can also help when overall class separation matters across thresholds."
        return text

    return (
        "Choose the regression model you want to explain based on the type of performance you trust most. "
        "R2 Score is useful when you care about overall explanatory strength, while MAE and RMSE are better when you want to focus on prediction error size in the original target units."
    )


def get_shap_intro_text():
    return (
        "SHAP explains how each feature contributes to a model prediction. "
        "A positive SHAP value pushes the prediction upward, while a negative SHAP value pushes it downward. "
        "The larger the absolute SHAP value, the more strongly that feature influenced the model's output."
    )