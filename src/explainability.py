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
    base_values = getattr(shap_explanation, "base_values", None)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values, dtype=float)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError("SHAP values could not be converted into a 2D numeric array.")

    if base_values is None:
        base_values = np.zeros(shape=(shap_values.shape[0],), dtype=float)

    base_values = np.array(base_values, dtype=float)

    if base_values.ndim > 1:
        base_values = base_values[:, 0]

    if base_values.ndim == 0:
        base_values = np.repeat(float(base_values), shap_values.shape[0])

    feature_importance = pd.DataFrame({
        "Feature": X_explain.columns,
        "Mean |SHAP Value|": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Mean |SHAP Value|", ascending=False).reset_index(drop=True)

    return {
        "X_explain": X_explain,
        "shap_values": shap_values,
        "base_values": base_values,
        "feature_importance_df": feature_importance
    }


def plot_shap_importance_bar(feature_importance_df, top_n=12):
    if feature_importance_df.empty:
        return None

    plot_df = feature_importance_df.head(top_n).copy()
    plot_df["Display Feature"] = plot_df["Feature"].apply(lambda x: _truncate_feature_name(x, max_len=28))

    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=140)
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

    ax.set_title("Most influential features", fontsize=10.5, pad=10)
    ax.set_xlabel("Average impact strength", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#334155", labelsize=8)
    ax.tick_params(axis="y", colors="#334155", labelsize=8)
    ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.18)
    return fig


def plot_shap_summary_figure(shap_values, X_explain, max_display=12):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)
    display_X = _build_display_frame(X_explain, max_len=26)

    plt.close("all")
    fig = plt.figure(figsize=(6.3, 3.9), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    shap.summary_plot(
        shap_values,
        display_X,
        max_display=max_display,
        show=False,
        plot_size=None
    )

    current_fig = plt.gcf()
    current_fig.set_size_inches(6.3, 3.9)
    current_fig.patch.set_facecolor("#F6F8FC")

    for ax in current_fig.axes:
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

    current_fig.subplots_adjust(left=0.34, right=0.96, top=0.90, bottom=0.18)
    return current_fig


def plot_shap_waterfall_figure(base_values, shap_values, X_explain, sample_index=0, max_display=8):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)
    base_values = np.array(base_values, dtype=float)

    if X_explain.empty:
        return None

    sample_index = int(sample_index)
    sample_index = max(0, min(sample_index, len(X_explain) - 1))

    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=float(base_values[sample_index]),
        data=X_explain.iloc[sample_index].values,
        feature_names=list(X_explain.columns)
    )

    plt.close("all")
    fig = plt.figure(figsize=(6.1, 3.8), dpi=135)
    fig.patch.set_facecolor("#F6F8FC")

    shap.plots.waterfall(explanation, max_display=max_display, show=False)

    current_fig = plt.gcf()
    current_fig.set_size_inches(6.1, 3.8)
    current_fig.patch.set_facecolor("#F6F8FC")

    for ax in current_fig.axes:
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    current_fig.subplots_adjust(left=0.26, right=0.97, top=0.90, bottom=0.17)
    return current_fig


def plot_shap_feature_effect_figure(shap_values, X_explain, feature_name):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)

    if X_explain.empty or feature_name not in X_explain.columns:
        return None

    feature_idx = list(X_explain.columns).index(feature_name)
    feature_vals = X_explain[feature_name].values
    feature_shap = shap_values[:, feature_idx]

    fig, ax = plt.subplots(figsize=(4.9, 3.2), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")
    ax.set_facecolor("#FFFFFF")

    scatter = ax.scatter(
        feature_vals,
        feature_shap,
        s=24,
        alpha=0.75,
        c=feature_vals,
        cmap="coolwarm",
        edgecolors="none"
    )

    ax.axhline(0, color="#94A3B8", linestyle="--", linewidth=1)
    ax.set_title(f"How the model reacts when {feature_name} changes", fontsize=10, pad=10)
    ax.set_xlabel(feature_name, fontsize=9)
    ax.set_ylabel("Impact on prediction", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#334155", labelsize=8)
    ax.tick_params(axis="y", colors="#334155", labelsize=8)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor("#CBD5E1")
    cbar.ax.tick_params(labelsize=8, colors="#334155")

    fig.subplots_adjust(left=0.13, right=0.92, top=0.88, bottom=0.18)
    return fig


def get_waterfall_interpretation(base_values, shap_values, X_explain, sample_index=0, top_k=3):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)
    base_values = np.array(base_values, dtype=float)

    if X_explain.empty:
        return "This chart could not be interpreted because no usable sample was available."

    sample_index = int(sample_index)
    sample_index = max(0, min(sample_index, len(X_explain) - 1))

    row_shap = shap_values[sample_index]
    row_values = X_explain.iloc[sample_index]

    pred_value = float(base_values[sample_index] + row_shap.sum())
    direction = "higher" if row_shap.sum() >= 0 else "lower"

    top_idx = np.argsort(np.abs(row_shap))[::-1][:top_k]
    pieces = []

    for idx in top_idx:
        feat = X_explain.columns[idx]
        val = row_values.iloc[idx]
        contrib = row_shap[idx]
        sign_text = "pushed the result upward" if contrib > 0 else "pulled the result downward"
        pieces.append(f"{feat}={val:.3f} {sign_text} by {abs(contrib):.3f}")

    joined = "; ".join(pieces)

    direction_text = "higher" if direction == "higher" else "lower"

    return (
        f"This chart explains why one specific result came out this way. "
        f"The model starts from its usual baseline, and then each feature moves the prediction step by step. "
        f"The selected row ends around {pred_value:.3f}, which is {direction_text} than the starting point. "
        f"The strongest reasons are: {joined}."
    )


def get_feature_effect_interpretation(shap_values, X_explain, feature_name):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)

    if X_explain.empty or feature_name not in X_explain.columns:
        return "This chart could not be interpreted because the selected feature was not available."

    feature_idx = list(X_explain.columns).index(feature_name)
    feature_vals = X_explain[feature_name].values
    feature_shap = shap_values[:, feature_idx]

    if np.std(feature_vals) == 0 or np.std(feature_shap) == 0:
        return (
            f"This chart shows how {feature_name} affects the model, "
            f"but this feature does not vary enough in the available sample to reveal a clear pattern."
        )

    corr = np.corrcoef(feature_vals, feature_shap)[0, 1]

    if corr >= 0.35:
        trend = "when this feature increases, the model usually pushes the result upward"
    elif corr <= -0.35:
        trend = "when this feature increases, the model usually pulls the result downward"
    else:
        trend = "the effect is more mixed, so the relationship does not follow a simple straight pattern"

    shap_strength = np.mean(np.abs(feature_shap))

    return (
        f"This chart shows how the model usually reacts when one feature changes. "
        f"For {feature_name}, the overall pattern suggests that {trend}. "
        f"The average strength of this effect is about {shap_strength:.3f}."
    )


def get_shap_selection_guidance(problem_type, has_roc_auc=False):
    if problem_type == "classification":
        return (
            "Start with the overall results table and choose the model that looks most trustworthy. "
            "A strong starting point is usually the model that leads on Accuracy or F1 Score. "
            "If one model is clearly ahead, it makes sense to use that one for SHAP. "
            "If the scores are close, it can help to prefer models that feel more stable and easier to explain."
        )

    return (
        "Choose the regression model that looks strongest in the overall results table. "
        "A high R2 Score together with lower error values is usually a good starting signal. "
        "If several models look similar, it often helps to choose the one that seems more stable and easier to explain."
    )


def get_shap_intro_text():
    return (
        "SHAP explains which features influenced a model result and by how much. "
        "Positive values push the prediction upward, while negative values pull it downward. "
        "The larger the value, the stronger that feature's influence."
    )
