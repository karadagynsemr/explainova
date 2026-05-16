import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import accuracy_score, r2_score


COLORS = {
    "model": "#3B82F6",
    "shap": "#8B5CF6",
    "shap_dark": "#6D28D9",
    "preprocessing": "#10B981",
    "preprocessing_dark": "#059669",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "slate": "#334155",
    "border": "#CBD5E1",
    "grid": "#E2E8F0",
    "fig_bg": "#F6F8FC",
}


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


def _predict_behavior_score(trained_model, X, problem_type):
    if problem_type == "classification":
        if hasattr(trained_model, "predict_proba"):
            proba = trained_model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        return trained_model.predict(X)

    return trained_model.predict(X)


def _apply_explainability_axis_style(ax, grid_axis="y"):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#334155", labelsize=8)
    ax.tick_params(axis="y", colors="#334155", labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E2E8F0", linestyle="--", linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)


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
    fig.patch.set_facecolor(COLORS["fig_bg"])
    ax.set_facecolor("#FFFFFF")

    ax.barh(
        plot_df["Display Feature"][::-1],
        plot_df["Mean |SHAP Value|"][::-1],
        color=COLORS["shap"],
        edgecolor=COLORS["shap_dark"],
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
    fig.patch.set_facecolor(COLORS["fig_bg"])

    shap.summary_plot(
        shap_values,
        display_X,
        max_display=max_display,
        show=False,
        plot_size=None
    )

    current_fig = plt.gcf()
    current_fig.set_size_inches(6.3, 3.9)
    current_fig.patch.set_facecolor(COLORS["fig_bg"])

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


def _format_positive_class_reference(positive_class_label=None):
    if positive_class_label is None or str(positive_class_label).strip() == "":
        return "class 1 / the positive class"
    label_text = str(positive_class_label).strip()
    if label_text in ["1", "1.0"]:
        return "class 1"
    return f"the positive class ({label_text})"


def plot_shap_feature_effect_figure(
        shap_values,
        X_explain,
        feature_name,
        problem_type=None,
        positive_class_label=None
):
    X_explain = _sanitize_feature_frame(X_explain)
    shap_values = np.array(shap_values, dtype=float)

    if X_explain.empty or feature_name not in X_explain.columns:
        return None

    feature_idx = list(X_explain.columns).index(feature_name)
    feature_vals = X_explain[feature_name].values
    feature_shap = shap_values[:, feature_idx]

    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")
    ax.set_facecolor("#FFFFFF")

    point_colors = np.where(feature_shap >= 0, COLORS["preprocessing"], COLORS["danger"])
    scatter = ax.scatter(
        feature_vals,
        feature_shap,
        s=24,
        alpha=0.75,
        c=point_colors,
        edgecolors="none"
    )

    ax.axhline(0, color="#94A3B8", linestyle="--", linewidth=1)
    if problem_type == "classification":
        target_text = _format_positive_class_reference(positive_class_label)
        title = f"How {feature_name} affects movement toward {target_text}"
        ylabel = f"Impact toward {target_text}"
    else:
        title = f"How the model reacts when {feature_name} changes"
        ylabel = "Impact on prediction"

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel(feature_name, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#334155", labelsize=8)
    ax.tick_params(axis="y", colors="#334155", labelsize=8)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.18)
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


def get_feature_effect_interpretation(
        shap_values,
        X_explain,
        feature_name,
        problem_type=None,
        positive_class_label=None
):
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

    if problem_type == "classification":
        target_text = _format_positive_class_reference(positive_class_label)
        if corr >= 0.35:
            trend = f"higher values usually increase movement toward {target_text}"
        elif corr <= -0.35:
            trend = f"higher values usually move the model away from {target_text}"
        else:
            trend = f"the effect on {target_text} is mixed, so the relationship is not a simple straight pattern"
    else:
        if corr >= 0.35:
            trend = "when this feature increases, the model usually pushes the predicted value upward"
        elif corr <= -0.35:
            trend = "when this feature increases, the model usually pulls the predicted value downward"
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


def get_shap_intro_text(problem_type=None, positive_class_label=None):
    if problem_type == "classification":
        target_text = _format_positive_class_reference(positive_class_label)
        return (
            f"SHAP explains which features moved the model toward or away from {target_text}. "
            f"Positive SHAP values increase the model's support for {target_text}; negative values decrease it. "
            "The larger the value, the stronger that feature's influence."
        )

    return (
        "SHAP explains which features influenced a model result and by how much. "
        "Positive values push the predicted value upward, while negative values pull it downward. "
        "The larger the value, the stronger that feature's influence."
    )


def compute_kfold_stability(models, X, y, problem_type, n_splits=5):
    X = _sanitize_feature_frame(X)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if len(X) < 2:
        return pd.DataFrame()

    class_counts = y.value_counts() if problem_type == "classification" else pd.Series(dtype=int)
    if problem_type == "classification" and not class_counts.empty:
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            return pd.DataFrame()
        n_splits = max(2, min(n_splits, min_class_count))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = {"Accuracy": "accuracy", "F1 Score": "f1_weighted"}
        if y.nunique() == 2:
            scoring["ROC AUC"] = "roc_auc"
    else:
        n_splits = max(2, min(n_splits, len(X)))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = {"R2 Score": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}

    rows = []
    for model_name, model in models.items():
        try:
            scores = cross_validate(
                clone(model),
                X,
                y,
                cv=splitter,
                scoring=scoring,
                n_jobs=None,
                error_score=np.nan
            )
        except Exception:
            continue

        row = {"Model": model_name, "Folds": n_splits}
        for metric_name in scoring.keys():
            values = np.array(scores[f"test_{metric_name}"], dtype=float)
            if metric_name in ["MAE", "RMSE"]:
                values = np.abs(values)
            row[f"{metric_name} Mean"] = float(np.nanmean(values))
            row[f"{metric_name} Std"] = float(np.nanstd(values))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_kfold_stability_figure(kfold_df, problem_type, metric_name=None):
    if kfold_df is None or kfold_df.empty:
        return None

    if metric_name is None:
        metric_name = "Accuracy" if problem_type == "classification" else "R2 Score"

    metric = f"{metric_name} Mean"
    std_metric = f"{metric_name} Std"
    if metric not in kfold_df.columns:
        return None

    lower_is_better = metric_name in ["MAE", "RMSE"]
    plot_df = kfold_df.sort_values(metric, ascending=not lower_is_better).copy()
    labels = [_truncate_feature_name(item, max_len=24) for item in plot_df["Model"]]

    fig, ax = plt.subplots(figsize=(4.4, 2.65), dpi=140)
    fig.patch.set_facecolor(COLORS["fig_bg"])

    ax.barh(
        labels,
        plot_df[metric],
        xerr=plot_df[std_metric] if std_metric in plot_df.columns else None,
        color=COLORS["preprocessing"],
        edgecolor=COLORS["preprocessing_dark"],
        linewidth=0.8,
        alpha=0.92
    )
    ax.set_title(f"{metric_name} stability across folds", fontsize=10.2, pad=9)
    ax.set_xlabel(metric_name, fontsize=8.5)
    ax.set_ylabel("")
    _apply_explainability_axis_style(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def get_kfold_interpretation(kfold_df, problem_type, metric_name=None, current_leader=None):
    if kfold_df is None or kfold_df.empty:
        return "Stability could not be measured because cross-validation did not return usable results."

    if metric_name is None:
        metric_name = "Accuracy" if problem_type == "classification" else "R2 Score"

    metric = f"{metric_name} Mean"
    std_metric = f"{metric_name} Std"
    if metric not in kfold_df.columns:
        return "Stability could not be summarized because the expected metric was not available."

    lower_is_better = metric_name in ["MAE", "RMSE"]
    best = kfold_df.sort_values(metric, ascending=lower_is_better).iloc[0]
    spread = float(best.get(std_metric, 0.0))
    if spread <= 0.03:
        stability = "quite stable"
    elif spread <= 0.08:
        stability = "reasonably stable, with some fold-to-fold movement"
    else:
        stability = "sensitive to which rows are used for training"

    base_text = (
        f"K-fold validation repeats the evaluation across multiple train/test splits. "
        f"Using {metric_name}, {best['Model']} has the strongest average result in this stability check, with variation that appears {stability}. "
        f"This does not replace the main model ranking; it shows whether performance remains consistent when the data split changes."
    )

    if current_leader and str(best["Model"]) != str(current_leader):
        return (
            base_text + f" The main dashboard leader is {current_leader}, while K-fold highlights {best['Model']}. "
            "This can happen because the dashboard is based on one held-out split, while K-fold averages several splits. "
            "Treat this as a stability signal: if the difference is small, the simpler or more explainable model can still be preferred; if the gap is large, compare both models before choosing."
        )

    return base_text


def compute_learning_curve_data(model, X, y, problem_type, cv=4):
    X = _sanitize_feature_frame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if len(X) < 3:
        return None
    scoring = "accuracy" if problem_type == "classification" else "r2"

    if problem_type == "classification":
        min_class_count = int(y.value_counts().min())
        if min_class_count < 2:
            return None
        cv = max(2, min(cv, min_class_count))
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv = max(2, min(cv, len(X)))
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    train_sizes, train_scores, validation_scores = learning_curve(
        clone(model),
        X,
        y,
        cv=splitter,
        scoring=scoring,
        train_sizes=np.linspace(0.3, 1.0, 5),
        n_jobs=None,
        error_score=np.nan
    )

    return {
        "train_sizes": train_sizes,
        "train_mean": np.nanmean(train_scores, axis=1),
        "train_std": np.nanstd(train_scores, axis=1),
        "validation_mean": np.nanmean(validation_scores, axis=1),
        "validation_std": np.nanstd(validation_scores, axis=1),
        "score_name": "Accuracy" if problem_type == "classification" else "R2 Score"
    }


def plot_learning_curve_figure(curve_data):
    if not curve_data:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 3.25), dpi=140)
    fig.patch.set_facecolor(COLORS["fig_bg"])

    train_sizes = curve_data["train_sizes"]
    train_mean = curve_data["train_mean"]
    train_std = curve_data["train_std"]
    validation_mean = curve_data["validation_mean"]
    validation_std = curve_data["validation_std"]

    ax.plot(train_sizes, train_mean, marker="o", color="#2563EB", label="Training score", linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#2563EB", alpha=0.13)
    ax.plot(train_sizes, validation_mean, marker="o", color="#10B981", label="Validation score", linewidth=2)
    ax.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color="#10B981", alpha=0.13)

    ax.set_title("Learning curve", fontsize=10.5, pad=10)
    ax.set_xlabel("Training rows used", fontsize=8.5)
    ax.set_ylabel(curve_data.get("score_name", "Score"), fontsize=8.5)
    _apply_explainability_axis_style(ax, grid_axis="both")
    ax.legend(fontsize=7.5, loc="best", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    fig.tight_layout()
    return fig


def get_learning_curve_interpretation(curve_data):
    if not curve_data:
        return "The learning curve could not be summarized."

    train_last = float(curve_data["train_mean"][-1])
    validation_last = float(curve_data["validation_mean"][-1])
    gap = train_last - validation_last
    previous_validation = float(curve_data["validation_mean"][-2]) if len(curve_data["validation_mean"]) > 1 else validation_last
    improvement = validation_last - previous_validation

    if gap > 0.12:
        fit_text = "The training score is materially higher than the validation score, indicating possible overfitting."
    elif validation_last < 0.45:
        fit_text = "Both scores are relatively low, indicating that feature quality or model choice may need review."
    else:
        fit_text = "Training and validation scores are reasonably aligned, indicating a healthier fit profile."

    if improvement > 0.02:
        data_text = "Validation performance is still improving near the final training size, so additional data may be beneficial."
    else:
        data_text = "Validation performance is flattening, so additional rows alone may have limited impact."

    return f"{fit_text} {data_text}"


def compute_pdp_ice_data(
        trained_model,
        X,
        feature_name,
        problem_type,
        grid_points=12,
        ice_samples=30,
        positive_class_label=None
):
    X = _sanitize_feature_frame(X)
    if X.empty or feature_name not in X.columns:
        return None

    feature_values = X[feature_name].dropna()
    unique_values = np.sort(feature_values.unique())
    if len(unique_values) <= grid_points:
        grid = unique_values
    else:
        grid = np.quantile(feature_values, np.linspace(0.05, 0.95, grid_points))
        grid = np.unique(grid)

    if len(grid) == 0:
        return None

    sample_X = sample_explanation_data(X, max_samples=ice_samples).reset_index(drop=True)
    ice_lines = []
    pdp_values = []

    for value in grid:
        modified = sample_X.copy()
        modified[feature_name] = value
        preds = np.array(_predict_behavior_score(trained_model, modified, problem_type), dtype=float)
        ice_lines.append(preds)
        pdp_values.append(float(np.mean(preds)))

    ice_matrix = np.array(ice_lines).T
    if problem_type == "classification":
        score_label = f"Probability of {_format_positive_class_reference(positive_class_label)}"
    else:
        score_label = "Predicted value"

    return {
        "feature_name": feature_name,
        "grid": np.array(grid, dtype=float),
        "pdp": np.array(pdp_values, dtype=float),
        "ice": ice_matrix,
        "score_label": score_label,
        "problem_type": problem_type,
        "positive_class_label": positive_class_label
    }


def plot_pdp_ice_figure(pdp_ice_data):
    if not pdp_ice_data:
        return None

    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    grid = pdp_ice_data["grid"]
    ice = pdp_ice_data["ice"]
    for row in ice:
        ax.plot(grid, row, color="#94A3B8", linewidth=0.75, alpha=0.25)

    trend_color = COLORS["preprocessing"] if pdp_ice_data["pdp"][-1] >= pdp_ice_data["pdp"][0] else COLORS["danger"]
    ax.plot(grid, pdp_ice_data["pdp"], color=trend_color, linewidth=2.6, marker="o", label="Average behavior")
    ax.set_title(f"What changes when {pdp_ice_data['feature_name']} changes?", fontsize=10.5, pad=10)
    ax.set_xlabel(pdp_ice_data["feature_name"], fontsize=8.5)
    ax.set_ylabel(pdp_ice_data["score_label"], fontsize=8.5)
    _apply_explainability_axis_style(ax, grid_axis="both")
    ax.legend(fontsize=7.5, loc="best", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    fig.tight_layout()
    return fig


def get_pdp_ice_interpretation(pdp_ice_data):
    if not pdp_ice_data:
        return "The feature behavior chart could not be summarized."

    grid = pdp_ice_data["grid"]
    pdp = pdp_ice_data["pdp"]
    if len(grid) < 2:
        return "This feature does not vary enough to show a useful what-changes-when-it-changes pattern."

    change = float(pdp[-1] - pdp[0])
    feature_name = pdp_ice_data["feature_name"]
    problem_type = pdp_ice_data.get("problem_type")
    if problem_type == "classification":
        target_text = _format_positive_class_reference(pdp_ice_data.get("positive_class_label"))
        if change > 0:
            direction = f"increases the model's probability for {target_text}"
        elif change < 0:
            direction = f"decreases the model's probability for {target_text}"
        else:
            direction = f"does not show a clear directional change for {target_text}"
    else:
        if change > 0:
            direction = "is associated with a higher predicted value"
        elif change < 0:
            direction = "is associated with a lower predicted value"
        else:
            direction = "does not show a clear directional change in the predicted value"

    return (
        f"The chart varies only {feature_name} while holding the remaining feature values fixed. "
        f"Across the analysis sample, moving from lower to higher values {direction}. "
        f"Variation among the gray ICE lines indicates that the feature effect differs across individual rows."
    )


def build_counterfactual_table(trained_model, X, sample_index, problem_type, feature_names, max_features=6):
    X = _sanitize_feature_frame(X).reset_index(drop=True)
    if X.empty:
        return pd.DataFrame(), "No usable rows were available for what-if analysis."

    sample_index = max(0, min(int(sample_index), len(X) - 1))
    row = X.iloc[[sample_index]].copy()
    before_pred = trained_model.predict(row)[0]
    before_score = float(_predict_behavior_score(trained_model, row, problem_type)[0])

    rows = []
    for feature in feature_names[:max_features]:
        if feature not in X.columns:
            continue

        values = X[feature].dropna()
        if values.nunique() <= 1:
            continue

        candidates = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 11)))
        best = None

        for candidate in candidates:
            modified = row.copy()
            modified[feature] = candidate
            after_pred = trained_model.predict(modified)[0]
            after_score = float(_predict_behavior_score(trained_model, modified, problem_type)[0])
            score_change = after_score - before_score
            pred_changed = after_pred != before_pred

            if problem_type == "classification" and pred_changed:
                distance = abs(float(candidate) - float(row.iloc[0][feature]))
                option = (distance, candidate, after_pred, after_score, score_change)
                if best is None or option[0] < best[0]:
                    best = option
            elif problem_type == "regression":
                option = (abs(score_change), candidate, after_pred, after_score, score_change)
                if best is None or option[0] > best[0]:
                    best = option

        if best is not None:
            _, candidate, after_pred, after_score, score_change = best
            rows.append({
                "Feature": feature,
                "Current Value": float(row.iloc[0][feature]),
                "What-if Value": float(candidate),
                "Current Result": float(before_score),
                "What-if Result": float(after_score),
                "Change": float(score_change),
                "Predicted Class After": after_pred if problem_type == "classification" else ""
            })

    df = pd.DataFrame(rows)
    if df.empty:
        note = (
            "No single-feature change in the tested value range changed the predicted class. "
            "That usually means this row is not easy to flip with one simple change."
            if problem_type == "classification"
            else
            "No useful what-if movement was found for the selected features."
        )
    else:
        note = (
            "This is a simple what-if search: it changes one feature at a time and checks how the model response moves. "
            "It should be read as model behavior, not as a guaranteed real-world recommendation."
        )

    return df, note


def build_surrogate_tree(trained_model, X, problem_type, max_depth=3):
    X = _sanitize_feature_frame(X).reset_index(drop=True)
    if X.empty:
        return None

    model_predictions = trained_model.predict(X)
    if problem_type == "classification":
        surrogate = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, random_state=42)
    else:
        surrogate = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=5, random_state=42)

    surrogate.fit(X, model_predictions)
    surrogate_predictions = surrogate.predict(X)
    if problem_type == "classification":
        fidelity = accuracy_score(model_predictions, surrogate_predictions)
        fidelity_label = "Agreement with original model"
    else:
        fidelity = r2_score(model_predictions, surrogate_predictions)
        fidelity_label = "Approximation strength"

    rules = export_text(
        surrogate,
        feature_names=[_truncate_feature_name(col, max_len=32) for col in X.columns],
        max_depth=max_depth
    )

    return {
        "tree": surrogate,
        "rules": rules,
        "fidelity": float(fidelity),
        "fidelity_label": fidelity_label,
        "feature_names": list(X.columns)
    }


def plot_surrogate_tree_figure(surrogate_outputs, problem_type):
    if not surrogate_outputs:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=130)
    fig.patch.set_facecolor("#F6F8FC")
    ax.set_facecolor("#FFFFFF")
    plot_tree(
        surrogate_outputs["tree"],
        feature_names=[_truncate_feature_name(col, max_len=18) for col in surrogate_outputs["feature_names"]],
        filled=True,
        rounded=True,
        fontsize=6.8,
        impurity=False,
        ax=ax
    )
    ax.set_title("Simple rule model that imitates the selected model", fontsize=10.5, pad=10)
    fig.tight_layout()
    return fig


def get_surrogate_interpretation(surrogate_outputs):
    if not surrogate_outputs:
        return "The simple rule model could not be generated."

    fidelity = surrogate_outputs["fidelity"]
    if fidelity >= 0.85:
        quality = "does a strong job of imitating"
    elif fidelity >= 0.65:
        quality = "captures a useful simplified version of"
    else:
        quality = "only loosely imitates"

    return (
        f"This small decision tree {quality} the selected model "
        f"({surrogate_outputs['fidelity_label']}: {fidelity:.3f}). "
        f"It is not replacing the original model; it gives a simpler rule-based sketch of the model's behavior."
    )
