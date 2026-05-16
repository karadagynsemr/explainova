import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


COLORS = {
    "model": "#3B82F6",
    "model_dark": "#1D4ED8",
    "model_soft": "#93C5FD",
    "shap": "#8B5CF6",
    "preprocessing": "#10B981",
    "warning": "#F59E0B",
    "matrix": "#14B8A6",
    "danger": "#EF4444",
}

PRIMARY = COLORS["model"]
PRIMARY_DARK = COLORS["model_dark"]
SECONDARY = COLORS["model_soft"]
SUCCESS = COLORS["preprocessing"]
WARNING = COLORS["warning"]
DANGER = COLORS["danger"]
INK = "#0F172A"
SLATE = "#334155"
BORDER = "#CBD5E1"
GRID = "#E2E8F0"
FIG_BG = "#F6F8FC"
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "explainova_confusion",
    ["#ECFEFF", COLORS["matrix"], "#0F766E"]
)


def _apply_clean_axis_style(ax, grid_axis="y"):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(axis="x", colors=SLATE, labelsize=7.5)
    ax.tick_params(axis="y", colors=SLATE, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linestyle="--", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def _truncate_label(text, max_len=28):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _metric_info(problem_type):
    if problem_type == "classification":
        return "Accuracy", "Correct prediction rate"
    return "R2 Score", "Explained variation"


def plot_confusion_matrix_figure(confusion_matrix_array, class_labels):
    fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    im = ax.imshow(confusion_matrix_array, aspect="auto", cmap=CONFUSION_CMAP)
    ax.set_title("Prediction distribution", fontsize=10.5, pad=8)
    ax.set_xlabel("Predicted class", fontsize=8.5)
    ax.set_ylabel("Actual class", fontsize=8.5)

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_yticklabels(class_labels, fontsize=7.5)

    for i in range(confusion_matrix_array.shape[0]):
        for j in range(confusion_matrix_array.shape[1]):
            val = confusion_matrix_array[i, j]
            rgba = im.cmap(im.norm(val))
            luminance = (0.299 * rgba[0]) + (0.587 * rgba[1]) + (0.114 * rgba[2])
            text_color = INK if luminance > 0.58 else "#F8FAFC"
            annotation = ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight=800,
                color=text_color
            )
            annotation.set_gid("confusion-cell-annotation")

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(BORDER)
    cbar.ax.tick_params(labelsize=7.5, colors=SLATE)

    fig.tight_layout()
    return fig


def plot_single_metric_comparison_figure(results_df, metric_name):
    if results_df.empty or metric_name not in results_df.columns:
        return None

    df = results_df.copy().sort_values(metric_name, ascending=metric_name in ["MAE", "RMSE"])
    colors = [PRIMARY if idx == 0 else SECONDARY for idx in range(len(df))]

    fig, ax = plt.subplots(figsize=(4.0, 2.7), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    bars = ax.bar(
        [_truncate_label(name, max_len=22) for name in df["Model"]],
        df[metric_name],
        color=colors,
        edgecolor=PRIMARY_DARK,
        linewidth=0.8,
        alpha=0.92
    )

    ax.set_title(f"{metric_name} comparison", fontsize=10, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(metric_name, fontsize=8.3)

    _apply_clean_axis_style(ax)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=SLATE
        )

    fig.tight_layout()
    return fig


def plot_metric_grid(results_df, metrics):
    figures = {}

    for metric in metrics:
        fig = plot_single_metric_comparison_figure(results_df, metric)
        if fig is not None:
            figures[metric] = fig

    return figures


def plot_model_leaderboard_figure(results_df, problem_type, top_n=6):
    if results_df is None or results_df.empty:
        return None

    primary_metric, metric_label = _metric_info(problem_type)
    if primary_metric not in results_df.columns:
        return None

    plot_df = (
        results_df[["Model", primary_metric]]
        .sort_values(primary_metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    plot_df["Display Model"] = plot_df["Model"].apply(lambda name: _truncate_label(name, max_len=26))

    fig, ax = plt.subplots(figsize=(4.8, 2.9), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    colors = [PRIMARY] + [SECONDARY] * max(len(plot_df) - 1, 0)
    bars = ax.barh(
        plot_df["Display Model"][::-1],
        plot_df[primary_metric][::-1],
        color=colors[::-1],
        edgecolor=PRIMARY_DARK,
        linewidth=0.8,
        alpha=0.92
    )

    ax.set_title("Overall model ranking", fontsize=10.5, pad=10)
    ax.set_xlabel(metric_label, fontsize=8.5)
    ax.set_ylabel("")
    _apply_clean_axis_style(ax, grid_axis="x")

    max_val = max(float(plot_df[primary_metric].max()), 0.001)
    ax.set_xlim(left=min(0.0, float(plot_df[primary_metric].min()) * 1.05), right=max_val * 1.18)

    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.3,
            color=SLATE
        )

    fig.tight_layout()
    return fig


def plot_roc_curve_figure(detailed_results):
    has_any_roc = False
    fig, ax = plt.subplots(figsize=(3.9, 3.0), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    palette = [PRIMARY, SECONDARY, SUCCESS, WARNING, "#EF4444"]

    for idx, (model_name, details) in enumerate(detailed_results.items()):
        roc_data = details.get("roc_data")

        if roc_data is None:
            continue

        has_any_roc = True
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            label=f"{_truncate_label(model_name, 18)} (AUC={roc_data['auc']:.3f})",
            linewidth=1.9,
            color=palette[idx % len(palette)]
        )

    if not has_any_roc:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.2)
    ax.set_title("Class separation performance", fontsize=10.5, pad=8)
    ax.set_xlabel("False positive rate", fontsize=8.5)
    ax.set_ylabel("True positive rate", fontsize=8.5)

    _apply_clean_axis_style(ax, grid_axis="both")
    ax.legend(fontsize=6.8, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)

    fig.tight_layout()
    return fig


def build_outlier_dataframe(report):
    return pd.DataFrame({
        "Column": list(report["outlier_report"].keys()),
        "Outliers (1.5 IQR)": list(report["outlier_report"].values()),
        "Extreme Outliers (3 IQR)": list(report["extreme_outlier_report"].values())
    })


def build_target_correlation_table(X, y, target_name="target", top_n=10):
    if X.empty:
        return pd.DataFrame()

    numeric_X = X.select_dtypes(include=["number"]).copy()
    if numeric_X.empty:
        return pd.DataFrame()

    y_series = pd.Series(y, name=target_name, index=X.index)
    combined = numeric_X.copy()
    combined[target_name] = y_series

    corr_series = combined.corr(numeric_only=True)[target_name].drop(labels=[target_name], errors="ignore")
    corr_series = corr_series.dropna()

    if corr_series.empty:
        return pd.DataFrame()

    corr_df = corr_series.reset_index()
    corr_df.columns = ["Feature", "Correlation with Target"]
    corr_df["Absolute Correlation"] = corr_df["Correlation with Target"].abs()
    corr_df = corr_df.sort_values(by="Absolute Correlation", ascending=False).head(top_n).reset_index(drop=True)

    return corr_df


def plot_correlation_heatmap_figure(X, y, target_name="Target", top_n=8):
    corr_table = build_target_correlation_table(X, y, target_name=target_name, top_n=top_n)

    if corr_table.empty:
        return None

    selected_features = corr_table["Feature"].tolist()

    numeric_X = X.select_dtypes(include=["number"]).copy()
    combined = numeric_X[selected_features].copy()
    combined[target_name] = pd.Series(y, name=target_name, index=X.index)

    corr_matrix = combined.corr(numeric_only=True)

    label_count = len(corr_matrix.columns)
    fig_size = max(4.6, min(6.2, label_count * 0.48))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Relationship heatmap", fontsize=10.5, pad=10)

    display_labels = [_truncate_label(col, max_len=16) for col in corr_matrix.columns]
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(display_labels, rotation=42, ha="right", fontsize=6.5)
    ax.set_yticklabels(display_labels, fontsize=6.8)

    if label_count <= 8:
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                val = corr_matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.55 else INK
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.8, color=text_color)

    for spine in ax.spines.values():
        spine.set_color(BORDER)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(BORDER)
    cbar.ax.tick_params(labelsize=8, colors=SLATE)

    fig.subplots_adjust(left=0.25, right=0.94, top=0.90, bottom=0.28)
    return fig


def plot_correlation_profile_figure(corr_table, top_n=8):
    if corr_table is None or corr_table.empty:
        return None

    plot_df = corr_table.head(top_n).copy()
    plot_df["Display Feature"] = plot_df["Feature"].apply(lambda item: _truncate_label(item, max_len=24))
    plot_df = plot_df.sort_values("Correlation with Target", ascending=True)

    fig, ax = plt.subplots(figsize=(4.9, 3.1), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    colors = [SUCCESS if value >= 0 else "#EF4444" for value in plot_df["Correlation with Target"]]
    bars = ax.barh(
        plot_df["Display Feature"],
        plot_df["Correlation with Target"],
        color=colors,
        edgecolor="#CBD5E1",
        linewidth=0.8,
        alpha=0.95
    )

    ax.axvline(0, color="#94A3B8", linewidth=1.0)
    ax.set_title("Strongest relationships with target", fontsize=10.5, pad=10)
    ax.set_xlabel("Correlation", fontsize=8.5)
    ax.set_ylabel("")
    _apply_clean_axis_style(ax, grid_axis="x")

    for bar, value in zip(bars, plot_df["Correlation with Target"]):
        offset = 0.03 if value >= 0 else -0.03
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=7.2,
            color=SLATE
        )

    fig.tight_layout()
    return fig


def plot_local_contribution_figure(local_contribution_df, top_n=6):
    if local_contribution_df is None or local_contribution_df.empty:
        return None

    plot_df = local_contribution_df.head(top_n).copy()
    plot_df["Display Feature"] = plot_df["Feature"].apply(lambda item: _truncate_label(item, max_len=24))
    plot_df = plot_df.sort_values("SHAP Contribution", ascending=True)

    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    colors = [SUCCESS if value >= 0 else DANGER for value in plot_df["SHAP Contribution"]]
    bars = ax.barh(
        plot_df["Display Feature"],
        plot_df["SHAP Contribution"],
        color=colors,
        edgecolor="#CBD5E1",
        linewidth=0.8,
        alpha=0.95
    )

    ax.axvline(0, color="#94A3B8", linewidth=1.0)
    ax.set_title("Main drivers of this result", fontsize=10.5, pad=10)
    ax.set_xlabel("Contribution direction and strength", fontsize=8.5)
    ax.set_ylabel("")
    _apply_clean_axis_style(ax, grid_axis="x")

    max_width = max(np.abs(plot_df["SHAP Contribution"]).max(), 0.001)
    for bar, value in zip(bars, plot_df["SHAP Contribution"]):
        offset = max_width * 0.05
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=ha,
            fontsize=7.2,
            color=SLATE
        )

    fig.tight_layout()
    return fig


def plot_feature_behavior_summary_figure(behavior_df, top_n=8):
    if behavior_df is None or behavior_df.empty:
        return None

    plot_df = behavior_df.head(top_n).copy()
    plot_df["Display Feature"] = plot_df["Feature"].apply(lambda item: _truncate_label(item, max_len=24))
    plot_df = plot_df.sort_values("Average Strength", ascending=True)

    color_map = {
        "Higher values usually raise prediction": SUCCESS,
        "Higher values usually lower prediction": "#EF4444",
        "Higher values move toward class 1": SUCCESS,
        "Higher values move away from class 1": "#EF4444",
        "Mixed / non-linear effect": WARNING,
        "No clear pattern": "#94A3B8",
    }
    colors = [color_map.get(item, SECONDARY) for item in plot_df["Typical Pattern"]]

    fig, ax = plt.subplots(figsize=(5.1, 3.2), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    bars = ax.barh(
        plot_df["Display Feature"],
        plot_df["Average Strength"],
        color=colors,
        edgecolor="#CBD5E1",
        linewidth=0.8,
        alpha=0.95
    )

    ax.set_title("Typical behavior of key features", fontsize=10.5, pad=10)
    ax.set_xlabel("Average impact strength", fontsize=8.5)
    ax.set_ylabel("")
    _apply_clean_axis_style(ax, grid_axis="x")

    max_val = max(float(plot_df["Average Strength"].max()), 0.001)
    for bar, value in zip(bars, plot_df["Average Strength"]):
        ax.text(
            value + (max_val * 0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=SLATE
        )

    fig.tight_layout()
    return fig


def get_confusion_matrix_interpretation(confusion_matrix_array, class_labels):
    total_samples = confusion_matrix_array.sum()
    diagonal_sum = confusion_matrix_array.diagonal().sum()

    if total_samples == 0:
        return "There were not enough samples to interpret this confusion matrix."

    accuracy_like = diagonal_sum / total_samples

    if accuracy_like >= 0.85:
        quality = "The model places most examples into the correct class."
    elif accuracy_like >= 0.65:
        quality = "The model is generally useful, but it still mixes some classes from time to time."
    else:
        quality = "The model is mixing classes quite often, so the results should be interpreted carefully."

    class_text = f"Off-diagonal cells show where classes such as {', '.join(class_labels)} are being confused with one another."
    return f"{quality} {class_text}"


def get_roc_interpretation(detailed_results):
    available = []

    for model_name, details in detailed_results.items():
        roc_data = details.get("roc_data")
        if roc_data is not None:
            available.append((model_name, roc_data["auc"]))

    if not available:
        return "The ROC chart could not be generated. This usually happens when the setup is not binary classification or the model could not produce suitable probability scores."

    best_model_name, best_auc = max(available, key=lambda x: x[1])

    if best_auc >= 0.90:
        strength = "very strong"
    elif best_auc >= 0.80:
        strength = "strong"
    elif best_auc >= 0.70:
        strength = "acceptable"
    else:
        strength = "limited"

    return (
        f"This chart shows how cleanly the model separates the two classes. "
        f"The best curve belongs to {best_model_name} with an AUC of {best_auc:.3f}. "
        f"That suggests {strength} separation power."
    )


def get_metric_commentary(results_df, metric_name, problem_type):
    if results_df.empty or metric_name not in results_df.columns:
        return "A summary comment could not be generated for this metric."

    ascending = metric_name in ["MAE", "RMSE"]
    best_row = results_df.sort_values(by=metric_name, ascending=ascending).iloc[0]
    model_name = best_row["Model"]
    value = float(best_row[metric_name])

    if problem_type == "classification":
        explanations = {
            "Accuracy": "stands out on overall correctness",
            "Precision": "looks stronger at reducing false alarms",
            "Recall": "looks stronger at capturing true positives",
            "F1 Score": "appears to strike the best balance",
            "ROC AUC": "offers the clearest class separation",
        }
        detail = explanations.get(metric_name, "delivers the best result on this metric")
        return f"In the {metric_name} chart, {model_name} leads with a score of {value:.3f} and {detail}."

    if metric_name == "R2 Score":
        return f"Based on R2, {model_name} is the strongest option for explaining the target, with a score of {value:.3f}."
    if metric_name == "MAE":
        return f"For MAE, lower is better. Here, {model_name} keeps the average error lowest at {value:.3f}."
    if metric_name == "RMSE":
        return f"RMSE penalizes larger mistakes more heavily, and {model_name} leads here with {value:.3f}."

    return f"{model_name} delivers the best {metric_name} result with {value:.3f}."


def get_model_recommendation_text(results_df, problem_type):
    if results_df is None or results_df.empty:
        return "A recommendation could not be generated yet because there is no model comparison."

    primary_metric, metric_label = _metric_info(problem_type)
    if primary_metric not in results_df.columns:
        return "The main metric required for comparison was not available."

    ranking = results_df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
    winner = ranking.iloc[0]
    winner_name = winner["Model"]
    winner_score = float(winner[primary_metric])

    if len(ranking) > 1:
        runner_up_score = float(ranking.iloc[1][primary_metric])
        gap = winner_score - runner_up_score
    else:
        gap = 0.0

    if gap >= 0.05:
        gap_text = "It separates clearly from the rest of the field."
    elif gap >= 0.02:
        gap_text = "It has a small but meaningful edge over the alternatives."
    else:
        gap_text = "It ranks first, but the margin versus the other models is still quite close."

    audience_text = (
        "That makes it a sensible first choice to carry forward."
        if gap >= 0.02 else
        "In this case, explainability and stability may matter almost as much as the score itself."
    )

    return (
        f"Overall, {winner_name} looks like the strongest candidate. "
        f"It reaches {winner_score:.3f} on {metric_label}. "
        f"{gap_text} {audience_text}"
    )


def get_correlation_profile_interpretation(corr_table):
    if corr_table is None or corr_table.empty:
        return "A correlation summary could not be generated."

    strongest = corr_table.iloc[0]
    feature = strongest["Feature"]
    corr_value = float(strongest["Correlation with Target"])
    direction = "moves in the same direction" if corr_value >= 0 else "moves in the opposite direction"

    strength = abs(corr_value)
    if strength >= 0.70:
        strength_text = "quite strong"
    elif strength >= 0.40:
        strength_text = "moderate"
    else:
        strength_text = "light to moderate"

    return (
        f"The most noticeable relationship appears in {feature}. "
        f"This feature {direction} with the target, and the relationship looks {strength_text} "
        f"(correlation: {corr_value:.2f}). "
        f"This section does not prove causation; it simply highlights patterns that move together."
    )


def build_error_analysis_table(details, problem_type, top_n=10):
    y_test = pd.Series(details.get("y_test", []), name="Actual").reset_index(drop=True)
    y_pred = pd.Series(details.get("y_pred", []), name="Predicted").reset_index(drop=True)

    if y_test.empty or y_pred.empty:
        return pd.DataFrame()

    if problem_type == "classification":
        error_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        error_df = error_df[error_df["Actual"] != error_df["Predicted"]]

        if error_df.empty:
            return pd.DataFrame({
                "Actual": ["No errors"],
                "Predicted": ["No errors"],
                "Count": [0]
            })

        return (
            error_df
            .groupby(["Actual", "Predicted"])
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    residuals = y_test.astype(float) - y_pred.astype(float)
    return (
        pd.DataFrame({
            "Actual": y_test.astype(float),
            "Predicted": y_pred.astype(float),
            "Residual": residuals.astype(float),
            "Absolute Error": residuals.abs().astype(float)
        })
        .sort_values("Absolute Error", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_error_analysis_figure(error_df, problem_type):
    if error_df is None or error_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
    fig.patch.set_facecolor(FIG_BG)

    if problem_type == "classification":
        if "No errors" in error_df.astype(str).values:
            ax.text(0.5, 0.5, "No classification errors found", ha="center", va="center", fontsize=10, color=SLATE)
            ax.axis("off")
            return fig

        plot_df = error_df.copy()
        plot_df["Pair"] = plot_df.apply(lambda row: f"{row['Actual']} -> {row['Predicted']}", axis=1)
        plot_df = plot_df.sort_values("Count", ascending=True)
        ax.barh(plot_df["Pair"], plot_df["Count"], color=DANGER, edgecolor="#EA580C", linewidth=0.8, alpha=0.92)
        ax.set_title("Most common wrong predictions", fontsize=10.5, pad=10)
        ax.set_xlabel("Error count", fontsize=8.5)
        ax.set_ylabel("")
        _apply_clean_axis_style(ax, grid_axis="x")
    else:
        plot_df = error_df.sort_values("Absolute Error", ascending=True).copy()
        labels = [f"Row {idx}" for idx in plot_df.index]
        colors = ["#EF4444" if value >= 0 else PRIMARY for value in plot_df["Residual"]]
        ax.barh(labels, plot_df["Residual"], color=colors, edgecolor="#CBD5E1", linewidth=0.8, alpha=0.92)
        ax.axvline(0, color="#94A3B8", linewidth=1.0)
        ax.set_title("Largest prediction errors", fontsize=10.5, pad=10)
        ax.set_xlabel("Actual minus predicted", fontsize=8.5)
        ax.set_ylabel("")
        _apply_clean_axis_style(ax, grid_axis="x")

    fig.tight_layout()
    return fig


def get_error_analysis_interpretation(error_df, problem_type):
    if error_df is None or error_df.empty:
        return "Error analysis could not be generated because prediction details were not available."

    if problem_type == "classification":
        if "No errors" in error_df.astype(str).values:
            return "The selected test split did not contain classification errors for this model."

        top = error_df.iloc[0]
        return (
            f"This section focuses only on mistakes. The most common mix-up is actual {top['Actual']} "
            f"being predicted as {top['Predicted']} ({int(top['Count'])} time(s)). "
            f"This helps users inspect where the model needs the most caution."
        )

    biggest = error_df.iloc[0]
    return (
        f"This section highlights the largest numeric mistakes. The biggest shown error is about "
        f"{float(biggest['Absolute Error']):.3f}. Positive residuals mean the model predicted too low; "
        f"negative residuals mean it predicted too high."
    )
