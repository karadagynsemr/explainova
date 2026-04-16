import pandas as pd
import matplotlib.pyplot as plt


def _apply_clean_axis_style(ax):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#334155", labelsize=7)
    ax.tick_params(axis="y", colors="#334155", labelsize=8)
    ax.title.set_color("#0F172A")
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")
    ax.grid(axis="y", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_confusion_matrix_figure(confusion_matrix_array, class_labels):
    fig, ax = plt.subplots(figsize=(4.1, 3.3), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    im = ax.imshow(confusion_matrix_array, aspect="auto", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=10, pad=8)
    ax.set_xlabel("Predicted", fontsize=8.5)
    ax.set_ylabel("Actual", fontsize=8.5)

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_yticklabels(class_labels, fontsize=7.5)

    max_val = confusion_matrix_array.max() if confusion_matrix_array.size > 0 else 0

    for i in range(confusion_matrix_array.shape[0]):
        for j in range(confusion_matrix_array.shape[1]):
            val = confusion_matrix_array[i, j]
            text_color = "white" if max_val > 0 and val > max_val * 0.55 else "#0F172A"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7.5, color=text_color)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor("#CBD5E1")
    cbar.ax.tick_params(labelsize=7.5, colors="#334155")

    fig.tight_layout()
    return fig


def plot_single_metric_comparison_figure(results_df, metric_name):
    if results_df.empty or metric_name not in results_df.columns:
        return None

    df = results_df.copy()

    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    bars = ax.bar(
        df["Model"],
        df[metric_name],
        color="#6366F1",
        edgecolor="#4F46E5",
        linewidth=0.8,
        alpha=0.9
    )

    ax.set_title(metric_name, fontsize=10, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(metric_name, fontsize=8)

    _apply_clean_axis_style(ax)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=7)

    for bar in bars:
        height = bar.get_height()
        label = f"{height:.3f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#334155"
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


def plot_roc_curve_figure(detailed_results):
    has_any_roc = False
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    palette = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444"]

    for idx, (model_name, details) in enumerate(detailed_results.items()):
        roc_data = details.get("roc_data")

        if roc_data is None:
            continue

        has_any_roc = True
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            label=f"{model_name} (AUC={roc_data['auc']:.3f})",
            linewidth=2,
            color=palette[idx % len(palette)]
        )

    if not has_any_roc:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.4)
    ax.set_title("ROC Curve", fontsize=10, pad=8)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)

    _apply_clean_axis_style(ax)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=7, loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0")

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


def plot_correlation_heatmap_figure(X, y, target_name="Target", top_n=10):
    corr_table = build_target_correlation_table(X, y, target_name=target_name, top_n=top_n)

    if corr_table.empty:
        return None

    selected_features = corr_table["Feature"].tolist()

    numeric_X = X.select_dtypes(include=["number"]).copy()
    combined = numeric_X[selected_features].copy()
    combined[target_name] = pd.Series(y, name=target_name, index=X.index)

    corr_matrix = combined.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=140)
    fig.patch.set_facecolor("#F6F8FC")

    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Correlation Heatmap", fontsize=10, pad=10)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=28, ha="right", fontsize=7)
    ax.set_yticklabels(corr_matrix.index, fontsize=7)

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "#0F172A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.2, color=text_color)

    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor("#CBD5E1")
    cbar.ax.tick_params(labelsize=8, colors="#334155")

    fig.tight_layout()
    return fig


def get_confusion_matrix_interpretation(confusion_matrix_array, class_labels):
    total_samples = confusion_matrix_array.sum()
    diagonal_sum = confusion_matrix_array.diagonal().sum()

    if total_samples == 0:
        return "No samples were available to interpret the confusion matrix."

    accuracy_like = diagonal_sum / total_samples

    if accuracy_like >= 0.85:
        quality = "Most predictions fall on the diagonal, so the model is separating classes quite well."
    elif accuracy_like >= 0.65:
        quality = "A fair number of predictions are correct, but some classes are still being confused."
    else:
        quality = "The model is mixing classes frequently, so predictions should be interpreted carefully."

    class_text = f"Diagonal cells show correct predictions for classes such as: {', '.join(class_labels)}."
    return f"{quality} {class_text}"


def get_roc_interpretation(detailed_results):
    available = []

    for model_name, details in detailed_results.items():
        roc_data = details.get("roc_data")
        if roc_data is not None:
            available.append((model_name, roc_data["auc"]))

    if not available:
        return "ROC curve could not be shown because this is not a compatible binary classification setup or the models could not produce suitable probability scores."

    best_model_name, best_auc = max(available, key=lambda x: x[1])

    if best_auc >= 0.90:
        strength = "excellent"
    elif best_auc >= 0.80:
        strength = "strong"
    elif best_auc >= 0.70:
        strength = "acceptable"
    else:
        strength = "limited"

    return (
        f"ROC curve shows how well models separate the two classes across thresholds. "
        f"Closer to the top-left is better. "
        f"The best ROC result here is {best_model_name} with AUC {best_auc:.3f}, which indicates {strength} separation."
    )


def get_metric_commentary(results_df, metric_name, problem_type):
    if results_df.empty or metric_name not in results_df.columns:
        return "This metric could not be summarized."

    ascending = metric_name in ["MAE", "RMSE"]
    best_row = results_df.sort_values(by=metric_name, ascending=ascending).iloc[0]

    model_name = best_row["Model"]
    value = best_row[metric_name]

    if problem_type == "classification":
        if metric_name in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
            return f"{metric_name} comparison suggests that {model_name} performs best on this criterion with a score of {value:.4f}."
    else:
        if metric_name == "R2 Score":
            return f"R2 comparison suggests that {model_name} explains the target best with a score of {value:.4f}."
        if metric_name in ["MAE", "RMSE"]:
            return f"For {metric_name}, lower values are better. {model_name} performs best here with {value:.4f}."

    return f"{model_name} performs best on {metric_name} with {value:.4f}."