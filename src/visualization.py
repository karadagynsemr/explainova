import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix_figure(confusion_matrix_array, class_labels):
    fig, ax = plt.subplots(figsize=(3.8, 3.1), dpi=120)

    im = ax.imshow(confusion_matrix_array, aspect="auto")

    ax.set_title("Confusion Matrix", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)

    for i in range(confusion_matrix_array.shape[0]):
        for j in range(confusion_matrix_array.shape[1]):
            ax.text(j, i, str(confusion_matrix_array[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_model_comparison_figure(results_df, problem_type):
    df = results_df.copy()

    if df.empty:
        return None

    metric_columns = [
        col for col in df.columns
        if col not in ["Model", "Problem Type"]
    ]

    if len(metric_columns) == 0:
        return None

    if problem_type == "classification":
        primary_metric = "Accuracy" if "Accuracy" in df.columns else metric_columns[0]
    else:
        primary_metric = "R2 Score" if "R2 Score" in df.columns else metric_columns[0]

    fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=120)
    ax.bar(df["Model"], df[primary_metric])
    ax.set_title(f"{primary_metric} Comparison", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel(primary_metric, fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()

    return fig


def plot_roc_curve_figure(detailed_results):
    has_any_roc = False
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=120)

    for model_name, details in detailed_results.items():
        roc_data = details.get("roc_data")

        if roc_data is None:
            continue

        has_any_roc = True
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            label=f"{model_name} (AUC={roc_data['auc']:.3f})"
        )

    if not has_any_roc:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve", fontsize=10)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()

    return fig


def build_outlier_dataframe(report):
    return pd.DataFrame({
        "Column": list(report["outlier_report"].keys()),
        "Outliers (1.5 IQR)": list(report["outlier_report"].values()),
        "Extreme Outliers (3 IQR)": list(report["extreme_outlier_report"].values())
    })


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