"""
phase4_evaluation.py
--------------------
Phase 4 - NLP Evaluation Metrics

This script:
  1. Loads the trained models and train/test split from  models/
  2. Generates predictions on the test set for both targets
  3. Outputs detailed classification reports (Precision, Recall, F1, Accuracy)
     for both Ticket Type and Ticket Priority
  4. Creates and saves confusion matrices using Matplotlib/Seaborn
     for visual inspection of category-level predictions

Outputs:
  - Console: Full classification reports for both targets
  - outputs/confusion_matrix_ticket_type.png
  - outputs/confusion_matrix_ticket_priority.png
"""

# -- imports -----------------------------------------------------------
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# =====================================================================
#           CONFUSION MATRIX VISUALIZATION HELPER
# =====================================================================
def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """
    Create a publication-quality confusion matrix heatmap.

    Parameters
    ----------
    y_true    : array-like  - true labels
    y_pred    : array-like  - predicted labels
    labels    : list[str]   - ordered class names
    title     : str         - plot title
    save_path : str         - file path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    # overlay percentages in smaller font
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.72,
                f"({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=8, color="gray",
            )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"     Saved -> {os.path.abspath(save_path)}")


# =====================================================================
#                      MAIN  EXECUTION
# =====================================================================
def main():
    # -- paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # -- 1. load models & split ----------------------------------------
    print("=" * 65)
    print("  PHASE 4 -- NLP Evaluation Metrics")
    print("=" * 65)

    with open(os.path.join(MODELS_DIR, "svc_ticket_type.pkl"), "rb") as f:
        svc_type = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "lr_ticket_priority.pkl"), "rb") as f:
        lr_priority = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "multi_rf_model.pkl"), "rb") as f:
        multi_rf = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "train_test_split.pkl"), "rb") as f:
        split = pickle.load(f)

    X_test = split["X_test"]
    y_type_test = split["y_type_test"]
    y_pri_test = split["y_pri_test"]

    print(f"\n[OK] Loaded models and test set ({X_test.shape[0]} samples)")

    # -- 2. generate predictions ----------------------------------------
    pred_type_svc = svc_type.predict(X_test)
    pred_pri_lr = lr_priority.predict(X_test)

    y_multi_test = np.column_stack([y_type_test, y_pri_test])
    pred_multi_rf = multi_rf.predict(X_test)

    # -- 3. classification reports -------------------------------------

    # ---- 3a. LinearSVC -> Ticket Type --------------------------------
    type_labels = sorted(set(y_type_test))
    pri_labels = sorted(set(y_pri_test))

    print("\n" + "=" * 65)
    print("  REPORT 1: Ticket Type  (LinearSVC)")
    print("=" * 65)
    print(classification_report(
        y_type_test, pred_type_svc,
        target_names=type_labels,
        digits=4,
        zero_division=0,
    ))

    # ---- 3b. LogisticRegression -> Ticket Priority -------------------
    print("=" * 65)
    print("  REPORT 2: Ticket Priority  (Logistic Regression)")
    print("=" * 65)
    print(classification_report(
        y_pri_test, pred_pri_lr,
        target_names=pri_labels,
        digits=4,
        zero_division=0,
    ))

    # ---- 3c. MultiOutput RandomForest -> Both targets ----------------
    print("=" * 65)
    print("  REPORT 3: Ticket Type  (RandomForest MultiOutput)")
    print("=" * 65)
    print(classification_report(
        y_multi_test[:, 0], pred_multi_rf[:, 0],
        target_names=type_labels,
        digits=4,
        zero_division=0,
    ))

    print("=" * 65)
    print("  REPORT 4: Ticket Priority  (RandomForest MultiOutput)")
    print("=" * 65)
    print(classification_report(
        y_multi_test[:, 1], pred_multi_rf[:, 1],
        target_names=pri_labels,
        digits=4,
        zero_division=0,
    ))

    # -- 4. overall accuracy summary -----------------------------------
    print("=" * 65)
    print("  OVERALL ACCURACY SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<40s} {'Accuracy':>10s}")
    print("-" * 55)
    print(f"  {'LinearSVC (Ticket Type)':<40s} {accuracy_score(y_type_test, pred_type_svc):>10.4f}")
    print(f"  {'LogReg (Ticket Priority)':<40s} {accuracy_score(y_pri_test, pred_pri_lr):>10.4f}")
    print(f"  {'RF MultiOut (Ticket Type)':<40s} {accuracy_score(y_multi_test[:, 0], pred_multi_rf[:, 0]):>10.4f}")
    print(f"  {'RF MultiOut (Ticket Priority)':<40s} {accuracy_score(y_multi_test[:, 1], pred_multi_rf[:, 1]):>10.4f}")
    print("-" * 55)

    # -- 5. confusion matrices -----------------------------------------
    print("\n-- Generating Confusion Matrices -------------------------------")

    # Ticket Type — LinearSVC
    plot_confusion_matrix(
        y_type_test, pred_type_svc,
        labels=type_labels,
        title="Confusion Matrix: Ticket Type (LinearSVC)",
        save_path=os.path.join(OUTPUTS_DIR, "confusion_matrix_ticket_type.png"),
    )

    # Ticket Priority — Logistic Regression
    plot_confusion_matrix(
        y_pri_test, pred_pri_lr,
        labels=pri_labels,
        title="Confusion Matrix: Ticket Priority (Logistic Regression)",
        save_path=os.path.join(OUTPUTS_DIR, "confusion_matrix_ticket_priority.png"),
    )

    # Ticket Type — RandomForest MultiOutput
    plot_confusion_matrix(
        y_multi_test[:, 0], pred_multi_rf[:, 0],
        labels=type_labels,
        title="Confusion Matrix: Ticket Type (RandomForest MultiOutput)",
        save_path=os.path.join(OUTPUTS_DIR, "confusion_matrix_rf_ticket_type.png"),
    )

    # Ticket Priority — RandomForest MultiOutput
    plot_confusion_matrix(
        y_multi_test[:, 1], pred_multi_rf[:, 1],
        labels=pri_labels,
        title="Confusion Matrix: Ticket Priority (RandomForest MultiOutput)",
        save_path=os.path.join(OUTPUTS_DIR, "confusion_matrix_rf_ticket_priority.png"),
    )

    print("\n" + "=" * 65)
    print(">> Phase 4 complete. Proceed to Phase 5 (Operations Insights).")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
