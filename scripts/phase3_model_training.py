"""
phase3_model_training.py
------------------------
Phase 3 - Multi-Output Model Training

This script:
  1. Loads the TF-IDF matrix and target labels from  models/
  2. Splits data into 80/20 train/test sets (stratified where possible)
  3. Trains two separate classifiers:
       - LinearSVC          for Ticket Type    (Category)
       - LogisticRegression for Ticket Priority
  4. Also trains a MultiOutputClassifier with RandomForest as a baseline
  5. Compares training & test accuracy for all approaches
  6. Saves the best models and the train/test split to  models/

Why separate classifiers?
  LinearSVC and LogisticRegression are fast, memory-efficient, and
  perform very well on high-dimensional sparse TF-IDF features.
  Training one per target lets us tune each independently.
  The MultiOutput RandomForest is included as a comparison baseline.
"""

# -- imports -----------------------------------------------------------
import os
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score


# =====================================================================
#                      MAIN  EXECUTION
# =====================================================================
def main():
    # -- paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    # -- 1. load TF-IDF matrix & targets -------------------------------
    print("=" * 65)
    print("  PHASE 3 -- Multi-Output Model Training")
    print("=" * 65)

    with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
        X = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "target_labels.pkl"), "rb") as f:
        targets = pickle.load(f)

    y_type = targets["Ticket_Type"]
    y_priority = targets["Ticket_Priority"]

    print(f"\n[OK] Loaded TF-IDF matrix : {X.shape}")
    print(f"     Ticket Type classes  : {sorted(set(y_type))}")
    print(f"     Priority classes     : {sorted(set(y_priority))}")

    # -- 2. train/test split (80/20) -----------------------------------
    print("\n-- Train/Test Split (80/20) ------------------------------------")

    # We split using y_type for stratification to keep category balance
    X_train, X_test, \
    y_type_train, y_type_test, \
    y_pri_train, y_pri_test = train_test_split(
        X, y_type, y_priority,
        test_size=0.20,
        random_state=42,
        stratify=y_type,
    )

    print(f"     Training set : {X_train.shape[0]} samples")
    print(f"     Test set     : {X_test.shape[0]} samples")

    # -- 3. Train individual classifiers --------------------------------
    results = {}

    # ---- 3a. LinearSVC for Ticket Type (Category) --------------------
    print("\n-- Model 1: LinearSVC for Ticket Type -------------------------")
    t0 = time.time()
    svc_type = LinearSVC(
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
    )
    svc_type.fit(X_train, y_type_train)
    elapsed = time.time() - t0

    train_acc = accuracy_score(y_type_train, svc_type.predict(X_train))
    test_acc = accuracy_score(y_type_test, svc_type.predict(X_test))

    print(f"     Train accuracy : {train_acc:.4f}")
    print(f"     Test accuracy  : {test_acc:.4f}")
    print(f"     Training time  : {elapsed:.2f}s")
    results["LinearSVC (Type)"] = {"train": train_acc, "test": test_acc}

    # ---- 3b. LogisticRegression for Ticket Priority ------------------
    print("\n-- Model 2: LogisticRegression for Ticket Priority ------------")
    t0 = time.time()
    lr_priority = LogisticRegression(
        C=1.0,
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    lr_priority.fit(X_train, y_pri_train)
    elapsed = time.time() - t0

    train_acc = accuracy_score(y_pri_train, lr_priority.predict(X_train))
    test_acc = accuracy_score(y_pri_test, lr_priority.predict(X_test))

    print(f"     Train accuracy : {train_acc:.4f}")
    print(f"     Test accuracy  : {test_acc:.4f}")
    print(f"     Training time  : {elapsed:.2f}s")
    results["LogReg (Priority)"] = {"train": train_acc, "test": test_acc}

    # ---- 3c. MultiOutputClassifier + RandomForest (baseline) ---------
    print("\n-- Model 3: MultiOutputClassifier + RandomForest (both) -------")
    t0 = time.time()

    # Combine both targets into a 2D array for multi-output
    y_multi_train = np.column_stack([y_type_train, y_pri_train])
    y_multi_test = np.column_stack([y_type_test, y_pri_test])

    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    multi_rf = MultiOutputClassifier(rf_base, n_jobs=-1)
    multi_rf.fit(X_train, y_multi_train)
    elapsed = time.time() - t0

    y_multi_pred_train = multi_rf.predict(X_train)
    y_multi_pred_test = multi_rf.predict(X_test)

    # Per-target accuracy for the multi-output model
    rf_type_train = accuracy_score(y_multi_train[:, 0], y_multi_pred_train[:, 0])
    rf_type_test = accuracy_score(y_multi_test[:, 0], y_multi_pred_test[:, 0])
    rf_pri_train = accuracy_score(y_multi_train[:, 1], y_multi_pred_train[:, 1])
    rf_pri_test = accuracy_score(y_multi_test[:, 1], y_multi_pred_test[:, 1])

    print(f"     Ticket Type  -> Train: {rf_type_train:.4f}  |  Test: {rf_type_test:.4f}")
    print(f"     Priority     -> Train: {rf_pri_train:.4f}  |  Test: {rf_pri_test:.4f}")
    print(f"     Training time  : {elapsed:.2f}s")
    results["RF MultiOut (Type)"] = {"train": rf_type_train, "test": rf_type_test}
    results["RF MultiOut (Pri)"] = {"train": rf_pri_train, "test": rf_pri_test}

    # -- 4. comparison table -------------------------------------------
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<30s} {'Train Acc':>10s} {'Test Acc':>10s}")
    print("-" * 55)
    for name, scores in results.items():
        print(f"  {name:<30s} {scores['train']:>10.4f} {scores['test']:>10.4f}")
    print("-" * 55)

    # -- 5. save models & split ----------------------------------------
    artifacts = {
        "svc_ticket_type.pkl": svc_type,
        "lr_ticket_priority.pkl": lr_priority,
        "multi_rf_model.pkl": multi_rf,
        "train_test_split.pkl": {
            "X_train": X_train, "X_test": X_test,
            "y_type_train": y_type_train, "y_type_test": y_type_test,
            "y_pri_train": y_pri_train, "y_pri_test": y_pri_test,
        },
    }

    print(f"\n[OK] Saving models to {os.path.abspath(MODELS_DIR)}/")
    for fname, obj in artifacts.items():
        fpath = os.path.join(MODELS_DIR, fname)
        with open(fpath, "wb") as f:
            pickle.dump(obj, f)
        print(f"     - {fname:<30s} ({os.path.getsize(fpath) / 1024:.1f} KB)")

    print("=" * 65)
    print("\n>> Phase 3 complete. Proceed to Phase 4 (Evaluation Metrics).\n")


if __name__ == "__main__":
    main()
