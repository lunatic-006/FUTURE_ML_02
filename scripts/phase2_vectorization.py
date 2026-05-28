"""
phase2_vectorization.py
-----------------------
Phase 2 - Feature Extraction (TF-IDF Vectorization)

This script:
  1. Loads the cleaned dataset from  data/processed/tickets_cleaned.csv
  2. Transforms the Cleaned_Text column into numerical features
     using scikit-learn's TfidfVectorizer.
  3. Configures key parameters:
       - max_features=5000   (caps vocabulary to prevent memory issues)
       - ngram_range=(1, 2)  (captures unigrams + bigrams for key phrases)
       - sublinear_tf=True   (applies log-scaling for better term weighting)
       - max_df=0.95         (ignores terms appearing in >95% of docs)
       - min_df=2            (ignores terms appearing in fewer than 2 docs)
  4. Saves the fitted vectorizer and the sparse TF-IDF matrix to  models/
  5. Prints a summary of the feature space

Outputs:
  - models/tfidf_vectorizer.pkl   : fitted TfidfVectorizer object
  - models/tfidf_matrix.pkl       : sparse TF-IDF feature matrix
  - models/target_labels.pkl      : target labels (Ticket Type + Priority)
"""

# -- imports -----------------------------------------------------------
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =====================================================================
#                      MAIN  EXECUTION
# =====================================================================
def main():
    # -- paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tickets_cleaned.csv")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -- 1. load cleaned data ------------------------------------------
    print("=" * 65)
    print("  PHASE 2 -- Feature Extraction (TF-IDF Vectorization)")
    print("=" * 65)

    df = pd.read_csv(PROCESSED_PATH)
    print(f"\n[OK] Loaded cleaned dataset : {os.path.abspath(PROCESSED_PATH)}")
    print(f"     Shape                  : {df.shape[0]} rows x {df.shape[1]} columns")

    # handle any NaN in Cleaned_Text (safety check)
    df["Cleaned_Text"] = df["Cleaned_Text"].fillna("")

    # -- 2. configure & fit TF-IDF vectorizer --------------------------
    print("\n-- TF-IDF Vectorizer Configuration ----------------------------")
    tfidf_params = {
        "max_features": 5000,       # cap vocabulary size for memory efficiency
        "ngram_range": (1, 2),      # unigrams + bigrams to capture key phrases
        "sublinear_tf": True,       # apply log normalization (1 + log(tf))
        "max_df": 0.95,             # ignore terms in >95% of documents
        "min_df": 2,                # ignore terms in fewer than 2 documents
        "strip_accents": "unicode", # normalize accented characters
    }

    for param, value in tfidf_params.items():
        print(f"     {param:20s} : {value}")

    vectorizer = TfidfVectorizer(**tfidf_params)

    print("\n[..] Fitting TF-IDF vectorizer on Cleaned_Text ...")
    tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Text"])

    # -- 3. inspect the feature space ----------------------------------
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n[OK] TF-IDF matrix built successfully!")
    print(f"     Matrix shape   : {tfidf_matrix.shape}  "
          f"({tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features)")
    print(f"     Matrix dtype   : {tfidf_matrix.dtype}")
    print(f"     Sparsity       : {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")
    print(f"     Non-zero vals  : {tfidf_matrix.nnz:,}")

    print(f"\n-- Sample features (first 20 of {len(feature_names)}) --------")
    print(f"     {list(feature_names[:20])}")

    print(f"\n-- Sample bigram features -------------------------------------")
    bigrams = [f for f in feature_names if " " in f]
    print(f"     Total bigrams  : {len(bigrams)}")
    print(f"     Examples       : {bigrams[:15]}")

    # -- 4. prepare target labels --------------------------------------
    targets = {
        "Ticket_Type": df["Ticket Type"].values,
        "Ticket_Priority": df["Ticket Priority"].values,
    }

    print(f"\n-- Target labels prepared -------------------------------------")
    print(f"     Ticket Type    : {len(set(targets['Ticket_Type']))} classes  -> {sorted(set(targets['Ticket_Type']))}")
    print(f"     Ticket Priority: {len(set(targets['Ticket_Priority']))} classes  -> {sorted(set(targets['Ticket_Priority']))}")

    # -- 5. save artifacts ---------------------------------------------
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
    labels_path = os.path.join(MODELS_DIR, "target_labels.pkl")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    with open(labels_path, "wb") as f:
        pickle.dump(targets, f)

    print(f"\n[OK] Artifacts saved to {os.path.abspath(MODELS_DIR)}/")
    print(f"     - tfidf_vectorizer.pkl  ({os.path.getsize(vectorizer_path) / 1024:.1f} KB)")
    print(f"     - tfidf_matrix.pkl      ({os.path.getsize(matrix_path) / 1024:.1f} KB)")
    print(f"     - target_labels.pkl     ({os.path.getsize(labels_path) / 1024:.1f} KB)")
    print("=" * 65)
    print("\n>> Phase 2 complete. Proceed to Phase 3 (Model Training).\n")


if __name__ == "__main__":
    main()
