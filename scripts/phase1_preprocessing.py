"""
phase1_preprocessing.py
-----------------------
Phase 1 - Data Setup & Text Preprocessing

This script:
  1. Loads the raw support-ticket CSV from  data/raw/customer_support_tickets.csv
  2. Performs exploratory sanity checks (shape, nulls, class balance)
  3. Applies a robust text-cleaning pipeline:
       - lowercase conversion
       - punctuation / special-character removal
       - English stop-word elimination  (NLTK)
       - WordNet lemmatization          (NLTK)
  4. Saves the cleaned dataset to  data/processed/tickets_cleaned.csv

Dataset columns used:
  - Ticket Description  : raw text to clean
  - Ticket Type         : target label 1 (Category)
  - Ticket Priority     : target label 2 (Priority)
"""

# -- imports -----------------------------------------------------------
import os
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -- download required NLTK data (only on first run) ------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)


# =====================================================================
#                     TEXT-CLEANING PIPELINE
# =====================================================================
class TextCleaner:
    """
    Reusable text-cleaning pipeline.

    Steps (in order):
        1. Lowercase
        2. Remove URLs, email addresses, zip codes
        3. Remove template placeholders like {product_purchased}
        4. Remove punctuation & digits
        5. Tokenize
        6. Remove English stop-words
        7. Lemmatize each token
        8. Rejoin into a single string
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # pre-compiled regex patterns for speed
        self._re_url = re.compile(r"https?://\S+|www\.\S+")
        self._re_email = re.compile(r"\S+@\S+\.\S+")
        self._re_placeholder = re.compile(r"\{[^}]+\}")       # e.g. {product_purchased}
        self._re_zipcode = re.compile(r"\b\d{5}(?:-\d{4})?\b")
        self._re_numbers = re.compile(r"\d+")
        self._re_extra_spaces = re.compile(r"\s{2,}")

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to a single string."""

        if not isinstance(text, str):
            return ""

        # 1 - lowercase
        text = text.lower()

        # 2 - strip URLs, emails, zip codes
        text = self._re_url.sub("", text)
        text = self._re_email.sub("", text)
        text = self._re_zipcode.sub("", text)

        # 3 - remove template placeholders
        text = self._re_placeholder.sub("", text)

        # 4 - remove punctuation & digits
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = self._re_numbers.sub("", text)

        # 5 - tokenize
        tokens = word_tokenize(text)

        # 6 - remove stop-words
        tokens = [t for t in tokens if t not in self.stop_words]

        # 7 - lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # 8 - rejoin & collapse whitespace
        cleaned = " ".join(tokens)
        cleaned = self._re_extra_spaces.sub(" ", cleaned).strip()

        return cleaned


# =====================================================================
#                      MAIN  EXECUTION
# =====================================================================
def main():
    # -- paths ---------------------------------------------------------
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "customer_support_tickets.csv")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    PROCESSED_PATH = os.path.join(PROCESSED_DIR, "tickets_cleaned.csv")

    # -- 1. load raw data ----------------------------------------------
    print("=" * 65)
    print("  PHASE 1 -- Data Setup & Text Preprocessing")
    print("=" * 65)

    df = pd.read_csv(RAW_PATH)
    print(f"\n[OK] Loaded dataset   : {os.path.abspath(RAW_PATH)}")
    print(f"     Shape            : {df.shape[0]} rows  x  {df.shape[1]} columns")
    print(f"     Columns          : {list(df.columns)}")

    # -- 2. sanity checks ----------------------------------------------
    print("\n-- Missing values (non-zero only) -----------------------------")
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    if len(missing_nonzero) > 0:
        print(missing_nonzero.to_string())
    else:
        print("   None -- dataset is complete")

    print("\n-- Ticket Type distribution (Category target) -----------------")
    print(df["Ticket Type"].value_counts().to_string())

    print("\n-- Ticket Priority distribution (Priority target) -------------")
    print(df["Ticket Priority"].value_counts().to_string())

    # -- 3. apply text cleaning ----------------------------------------
    print("\n[..] Cleaning text (lowercase -> punctuation -> stopwords -> lemmatize) ...")
    cleaner = TextCleaner()
    df["Cleaned_Text"] = df["Ticket Description"].apply(cleaner.clean)

    # quick before / after peek
    print("\n-- Before vs After (first 3 rows) ------------------------------")
    for idx in range(min(3, len(df))):
        orig = df.iloc[idx]["Ticket Description"]
        clean = df.iloc[idx]["Cleaned_Text"]
        print(f"\n   [Row {idx}]")
        print(f"   ORIGINAL : {orig[:120]}...")
        print(f"   CLEANED  : {clean[:120]}")

    # -- 4. save processed data ----------------------------------------
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # keep only the columns needed for modelling
    cols_to_keep = ["Ticket ID", "Ticket Type", "Ticket Priority",
                    "Ticket Description", "Cleaned_Text"]
    df_out = df[cols_to_keep].copy()
    df_out.to_csv(PROCESSED_PATH, index=False)

    print(f"\n[OK] Cleaned dataset saved -> {os.path.abspath(PROCESSED_PATH)}")
    print(f"     Shape : {df_out.shape[0]} rows  x  {df_out.shape[1]} columns")
    print(f"     Columns saved : {list(df_out.columns)}")
    print("=" * 65)
    print("\n>> Phase 1 complete. Proceed to Phase 2 (Feature Extraction).\n")


if __name__ == "__main__":
    main()
