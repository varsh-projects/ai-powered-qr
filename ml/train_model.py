"""
ml/train_model.py
LinkShield AI - Machine Learning Model Training Script

This script performs end-to-end training and evaluation of the malicious URL
detection model:
1. Loads dataset from CSV (exact specified path with fallbacks).
2. Validates schema and cleans data (removes duplicates, nulls, invalid labels).
3. Normalizes labels to integer 0 (Benign) and 1 (Malicious).
4. Extracts deterministic features using ml/feature_extractor.py.
5. Splits data into stratified train/test sets using a fixed random seed.
6. Trains a RandomForestClassifier.
7. Evaluates honestly on the held-out test set (Accuracy, Precision, Recall, F1, CM).
8. Saves the trained model and feature schema to ml/model.joblib.
"""

import os
import sys
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure the project root is in sys.path so ml module can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the reusable feature extractor module
try:
    from ml.feature_extractor import (
        extract_features,
        get_feature_names,
        load_model,
        predict_url,
        FEATURE_NAMES,
    )
except ImportError:
    # If run directly inside ml/ directory
    from feature_extractor import (
        extract_features,
        get_feature_names,
        load_model,
        predict_url,
        FEATURE_NAMES,
    )


# Default dataset paths relative to PROJECT_ROOT
DEFAULT_DATASET_REL_PATH = os.path.join("datasets", "phishing_url_dataset_unique.csv")
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, DEFAULT_DATASET_REL_PATH)
DEFAULT_MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "ml", "model.joblib")


def resolve_dataset_path(custom_path: str = None) -> str:
    """
    Locates the dataset CSV file.
    Resolves relative paths reliably from PROJECT_ROOT, making execution
    completely independent of the terminal's current working directory.
    """
    if custom_path:
        if os.path.isabs(custom_path):
            return custom_path
        return os.path.join(PROJECT_ROOT, custom_path)

    return DEFAULT_DATASET_PATH


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Loads dataset, checks existence, validates required columns,
    and removes bad data (nulls, duplicates, invalid labels).
    """
    print("=" * 70)
    print("1. LOADING AND VALIDATING DATASET")
    print("=" * 70)

    # Step 1.1: Check file existence
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Dataset file not found at: {csv_path}")
        print("Please check that the CSV file exists or specify --dataset <path>.")
        sys.exit(1)

    print(f"Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to read dataset CSV: {e}")
        sys.exit(1)

    initial_count = len(df)
    print(f"Initial raw record count: {initial_count:,}")

    # Step 1.2: Validate required columns
    required_columns = {"url", "label"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        print(f"\n[ERROR] Missing required columns in dataset: {missing_cols}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    if "source" in df.columns:
        print("Note: 'source' column detected in dataset. It will NOT be used as a model feature.")

    # Step 1.3: Clean URL column (remove nulls and whitespace-only URLs)
    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""].copy()
    after_url_clean = len(df)
    if initial_count - after_url_clean > 0:
        print(f"Removed {initial_count - after_url_clean:,} rows with missing or blank URLs.")

    # Step 1.4: Remove duplicate URLs
    df = df.drop_duplicates(subset=["url"]).copy()
    after_dedup = len(df)
    if after_url_clean - after_dedup > 0:
        print(f"Removed {after_url_clean - after_dedup:,} duplicate URLs.")

    # Step 1.5: Validate and normalize labels to integer 0 and 1
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Filter out any labels outside of {0, 1}
    valid_mask = df["label"].isin([0, 1])
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"Removed {invalid_count:,} rows with invalid label values (not 0 or 1).")
        df = df[valid_mask].copy()

    final_count = len(df)
    if final_count == 0:
        print("\n[ERROR] No valid data remaining after cleaning!")
        sys.exit(1)

    # Step 1.6: Print dataset size and class distribution
    print(f"\nCleaned dataset size: {final_count:,} records")
    print("\nClass Distribution:")
    class_counts = df["label"].value_counts().to_dict()
    benign_count = class_counts.get(0, 0)
    malicious_count = class_counts.get(1, 0)
    print(f" - Class 0 (Benign):    {benign_count:,} ({benign_count / final_count * 100:.2f}%)")
    print(f" - Class 1 (Malicious): {malicious_count:,} ({malicious_count / final_count * 100:.2f}%)")

    # Check for stratified split feasibility
    if benign_count < 5 or malicious_count < 5:
        print("\n[ERROR] Dataset is too small or unbalanced for stratified train/test splitting.")
        print(f"Found: Benign={benign_count}, Malicious={malicious_count}. Minimum required per class: 5.")
        sys.exit(1)

    return df


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features for all URLs in the dataframe using the reusable feature extractor.
    Guarantees deterministic feature ordering and excludes non-feature columns (like 'source').
    """
    print("\n" + "=" * 70)
    print("2. EXTRACTING NUMERIC FEATURES")
    print("=" * 70)
    print(f"Extracting {len(FEATURE_NAMES)} static lexical features from {len(df):,} URLs...")

    features_records = [extract_features(url) for url in df["url"]]
    X = pd.DataFrame(features_records)[FEATURE_NAMES]

    print(f"Feature matrix shape: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"Features: {', '.join(FEATURE_NAMES[:6])} ... (+{len(FEATURE_NAMES)-6} more)")
    return X


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray, model_output_path: str):
    """
    Splits data into stratified train/test sets, trains a RandomForestClassifier,
    evaluates on the test set, and saves model artifacts.
    """
    print("\n" + "=" * 70)
    print("3. SPLITTING DATA (STRATIFIED 80/20)")
    print("=" * 70)

    # Fixed random_state=42 for deterministic reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Testing set:  {X_test.shape[0]:,} samples")

    print("\n" + "=" * 70)
    print("4. TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 70)
    print("Training RandomForestClassifier (n_estimators=100, random_state=42, n_jobs=-1)...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("Model training completed successfully.")

    print("\n" + "=" * 70)
    print("5. EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    y_pred = model.predict(X_test)

    # Compute key classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:   {accuracy * 100:.2f}% ({accuracy:.6f})")
    print(f"Precision:  {precision * 100:.2f}% ({precision:.6f})")
    print(f"Recall:     {recall * 100:.2f}% ({recall:.6f})")
    print(f"F1-Score:   {f1 * 100:.2f}% ({f1:.6f})")

    # Formatted confusion matrix display
    # Confusion matrix structure:
    # [[True Negatives (TN), False Positives (FP)],
    #  [False Negatives (FN), True Positives (TP)]]
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(f"                 Predicted Benign (0)   Predicted Malicious (1)")
    print(f"Actual Benign:        {tn:6d} (TN)            {fp:6d} (FP)")
    print(f"Actual Malicious:     {fn:6d} (FN)            {tp:6d} (TP)")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign (0)", "Malicious (1)"],
            digits=4,
        )
    )

    # Step 6: Save model and feature schema
    print("=" * 70)
    print("6. SAVING MODEL ARTIFACT AND SCHEMA")
    print("=" * 70)

    # Ensure target directory exists
    output_dir = os.path.dirname(model_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Prepare bundle containing model, feature schema, and evaluation metrics
    artifact_payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "test_samples": int(len(y_test)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
    }

    joblib.dump(artifact_payload, model_output_path)
    print(f"Model and feature schema saved to: {model_output_path}")
    print(f"Artifact size: {os.path.getsize(model_output_path) / (1024 * 1024):.2f} MB")

    # Step 7: Sanity verification of saved artifact
    print("\n" + "=" * 70)
    print("7. VERIFYING SAVED MODEL INFERENCE")
    print("=" * 70)
    loaded_model, loaded_features = load_model(model_output_path)
    assert loaded_features == FEATURE_NAMES, "Saved feature schema mismatch!"

    sample_tests = [
        "https://www.google.com",
        "http://110.37.26.193:54956/bin.sh",
        "https://sentry5.obor1shwron8.ru/4ylkytvt",
    ]

    for sample_url in sample_tests:
        result = predict_url(sample_url, loaded_model, loaded_features)
        print(f"URL: {result['url']}")
        print(f" -> Label: {result['label']} | Malicious Prob: {result['malicious_probability'] * 100:.1f}%\n")

    print("=" * 70)
    print("TRAINING PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LinkShield AI - ML Model Training Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to phishing URL dataset CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_MODEL_OUTPUT,
        help="Path to save trained model.joblib.",
    )
    args = parser.parse_args()

    # 1. Resolve path
    dataset_path = resolve_dataset_path(args.dataset)

    # 2. Load and validate
    df = load_and_validate_data(dataset_path)

    # 3. Extract features
    X = extract_all_features(df)
    y = df["label"].values

    # 4. Train, evaluate, and save
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    train_and_evaluate(X, y, output_path)


if __name__ == "__main__":
    main()
