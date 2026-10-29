from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"D:\DiabetesCareAI")
DATA_PATH = BASE_DIR / "data" / "diabetic_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("             DiabetesCare AI - Model Training")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH, na_values=["?"])
print(f"Dataset rows: {len(df):,}")
print(f"Dataset columns: {len(df.columns)}")

required = {
    "patient_nbr", "readmitted", "encounter_id", "age", "gender",
    "race", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "max_glu_serum", "A1Cresult"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

print("\n[2/6] Creating the target...")
# Positive = readmitted within 30 days.
# Negative = not readmitted within 30 days (NO or >30).
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df["readmitted_within_30_days"] = (
    df["readmitted"] == "<30"
).astype(int)

print("Target counts:")
print(df["readmitted_within_30_days"].value_counts().rename(
    index={0: "Not within 30 days", 1: "Within 30 days"}
).to_string())

# Exclude identifiers and the outcome from predictive features.
feature_columns = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
]

# Keep encounters from the same patient on only one side of the split.
# The dataset contains repeated encounters for some patients.
groups = df["patient_nbr"].astype(str)
X = df[feature_columns].copy()
y = df["readmitted_within_30_days"].copy()

print("\n[3/6] Splitting train/test by patient...")
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_patients = set(groups.iloc[train_idx])
test_patients = set(groups.iloc[test_idx])

if train_patients.intersection(test_patients):
    raise RuntimeError("Patient leakage detected between train and test!")

print(f"Training encounters: {len(X_train):,}")
print(f"Testing encounters:  {len(X_test):,}")
print(f"Distinct training patients: {len(train_patients):,}")
print(f"Distinct testing patients:  {len(test_patients):,}")
print(f"Training positive rate: {y_train.mean():.3%}")
print(f"Testing positive rate:  {y_test.mean():.3%}")

numeric_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
categorical_features = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

print("\n[4/6] Training logistic regression...")
model.fit(X_train, y_train)
print("Training completed.")

print("\n[5/6] Evaluating on the held-out test set...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "model": "Logistic Regression",
    "dataset": "UCI Diabetes 130-US Hospitals (1999-2008)",
    "target": "Readmission within 30 days",
    "train_encounters": int(len(X_train)),
    "test_encounters": int(len(X_test)),
    "train_distinct_patients": int(len(train_patients)),
    "test_distinct_patients": int(len(test_patients)),
    "positive_rate_test": float(y_test.mean()),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "average_precision": float(average_precision_score(
        y_test, probabilities
    )),
}

if y_test.nunique() == 2:
    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
else:
    metrics["roc_auc"] = None
    print("Warning: test set contains only one target class; ROC-AUC unavailable.")

print("\n--- Test Results ---")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Not readmitted within 30 days", "Readmitted within 30 days"],
    zero_division=0,
))

print("\n[6/6] Saving model and evaluation results...")
joblib.dump(model, MODEL_DIR / "diabetes_readmission_model.joblib")

with open(MODEL_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

feature_info = {
    "features": feature_columns,
    "target_column": "readmitted_within_30_days",
    "positive_class": 1,
    "positive_class_meaning": "Readmitted within 30 days",
    "dataset_source": (
        "UCI Machine Learning Repository: "
        "Diabetes 130-US Hospitals for Years 1999-2008"
    ),
}
with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
    json.dump(feature_info, f, indent=2)

print(f"\nModel saved to: {MODEL_DIR / 'diabetes_readmission_model.joblib'}")
print(f"Metrics saved to: {MODEL_DIR / 'evaluation_metrics.json'}")
print(f"Model info saved to: {MODEL_DIR / 'model_info.json'}")
print("\nTraining run finished successfully.")
print("Note: This is an educational prototype, not a clinical tool.")
