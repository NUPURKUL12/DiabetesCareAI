from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

BASE = Path(r"D:\DiabetesCareAI")
DATA = BASE / "data" / "diabetic_data.csv"
OUT = BASE / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

print("=" * 65)
print("DIABETESCARE AI - MODEL TOURNAMENT")
print("=" * 65)

df = pd.read_csv(DATA, na_values=["?"])
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df["target"] = (df["readmitted"] == "<30").astype(int)

# Prediction point: discharge. These features must be known by then.
features = [
    "age", "gender", "race",
    "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "max_glu_serum", "A1Cresult",
]

numeric = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
categorical = [c for c in features if c not in numeric]

X = df[features].copy()
y = df["target"].copy()
groups = df["patient_nbr"].astype(str)

print(f"Dataset encounters: {len(df):,}")
print(f"Readmission-within-30-days rate: {y.mean():.2%}")

# Split patients into 60% train, 20% validation, 20% test.
# No patient can appear in more than one split.
split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
trainval_idx, test_idx = next(split1.split(X, y, groups))

split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
train_rel, val_rel = next(
    split2.split(
        X.iloc[trainval_idx],
        y.iloc[trainval_idx],
        groups.iloc[trainval_idx],
    )
)
train_idx = trainval_idx[train_rel]
val_idx = trainval_idx[val_rel]

train_patients = set(groups.iloc[train_idx])
val_patients = set(groups.iloc[val_idx])
test_patients = set(groups.iloc[test_idx])

assert not train_patients.intersection(val_patients)
assert not train_patients.intersection(test_patients)
assert not val_patients.intersection(test_patients)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(f"Training:   {len(X_train):,} encounters")
print(f"Validation: {len(X_val):,} encounters")
print(f"Test:       {len(X_test):,} encounters")
print("Patient overlap check: passed")

# Sparse one-hot encoding works with Logistic Regression and Random Forest.
sparse_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

# HistGradientBoosting requires dense input, so it gets a separate
# equivalent preprocessing pipeline.
dense_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )),
    ]), categorical),
])

models = {
    "Logistic Regression": Pipeline([
        ("prep", clone(sparse_preprocessor)),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("prep", clone(sparse_preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ]),
    "HistGradientBoosting": Pipeline([
        ("prep", clone(dense_preprocessor)),
        ("model", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
        )),
    ]),
}

results = []
trained = {}

for name, pipeline in models.items():
    print(f"\n{'-' * 55}")
    print(f"Training: {name}")
    start = time.time()

    if name == "HistGradientBoosting":
        # Compensate for the imbalanced target during training.
        weights = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        pipeline.fit(X_train, y_train)

    elapsed = time.time() - start
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    # Choose a threshold on validation data only.
    # Maximize F1; the final test set is not used for tuning.
    thresholds = np.arange(0.10, 0.81, 0.02)
    scored = []
    for threshold in thresholds:
        val_pred = (val_probs >= threshold).astype(int)
        scored.append((
            f1_score(y_val, val_pred, zero_division=0),
            threshold,
        ))
    best_val_f1, best_threshold = max(scored, key=lambda item: item[0])

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_threshold).astype(int)

    metrics = {
        "model": name,
        "training_seconds": round(elapsed, 2),
        "validation_best_f1": round(float(best_val_f1), 4),
        "threshold_selected_on_validation": round(float(best_threshold), 2),
        "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
        "test_precision": round(float(precision_score(
            y_test, test_pred, zero_division=0
        )), 4),
        "test_recall": round(float(recall_score(
            y_test, test_pred, zero_division=0
        )), 4),
        "test_f1": round(float(f1_score(
            y_test, test_pred, zero_division=0
        )), 4),
        "test_average_precision": round(float(
            average_precision_score(y_test, test_probs)
        ), 4),
        "test_roc_auc": round(float(roc_auc_score(
            y_test, test_probs
        )), 4),
    }

    results.append(metrics)
    trained[name] = pipeline
    print(json.dumps(metrics, indent=2))

# Rank using test average precision as the primary metric.
# This is appropriate for a rare positive class. ROC-AUC is also reported.
table = pd.DataFrame(results).sort_values(
    ["test_average_precision", "test_roc_auc"],
    ascending=False,
)
table.to_csv(OUT / "model_comparison.csv", index=False)

with open(OUT / "model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON (higher is better for all metrics below)")
print("=" * 65)
print(table.to_string(index=False))

# Save the top candidate, but do not call it clinically validated.
winner = str(table.iloc[0]["model"])
joblib.dump(trained[winner], OUT / "best_candidate_model.joblib")

with open(OUT / "selection.json", "w", encoding="utf-8") as f:
    json.dump({
        "selected_candidate": winner,
        "selection_metric": "test_average_precision",
        "warning": (
            "Research prototype only. The selected model is a candidate, "
            "not a clinically validated model. Confirm prediction-time "
            "features and perform further validation before deployment."
        ),
    }, f, indent=2)

print(f"\nTop candidate by test average precision: {winner}")
print(f"Results saved in: {OUT}")
print("This experiment is for model comparison, not clinical deployment.")
