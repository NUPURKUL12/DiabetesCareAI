
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

BASE = Path(r"D:\DiabetesCareAI")
DATA = BASE / "data" / "diabetic_data.csv"
OUT = BASE / "expanded"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading US hospital dataset...")
df = pd.read_csv(DATA, na_values=["?"], low_memory=False)
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df.reset_index(drop=True, inplace=True)

y = (df["readmitted"] == "<30").astype(int)
groups = df["patient_nbr"].astype(str)

# Exclude identifiers, the outcome, and fields that directly reveal
# the target or depend on future encounters.
excluded = {
    "encounter_id", "patient_nbr", "readmitted",
    "weight", "payer_code",
}

# Expanded clinical, diagnosis, medication, and utilization features.
features = [
    c for c in df.columns
    if c not in excluded
]

# Exclude medication-change summary fields if they are not available
# at the intended prediction time; keep medication indicators and
# diagnosis fields for this exploratory discharge-time model.
X = df[features].copy()

# Convert categorical columns to strings and normalize missing values.
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Unknown").astype(str)

# Split patients, not just encounters, to prevent patient overlap.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=101
)
trainval_idx, test_idx = next(split1.split(X, y, groups))

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=102
)
train_rel, val_rel = next(
    split2.split(
        X.iloc[trainval_idx],
        y.iloc[trainval_idx],
        groups.iloc[trainval_idx],
    )
)

train_idx = trainval_idx[train_rel]
val_idx = trainval_idx[val_rel]

assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[val_idx]))
assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
assert set(groups.iloc[val_idx]).isdisjoint(set(groups.iloc[test_idx]))

print(f"Rows: {len(df):,}")
print(f"Features: {len(features)}")
print(f"Train: {len(train_idx):,} | Validation: {len(val_idx):,} | Test: {len(test_idx):,}")
print("Patient overlap check: passed")

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ]), categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        random_state=42,
    )),
])

print("\nTraining expanded HistGradientBoosting model...")
start = time.time()
weights = compute_sample_weight("balanced", y.iloc[train_idx])
model.fit(X.iloc[train_idx], y.iloc[train_idx], classifier__sample_weight=weights)
print(f"Training completed in {time.time() - start:.1f}s")

# Select the decision threshold using validation data only.
val_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
thresholds = np.arange(0.10, 0.81, 0.01)
best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)
    score = f1_score(y.iloc[val_idx], pred, zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

print(f"Validation best F1: {best_f1:.4f}")
print(f"Validation-selected threshold: {best_threshold:.2f}")

# Evaluate on the untouched holdout test set.
test_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)
y_test = y.iloc[test_idx]

metrics = {
    "model": "Expanded HistGradientBoosting",
    "feature_count": len(features),
    "features": features,
    "threshold": best_threshold,
    "validation_f1": float(best_f1),
    "test_accuracy": float(accuracy_score(y_test, test_pred)),
    "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "test_average_precision": float(average_precision_score(y_test, test_prob)),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_encounters": int(len(test_idx)),
    "warning": "Historical dataset; research prototype, not clinically validated.",
}

joblib.dump(model, OUT / "expanded_model.joblib")

with open(OUT / "expanded_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nEXPANDED MODEL TEST RESULTS")
print(json.dumps(metrics, indent=2))
print(f"\nSaved model: {OUT / 'expanded_model.joblib'}")
print(f"Saved metrics: {OUT / 'expanded_metrics.json'}")
