
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

BASE = Path(r"D:\DiabetesCareAI")
OUT = BASE / "tuning"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "data" / "diabetic_data.csv", na_values=["?"])
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
y = (df["readmitted"] == "<30").astype(int)

features = [
    "age", "gender", "race", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "max_glu_serum", "A1Cresult",
]
numeric = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
categorical = [f for f in features if f not in numeric]

X = df[features].copy()
groups = df["patient_nbr"].astype(str)

# Fresh patient-level 60/20/20 split.
first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=101)
trainval, test = next(first.split(X, y, groups))

second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=102)
tr_rel, val_rel = next(
    second.split(X.iloc[trainval], y.iloc[trainval], groups.iloc[trainval])
)
train = trainval[tr_rel]
val = trainval[val_rel]

for a, b in [(train, val), (train, test), (val, test)]:
    assert set(groups.iloc[a]).isdisjoint(set(groups.iloc[b]))

Xtr, ytr = X.iloc[train], y.iloc[train]
Xv, yv = X.iloc[val], y.iloc[val]
Xte, yte = X.iloc[test], y.iloc[test]

def make_preprocessor(dense=False):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore", sparse_output=not dense
            )),
        ]), categorical),
    ])

candidates = [
    ("Logistic C=0.3", LogisticRegression(
        C=0.3, max_iter=1000, class_weight="balanced", random_state=42
    ), False, False),
    ("Logistic C=1", LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=42
    ), False, False),
    ("Random Forest 150", RandomForestClassifier(
        n_estimators=150, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ), False, False),
    ("Random Forest 250", RandomForestClassifier(
        n_estimators=250, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1, random_state=43
    ), False, False),
    ("HistGB 100", HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=True, random_state=42
    ), True, True),
    ("HistGB 200", HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=20,
        l2_regularization=1.0, early_stopping=True, random_state=42
    ), True, True),
    ("HistGB 150 deeper", HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=2.0, early_stopping=True, random_state=42
    ), True, True),
]

rows = []
fitted = {}

print(f"Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}")
print("Patient overlap: none")
print("\nSelecting configurations using validation data only.")

for name, estimator, dense, use_weights in candidates:
    print(f"\nTraining {name}...")
    pipe = Pipeline([
        ("prep", make_preprocessor(dense)),
        ("model", estimator),
    ])

    start = time.time()
    if use_weights:
        weights = compute_sample_weight("balanced", ytr)
        pipe.fit(Xtr, ytr, model__sample_weight=weights)
    else:
        pipe.fit(Xtr, ytr)

    elapsed = time.time() - start
    val_probs = pipe.predict_proba(Xv)[:, 1]
    ap = average_precision_score(yv, val_probs)

    rows.append({
        "model": name,
        "validation_average_precision": float(ap),
        "training_seconds": round(elapsed, 2),
    })
    fitted[name] = (pipe, val_probs)

    print(f"Validation average precision: {ap:.4f} | Time: {elapsed:.2f}s")

leaderboard = pd.DataFrame(rows).sort_values(
    "validation_average_precision", ascending=False
)
leaderboard.to_csv(OUT / "validation_leaderboard.csv", index=False)

winner = str(leaderboard.iloc[0]["model"])
best_pipe, val_probs = fitted[winner]

# Tune the decision threshold on validation only.
thresholds = np.arange(0.10, 0.81, 0.01)
threshold_scores = [
    (f1_score(yv, val_probs >= t, zero_division=0), t)
    for t in thresholds
]
val_f1, threshold = max(threshold_scores, key=lambda item: item[0])

print("\nVALIDATION LEADERBOARD")
print(leaderboard.to_string(index=False))
print(f"\nSelected configuration: {winner}")
print(f"Validation-selected threshold: {threshold:.2f}")
print(f"Validation F1: {val_f1:.4f}")

# Test set is used only after model/configuration selection is complete.
test_probs = best_pipe.predict_proba(Xte)[:, 1]
test_pred = (test_probs >= threshold).astype(int)

test_metrics = {
    "selected_model": winner,
    "threshold_selected_on_validation": float(threshold),
    "test_accuracy": float(accuracy_score(yte, test_pred)),
    "test_precision": float(precision_score(yte, test_pred, zero_division=0)),
    "test_recall": float(recall_score(yte, test_pred, zero_division=0)),
    "test_f1": float(f1_score(yte, test_pred, zero_division=0)),
    "test_average_precision": float(average_precision_score(yte, test_probs)),
    "test_roc_auc": float(roc_auc_score(yte, test_probs)),
    "test_encounters": int(len(yte)),
    "test_positive_rate": float(yte.mean()),
    "warning": "Research prototype; not clinically validated.",
}

print("\nFINAL HOLDOUT TEST RESULTS")
print(json.dumps(test_metrics, indent=2))

joblib.dump(best_pipe, OUT / "selected_model.joblib")
with open(OUT / "final_test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)

print(f"\nSaved results to: {OUT}")
print("Done. The test set was not used to choose the model or threshold.")
