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
    precision_score, recall_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

BASE = Path(r"D:\DiabetesCareAI")
DATA = BASE / "data" / "diabetic_data.csv"
OUT = BASE / "readmission_tournament_v2"
OUT.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

print("=" * 68)
print("DIABETESCARE AI - READMISSION MODEL TOURNAMENT V2")
print("Validation-based selection; patient-separated test evaluation")
print("=" * 68)

df = pd.read_csv(DATA, na_values=["?"])
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
categorical = [c for c in features if c not in numeric]

X = df[features].copy()
groups = df["patient_nbr"].astype(str)

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

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(f"Dataset encounters: {len(df):,}")
print(f"Features: {len(features)}")
print(f"Train / validation / test: {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
print("Patient leakage check: PASSED")

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
started = time.time()

for name, pipeline in models.items():
    print(f"\n--- Training {name} ---")
    t0 = time.time()

    if name == "HistGradientBoosting":
        weights = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_ap = average_precision_score(y_val, val_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    thresholds = np.arange(0.05, 0.81, 0.01)
    scored = [
        (
            f1_score(y_val, (val_prob >= t).astype(int), zero_division=0),
            float(t),
        )
        for t in thresholds
    ]
    best_val_f1, best_threshold = max(scored, key=lambda z: z[0])

    results.append({
        "model": name,
        "validation_average_precision": float(val_ap),
        "validation_roc_auc": float(val_auc),
        "validation_best_f1": float(best_val_f1),
        "validation_threshold": best_threshold,
        "training_seconds": round(time.time() - t0, 2),
    })
    trained[name] = pipeline

    print(f"Validation AP: {val_ap:.4f}")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Validation F1: {best_val_f1:.4f}")
    print(f"Validation threshold: {best_threshold:.2f}")

# Choose the model using validation AP only.
results.sort(
    key=lambda r: (
        r["validation_average_precision"],
        r["validation_roc_auc"],
    ),
    reverse=True,
)
winner_info = results[0]
winner_name = winner_info["model"]
winner = trained[winner_name]
threshold = winner_info["validation_threshold"]

# Test set is used once for final evaluation of the selected candidate.
test_prob = winner.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

test_metrics = {
    "selected_model": winner_name,
    "selection_metric": "validation_average_precision",
    "threshold_selected_on_validation": threshold,
    "test_accuracy": float(accuracy_score(y_test, test_pred)),
    "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "test_average_precision": float(average_precision_score(y_test, test_prob)),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_confusion_matrix_tn_fp_fn_tp": confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel().tolist(),
    "test_encounters": int(len(y_test)),
    "test_positive_rate": float(y_test.mean()),
    "test_distinct_patients": int(len(test_patients)),
    "warning": (
        "Research prototype only. Historical 1999-2008 dataset; "
        "not clinically validated and not ready for clinical decisions."
    ),
}

# Save into a NEW directory; preserve existing model artifacts.
model_path = OUT / "selected_readmission_candidate.joblib"
joblib.dump({
    "model": winner,
    "feature_columns": features,
    "threshold": threshold,
    "target": "readmission within 30 days",
    "selection_metric": "validation_average_precision",
    "warning": test_metrics["warning"],
}, model_path, compress=3)

(OUT / "validation_comparison.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
(OUT / "selected_test_metrics.json").write_text(
    json.dumps(test_metrics, indent=2), encoding="utf-8"
)

print("\n" + "=" * 68)
print("VALIDATION RANKING")
print("=" * 68)
for rank, row in enumerate(results, start=1):
    print(
        f"{rank}. {row['model']}: "
        f"validation AP={row['validation_average_precision']:.4f}, "
        f"ROC-AUC={row['validation_roc_auc']:.4f}"
    )

print("\nSELECTED CANDIDATE - UNTOUCHED TEST RESULTS")
print(json.dumps(test_metrics, indent=2))
print("\nSaved candidate:", model_path)
print("Saved comparison:", OUT / "validation_comparison.json")
print("Saved test report:", OUT / "selected_test_metrics.json")
print("Elapsed seconds:", round(time.time() - started, 2))
