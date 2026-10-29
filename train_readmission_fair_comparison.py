import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

BASE = Path(r"D:\DiabetesCareAI")
DATA = BASE / "data" / "diabetic_data.csv"
OUT = BASE / "readmission_fair_comparison"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("READMISSION FAIR COMPARISON - SHARED PATIENT-GROUPED SPLIT")
print("=" * 68)

df = pd.read_csv(DATA, na_values=["?"], low_memory=False)
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df.reset_index(drop=True, inplace=True)

y = (df["readmitted"] == "<30").astype(int)
groups = df["patient_nbr"].astype(str)

excluded = {"encounter_id", "patient_nbr", "readmitted", "weight", "payer_code"}
expanded_features = [c for c in df.columns if c not in excluded]
small_features = [
    "age", "gender", "race", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "max_glu_serum", "A1Cresult",
]
small_features = [c for c in small_features if c in df.columns]

# One shared 60/20/20 split for all models.
split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=101)
trainval_idx, test_idx = next(split1.split(df, y, groups))

split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=102)
train_rel, val_rel = next(
    split2.split(df.iloc[trainval_idx], y.iloc[trainval_idx],
                 groups.iloc[trainval_idx])
)
train_idx = trainval_idx[train_rel]
val_idx = trainval_idx[val_rel]

train_patients = set(groups.iloc[train_idx])
val_patients = set(groups.iloc[val_idx])
test_patients = set(groups.iloc[test_idx])
assert not (train_patients & val_patients)
assert not (train_patients & test_patients)
assert not (val_patients & test_patients)

print(f"Rows: {len(df):,}")
print(f"Expanded feature count: {len(expanded_features)}")
print(f"Baseline feature count: {len(small_features)}")
print(f"Train/validation/test: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
print("Patient overlap check: PASSED")

def make_expanded_model(features):
    X = df[features]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessor", prep),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=42
        )),
    ])

def make_small_hgb(features):
    X = df[features]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )),
        ]), categorical),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=20,
            l2_regularization=1.0, early_stopping=True, random_state=42
        )),
    ])

def make_small_rf(features):
    X = df[features]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )),
    ])

candidates = {
    "Expanded HistGradientBoosting (45 features)": (
        make_expanded_model(expanded_features), expanded_features, "classifier"
    ),
    "Baseline HistGradientBoosting (16 features)": (
        make_small_hgb(small_features), small_features, "model"
    ),
    "Baseline Random Forest (16 features)": (
        make_small_rf(small_features), small_features, "model"
    ),
}

results = []
trained = {}
start = time.time()

for name, (model, features, classifier_step) in candidates.items():
    print(f"\n--- Training {name} ---")
    X_train = df.iloc[train_idx][features]
    X_val = df.iloc[val_idx][features]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, **{f"{classifier_step}__sample_weight": weights})

    val_prob = model.predict_proba(X_val)[:, 1]
    val_ap = average_precision_score(y_val, val_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    thresholds = np.arange(0.05, 0.81, 0.01)
    threshold_scores = [
        (f1_score(y_val, (val_prob >= t).astype(int), zero_division=0), float(t))
        for t in thresholds
    ]
    val_f1, threshold = max(threshold_scores, key=lambda item: item[0])

    results.append({
        "model": name,
        "feature_count": len(features),
        "validation_average_precision": float(val_ap),
        "validation_roc_auc": float(val_auc),
        "validation_best_f1": float(val_f1),
        "validation_threshold": threshold,
    })
    trained[name] = (model, features, threshold)
    print(f"Validation AP: {val_ap:.4f}")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Validation F1: {val_f1:.4f} at threshold {threshold:.2f}")

# Select using validation AP only; never rank by test results.
results.sort(
    key=lambda r: (r["validation_average_precision"], r["validation_roc_auc"]),
    reverse=True
)
winner_info = results[0]
winner_name = winner_info["model"]
winner, winner_features, threshold = trained[winner_name]

X_test = df.iloc[test_idx][winner_features]
y_test = y.iloc[test_idx]
test_prob = winner.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

test_metrics = {
    "selected_model": winner_name,
    "feature_count": len(winner_features),
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
    "test_distinct_patients": int(len(test_patients)),
    "warning": (
        "Historical 1999-2008 data; research prototype only, not clinically "
        "validated. Prediction-time availability of features must be verified."
    ),
}

model_path = OUT / "selected_model.joblib"
joblib.dump({
    "model": winner,
    "feature_columns": winner_features,
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
for i, row in enumerate(results, 1):
    print(
        f"{i}. {row['model']} | features={row['feature_count']} | "
        f"validation AP={row['validation_average_precision']:.4f} | "
        f"ROC-AUC={row['validation_roc_auc']:.4f}"
    )
print("\nSELECTED MODEL - SHARED UNTOUCHED TEST SPLIT")
print(json.dumps(test_metrics, indent=2))
print("\nSaved model:", model_path)
print("Saved validation comparison:", OUT / "validation_comparison.json")
print("Saved test metrics:", OUT / "selected_test_metrics.json")
print("Elapsed seconds:", round(time.time() - start, 2))
