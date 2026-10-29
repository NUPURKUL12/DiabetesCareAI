import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve
)

ROOT = Path(r"D:\DiabetesCareAI")
CDC_PATH = ROOT / "external_data" / "cdc_diabetes" / "cdc_diabetes_health_indicators.csv"
HOSPITAL_PATH = ROOT / "data" / "diabetic_data.csv"
OUT = ROOT / "unified_training"
OUT.mkdir(parents=True, exist_ok=True)

START = time.time()
SEED = 42


def prepare_source(path, target, task_name, prefix, drop_columns):
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in {path}")

    y_raw = df[target]

    if task_name == "cdc_diabetes_status":
        y = pd.to_numeric(y_raw, errors="raise").astype(int)
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("CDC target must contain only 0 and 1.")
    else:
        # Positive class: readmitted within 30 days.
        y = (y_raw.astype(str).str.strip() == "<30").astype(int)

    excluded = set([target] + drop_columns)
    feature_df = df.drop(
        columns=[c for c in excluded if c in df.columns]
    ).copy()

    encoded = pd.DataFrame(index=feature_df.index)

    for col in feature_df.columns:
        s = feature_df[col].replace(["?", "Unknown/Invalid", "None"], np.nan)

        numeric = pd.to_numeric(s, errors="coerce")
        nonmissing = int(s.notna().sum())

        if nonmissing == 0 or numeric.notna().sum() >= 0.95 * nonmissing:
            encoded[col] = numeric.astype(float)
        else:
            # Stable category coding for this dataset and feature.
            codes, uniques = pd.factorize(s.astype("string"), sort=True)
            values = codes.astype(float)
            values[codes < 0] = np.nan
            encoded[col] = values

    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded.columns = [f"{prefix}__{c}" for c in encoded.columns]

    # The task identifier is an explicit feature in the single model.
    encoded.insert(0, "task_id", 0 if task_name == "cdc_diabetes_status" else 1)

    indices = np.arange(len(y))
    train_idx, temporary_idx = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temporary_idx, test_size=0.50, random_state=SEED + 1,
        stratify=y.iloc[temporary_idx]
    )

    print(f"\n{task_name}")
    print(f"  Records: {len(df):,}")
    print(f"  Features: {len(feature_df)}")
    print(f"  Positive label rate: {y.mean():.4f}")
    print(
        f"  Train/validation/test: "
        f"{len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}"
    )

    return {
        "name": task_name,
        "X": encoded,
        "y": y.reset_index(drop=True),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "features": list(encoded.columns[1:]),
    }


def best_f1_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "records": int(len(y)),
        "actual_positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(predictions)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y, probabilities))
        if len(np.unique(y)) == 2 else None
    )
    return result


print("=" * 68)
print("DIABETESCARE AI - SINGLE UNIFIED MODEL TRAINING")
print("=" * 68)

cdc = prepare_source(
    CDC_PATH, "Diabetes_binary", "cdc_diabetes_status",
    "cdc", ["ID"]
)
hospital = prepare_source(
    HOSPITAL_PATH, "readmitted", "hospital_readmission_30d",
    "hospital", ["encounter_id", "patient_nbr"]
)

# Each source has its own namespaced features. Unavailable features remain
# NaN; task_id identifies which outcome the row's label represents.
all_columns = list(dict.fromkeys(
    list(cdc["X"].columns) + list(hospital["X"].columns)
))

def partition(source, split):
    idx = source[f"{split}_idx"]
    X = source["X"].iloc[idx].reindex(columns=all_columns)
    y = source["y"].iloc[idx].to_numpy()
    return X, y

X_train_parts, y_train_parts = [], []
X_val_parts, y_val_parts = [], []
X_test_parts, y_test_parts = [], []

for source in (cdc, hospital):
    X, y = partition(source, "train")
    X_train_parts.append(X)
    y_train_parts.append(y)

    X, y = partition(source, "val")
    X_val_parts.append(X)
    y_val_parts.append(y)

    X, y = partition(source, "test")
    X_test_parts.append(X)
    y_test_parts.append(y)

X_train = pd.concat(X_train_parts, ignore_index=True)
y_train = np.concatenate(y_train_parts)
X_val = pd.concat(X_val_parts, ignore_index=True)
y_val = np.concatenate(y_val_parts)
X_test = pd.concat(X_test_parts, ignore_index=True)
y_test = np.concatenate(y_test_parts)

# One estimator, trained on both tasks with the task identifier included.
model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.10,
)

print("\nTraining ONE combined HistGradientBoosting classifier...")
print("Combined training rows:", f"{len(X_train):,}")
print("Combined feature columns:", len(all_columns))

model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

# Tune thresholds separately by task using validation rows only.
val_results = {}
test_results = {}
thresholds = {}

offset = 0
for source in (cdc, hospital):
    n = len(source["val_idx"])
    yv = y_val[offset:offset + n]
    pv = val_prob[offset:offset + n]
    thresholds[source["name"]] = best_f1_threshold(yv, pv)
    val_results[source["name"]] = evaluate(
        yv, pv, thresholds[source["name"]]
    )
    offset += n

offset = 0
for source in (cdc, hospital):
    n = len(source["test_idx"])
    yt = y_test[offset:offset + n]
    pt = test_prob[offset:offset + n]
    test_results[source["name"]] = evaluate(
        yt, pt, thresholds[source["name"]]
    )
    offset += n

bundle = {
    "model": model,
    "feature_columns": all_columns,
    "task_features": {
        "cdc_diabetes_status": cdc["features"],
        "hospital_readmission_30d": hospital["features"],
    },
    "task_ids": {
        "cdc_diabetes_status": 0,
        "hospital_readmission_30d": 1,
    },
    "thresholds": thresholds,
    "encoding_note": (
        "Categorical values are factorized per source feature. "
        "Supply feature values in the same encoding used during training."
    ),
    "warning": (
        "Research prototype. CDC task is survey-based status classification; "
        "hospital task uses historical 1999-2008 data. Not clinically validated."
    ),
}

model_path = ROOT / "unified_model.joblib"
joblib.dump(bundle, model_path, compress=3)

report = {
    "model_artifact": str(model_path),
    "model_type": type(model).__name__,
    "single_estimator": True,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "unified_feature_columns": len(all_columns),
    "thresholds_selected_on_validation": thresholds,
    "validation_metrics_by_task": val_results,
    "test_metrics_by_task": test_results,
    "datasets": {
        "CDC": {
            "records": int(len(cdc["y"])),
            "task": "survey-based diabetes status",
            "features": len(cdc["features"]),
        },
        "Hospital": {
            "records": int(len(hospital["y"])),
            "task": "30-day readmission label",
            "features": len(hospital["features"]),
        },
    },
    "limitations": [
        "No patient-level linkage between the two datasets.",
        "The shared classifier uses task-specific feature namespaces.",
        "Hospital data is historical (1999-2008).",
        "No independent external validation or clinical validation.",
        "Categorical feature encoding must be preserved for inference.",
    ],
    "elapsed_seconds": round(time.time() - START, 2),
}

report_path = OUT / "unified_training_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n" + "=" * 68)
print("UNIFIED MODEL TEST RESULTS")
print("=" * 68)
for task, metrics in test_results.items():
    print(f"\n{task}")
    for key in (
        "accuracy", "precision", "recall", "f1",
        "average_precision", "roc_auc", "threshold",
        "records", "confusion_matrix_tn_fp_fn_tp"
    ):
        print(f"  {key}: {metrics[key]}")

print("\nSaved ONE unified model:", model_path)
print("Report:", report_path)
print("Elapsed seconds:", report["elapsed_seconds"])
print("\nDo not describe this as clinically validated.")
