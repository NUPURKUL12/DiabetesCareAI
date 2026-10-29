
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(r"D:\DiabetesCareAI")
OUT = BASE / "tuning"

df = pd.read_csv(
    BASE / "data" / "diabetic_data.csv",
    na_values=["?"],
    low_memory=False,
)
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df = df.reset_index(drop=True)

y = (df["readmitted"] == "<30").astype(int)

features = [
    "age", "gender", "race", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "max_glu_serum", "A1Cresult",
]

X = df[features].copy()
groups = df["patient_nbr"].astype(str)

# Reproduce the holdout test split used by tune_models.py.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=101
)
trainval_idx, test_idx = next(splitter.split(X, y, groups))

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
test_rows = df.iloc[test_idx]

model = joblib.load(OUT / "selected_model.joblib")

with open(OUT / "final_test_metrics.json", encoding="utf-8") as f:
    saved_metrics = json.load(f)

threshold = saved_metrics["threshold_selected_on_validation"]
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print(f"Fixed threshold: {threshold:.3f}")
print(f"Holdout encounters: {len(test_idx):,}")
print("\nOverall holdout metrics:")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")

results = []

# Use broad demographic categories already present in the dataset.
for column in ["age", "gender", "race"]:
    subgroup = test_rows[column].fillna("Unknown").astype(str)

    for category in sorted(subgroup.unique()):
        mask = subgroup.to_numpy() == category
        count = int(mask.sum())

        # Avoid drawing conclusions from very small groups.
        if count < 100:
            continue

        actual = y_test.iloc[mask]
        predicted = predictions[mask]
        positive_count = int(actual.sum())

        results.append({
            "grouping": column,
            "category": category,
            "encounters": count,
            "actual_readmission_rate": float(actual.mean()),
            "precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "recall": (
                float(recall_score(actual, predicted, zero_division=0))
                if positive_count else None
            ),
            "prediction_rate": float(predicted.mean()),
        })

report = pd.DataFrame(results)
report.to_csv(OUT / "subgroup_audit.csv", index=False)

print("\nSUBGROUP AUDIT")
print(report.to_string(
    index=False,
    float_format=lambda value: f"{value:.3f}"
))

print(f"\nSaved: {OUT / 'subgroup_audit.csv'}")
print(
    "\nResearch audit only. Differences between groups need investigation; "
    "they do not by themselves establish discrimination or causation."
)
