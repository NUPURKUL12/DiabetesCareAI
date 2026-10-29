from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(r"D:\DiabetesCareAI")
DATA = BASE / "data" / "diabetic_data.csv"
MODEL = BASE / "tuning" / "selected_model.joblib"
OUT = BASE / "tuning"

df = pd.read_csv(DATA, na_values=["?"], low_memory=False)
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
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

# Re-create the exact split used by tune_models.py.
first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=101)
trainval, test = next(first.split(X, y, groups))

second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=102)
train_rel, val_rel = next(
    second.split(
        X.iloc[trainval],
        y.iloc[trainval],
        groups.iloc[trainval],
    )
)
val_idx = trainval[val_rel]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

model = joblib.load(MODEL)

print("Calculating validation-set permutation importance...")
result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)

importance = pd.DataFrame({
    "feature": features,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)

importance.to_csv(OUT / "feature_importance.csv", index=False)

print("\nFeature importance (higher means more useful to this model):")
print(importance.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

print(f"\nSaved: {OUT / 'feature_importance.csv'}")
print(
    "\nInterpretation caution: importance measures predictive usefulness, "
    "not causation. Correlated features can share or mask importance."
)