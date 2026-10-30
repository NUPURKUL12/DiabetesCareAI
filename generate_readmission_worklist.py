import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(r"D:\DiabetesCareAI")
DATA_PATH = BASE / "data" / "diabetic_data.csv"
MODEL_PATH = BASE / "readmission_fair_comparison" / "selected_model.joblib"
TIER_PATH = BASE / "readmission_fair_comparison" / "risk_tier_analysis.json"
METRICS_PATH = BASE / "readmission_fair_comparison" / "selected_test_metrics.json"

OUT_DIR = BASE / "readmission_fair_comparison"
WORKLIST_PATH = OUT_DIR / "retrospective_patient_worklist.csv"
SUMMARY_PATH = OUT_DIR / "worklist_summary.json"

print("=" * 70)
print("GENERATING RETROSPECTIVE READMISSION WORKLIST")
print("=" * 70)

# Load the exact selected model and its expected features.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["feature_columns"]

# Load the same source data and apply the original label filtering.
df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False)
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df.reset_index(drop=True, inplace=True)

y = (df["readmitted"] == "<30").astype(int)
groups = df["patient_nbr"].astype(str)

# Reproduce the original patient-grouped 60/20/20 split.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=101
)
trainval_idx, test_idx = next(split1.split(df, y, groups))

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=102
)
train_rel, val_rel = next(
    split2.split(
        df.iloc[trainval_idx],
        y.iloc[trainval_idx],
        groups.iloc[trainval_idx]
    )
)

train_idx = trainval_idx[train_rel]
val_idx = trainval_idx[val_rel]

# Confirm no patient appears in more than one split.
train_patients = set(groups.iloc[train_idx])
val_patients = set(groups.iloc[val_idx])
test_patients = set(groups.iloc[test_idx])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

print("Patient-group separation verified.")

# Recalculate the tier cutoffs from validation predictions.
val_prob = model.predict_proba(df.iloc[val_idx][features])[:, 1]
low_cut, high_cut = np.quantile(val_prob, [1/3, 2/3])

# Score the original held-out test encounters.
test = df.iloc[test_idx].copy()
test_scores = model.predict_proba(test[features])[:, 1]

def assign_tier(score):
    if score < low_cut:
        return "Low"
    if score < high_cut:
        return "Medium"
    return "High"

tiers = [assign_tier(score) for score in test_scores]

# Build an anonymized worklist. Do not include patient IDs,
# encounter IDs, or individual actual readmission outcomes.
worklist = pd.DataFrame({
    "predicted_score": test_scores,
    "risk_tier": tiers,
})

# Add useful encounter-level context.
worklist["age_group"] = test["age"].fillna("Not recorded").astype(str).to_numpy()
worklist["length_of_stay_days"] = test["time_in_hospital"].to_numpy()
worklist["lab_procedures"] = test["num_lab_procedures"].to_numpy()
worklist["procedures"] = test["num_procedures"].to_numpy()
worklist["medication_count"] = test["num_medications"].to_numpy()
worklist["prior_outpatient_visits"] = test["number_outpatient"].to_numpy()
worklist["prior_emergency_visits"] = test["number_emergency"].to_numpy()
worklist["prior_inpatient_visits"] = test["number_inpatient"].to_numpy()
worklist["diagnosis_count"] = test["number_diagnoses"].to_numpy()
worklist["max_glucose_result"] = (
    test["max_glu_serum"].fillna("Not recorded").astype(str).to_numpy()
)
worklist["a1c_result"] = (
    test["A1Cresult"].fillna("Not recorded").astype(str).to_numpy()
)

# Sort High first, then by descending model score.
priority_order = {"High": 1, "Medium": 2, "Low": 3}
worklist["_priority_order"] = worklist["risk_tier"].map(priority_order)

worklist = worklist.sort_values(
    ["_priority_order", "predicted_score"],
    ascending=[True, False]
).reset_index(drop=True)

worklist.insert(
    0,
    "worklist_reference",
    [f"ENC-{i:06d}" for i in range(1, len(worklist) + 1)]
)

worklist["priority"] = worklist["risk_tier"].map({
    "High": "Priority 1",
    "Medium": "Priority 2",
    "Low": "Priority 3",
})

worklist.drop(columns=["_priority_order"], inplace=True)
worklist["predicted_score"] = worklist["predicted_score"].round(6)

# Verify against the already saved evaluation results.
saved_tiers = json.loads(TIER_PATH.read_text(encoding="utf-8"))
saved_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

tier_summary = []
for name in ["Low", "Medium", "High"]:
    part = worklist[worklist["risk_tier"] == name]
    original = next(
        item for item in saved_tiers["test_results_by_tier"]
        if item["risk_tier"] == name
    )

    assert len(part) == original["encounters"]

    tier_summary.append({
        "risk_tier": name,
        "encounters": int(len(part)),
        "mean_predicted_score": float(part["predicted_score"].mean()),
        "reference_observed_readmission_rate": original[
            "observed_readmission_rate"
        ],
    })

assert len(worklist) == saved_metrics["test_encounters"]
assert test["patient_nbr"].nunique() == saved_metrics["test_distinct_patients"]

# Save the anonymized worklist.
worklist.to_csv(WORKLIST_PATH, index=False, encoding="utf-8-sig")

summary = {
    "status": "success",
    "purpose": "Retrospective held-out test-set prioritization demonstration",
    "source_dataset": "UCI Diabetes 130-US Hospitals, 1999-2008",
    "model": saved_metrics["selected_model"],
    "test_encounters": int(len(worklist)),
    "distinct_test_patients": int(test["patient_nbr"].nunique()),
    "validation_tier_cutoffs": {
        "low_to_medium": float(low_cut),
        "medium_to_high": float(high_cut),
    },
    "tier_summary": tier_summary,
    "worklist_file": str(WORKLIST_PATH),
    "limitations": [
        "This is a retrospective research demonstration, not a live patient worklist.",
        "Individual patient and encounter identifiers are not included.",
        "Individual actual readmission outcomes are not included.",
        "Risk scores are not clinically calibrated probabilities.",
        "Tier boundaries are relative validation-set tertiles, not medical decision thresholds.",
        "Some model features may only be available at or near discharge; prediction-time availability must be assessed.",
        "The source data is historical and requires external validation.",
    ],
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nWORKLIST GENERATED SUCCESSFULLY")
print("Total encounters:", len(worklist))
print("Distinct patients:", test["patient_nbr"].nunique())

print("\nRISK TIER COUNTS")
print(worklist["risk_tier"].value_counts().reindex(
    ["High", "Medium", "Low"]
).to_string())

print("\nFIRST 10 WORKLIST ENTRIES")
print(worklist.head(10).to_string(index=False))

print("\nSAVED FILES")
print(WORKLIST_PATH)
print(SUMMARY_PATH)

print("\nAll verification checks passed.")
