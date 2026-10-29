import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(r"D:\DiabetesCareAI")
DATA = BASE / "data" / "diabetic_data.csv"
MODEL_PATH = BASE / "readmission_fair_comparison" / "selected_model.joblib"
OUT = BASE / "readmission_fair_comparison"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["feature_columns"]

df = pd.read_csv(DATA, na_values=["?"], low_memory=False)
df = df[df["readmitted"].isin(["<30", "NO", ">30"])].copy()
df.reset_index(drop=True, inplace=True)

y = (df["readmitted"] == "<30").astype(int)
groups = df["patient_nbr"].astype(str)

split1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=101)
trainval_idx, test_idx = next(split1.split(df, y, groups))

split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=102)
train_rel, val_rel = next(
    split2.split(
        df.iloc[trainval_idx],
        y.iloc[trainval_idx],
        groups.iloc[trainval_idx]
    )
)
val_idx = trainval_idx[val_rel]

val_prob = model.predict_proba(df.iloc[val_idx][features])[:, 1]
test_prob = model.predict_proba(df.iloc[test_idx][features])[:, 1]

# Relative tiers: validation-score tertiles, not clinical cutoffs.
low_cut, high_cut = np.quantile(val_prob, [1/3, 2/3])

def tier(p):
    if p < low_cut:
        return "Low"
    if p < high_cut:
        return "Medium"
    return "High"

test = pd.DataFrame({
    "predicted_probability": test_prob,
    "actual_readmission_within_30_days": y.iloc[test_idx].to_numpy(),
})
test["risk_tier"] = [tier(p) for p in test["predicted_probability"]]

summary = []
for name in ["Low", "Medium", "High"]:
    part = test[test["risk_tier"] == name]
    summary.append({
        "risk_tier": name,
        "encounters": int(len(part)),
        "mean_predicted_probability": float(part["predicted_probability"].mean()),
        "observed_readmission_rate": float(
            part["actual_readmission_within_30_days"].mean()
        ),
        "actual_readmissions": int(
            part["actual_readmission_within_30_days"].sum()
        ),
    })

result = {
    "task": "Historical readmission within 30 days",
    "tier_method": "Validation-set predicted probability tertiles",
    "validation_cutoffs": {
        "low_to_medium": float(low_cut),
        "medium_to_high": float(high_cut),
    },
    "test_results_by_tier": summary,
    "warning": (
        "Relative research-prototype tiers only. Probabilities are not "
        "clinically calibrated; thresholds are not medical decision rules. "
        "Historical 1999-2008 data; external validation is required."
    ),
}

(OUT / "risk_tier_analysis.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
pd.DataFrame(summary).to_csv(OUT / "risk_tier_analysis.csv", index=False)

print(json.dumps(result, indent=2))
print("\nSaved:")
print(OUT / "risk_tier_analysis.json")
print(OUT / "risk_tier_analysis.csv")
