import json
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(r"D:\DiabetesCareAI")
MODEL_PATH = BASE / "readmission_fair_comparison" / "selected_model.joblib"
TIERS_PATH = BASE / "readmission_fair_comparison" / "risk_tier_analysis.json"


def load_readmission_predictor():
    bundle = joblib.load(MODEL_PATH)
    tier_data = json.loads(TIERS_PATH.read_text(encoding="utf-8"))

    return {
        "model": bundle["model"],
        "features": bundle["feature_columns"],
        "low_medium_cutoff": tier_data["validation_cutoffs"]["low_to_medium"],
        "medium_high_cutoff": tier_data["validation_cutoffs"]["medium_to_high"],
    }


def predict_readmission(patient_data):
    """
    patient_data: dict containing the model's required 16 input features.

    Returns a research-only readmission score and provisional risk tier.
    This is not a clinical decision tool.
    """
    predictor = load_readmission_predictor()
    features = predictor["features"]
    missing = [name for name in features if name not in patient_data]

    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))

    row = pd.DataFrame(
        [{name: patient_data[name] for name in features}],
        columns=features,
    )

    probability = float(predictor["model"].predict_proba(row)[0, 1])

    if probability < predictor["low_medium_cutoff"]:
        tier = "Low"
    elif probability < predictor["medium_high_cutoff"]:
        tier = "Medium"
    else:
        tier = "High"

    return {
        "task": "Historical readmission within 30 days",
        "predicted_score": round(probability, 4),
        "risk_tier": tier,
        "tier_cutoffs": {
            "low_to_medium": round(predictor["low_medium_cutoff"], 4),
            "medium_to_high": round(predictor["medium_high_cutoff"], 4),
        },
        "warning": (
            "Research prototype only. Scores are not clinically calibrated. "
            "The model uses historical 1999-2008 hospital data and requires "
            "external validation before clinical use."
        ),
    }


if __name__ == "__main__":
    print("Readmission predictor loaded successfully.")
    predictor = load_readmission_predictor()
    print("Required input features:")
    print(json.dumps(predictor["features"], indent=2))
    print("Provisional tier cutoffs:")
    print(json.dumps({
        "low_to_medium": predictor["low_medium_cutoff"],
        "medium_to_high": predictor["medium_high_cutoff"],
    }, indent=2))
