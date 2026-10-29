import json
from pathlib import Path
import joblib

root = Path(r"D:\DiabetesCareAI")

artifacts = [
    ("Original readmission model", root / "tuning" / "selected_model.joblib"),
    ("Expanded readmission model", root / "expanded" / "expanded_model.joblib"),
    ("CDC diabetes model", root / "external_data" / "cdc_diabetes" / "model_results" / "cdc_diabetes_model.joblib"),
]

metrics = [
    root / "tuning" / "final_test_metrics.json",
    root / "expanded" / "expanded_metrics.json",
    root / "external_data" / "cdc_diabetes" / "model_results" / "cdc_diabetes_metrics.json",
]

for label, path in artifacts:
    print("\n" + "=" * 65)
    print(label)
    print("File:", path)
    if not path.exists():
        print("MISSING")
        continue

    model = joblib.load(path)
    print("Type:", type(model).__name__)
    print("Expected feature count:", getattr(model, "n_features_in_", "not exposed"))
    if hasattr(model, "feature_names_in_"):
        print("Features:", list(model.feature_names_in_))
    if hasattr(model, "named_steps"):
        print("Pipeline steps:", list(model.named_steps.keys()))
        for name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                print(name, "features:", list(step.feature_names_in_))

for path in metrics:
    print("\nMETRICS:", path)
    if path.exists():
        try:
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        except Exception as e:
            print("Read error:", e)
    else:
        print("MISSING")

print("\nInventory finished. Existing models were not modified.")
