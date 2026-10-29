from pathlib import Path
import importlib.util
import joblib

base = Path(r"D:\DiabetesCareAI")

models = {
    "Readmission": base / "readmission_fair_comparison" / "selected_model.joblib",
    "CDC diabetes status": base / "external_data" / "cdc_diabetes" / "model_results" / "cdc_diabetes_model.joblib",
}

for name, path in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("Path:", path)
    if not path.exists():
        print("MODEL FILE NOT FOUND")
        continue

    bundle = joblib.load(path)
    print("Saved object type:", type(bundle).__name__)

    if isinstance(bundle, dict):
        print("Keys:", list(bundle.keys()))
        for key, value in bundle.items():
            if key in ("model", "classifier", "pipeline"):
                print(f"{key} type:", type(value).__name__)
            elif key in ("feature_columns", "features", "threshold", "selected_threshold"):
                print(f"{key}:", value)

print("\n" + "=" * 60)
print("Available web frameworks:")
for package in ("fastapi", "flask", "uvicorn", "joblib", "pandas", "sklearn"):
    print(f"{package}: {'available' if importlib.util.find_spec(package) else 'not installed'}")
