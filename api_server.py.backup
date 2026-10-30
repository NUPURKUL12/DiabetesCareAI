import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent

READMISSION_MODEL_PATH = (
    BASE / "readmission_fair_comparison" / "selected_model.joblib"
)
RISK_TIERS_PATH = (
    BASE / "readmission_fair_comparison" / "risk_tier_analysis.json"
)
CDC_MODEL_PATH = (
    BASE / "external_data" / "cdc_diabetes" / "model_results"
    / "cdc_diabetes_model.joblib"
)
CDC_FEATURES_PATH = (
    BASE / "external_data" / "cdc_diabetes" / "model_results"
    / "feature_columns.json"
)
CDC_METRICS_PATH = (
    BASE / "external_data" / "cdc_diabetes" / "model_results"
    / "cdc_diabetes_metrics.json"
)

PORT = 8765


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_models():
    readmission_bundle = joblib.load(READMISSION_MODEL_PATH)
    tiers = load_json(RISK_TIERS_PATH)
    cdc_model = joblib.load(CDC_MODEL_PATH)
    cdc_features = load_json(CDC_FEATURES_PATH)
    cdc_metrics = load_json(CDC_METRICS_PATH)

    return {
        "readmission_model": readmission_bundle["model"],
        "readmission_features": readmission_bundle["feature_columns"],
        "low_medium_cutoff": tiers["validation_cutoffs"]["low_to_medium"],
        "medium_high_cutoff": tiers["validation_cutoffs"]["medium_to_high"],
        "cdc_model": cdc_model,
        "cdc_features": cdc_features,
        "cdc_threshold": cdc_metrics["threshold_selected_on_validation"],
    }


MODELS = load_models()


def validate_input(data, features):
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [feature for feature in features if feature not in data]
    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))

    row = {feature: data[feature] for feature in features}

    for name, value in row.items():
        if value is None or value == "":
            raise ValueError(f"Feature '{name}' cannot be empty.")

        if isinstance(value, (dict, list)):
            raise ValueError(f"Feature '{name}' must be a single value.")

    return pd.DataFrame([row], columns=features)


def predict_readmission(data):
    features = MODELS["readmission_features"]
    row = validate_input(data, features)

    probability = float(
        MODELS["readmission_model"].predict_proba(row)[0, 1]
    )

    if probability < MODELS["low_medium_cutoff"]:
        tier = "Low"
    elif probability < MODELS["medium_high_cutoff"]:
        tier = "Medium"
    else:
        tier = "High"

    return {
        "task": "Historical readmission within 30 days",
        "predicted_score": round(probability, 4),
        "risk_tier": tier,
        "tier_cutoffs": {
            "low_to_medium": round(MODELS["low_medium_cutoff"], 4),
            "medium_to_high": round(MODELS["medium_high_cutoff"], 4),
        },
        "warning": (
            "Research prototype only. The score is not clinically calibrated. "
            "The model uses historical 1999-2008 hospital data and requires "
            "external validation. This is not medical advice."
        ),
    }


def predict_diabetes_status(data):
    features = MODELS["cdc_features"]
    row = validate_input(data, features)

    probability = float(
        MODELS["cdc_model"].predict_proba(row)[0, 1]
    )
    threshold = MODELS["cdc_threshold"]

    return {
        "task": "CDC survey-based diabetes-status classification",
        "predicted_score": round(probability, 4),
        "threshold": round(threshold, 4),
        "predicted_class": int(probability >= threshold),
        "class_meaning": {
            "0": "Survey target class 0",
            "1": "Survey target class 1",
        },
        "warning": (
            "Survey-based research classifier, not a medical diagnosis "
            "or prediction of future diabetes onset. External validation "
            "is required."
        ),
    }


class APIHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "readmission_model_loaded": True,
                "cdc_diabetes_model_loaded": True,
                "endpoints": [
                    "GET /health",
                    "POST /api/readmission",
                    "POST /api/diabetes-status",
                ],
            })
        else:
            self.send_json(404, {"error": "Endpoint not found."})

    def do_POST(self):
        routes = {
            "/api/readmission": predict_readmission,
            "/api/diabetes-status": predict_diabetes_status,
        }

        if self.path not in routes:
            self.send_json(404, {"error": "Endpoint not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                self.send_json(400, {
                    "error": "Provide a non-empty JSON body under 1 MB."
                })
                return

            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            result = routes[self.path](data)
            self.send_json(200, result)

        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "Invalid JSON request body."})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {
                "error": (
                    "Prediction failed. Check the input types and server "
                    "console for details."
                )
            })
            raise

    def log_message(self, format_string, *args):
        print("[API]", format_string % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), APIHandler)
    print(f"DiabetesCare AI API running at http://127.0.0.1:{PORT}")
    print("Health check: http://127.0.0.1:8765/health")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API...")
    finally:
        server.server_close()

