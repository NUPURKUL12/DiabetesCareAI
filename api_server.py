import csv
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent

READMISSION_MODEL_PATH = (
    BASE / "readmission_fair_comparison" / "selected_model.joblib"
)
READMISSION_METRICS_PATH = (
    BASE / "readmission_fair_comparison" / "selected_test_metrics.json"
)
RISK_TIERS_PATH = (
    BASE / "readmission_fair_comparison" / "risk_tier_analysis.json"
)
WORKLIST_PATH = (
    BASE / "readmission_fair_comparison" / "retrospective_patient_worklist.csv"
)
WORKLIST_SUMMARY_PATH = (
    BASE / "readmission_fair_comparison" / "worklist_summary.json"
)
INTERVENTIONS_PATH = (
    BASE / "readmission_fair_comparison" / "interventions.json"
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

ALLOWED_STATUSES = {
    "Not reviewed",
    "Reviewed",
    "Follow-up planned",
    "Completed",
}

INTERVENTION_LOCK = Lock()


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


def load_worklist():
    if not WORKLIST_PATH.exists():
        raise FileNotFoundError(
            f"Worklist file not found: {WORKLIST_PATH}"
        )

    with WORKLIST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {
        "worklist_reference",
        "predicted_score",
        "risk_tier",
        "priority",
    }

    if not rows:
        raise ValueError("The retrospective worklist is empty.")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            "Worklist is missing required columns: "
            + ", ".join(sorted(missing))
        )

    references = [row["worklist_reference"] for row in rows]

    if len(references) != len(set(references)):
        raise ValueError("Duplicate worklist references were found.")

    for row in rows:
        row["predicted_score"] = float(row["predicted_score"])

        for field in (
            "length_of_stay_days",
            "lab_procedures",
            "procedures",
            "medication_count",
            "prior_outpatient_visits",
            "prior_emergency_visits",
            "prior_inpatient_visits",
            "diagnosis_count",
        ):
            if field in row:
                row[field] = int(row[field])

    return rows


def load_interventions():
    if not INTERVENTIONS_PATH.exists():
        return {}

    data = load_json(INTERVENTIONS_PATH)

    if not isinstance(data, dict):
        raise ValueError(
            "interventions.json must contain a JSON object."
        )

    return data


MODELS = load_models()
WORKLIST = load_worklist()
WORKLIST_BY_REFERENCE = {
    row["worklist_reference"]: row for row in WORKLIST
}
WORKLIST_COUNTS = {
    tier: sum(1 for row in WORKLIST if row["risk_tier"] == tier)
    for tier in ("High", "Medium", "Low")
}
INTERVENTIONS = load_interventions()


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


def get_metrics():
    readmission = load_json(READMISSION_METRICS_PATH)
    tiers = load_json(RISK_TIERS_PATH)
    cdc = load_json(CDC_METRICS_PATH)

    return {
        "status": "success",
        "readmission": {
            "task": "Historical readmission within 30 days",
            "dataset": "UCI Diabetes 130-US Hospitals (1999-2008)",
            "evaluation": "Patient-grouped held-out test set",
            "metrics": readmission,
            "risk_tiers": tiers,
        },
        "diabetes_status": {
            "task": "CDC survey-based diabetes-status classification",
            "dataset": "CDC Diabetes Health Indicators",
            "evaluation": "Stratified held-out test split",
            "metrics": cdc,
        },
        "data_coverage": [
            {
                "category": "Claims data",
                "status": "Not available",
                "details": (
                    "No comprehensive claims dataset. Payer codes and "
                    "utilization counts are not equivalent to claims."
                ),
            },
            {
                "category": "Electronic Health Records",
                "status": "Partial",
                "details": (
                    "Historical hospital encounter records, diagnoses, "
                    "procedures, medications and laboratory indicators. "
                    "Not a complete longitudinal EHR."
                ),
            },
            {
                "category": "Social Determinants of Health",
                "status": "Partial proxies",
                "details": (
                    "CDC survey includes income, education, healthcare "
                    "access and cost-related access indicators. "
                    "Not a comprehensive SDOH dataset."
                ),
            },
            {
                "category": "Behavioral and lifestyle data",
                "status": "Available in CDC survey",
                "details": (
                    "Includes physical activity, smoking, fruit and "
                    "vegetable consumption, heavy alcohol consumption "
                    "and health indicators."
                ),
            },
        ],
        "integration_limitations": [
            "The hospital and CDC datasets are independent and are not linked at patient level.",
            "The two models perform different prediction tasks and must not be combined into one patient risk score.",
            "Readmission scores are not clinically calibrated probabilities.",
            "Both models require external validation before any real-world clinical use.",
        ],
    }


def get_worklist(query):
    risk_tier = query.get("risk_tier", ["All"])[0]
    search_text = query.get("q", [""])[0].strip().lower()

    try:
        limit = int(query.get("limit", ["50"])[0])
        offset = int(query.get("offset", ["0"])[0])
    except ValueError:
        raise ValueError("limit and offset must be integers.")

    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200.")

    if offset < 0:
        raise ValueError("offset cannot be negative.")

    if risk_tier not in {"All", "High", "Medium", "Low"}:
        raise ValueError("risk_tier must be All, High, Medium or Low.")

    filtered = WORKLIST

    if risk_tier != "All":
        filtered = [
            row for row in filtered
            if row["risk_tier"] == risk_tier
        ]

    if search_text:
        filtered = [
            row for row in filtered
            if search_text in " ".join(str(value) for value in row.values()).lower()
        ]

    total = len(filtered)
    page = filtered[offset:offset + limit]

    return {
        "status": "success",
        "purpose": "Retrospective held-out test-set prioritization demonstration",
        "total": total,
        "limit": limit,
        "offset": offset,
        "risk_tier_filter": risk_tier,
        "tier_counts": WORKLIST_COUNTS,
        "records": page,
        "warning": (
            "Historical retrospective research worklist, not a live patient "
            "list or a clinical triage tool. Scores are not calibrated "
            "probabilities. Individual outcomes are not provided."
        ),
    }


def save_interventions():
    temporary_path = INTERVENTIONS_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(INTERVENTIONS, file, indent=2, ensure_ascii=False)
        file.write("\n")

    os.replace(temporary_path, INTERVENTIONS_PATH)


def get_interventions(query):
    reference = query.get("worklist_reference", [None])[0]

    with INTERVENTION_LOCK:
        if reference:
            if reference not in WORKLIST_BY_REFERENCE:
                raise ValueError("Unknown worklist reference.")

            record = INTERVENTIONS.get(reference)
            records = {reference: record} if record else {}
        else:
            records = dict(INTERVENTIONS)

    return {
        "status": "success",
        "count": len(records),
        "allowed_statuses": sorted(ALLOWED_STATUSES),
        "records": records,
    }


def update_intervention(data):
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    reference = data.get("worklist_reference")

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("worklist_reference is required.")

    reference = reference.strip()

    if reference not in WORKLIST_BY_REFERENCE:
        raise ValueError("Unknown worklist reference.")

    status = data.get("status")

    if status not in ALLOWED_STATUSES:
        raise ValueError(
            "status must be one of: " + ", ".join(sorted(ALLOWED_STATUSES))
        )

    follow_up_date = data.get("follow_up_date", "")
    if follow_up_date is None:
        follow_up_date = ""

    if not isinstance(follow_up_date, str):
        raise ValueError("follow_up_date must be a date string.")

    follow_up_date = follow_up_date.strip()

    if follow_up_date:
        try:
            datetime.strptime(follow_up_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("follow_up_date must use YYYY-MM-DD format.")

    notes = data.get("notes", "")

    if notes is None:
        notes = ""

    if not isinstance(notes, str):
        raise ValueError("notes must be text.")

    notes = notes.strip()

    if len(notes) > 1000:
        raise ValueError("notes must be 1000 characters or fewer.")

    record = {
        "worklist_reference": reference,
        "status": status,
        "follow_up_date": follow_up_date,
        "notes": notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    with INTERVENTION_LOCK:
        INTERVENTIONS[reference] = record
        save_interventions()

    return {
        "status": "success",
        "message": "Intervention record saved.",
        "record": record,
    }


class APIHandler(BaseHTTPRequestHandler):

    def send_json(self, status, payload):
        body = json.dumps(
            payload, indent=2, ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type", "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, OPTIONS"
        )
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length header.")

        if length <= 0 or length > 1_000_000:
            raise ValueError("Provide a non-empty JSON body under 1 MB.")

        raw = self.rfile.read(length)

        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("Invalid JSON request body.")

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/health":
            self.send_json(200, {
                "status": "ok",
                "readmission_model_loaded": True,
                "cdc_diabetes_model_loaded": True,
                "worklist_loaded": True,
                "worklist_records": len(WORKLIST),
                "endpoints": [
                    "GET /health",
                    "GET /api/metrics",
                    "GET /api/worklist",
                    "GET /api/interventions",
                    "POST /api/interventions",
                    "POST /api/readmission",
                    "POST /api/diabetes-status",
                ],
            })
            return

        if path == "/api/metrics":
            try:
                self.send_json(200, get_metrics())
            except Exception as exc:
                self.send_json(500, {
                    "error": "Unable to load evaluation metrics.",
                    "details": str(exc),
                })
            return

        if path == "/api/worklist":
            try:
                self.send_json(200, get_worklist(query))
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.send_json(500, {
                    "error": "Unable to load the retrospective worklist.",
                    "details": str(exc),
                })
            return

        if path == "/api/interventions":
            try:
                self.send_json(200, get_interventions(query))
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.send_json(500, {
                    "error": "Unable to load intervention records.",
                    "details": str(exc),
                })
            return

        self.send_json(404, {"error": "Endpoint not found."})

    def do_POST(self):
        path = urlsplit(self.path).path

        if path == "/api/interventions":
            try:
                data = self.read_json_body()
                result = update_intervention(data)
                self.send_json(200, result)
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.send_json(500, {
                    "error": "Unable to save intervention record.",
                    "details": str(exc),
                })
            return

        routes = {
            "/api/readmission": predict_readmission,
            "/api/diabetes-status": predict_diabetes_status,
        }

        if path not in routes:
            self.send_json(404, {"error": "Endpoint not found."})
            return

        try:
            data = self.read_json_body()
            result = routes[path](data)
            self.send_json(200, result)

        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})

        except Exception:
            self.send_json(500, {
                "error": (
                    "Prediction failed. Check the input types and "
                    "server console for details."
                )
            })
            raise

    def log_message(self, format_string, *args):
        print("[API]", format_string % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), APIHandler)

    print(f"DiabetesCare AI API running at http://127.0.0.1:{PORT}")
    print(f"Health check: http://127.0.0.1:{PORT}/health")
    print(f"Metrics: http://127.0.0.1:{PORT}/api/metrics")
    print(f"Worklist: http://127.0.0.1:{PORT}/api/worklist")
    print(f"Interventions: http://127.0.0.1:{PORT}/api/interventions")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API...")
    finally:
        server.server_close()
