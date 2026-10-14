"""
backend/app.py
LinkShield AI - Flask Backend Service

This service provides real-time URL risk analysis by serving the trained
machine learning model from ml/model.joblib.

Endpoints:
  GET  /api/health - Service health check and model loading status.
  POST /api/scan   - Analyze a URL string and return classification and risk indicators.
"""

import os
import sys
from typing import Dict, Any, List
from flask import Flask, request, jsonify, render_template

# Ensure project root is in sys.path so ml module can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.feature_extractor import (
    load_model,
    predict_url,
    extract_features,
    FEATURE_NAMES,
)

# Initialize Flask application
app = Flask(__name__)

# Model configuration
MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "model.joblib")

# Global model state loaded once at startup
MODEL = None
FEATURE_COLUMNS = None
MODEL_LOAD_ERROR = None

try:
    MODEL, FEATURE_COLUMNS = load_model(MODEL_PATH)
    print(f"[INFO] Successfully loaded ML model from: {MODEL_PATH}")
    print(f"[INFO] Feature schema contains {len(FEATURE_COLUMNS)} features.")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"[ERROR] Failed to load model artifact: {e}", file=sys.stderr)


def derive_risk_indicators(features: Dict[str, Any]) -> List[str]:
    """
    Derives clear, human-understandable risk indicators from the static features.
    Does not visit or fetch the URL.
    """
    indicators: List[str] = []

    # 1. Raw IP address check
    if features.get("is_ip") == 1:
        indicators.append("Uses a raw IP address instead of a standard domain name.")

    # 2. Obfuscation check (@ symbol)
    if features.get("num_at", 0) > 0:
        indicators.append("Contains '@' symbol (often used to obscure destination hostnames).")

    # 3. Encryption check
    if features.get("is_https", 0) == 0:
        indicators.append("Does not use HTTPS encryption (insecure HTTP or missing scheme).")

    # 4. Phishing keywords check
    keywords = [
        "login", "verify", "update", "secure", "account",
        "banking", "signin", "confirm", "auth"
    ]
    detected_kws = [kw for kw in keywords if features.get(f"kw_{kw}", 0) == 1]
    if detected_kws:
        indicators.append(
            f"Contains high-risk phishing keywords ({len(detected_kws)} detected: {', '.join(detected_kws)})."
        )

    # 5. Excessive URL length
    if features.get("url_length", 0) > 75:
        indicators.append(f"Unusually long URL length ({features.get('url_length')} characters).")

    # 6. Typosquatting / hyphen check
    if features.get("num_hyphens", 0) > 3:
        indicators.append(
            f"Excessive use of hyphens ({features.get('num_hyphens')} detected, typical of typosquatting domains)."
        )

    # 7. Deep subdomains check
    if features.get("num_subdomains", 0) > 2:
        indicators.append(
            f"Unusually high number of subdomains ({features.get('num_subdomains')} detected)."
        )

    # 8. Percent encoding
    if features.get("has_url_encoding", 0) == 1:
        encoded_count = features.get("num_encoded_chars", 0)
        indicators.append(
            f"Contains URL percent-encoding ({encoded_count} encoded sequence{'s' if encoded_count > 1 else ''})."
        )

    # If no risk heuristics were triggered
    if not indicators:
        indicators.append("No prominent structural or lexical threat indicators detected.")

    return indicators


@app.after_request
def apply_cors_headers(response):
    """
    Enables CORS restricted to local development origins.
    """
    origin = request.headers.get("Origin")
    # Restrict CORS to standard local development origins
    if origin and (
        origin.startswith("http://localhost:")
        or origin.startswith("http://127.0.0.1:")
    ):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint: serves the premium cybersecurity dashboard for web browsers,
    or returns a JSON API overview if requested with Accept: application/json.
    """
    accept_header = request.headers.get("Accept", "")
    if accept_header.startswith("application/json") and "text/html" not in accept_header:
        return jsonify({
            "service": "LinkShield AI API",
            "status": "online",
            "model_loaded": MODEL is not None,
            "endpoints": {
                "health": "/api/health",
                "scan": "/api/scan (POST)"
            },
            "usage": {
                "method": "POST",
                "url": "/api/scan",
                "headers": {"Content-Type": "application/json"},
                "body": {"url": "https://example.com"}
            }
        }), 200

    return render_template("index.html")


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Returns service health and model loading status.
    """
    is_healthy = MODEL is not None
    response_data = {
        "status": "ok" if is_healthy else "degraded",
        "service": "LinkShield AI API",
        "model_loaded": is_healthy,
        "feature_count": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
        "supported_classes": ["Safe", "Malicious"],
    }

    if not is_healthy and MODEL_LOAD_ERROR:
        response_data["error"] = MODEL_LOAD_ERROR
        return jsonify(response_data), 503

    return jsonify(response_data), 200


@app.route("/api/scan", methods=["GET"])
@app.route("/scan", methods=["GET"])
def scan_info():
    """
    Friendly guide if a user navigates to /api/scan via browser GET.
    """
    accept_header = request.headers.get("Accept", "")
    if "text/html" in accept_header:
        return render_template("index.html")

    return jsonify({
        "error": "Method Not Allowed",
        "message": "The scan endpoint expects a POST request with a JSON body.",
        "example_request": {
            "method": "POST",
            "url": "/api/scan",
            "headers": {"Content-Type": "application/json"},
            "body": {"url": "https://example.com"}
        }
    }), 405


@app.route("/scan", methods=["POST", "OPTIONS"])
@app.route("/api/scan", methods=["POST", "OPTIONS"])
def scan_url():
    """
    URL scanning endpoint.
    Accepts JSON: {"url": "https://example.com"}
    Returns predicted label ('Safe' or 'Malicious'), risk score, and feature-derived risk indicators.
    """
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return ("", 204)

    # Check if model is ready
    if MODEL is None:
        return (
            jsonify(
                {
                    "error": "ML model is not loaded. Please ensure ml/model.joblib exists.",
                    "details": MODEL_LOAD_ERROR,
                }
            ),
            503,
        )

    # Validate that request content type is JSON
    if not request.is_json:
        return (
            jsonify(
                {
                    "error": "Invalid Content-Type. Request body must be valid JSON with 'Content-Type: application/json'."
                }
            ),
            400,
        )

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return (
            jsonify(
                {
                    "error": "Malformed JSON payload. Expected a JSON object with a 'url' property."
                }
            ),
            400,
        )

    # Validate URL field
    url_input = data.get("url")
    if url_input is None:
        return (
            jsonify({"error": "Missing required field 'url' in JSON body."}),
            400,
        )

    if not isinstance(url_input, str):
        return (
            jsonify({"error": "Field 'url' must be a string."}),
            400,
        )

    url_clean = url_input.strip()
    if not url_clean:
        return (
            jsonify({"error": "Field 'url' cannot be empty or whitespace only."}),
            400,
        )

    # Guard against unreasonable input length (DoS protection)
    if len(url_clean) > 4096:
        return (
            jsonify({"error": "URL length exceeds the maximum allowed limit of 4096 characters."}),
            400,
        )

    try:
        # Run ML model prediction via ml/feature_extractor.py helper
        inference_result = predict_url(url_clean, MODEL, FEATURE_COLUMNS)

        raw_prediction = inference_result["prediction"]
        malicious_prob = inference_result["malicious_probability"]
        extracted_features = inference_result["features"]

        # Map binary class strictly to Safe (0) or Malicious (1)
        label = "Malicious" if raw_prediction == 1 else "Safe"
        risk_score = round(malicious_prob * 100, 2)

        # Derive understandable risk indicators
        risk_indicators = derive_risk_indicators(extracted_features)

        response_payload = {
            "url": url_clean,
            "prediction": raw_prediction,
            "label": label,
            "risk_score": risk_score,
            "malicious_probability": malicious_prob,
            "risk_indicators": risk_indicators,
            "features": extracted_features,
        }

        return jsonify(response_payload), 200

    except Exception as e:
        # Return honest error rather than fabricated prediction
        return (
            jsonify(
                {
                    "error": "Failed to analyze URL due to an internal error.",
                    "details": str(e),
                }
            ),
            500,
        )


@app.errorhandler(404)
def resource_not_found(e):
    """
    Returns structured JSON with guidance when an invalid URL is accessed.
    """
    return (
        jsonify(
            {
                "error": "Not Found",
                "message": "The requested URL was not found on the server.",
                "available_endpoints": [
                    {"method": "GET", "path": "/api/health", "description": "Health check & model status"},
                    {"method": "POST", "path": "/api/scan", "description": "Scan URL with JSON body: {'url': '...'}"},
                ],
            }
        ),
        404,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting LinkShield AI Flask API on port {port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
