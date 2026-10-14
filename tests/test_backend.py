"""
tests/test_backend.py
Unit and integration tests for LinkShield AI Flask API.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app


class LinkShieldBackendTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_health_check(self):
        """Test GET /api/health returns 200 and indicates model is loaded."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertIn("Safe", data["supported_classes"])
        self.assertIn("Malicious", data["supported_classes"])

    def test_scan_benign_url(self):
        """Test POST /api/scan with known benign domain."""
        payload = {"url": "https://bronze.systems"}
        response = self.client.post("/api/scan", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["label"], "Safe")
        self.assertEqual(data["prediction"], 0)
        self.assertLessEqual(data["risk_score"], 50.0)
        self.assertIn("risk_indicators", data)
        self.assertIn("features", data)

    def test_scan_malicious_ip_url(self):
        """Test POST /api/scan with known malicious raw IP payload URL."""
        payload = {"url": "http://110.37.26.193:54956/bin.sh"}
        response = self.client.post("/api/scan", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["label"], "Malicious")
        self.assertEqual(data["prediction"], 1)
        self.assertGreaterEqual(data["risk_score"], 50.0)
        # Check that raw IP was flagged in indicators
        ip_flagged = any("IP address" in ind for ind in data["risk_indicators"])
        self.assertTrue(ip_flagged)

    def test_scan_missing_url(self):
        """Test POST /api/scan returns 400 when url field is missing."""
        response = self.client.post("/api/scan", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_scan_empty_url(self):
        """Test POST /api/scan returns 400 when url is empty or whitespace."""
        response = self.client.post("/api/scan", json={"url": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_scan_non_json_request(self):
        """Test POST /api/scan returns 400 when request body is not JSON."""
        response = self.client.post(
            "/api/scan",
            data="not json",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)

    def test_cors_headers(self):
        """Test CORS headers for local development origin."""
        response = self.client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"},
        )
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:3000",
        )

    def test_root_endpoint(self):
        """Test GET / returns 200 with HTML for browser and JSON when requested."""
        # Browser request (default)
        response_html = self.client.get("/")
        self.assertEqual(response_html.status_code, 200)
        self.assertIn(b"LinkShield AI", response_html.data)

        # API client request (Accept: application/json)
        response_json = self.client.get("/", headers={"Accept": "application/json"})
        self.assertEqual(response_json.status_code, 200)
        data = response_json.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_scan_get_guidance(self):
        """Test GET /api/scan returns 405 with helpful guide."""
        response = self.client.get("/api/scan")
        self.assertEqual(response.status_code, 405)
        data = response.get_json()
        self.assertIn("example_request", data)

    def test_custom_404_handler(self):
        """Test accessing invalid URL returns structured 404 JSON with endpoint guide."""
        response = self.client.get("/invalid-route")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data["error"], "Not Found")
        self.assertIn("available_endpoints", data)


if __name__ == "__main__":
    unittest.main()
