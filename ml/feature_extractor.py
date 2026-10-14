"""
ml/feature_extractor.py
LinkShield AI - Malicious URL Feature Extractor

This module extracts deterministic lexical and structural numeric features
from URL strings for machine learning training and real-time inference.
It uses purely static parsing (urllib.parse, ipaddress, re) and NEVER
performs network requests or visits URLs.
"""

import os
import re
import ipaddress
import urllib.parse
from typing import Dict, List, Any, Tuple, Optional
import joblib
import pandas as pd


# Standard list of keywords commonly observed in phishing and credential-harvesting URLs
SUSPICIOUS_KEYWORDS: List[str] = [
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "banking",
    "signin",
    "confirm",
    "auth",
]

# Canonical list of feature names in deterministic order
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "is_https",
    "is_ip",
    "num_dots",
    "num_hyphens",
    "num_dots_hostname",
    "num_hyphens_hostname",
    "num_subdomains",
    "num_special_chars",
    "num_at",
    "num_questionmark",
    "num_equal",
    "num_slash",
    "num_ampersand",
    "num_underscore",
    "num_encoded_chars",
    "has_url_encoding",
    "kw_login",
    "kw_verify",
    "kw_update",
    "kw_secure",
    "kw_account",
    "kw_banking",
    "kw_signin",
    "kw_confirm",
    "kw_auth",
    "num_suspicious_keywords",
]


def extract_features(url: Any) -> Dict[str, int]:
    """
    Extracts deterministic numeric features from a URL string.

    Parameters:
        url (str or Any): The input URL string to analyze.

    Returns:
        Dict[str, int]: A dictionary mapping feature names to numeric values.
    """
    # Safe string conversion and sanitization
    if url is None:
        raw_url = ""
    elif isinstance(url, str):
        raw_url = url.strip()
    else:
        raw_url = str(url).strip()

    url_lower = raw_url.lower()

    # 1. URL Length
    url_length = len(raw_url)

    # 2. HTTPS indicator
    # Check whether the URL explicitly uses HTTPS scheme
    is_https = 1 if url_lower.startswith("https://") else 0

    # 3. Parse URL components safely
    # If the URL lacks a scheme (e.g. "example.com/path"), prepend "http://" for parsing
    if "://" in raw_url:
        parse_target = raw_url
    else:
        parse_target = f"http://{raw_url}"

    hostname = ""
    path = ""
    query = ""

    try:
        parsed = urllib.parse.urlsplit(parse_target)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        # Check scheme if not already flagged
        if not is_https and parsed.scheme.lower() == "https":
            is_https = 1
    except Exception:
        # Fallback for malformed URLs
        hostname = ""
        path = ""
        query = ""

    # Hostname, path, and query lengths
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    # 4. Count dots and hyphens
    num_dots = raw_url.count(".")
    num_hyphens = raw_url.count("-")
    num_dots_hostname = hostname.count(".")
    num_hyphens_hostname = hostname.count("-")

    # 5. IP Address Check
    # Malicious URLs often use raw IPv4 or IPv6 addresses instead of domain names
    is_ip = 0
    if hostname:
        # Strip potential port or brackets for IPv6
        clean_host = hostname.strip("[]")
        if ":" in clean_host and not clean_host.startswith("::"):
            clean_host = clean_host.split(":")[0]
        try:
            ipaddress.ip_address(clean_host)
            is_ip = 1
        except ValueError:
            # Fallback regex for standard IPv4
            if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", clean_host):
                is_ip = 1

    # 6. Number of subdomains
    # If hostname is an IP or empty, subdomain count is 0
    if is_ip or not hostname:
        num_subdomains = 0
    else:
        # Split hostname by dots, filtering empty tokens
        parts = [p for p in hostname.split(".") if p]
        # Check for common second-level domains (e.g. .co.uk, .com.au, .gov.in)
        if (
            len(parts) > 2
            and len(parts[-1]) == 2
            and parts[-2] in {"co", "com", "org", "net", "edu", "gov", "ac"}
        ):
            num_subdomains = max(0, len(parts) - 3)
        else:
            num_subdomains = max(0, len(parts) - 2)

    # 7. Special character indicators
    num_special_chars = len(re.findall(r"[^a-zA-Z0-9]", raw_url))
    num_at = raw_url.count("@")
    num_questionmark = raw_url.count("?")
    num_equal = raw_url.count("=")
    num_slash = raw_url.count("/")
    num_ampersand = raw_url.count("&")
    num_underscore = raw_url.count("_")

    # 8. URL encoding indicators (e.g. %20, %3F, %2F)
    encoded_matches = re.findall(r"%[0-9a-fA-F]{2}", raw_url)
    num_encoded_chars = len(encoded_matches)
    has_url_encoding = 1 if num_encoded_chars > 0 else 0

    # 9. Suspicious keyword indicators
    keyword_features: Dict[str, int] = {}
    total_suspicious = 0
    for kw in SUSPICIOUS_KEYWORDS:
        present = 1 if kw in url_lower else 0
        keyword_features[f"kw_{kw}"] = present
        total_suspicious += present

    # Assemble complete feature dictionary matching canonical FEATURE_NAMES order
    features: Dict[str, int] = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "is_https": is_https,
        "is_ip": is_ip,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_dots_hostname": num_dots_hostname,
        "num_hyphens_hostname": num_hyphens_hostname,
        "num_subdomains": num_subdomains,
        "num_special_chars": num_special_chars,
        "num_at": num_at,
        "num_questionmark": num_questionmark,
        "num_equal": num_equal,
        "num_slash": num_slash,
        "num_ampersand": num_ampersand,
        "num_underscore": num_underscore,
        "num_encoded_chars": num_encoded_chars,
        "has_url_encoding": has_url_encoding,
        **keyword_features,
        "num_suspicious_keywords": total_suspicious,
    }

    return features


def extract_feature_vector(url: Any) -> List[int]:
    """
    Extracts features for a URL and returns them as a list ordered
    according to FEATURE_NAMES.
    """
    feat_dict = extract_features(url)
    return [feat_dict[name] for name in FEATURE_NAMES]


def get_feature_names() -> List[str]:
    """Returns the ordered list of feature column names."""
    return list(FEATURE_NAMES)


def load_model(model_path: str = "ml/model.joblib") -> Tuple[Any, List[str]]:
    """
    Safely loads the trained model and its feature schema.
    Works whether the file contains a dictionary payload or a model instance.

    Returns:
        Tuple[model, feature_names]: The loaded scikit-learn model and list of feature names.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_names = artifact.get("feature_names", FEATURE_NAMES)
    else:
        model = artifact
        feature_names = list(getattr(model, "feature_names_in_", FEATURE_NAMES))

    return model, feature_names


def predict_url(url: str, model: Any, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Helper function for real-time inference (e.g. inside Flask backend).
    Extracts features, aligns schema, and runs model prediction.

    Returns:
        dict: {
            "url": str,
            "prediction": int (0 = Benign, 1 = Malicious),
            "label": str ("Benign" or "Malicious"),
            "malicious_probability": float,
            "features": dict
        }
    """
    if feature_names is None:
        feature_names = get_feature_names()

    raw_features = extract_features(url)
    df_features = pd.DataFrame([raw_features])[feature_names]

    prediction = int(model.predict(df_features)[0])
    probabilities = model.predict_proba(df_features)[0]
    # Class 1 is malicious
    malicious_prob = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)

    return {
        "url": url,
        "prediction": prediction,
        "label": "Malicious" if prediction == 1 else "Benign",
        "malicious_probability": round(malicious_prob, 4),
        "features": raw_features,
    }
