import re
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def extract_features(url):
    """Extracts lexical and structural features from a given URL."""
    features = {}
    
    # 1. Lexical metrics
    features['url_length'] = len(url)
    features['has_ip'] = 1 if re.search(r'(([0-9]{1,3}\.){3}[0-9]{1,3})', url) else 0
    features['count_dots'] = url.count('.')
    features['count_at'] = url.count('@')
    features['count_hyphen'] = url.count('-')
    features['count_slash'] = url.count('/')
    
    # 2. Suspicious keyword check
    keywords = ['login', 'verify', 'account', 'update', 'secure', 'banking', 'signin', 'confirm', 'auth']
    features['suspicious_kw'] = sum(1 for kw in keywords if kw in url.lower())
    
    return features

def evaluate_url(url):
    """Analyzes the URL, calculates a risk score, and generates human-readable warnings."""
    feats = extract_features(url)
    
    risk_score = 0
    reasons = []
    
    if feats['has_ip']:
        risk_score += 45
        reasons.append("⚠️ Uses a raw IP address instead of a standard domain name.")
    if feats['count_at'] > 0:
        risk_score += 30
        reasons.append("⚠️ Contains an '@' symbol (often used to obscure destination paths).")
    if feats['suspicious_kw'] > 0:
        risk_score += 35
        reasons.append(f"⚠️ Contains high-risk phishing keywords ({feats['suspicious_kw']} detected).")
    if feats['url_length'] > 75:
        risk_score += 15
        reasons.append("⚠️ Unusually long URL string length.")
    if feats['count_hyphen'] > 3:
        risk_score += 15
        reasons.append("⚠️ Excessive use of hyphens, typical of typosquatting domains.")

    # Cap risk score at 100
    risk_score = min(risk_score, 100)
    
    # Verdict determination
    if risk_score >= 50:
        status = "Malicious"
    elif risk_score > 0:
        status = "Suspicious"
    else:
        status = "Safe"
        reasons.append("✅ No obvious threat indicators found.")
        
    return {
        "url": url,
        "status": status,
        "risk_score": risk_score,
        "reasons": reasons
    }