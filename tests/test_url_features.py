"""
Unit Tests for Phase 2 URL Lexical Feature Extractor and Inference Engine
=========================================================================
File: tests/test_url_features.py
"""

import math
import os
import joblib
import numpy as np
import pandas as pd
import pytest

from url_features import (
    extract_url_features,
    extract_url_features_df,
    extract_hostname,
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    SHORTENING_DOMAINS
)

def test_feature_names_order_and_count():
    """Confirms extracted dictionary contains exactly the 26 keys in FEATURE_NAMES in exact order."""
    test_url = "https://example.com/login"
    features = extract_url_features(test_url)
    keys = list(features.keys())
    assert keys == FEATURE_NAMES, "Feature keys do not match FEATURE_NAMES sequence"
    assert len(keys) == 26, f"Expected 26 features, got {len(keys)}"

def test_empty_string():
    """Confirms empty string parses gracefully without throwing exceptions and returns valid finite numbers."""
    features = extract_url_features("")
    assert isinstance(features, dict)
    assert len(features) == 26
    for k, v in features.items():
        assert not math.isnan(v), f"Feature {k} is NaN on empty string"
        assert not math.isinf(v), f"Feature {k} is Inf on empty string"

def test_url_with_no_scheme():
    """Confirms URLs without explicit http/https scheme prefix are handled reliably."""
    url = "github.com/torvalds/linux"
    features = extract_url_features(url)
    assert features["hostname_length"] > 0
    assert features["count_slashes"] == 2
    assert features["is_https"] == 0

def test_ip_address_hostnames():
    """Confirms IPv4 and IPv6 hostnames trigger is_ip_address = 1."""
    ipv4_url = "http://192.168.1.1/admin/login.php"
    feat_ipv4 = extract_url_features(ipv4_url)
    assert feat_ipv4["is_ip_address"] == 1
    assert feat_ipv4["subdomain_count"] == 0

    ipv6_url = "http://[2001:db8::1]:8080/index.html"
    feat_ipv6 = extract_url_features(ipv6_url)
    assert feat_ipv6["is_ip_address"] == 1

def test_url_with_at_symbol():
    """Confirms '@' symbol credential obfuscation flag is caught."""
    clean_url = "https://google.com/search"
    phish_url = "http://google.com@evil-attacker.com/login"
    assert extract_url_features(clean_url)["has_at_symbol"] == 0
    assert extract_url_features(phish_url)["has_at_symbol"] == 1

def test_is_https_flag():
    """Confirms is_https differentiates HTTPS from unencrypted HTTP and other schemes."""
    https_url = "https://secure-banking.com/account"
    http_url = "http://secure-banking.com/account"
    assert extract_url_features(https_url)["is_https"] == 1
    assert extract_url_features(http_url)["is_https"] == 0

def test_known_shortening_domains():
    """Confirms known shortener services trigger is_shortener = 1."""
    short_urls = [
        "https://bit.ly/3xY9zQ",
        "http://tinyurl.com/phish-target",
        "https://t.co/abc1234",
        "https://is.gd/update"
    ]
    for su in short_urls:
        feat = extract_url_features(su)
        assert feat["is_shortener"] == 1, f"Failed to detect shortener on {su}"

    legit_url = "https://wikipedia.org/wiki/Phishing"
    assert extract_url_features(legit_url)["is_shortener"] == 0

def test_unicode_and_punycode_urls():
    """Confirms internationalized domain names (IDNs) and punycode strings parse safely."""
    puny_url = "https://xn--bcher-kva.example.de/b%C3%BCcher"
    feat = extract_url_features(puny_url)
    assert feat["url_length"] > 0
    assert not math.isnan(feat["entropy"])

def test_very_long_url():
    """Confirms extremely long URLs with large query strings parse without overflow or error."""
    long_path = "a" * 800
    long_query = "x=" + ("1234567890" * 50)
    long_url = f"https://example.com/{long_path}?{long_query}"
    feat = extract_url_features(long_url)
    assert feat["url_length"] == len(long_url)
    assert feat["path_length"] >= 800
    assert feat["count_digits"] >= 500
    assert feat["entropy"] > 0

def test_no_nan_or_inf_in_features():
    """Ensures all extracted features are strictly finite real numbers across diverse edge cases."""
    edge_cases = [
        "",
        "http://",
        "https://",
        "://",
        "???",
        "http://.",
        "https://127.0.0.1",
        "ftp://files.server.org/data",
        "https://user:password@domain.com:8080/path?q=1#frag"
    ]
    for url in edge_cases:
        feat = extract_url_features(url)
        for name, val in feat.items():
            assert isinstance(val, (int, float, np.number)), f"{name} not numeric on {url}"
            assert not math.isnan(val), f"{name} is NaN on {url}"
            assert not math.isinf(val), f"{name} is Inf on {url}"

def test_model_predict_proba_inference():
    """Confirms url_model.pkl loads and successfully produces probabilities with extracted feature vectors."""
    assert os.path.exists("url_model.pkl"), "url_model.pkl not found!"
    payload = joblib.load("url_model.pkl")
    model = payload["model"]

    test_urls = [
        "https://www.google.com",
        "http://paypal-verification-security-login.com/signin",
        "https://github.com/features"
    ]
    df_feat = extract_url_features_df(test_urls)
    assert df_feat.shape == (3, 26)

    probs = model.predict_proba(df_feat)
    assert probs.shape == (3, 2), "Prediction probabilities shape mismatch"
    for p in probs:
        assert 0.0 <= p[0] <= 1.0
        assert 0.0 <= p[1] <= 1.0
        assert pytest.approx(sum(p), 0.001) == 1.0


def test_www_normalization_equivalence():
    """Confirms www and non-www URLs of the same domain yield identical extracted features."""
    feat_www = extract_url_features("https://www.github.com")
    feat_apex = extract_url_features("https://github.com")
    assert feat_www == feat_apex, f"Features differ between www and non-www versions: {feat_www} != {feat_apex}"
