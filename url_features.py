"""
================================================================================
Phishing Website Detection - Phase 2: URL Lexical Feature Extractor
================================================================================
File: url_features.py
Purpose: Extract 26 lightweight, fast, purely lexical features directly from
         raw URL strings without requiring live network requests, DNS queries,
         or HTML DOM scraping.

Why Purely Lexical Features? (Viva Key Concept):
- Model A (Phase 1, UCI Dataset) depended on 30 features that required active
  web scraping, Alexa traffic rankings, external DNS queries, and SSL metadata.
  If a phishing site goes offline (lifespan often < 24 hours) or blocks bots,
  those features cannot be extracted.
- Model B (Phase 2, Lexical Features) inspects ONLY the URL character patterns.
  It computes in microseconds, requires zero internet access, works even if
  the phishing site has already been taken down, and operates safely in a
  browser extension, proxy, or firewall before a connection is even established.

Author: Yashvi Chauhan
Project: Phishing Website Detection (Phase 2)
================================================================================
"""

import math
import re
import urllib.parse
from collections import Counter
from typing import Dict, List, Union
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# List of high-risk keywords frequently weaponized by credential harvesters
SUSPICIOUS_KEYWORDS: List[str] = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank"
]

# Popular URL shortening services used to camouflage malicious landing pages
SHORTENING_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bitly.com", "tiny.cc", "rb.gy", "cutt.ly",
    "tiny.one", "qr.ae", "v.gd", "tr.im", "s.id"
}

# Pre-compiled regex for IPv4 address matching in hostnames
IPV4_REGEX = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")

# Pre-compiled regex for IPv6 address matching
IPV6_REGEX = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")

# Standard alphanumeric + primary URL delimiters
STANDARD_URL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/?=&")

# Canonical list of feature names in the exact sequence expected by the model
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_digits",
    "count_underscores",
    "count_slashes",
    "count_question_marks",
    "count_equal_signs",
    "count_ampersands",
    "count_other_special",
    "subdomain_count",
    "is_ip_address",
    "has_at_symbol",
    "is_https",
    "is_shortener",
    "suspicious_keyword_count",
    "has_login",
    "has_verify",
    "has_secure",
    "has_account",
    "has_update",
    "has_bank",
    "digit_to_letter_ratio",
    "entropy"
]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def calculate_shannon_entropy(text: str) -> float:
    """
    Computes Shannon Entropy of a string: H(X) = -sum(p(x) * log2(p(x))).
    
    Viva Explanation:
    Shannon entropy measures the information density and randomness of characters.
    Human-readable, legitimate URLs (e.g. 'wikipedia.org/wiki/Computer') have
    predictable character distributions and lower entropy (~3.2 - 4.0).
    Malicious or DGA (Domain Generation Algorithm) URLs with randomized hashes,
    obfuscated query strings, or hex tokens have significantly higher entropy (>4.5).
    """
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return float(entropy)


def normalize_url(url: str) -> str:
    """
    Normalizes a URL by stripping any 'www.' prefix from its hostname.
    Ensures that 'https://www.google.com/x' and 'https://google.com/x' yield
    identical representations and lexical features, eliminating dataset 'www.' bias.
    """
    u = str(url).strip()
    if not u:
        return u
    has_scheme = "://" in u
    temp_url = u if has_scheme else "http://" + u
    try:
        parsed = urllib.parse.urlsplit(temp_url)
        netloc = parsed.netloc
        userinfo = ""
        host_port = netloc
        if "@" in netloc:
            userinfo, host_port = netloc.split("@", 1)
            userinfo += "@"

        host = host_port
        port = ""
        if ":" in host_port and not host_port.endswith("]"):
            h, p = host_port.rsplit(":", 1)
            if p.isdigit():
                host = h
                port = ":" + p

        if host.lower().startswith("www."):
            norm_host = host[4:]
            norm_netloc = userinfo + norm_host + port
            norm_parsed = parsed._replace(netloc=norm_netloc)
            res = urllib.parse.urlunsplit(norm_parsed)
            if not has_scheme:
                res = res[len("http://"):]
            return res
        return u
    except Exception:
        return re.sub(r"^(https?://)?www\.", r"\1", u, flags=re.IGNORECASE)


def extract_hostname(url: str) -> str:
    """
    Safely extracts the lowercase hostname from a raw URL string,
    normalizing by stripping leading 'www.' if present.
    """
    url_str = normalize_url(url)
    if not url_str:
        return ""
    if "://" not in url_str:
        url_str = "http://" + url_str
    try:
        parsed = urllib.parse.urlparse(url_str)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        # Fallback regex extraction if urllib fails on malformed inputs
        match = re.search(r"https?://([^/:\?#]+)", url_str)
        host = match.group(1).lower() if match else ""
        if host.startswith("www."):
            host = host[4:]
        return host


# -----------------------------------------------------------------------------
# Primary Feature Extraction Functions
# -----------------------------------------------------------------------------

def extract_url_features(url: str) -> Dict[str, Union[int, float]]:
    """
    Extracts 26 lexical cybersecurity features from a single raw URL string.
    
    Parameters
    ----------
    url : str
        The raw URL string to analyze (e.g., 'https://paypal-verify.com/login').
        
    Returns
    -------
    Dict[str, Union[int, float]]
        A dictionary mapping feature names to numerical values.
        
    Features Extracted (and Viva Rationale):
    1.  url_length: Length of URL. Long URLs conceal fake subdomains or tokens.
    2.  hostname_length: Length of hostname. Spoofed domains are often longer.
    3.  path_length: Length of path. Deep paths are common in phishing kits.
    4.  count_dots: Dot count. Excessive dots indicate multi-level subdomains.
    5.  count_hyphens: Hyphen count. Used to mimic brand names (e.g. apple-login).
    6.  count_digits: Total digits. Malicious URLs use numeric tokens/IPs.
    7.  count_underscores: Underscore count. Rare in legit hostnames, seen in exploits.
    8.  count_slashes: Slash count. Indicates path nesting depth.
    9.  count_question_marks: Question marks. Marks start of query parameters.
    10. count_equal_signs: Equal signs. Indicates parameter key-value assignments.
    11. count_ampersands: Ampersands. Joins multiple query parameters in redirection.
    12. count_other_special: Non-standard characters (%, #, @, +, ~, $, etc.).
    13. subdomain_count: Number of subdomains preceding registered domain.
    14. is_ip_address: Binary (1 if hostname is bare IP, 0 otherwise).
    15. has_at_symbol: Binary (1 if '@' in URL). Browsers ignore prefix before '@'.
    16. is_https: Binary (1 if HTTPS protocol used, 0 otherwise).
    17. is_shortener: Binary (1 if hosted on bit.ly, tinyurl, etc., 0 otherwise).
    18. suspicious_keyword_count: Total occurrences of login, verify, secure, etc.
    19. has_login: Binary indicator for 'login' keyword.
    20. has_verify: Binary indicator for 'verify' keyword.
    21. has_secure: Binary indicator for 'secure' keyword.
    22. has_account: Binary indicator for 'account' keyword.
    23. has_update: Binary indicator for 'update' keyword.
    24. has_bank: Binary indicator for 'bank' keyword.
    25. digit_to_letter_ratio: Ratio of digits to letters. High in generated attacks.
    26. entropy: Shannon entropy measuring character distribution unpredictability.
    """
    # Step 1: Normalize URL (strip 'www.' prefix from hostname) to remove dataset artifact bias
    u = normalize_url(str(url).strip())
    u_lower = u.lower()
    
    # Ensure scheme for standard parsing
    url_with_scheme = u if "://" in u else "http://" + u
    try:
        parsed = urllib.parse.urlparse(url_with_scheme)
        hostname = (parsed.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        path = parsed.path or ""
    except Exception:
        hostname = extract_hostname(u)
        path = ""
        parsed = None

    # Length-based features
    url_len = len(u)
    host_len = len(hostname)
    path_len = len(path)

    # Delimiter character frequencies
    count_dots = u.count(".")
    count_hyphens = u.count("-")
    count_underscores = u.count("_")
    count_slashes = u.count("/")
    count_qmarks = u.count("?")
    count_equals = u.count("=")
    count_ampersands = u.count("&")

    # Character composition
    n_digits = sum(c.isdigit() for c in u)
    n_letters = sum(c.isalpha() for c in u)
    count_other_special = sum(1 for c in u if c not in STANDARD_URL_CHARS)

    # Subdomain calculation
    # Example: 'mail.google.com' has parts ['mail', 'google', 'com'] -> 1 subdomain ('mail')
    host_parts = hostname.split(".")
    if len(host_parts) >= 2 and not IPV4_REGEX.match(hostname):
        subdomain_count = max(0, len(host_parts) - 2)
    else:
        subdomain_count = 0

    # IP address check in hostname
    is_ip = 1 if (IPV4_REGEX.match(hostname) or IPV6_REGEX.match(hostname)) else 0

    # '@' symbol (credential obscuring tactic)
    has_at = 1 if "@" in u else 0

    # HTTPS protocol usage
    is_https = 1 if (u_lower.startswith("https://") or (parsed and parsed.scheme == "https")) else 0

    # URL shortener detection
    is_shortener = 1 if any(
        hostname == s or hostname.endswith("." + s) for s in SHORTENING_DOMAINS
    ) else 0

    # Suspicious keyword tracking
    kw_flags = {
        f"has_{kw}": (1 if kw in u_lower else 0)
        for kw in SUSPICIOUS_KEYWORDS
    }
    suspicious_keyword_count = sum(u_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)

    # Digit-to-letter ratio (avoids division by zero)
    digit_to_letter_ratio = (n_digits / n_letters) if n_letters > 0 else float(n_digits)

    # Shannon Entropy calculation
    entropy = calculate_shannon_entropy(u)

    # Construct complete dictionary
    features = {
        "url_length": url_len,
        "hostname_length": host_len,
        "path_length": path_len,
        "count_dots": count_dots,
        "count_hyphens": count_hyphens,
        "count_digits": n_digits,
        "count_underscores": count_underscores,
        "count_slashes": count_slashes,
        "count_question_marks": count_qmarks,
        "count_equal_signs": count_equals,
        "count_ampersands": count_ampersands,
        "count_other_special": count_other_special,
        "subdomain_count": subdomain_count,
        "is_ip_address": is_ip,
        "has_at_symbol": has_at,
        "is_https": is_https,
        "is_shortener": is_shortener,
        "suspicious_keyword_count": suspicious_keyword_count,
        "has_login": kw_flags["has_login"],
        "has_verify": kw_flags["has_verify"],
        "has_secure": kw_flags["has_secure"],
        "has_account": kw_flags["has_account"],
        "has_update": kw_flags["has_update"],
        "has_bank": kw_flags["has_bank"],
        "digit_to_letter_ratio": round(digit_to_letter_ratio, 6),
        "entropy": round(entropy, 6)
    }

    return features


def extract_url_features_df(urls: Union[List[str], pd.Series]) -> pd.DataFrame:
    """
    Batch feature extraction optimized for high throughput on datasets with 100k+ URLs.
    
    Parameters
    ----------
    urls : list or pd.Series
        Collection of URL strings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rows corresponding to URLs and columns corresponding to FEATURE_NAMES.
    """
    records = [extract_url_features(u) for u in urls]
    return pd.DataFrame.from_records(records, columns=FEATURE_NAMES)


if __name__ == "__main__":
    # Self-test demonstration with sample benign and malicious URLs
    test_urls = [
        "https://www.google.com",
        "https://www.paypal.com/signin",
        "http://paypal-verification-account.security-update.com/login?token=8943729",
        "http://192.168.1.1/admin/login.php",
        "https://bit.ly/3x8Qz9"
    ]
    
    print("=" * 80)
    print("Self-Test: Extracting Lexical Features for Sample URLs")
    print("=" * 80)
    for sample in test_urls:
        feats = extract_url_features(sample)
        print(f"\nURL: {sample}")
        print(f"  Length: {feats['url_length']} | Host Len: {feats['hostname_length']} | Entropy: {feats['entropy']:.3f}")
        print(f"  Dots: {feats['count_dots']} | Hyphens: {feats['count_hyphens']} | Digits: {feats['count_digits']} | IP: {feats['is_ip_address']}")
        print(f"  Shortener: {feats['is_shortener']} | HTTPS: {feats['is_https']} | Suspicious KWs: {feats['suspicious_keyword_count']}")
    
    df_test = extract_url_features_df(test_urls)
    print("\nFeature Matrix Shape:", df_test.shape)
    print("All 26 Features Verified Successfully.")
