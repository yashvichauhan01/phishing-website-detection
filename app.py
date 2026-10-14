"""
================================================================================
Phishing Website Detection - Interactive Web Application (Streamlit)
================================================================================
File: app.py
Features:
1. 🔗 Real-Time URL Scanner (Phase 2):
   Inspects live URLs purely using 26 lexical features (from url_features.py)
   and evaluates via our Champion Model (url_model.pkl, XGBoost, 99.48% accuracy).
2. ⚙️ Manual 30-Feature Inspector (Phase 1):
   Allows manual configuration of all 30 UCI features to demonstrate Model A (model.pkl).
3. 📊 Model Comparison & Architecture (Model A vs Model B):
   Side-by-side comparative benchmarks, operational trade-offs, and limitations.
4. 🎓 Viva Prep Guide:
   Interactive questions and model explanations for examination defense.

Author: Yashvi Chauhan
Project: Phishing Website Detection (Phases 1 & 2)
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import the Phase 2 lexical feature extraction engine
from url_features import (
    extract_url_features,
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    SHORTENING_DOMAINS
)

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Phishing Website Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern cybersecurity dashboard aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .phishing-banner {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 1.2rem;
        border-radius: 8px;
        color: #991b1b;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(220, 38, 38, 0.08);
    }
    .legit-banner {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 1.2rem;
        border-radius: 8px;
        color: #166534;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(22, 163, 74, 0.08);
    }
    .metric-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-danger {
        color: #b91c1c;
        font-weight: 600;
    }
    .badge-safe {
        color: #15803d;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. MODEL LOADING HELPERS (CACHED)
# =============================================================================
@st.cache_resource
def load_phase2_url_model():
    """
    Loads Phase 2 Champion Model (Model B: url_model.pkl).
    Trained on 235,370 URLs from PhiUSIIL dataset with 26 purely lexical features.
    """
    model_path = "url_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file '{model_path}' not found! Please run 'python train_url_model.py' first.")
        st.stop()
    payload = joblib.load(model_path)
    return payload


@st.cache_resource
def load_phase1_uci_model():
    """
    Loads Phase 1 Champion Model (Model A: model.pkl).
    Trained on 11,055 samples from UCI Phishing dataset with 30 heuristic/DOM features.
    """
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None
    payload = joblib.load(model_path)
    return payload


# Load both models
phase2_payload = load_phase2_url_model()
phase1_payload = load_phase1_uci_model()

url_model = phase2_payload["model"]
url_model_name = phase2_payload.get("model_name", "XGBoost Classifier")
url_metrics = phase2_payload.get("metrics", {})


# =============================================================================
# 3. SIDEBAR: PERFORMANCE OVERVIEW & VIVA PREPARATION CHEATSHEET
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Phishing Detection AI")
    st.caption("Dual-Architecture Machine Learning Security Engine")

    st.markdown("---")
    st.subheader("🚀 Active Scanner Engine")
    st.markdown(f"**Architecture:** Model B (Lexical URL)")
    st.markdown(f"**Champion:** `{url_model_name}`")
    
    col_sb1, col_sb2 = st.columns(2)
    acc_val = url_metrics.get("accuracy", 0.9948) * 100
    f1_val = url_metrics.get("f1_phishing", 0.9940) * 100
    col_sb1.metric("Accuracy", f"{acc_val:.2f}%")
    col_sb2.metric("F1 (Phish)", f"{f1_val:.2f}%")
    
    st.caption(f"ROC-AUC (Phish): **{url_metrics.get('roc_auc_phishing', 0.9970)*100:.2f}%**")

    st.markdown("---")
    st.subheader("🎓 Viva Quick Reference")
    
    with st.expander("❓ Why purely lexical URL features?"):
        st.write(
            "**Zero Network Dependency:** Model A required active web crawling, HTML parsing, "
            "and live DNS/Alexa API queries. If a malicious site is already offline (most phishing "
            "campaigns last < 24 hrs) or blocks automated scrapers, Model A fails.\n\n"
            "**Microsecond Latency:** Lexical features compute directly from string syntax in "
            "< 0.1 ms, enabling real-time deployment in email filters, firewalls, and browser extensions."
        )

    with st.expander("❓ Why GroupShuffleSplit by Domain?"):
        st.write(
            "If multiple URLs share the same domain (e.g. `google.com/search` and `google.com/mail`), "
            "standard train-test split causes **Domain Leakage**. The model memorizes domain tokens "
            "instead of learning generalizable lexical patterns.\n\n"
            "`GroupShuffleSplit` groups by `hostname`, guaranteeing the test set evaluates "
            "100% unseen websites (Zero Data Leakage)."
        )

    with st.expander("❓ What is Shannon Entropy in URLs?"):
        st.write(
            "Shannon entropy $H = -\\sum p_i \\log_2(p_i)$ measures character randomness. "
            "Natural English URLs have predictable letter distributions (entropy ~3.2–3.9). "
            "Malicious URLs generated by Domain Generation Algorithms (DGAs) or token-obfuscation "
            "contain high randomness (entropy > 4.3)."
        )

    with st.expander("❓ Why target Phishing (label 0) metrics?"):
        st.write(
            "In cybersecurity, predicting legitimate sites correctly is trivial. What matters is "
            "**Precision** (not blocking benign user traffic) and **Recall** (catching stealthy "
            "threats). Evaluating metrics specifically on label 0 demonstrates true defensive capability."
        )


# =============================================================================
# 4. MAIN USER INTERFACE
# =============================================================================
st.markdown('<div class="main-header">🛡️ Phishing Website Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time lexical URL threat analysis & heuristic security classification powered by Machine Learning.</div>', unsafe_allow_html=True)

tab_scanner, tab_manual, tab_comparison = st.tabs([
    "🔗 Live URL Scanner (Phase 2 - Lexical Model)",
    "⚙️ Manual 30-Feature Inspector (Phase 1 - Model A)",
    "📊 Model Comparison & Architecture (Model A vs Model B)"
])


# -----------------------------------------------------------------------------
# TAB 1: LIVE URL SCANNER (PHASE 2 - LEXICAL FEATURES + XGBOOST)
# -----------------------------------------------------------------------------
with tab_scanner:
    st.subheader("Inspect Any Website URL")
    st.markdown("Enter a website URL or test preset threat profiles. The system extracts **26 lexical features** and evaluates them via **Model B (url_model.pkl)**.")

    # Quick test preset samples
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    preset_url = ""
    if col_demo1.button("📌 Test Legitimate Portal (Google)"):
        preset_url = "https://www.google.com"
    if col_demo2.button("⚠️ Test Suspicious IP Address"):
        preset_url = "http://192.168.1.105:8080/login/account-verification.php"
    if col_demo3.button("🚨 Test Phishing Brand Spoof"):
        preset_url = "http://paypal-verification-security-login.account-update.com/signin?token=92837498"

    input_url = st.text_input(
        "Enter Website URL to Analyze:",
        value=preset_url if preset_url else "https://github.com",
        placeholder="https://example.com/account/login"
    )

    if st.button("🔍 Scan URL Now", type="primary", use_container_width=True):
        raw_url = input_url.strip()
        if not raw_url:
            st.warning("Please enter a valid URL to analyze.")
        else:
            # 1. Extract 26 purely lexical features using url_features.py
            features_dict = extract_url_features(raw_url)
            
            # 2. Build feature vector aligned with model's expected column order
            input_vector = [features_dict[col] for col in FEATURE_NAMES]
            X_infer = pd.DataFrame([input_vector], columns=FEATURE_NAMES)

            # 3. Model inference
            pred = url_model.predict(X_infer)[0]
            probs = url_model.predict_proba(X_infer)[0]

            # Label 0: Phishing, Label 1: Legitimate
            phishing_prob = probs[0] * 100
            legit_prob = probs[1] * 100

            st.markdown("---")
            if pred == 0:
                st.markdown(f"""
                <div class="phishing-banner">
                    🚨 WARNING: PHISHING THREAT DETECTED!
                    <br><span style="font-size: 0.95rem; font-weight: normal;">
                    Phishing Probability: <b>{phishing_prob:.2f}%</b>. This URL displays structural and lexical anomalies characteristic of fraudulent credentials harvesting.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-banner">
                    🛡️ VERDICT: LEGITIMATE WEBSITE
                    <br><span style="font-size: 0.95rem; font-weight: normal;">
                    Legitimate Confidence: <b>{legit_prob:.2f}%</b>. This URL adheres to trusted, benign lexical syntax and domain conventions.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Key metric summary cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Phishing Risk Score", f"{phishing_prob:.1f}%")
            col_m2.metric("Legitimate Confidence", f"{legit_prob:.1f}%")
            col_m3.metric("Shannon Entropy", f"{features_dict['entropy']:.2f}")
            col_m4.metric("URL Length", f"{features_dict['url_length']} chars")

            st.progress(int(phishing_prob), text=f"Calculated Threat Level: {phishing_prob:.1f}%")

            # Threat Indicators Highlights
            st.markdown("### 🔍 Risk Factor Diagnostics")
            col_d1, col_d2, col_d3 = st.columns(3)

            with col_d1:
                is_https = features_dict["is_https"] == 1
                proto_text = "✅ HTTPS Encrypted" if is_https else "⚠️ HTTP (Unencrypted)"
                st.markdown(f"**Protocol:** {proto_text}")
                
                is_ip = features_dict["is_ip_address"] == 1
                ip_text = "🚨 Bare IP Address Detected" if is_ip else "✅ Standard Domain Name"
                st.markdown(f"**Host Identity:** {ip_text}")

            with col_d2:
                is_short = features_dict["is_shortener"] == 1
                short_text = "🚨 Known URL Shortener" if is_short else "✅ Direct Domain"
                st.markdown(f"**Redirection:** {short_text}")

                hyphens = features_dict["count_hyphens"]
                hyphen_text = f"🚨 {hyphens} Hyphens (Brand Spoofing Risk)" if hyphens >= 3 else f"✅ {hyphens} Hyphens"
                st.markdown(f"**Hyphen Count:** {hyphen_text}")

            with col_d3:
                kw_cnt = features_dict["suspicious_keyword_count"]
                kw_text = f"🚨 {kw_cnt} Sensitive Keywords Found" if kw_cnt > 0 else "✅ No Sensitive Keywords"
                st.markdown(f"**Keywords:** {kw_text}")

                entropy_val = features_dict["entropy"]
                ent_text = f"🚨 High Randomness ({entropy_val:.2f})" if entropy_val > 4.2 else f"✅ Normal ({entropy_val:.2f})"
                st.markdown(f"**Entropy:** {ent_text}")

            # Expandable detailed table with all 26 extracted features
            with st.expander("📋 Inspect All 26 Extracted Lexical Features", expanded=True):
                feat_table = []
                for name in FEATURE_NAMES:
                    val = features_dict[name]
                    # Risk assessment annotation
                    if name == "is_https":
                        assessment = "Safe (Encrypted)" if val == 1 else "Suspicious (Unencrypted)"
                    elif name in ("is_ip_address", "is_shortener", "has_at_symbol") and val == 1:
                        assessment = "High Risk Indicator"
                    elif name.startswith("has_") and val == 1:
                        assessment = "Suspicious Keyword Present"
                    elif name == "entropy" and val > 4.2:
                        assessment = "High Randomness (DGA / Hash)"
                    elif name == "count_hyphens" and val >= 3:
                        assessment = "Elevated Hyphen Count"
                    elif name == "subdomain_count" and val >= 2:
                        assessment = "Deep Subdomain Nesting"
                    else:
                        assessment = "Benign / Normal"

                    feat_table.append({
                        "Feature Name": name,
                        "Extracted Value": val,
                        "Cybersecurity Assessment": assessment
                    })

                st.dataframe(pd.DataFrame(feat_table), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: MANUAL 30-FEATURE INSPECTOR (PHASE 1 - MODEL A)
# -----------------------------------------------------------------------------
with tab_manual:
    st.subheader("Manual Feature Configuration (Phase 1 — Model A)")
    st.caption("Adjust all 30 UCI features manually to demonstrate how individual HTML/DOM/traffic signals influence Model A.")

    if phase1_payload is None:
        st.warning("Model A (`model.pkl`) not found. Please run `python train.py` to enable Phase 1 manual inspection.")
    else:
        model_a = phase1_payload["model"] if isinstance(phase1_payload, dict) else phase1_payload
        feature_names_a = phase1_payload.get("feature_names", []) if isinstance(phase1_payload, dict) else []

        options_map = {"Legitimate (+1)": 1, "Suspicious (0)": 0, "Phishing (-1)": -1}

        col_m1, col_m2, col_m3 = st.columns(3)
        manual_inputs = {}

        with col_m1:
            st.markdown("**🌐 URL & Domain Attributes**")
            manual_inputs['having_IPhaving_IP_Address'] = options_map[st.selectbox("IP in Address", list(options_map.keys()), index=0, key="m_ip")]
            manual_inputs['URLURL_Length'] = options_map[st.selectbox("URL Length", list(options_map.keys()), index=0, key="m_len")]
            manual_inputs['having_IP_Address'] = manual_inputs['having_IPhaving_IP_Address']
            manual_inputs['URL_Length'] = manual_inputs['URLURL_Length']
            manual_inputs['Shortining_Service'] = options_map[st.selectbox("Shortening Service", list(options_map.keys()), index=0, key="m_short")]
            manual_inputs['having_At_Symbol'] = options_map[st.selectbox("Having @ Symbol", list(options_map.keys()), index=0, key="m_at")]
            manual_inputs['double_slash_redirecting'] = options_map[st.selectbox("Double Slash Redirection", list(options_map.keys()), index=0, key="m_slash")]
            manual_inputs['Prefix_Suffix'] = options_map[st.selectbox("Prefix/Suffix Hyphen", list(options_map.keys()), index=0, key="m_hyphen")]
            manual_inputs['having_Sub_Domain'] = options_map[st.selectbox("Subdomains Count", list(options_map.keys()), index=0, key="m_sub")]
            manual_inputs['Domain_registeration_length'] = options_map[st.selectbox("Domain Reg. Length", list(options_map.keys()), index=0, key="m_reg")]
            manual_inputs['age_of_domain'] = options_map[st.selectbox("Age of Domain", list(options_map.keys()), index=0, key="m_age")]
            manual_inputs['DNSRecord'] = options_map[st.selectbox("DNS Record Exists", list(options_map.keys()), index=0, key="m_dns")]

        with col_m2:
            st.markdown("**🔒 Security & Page Elements**")
            manual_inputs['SSLfinal_State'] = options_map[st.selectbox("SSL Final State", list(options_map.keys()), index=0, key="m_ssl")]
            manual_inputs['Favicon'] = options_map[st.selectbox("Favicon External", list(options_map.keys()), index=0, key="m_fav")]
            manual_inputs['port'] = options_map[st.selectbox("Port Standard", list(options_map.keys()), index=0, key="m_port")]
            manual_inputs['HTTPS_token'] = options_map[st.selectbox("HTTPS Token in Domain", list(options_map.keys()), index=0, key="m_htok")]
            manual_inputs['Request_URL'] = options_map[st.selectbox("Request URL External", list(options_map.keys()), index=0, key="m_req")]
            manual_inputs['URL_of_Anchor'] = options_map[st.selectbox("URL of Anchor", list(options_map.keys()), index=0, key="m_anch")]
            manual_inputs['Links_in_tags'] = options_map[st.selectbox("Links in Tags", list(options_map.keys()), index=0, key="m_tags")]
            manual_inputs['SFH'] = options_map[st.selectbox("Server Form Handler (SFH)", list(options_map.keys()), index=0, key="m_sfh")]
            manual_inputs['Submitting_to_email'] = options_map[st.selectbox("Submit to Email", list(options_map.keys()), index=0, key="m_mail")]
            manual_inputs['Abnormal_URL'] = options_map[st.selectbox("Abnormal URL", list(options_map.keys()), index=0, key="m_abn")]

        with col_m3:
            st.markdown("**⚡ Traffic & Behavior**")
            manual_inputs['Redirect'] = options_map[st.selectbox("Redirect Count", ["Legitimate (+1)", "Suspicious (0)"], index=0, key="m_redir")]
            manual_inputs['on_mouseover'] = options_map[st.selectbox("Mouseover Status Bar", list(options_map.keys()), index=0, key="m_mouse")]
            manual_inputs['RightClick'] = options_map[st.selectbox("Right Click Disabled", list(options_map.keys()), index=0, key="m_rclick")]
            manual_inputs['popUpWidnow'] = options_map[st.selectbox("Popup Window with Input", list(options_map.keys()), index=0, key="m_popup")]
            manual_inputs['Iframe'] = options_map[st.selectbox("Iframe Redirection", list(options_map.keys()), index=0, key="m_iframe")]
            manual_inputs['web_traffic'] = options_map[st.selectbox("Web Traffic Rank", list(options_map.keys()), index=0, key="m_traf")]
            manual_inputs['Page_Rank'] = options_map[st.selectbox("Page Rank", list(options_map.keys()), index=0, key="m_prank")]
            manual_inputs['Google_Index'] = options_map[st.selectbox("Indexed in Google", list(options_map.keys()), index=0, key="m_gindex")]
            manual_inputs['Links_pointing_to_page'] = options_map[st.selectbox("Links Pointing to Page", list(options_map.keys()), index=0, key="m_links")]
            manual_inputs['Statistical_report'] = options_map[st.selectbox("Statistical Report", list(options_map.keys()), index=0, key="m_stat")]

        if st.button("⚡ Evaluate Manual Feature Vector with Model A", type="primary", use_container_width=True):
            m_vector = [manual_inputs.get(c, 0) for c in feature_names_a]
            X_manual = pd.DataFrame([m_vector], columns=feature_names_a)

            m_pred = model_a.predict(X_manual)[0]
            m_probs = model_a.predict_proba(X_manual)[0]

            if m_pred == 0:
                st.error(f"🚨 Model A Prediction: **Phishing Website** (Phishing Confidence: {m_probs[0]*100:.2f}%)")
            else:
                st.success(f"🛡️ Model A Prediction: **Legitimate Website** (Legitimate Confidence: {m_probs[1]*100:.2f}%)")


# -----------------------------------------------------------------------------
# TAB 3: MODEL COMPARISON & ARCHITECTURE (MODEL A vs MODEL B)
# -----------------------------------------------------------------------------
with tab_comparison:
    st.subheader("Architectural Comparison: Model A vs Model B")
    st.markdown("A head-to-head comparison between **Phase 1 (UCI Heuristic Model)** and **Phase 2 (URL-Only Lexical Model)**.")

    comparison_data = {
        "Metric / Property": [
            "Dataset Source",
            "Sample Count",
            "Features Count",
            "Feature Nature",
            "Data Split Strategy",
            "Champion Algorithm",
            "Overall Accuracy",
            "Phishing Precision",
            "Phishing Recall",
            "Phishing F1-Score",
            "Phishing ROC-AUC",
            "Execution Latency",
            "Network / Internet Dependency"
        ],
        "Model A (Phase 1 — UCI Features)": [
            "UCI Phishing Websites Dataset",
            "5,849 instances (deduplicated)",
            "30 features",
            "HTML DOM, Alexa Rank, DNS, SSL",
            "Stratified Random 80/20",
            "XGBoost Classifier",
            "95.21%",
            "95.37%",
            "94.70%",
            "95.04%",
            "99.33%",
            "2.0 – 5.0 seconds (Scraping + DNS)",
            "Required (Crawls live webpage & APIs)"
        ],
        "Model B (Phase 2 — Lexical URL)": [
            "PhiUSIIL Phishing URL Dataset",
            "235,370 instances (deduplicated)",
            "26 lexical features",
            "String syntax, entropy, delimiters",
            "GroupShuffleSplit (by Domain)",
            "XGBoost Classifier",
            "99.48%",
            "99.89%",
            "98.91%",
            "99.40%",
            "99.70%",
            "< 0.1 milliseconds (Pure string math)",
            "None (100% offline & client-safe)"
        ]
    }
    st.table(pd.DataFrame(comparison_data))

    st.markdown("---")
    st.markdown("### ⚖️ Architectural Trade-Offs & Limitations")

    col_lim1, col_lim2 = st.columns(2)

    with col_lim1:
        st.markdown("""
        #### 📌 Limitations of Model A (UCI Features)
        1. **High Latency & Resource Overhead:** Extracting DOM tags (`<a>`, `<iframe>`, `<form>`), measuring Alexa web traffic, and performing WHOIS DNS lookups takes **2 to 5 seconds** per URL.
        2. **Dead Website Failure:** Over 80% of phishing domains are taken down within 24–48 hours. When a site is offline or suspended, Model A cannot retrieve any page content, resulting in complete failure.
        3. **Vulnerable to Cloaking & Bot Detection:** Attackers employ server-side bot blockers or CAPTCHAs that return benign HTML to security scanners while serving phishing payloads to real human mobile users.
        4. **Data Leakage Risk:** Standard row-level splitting on small datasets can cause domain memorization if multiple URLs share the same website.
        """)

    with col_lim2:
        st.markdown("""
        #### 📌 Limitations of Model B (Lexical-Only)
        1. **Blind to Page Content & Form Handlers:** Model B evaluates solely the characters of the URL string. It cannot inspect what forms, scripts, or credential-harvesting inputs actually render on the page.
        2. **Compromised Legitimate Sites:** If an attacker compromises a high-reputation domain (e.g. `university.edu/student/survey`), the clean domain syntax may evade lexical triggers despite hosting malware.
        3. **Homoglyph & Benign-Looking URLs:** Attackers utilizing clean, newly registered domains without obvious hyphens or keywords (e.g., `portal-secure7.com`) can produce low lexical suspicion.
        4. **Production Recommendation:** Deploy **Model B as a microsecond edge filter** (in firewalls/browser extensions), and trigger deep DOM inspection (Model A) only on ambiguous cases.
        """)
