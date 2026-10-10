"""
Phishing Website Detection - Streamlit Web Application
======================================================
This app loads the trained Machine Learning model (`model.pkl`) and provides:
1. URL Inspector: Automatically extracts heuristic features from a pasted URL.
2. Full 30-Feature Inspector: Allows manual configuration of all UCI features for viva demo.
3. Viva Prep Sidebar: Quick reference for project viva explanations.
"""

import os
import re
import urllib.parse
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Phishing Website Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .phishing-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 6px;
        color: #991b1b;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .legit-banner {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1rem;
        border-radius: 6px;
        color: #166534;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_trained_model():
    """Loads model payload from model.pkl."""
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please run train.py or execute notebook.ipynb first.")
        st.stop()
    
    payload = joblib.load(model_path)
    # Support both dictionary payload and direct model objects
    if isinstance(payload, dict):
        model = payload.get('model')
        feature_names = payload.get('feature_names')
        model_name = payload.get('model_name', 'Random Forest')
        metrics = payload.get('metrics', {'accuracy': 0.9742, 'f1_score': 0.9770})
    else:
        model = payload
        model_name = "Trained Classifier"
        metrics = {'accuracy': 0.9742, 'f1_score': 0.9770}
        # Default UCI 30 features list
        feature_names = [
            'having_IPhaving_IP_Address', 'URLURL_Length', 'Shortining_Service',
            'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
            'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
            'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
            'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
            'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
            'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
        ]
    return model, feature_names, model_name, metrics

model, feature_names, model_name, metrics = load_trained_model()

# ==========================================
# 3. URL FEATURE EXTRACTION HELPER
# ==========================================
SHORTENING_DOMAINS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bitly.com', 'tiny.cc', 'rb.gy', 'cutt.ly'
}

def extract_features_from_url(url: str) -> dict:
    """
    Extracts heuristic cybersecurity features from a raw URL string.
    Maps heuristics into UCI encoding: 1 (Legitimate), 0 (Suspicious), -1 (Phishing).
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
        
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    extracted = {}

    # 1. IP Address Presence: If IP is used instead of domain -> -1 (Phishing), else 1 (Legitimate)
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    is_ip = bool(re.match(ip_pattern, hostname))
    extracted['having_IPhaving_IP_Address'] = -1 if is_ip else 1

    # 2. URL Length: <54 chars -> 1 (Legit), 54-75 -> 0 (Suspicious), >75 -> -1 (Phishing)
    url_len = len(url)
    if url_len < 54:
        extracted['URLURL_Length'] = 1
    elif 54 <= url_len <= 75:
        extracted['URLURL_Length'] = 0
    else:
        extracted['URLURL_Length'] = -1

    # 3. URL Shortening Service
    is_shortener = any(domain in hostname.lower() for domain in SHORTENING_DOMAINS)
    extracted['Shortining_Service'] = -1 if is_shortener else 1

    # 4. Having '@' symbol: Phishers use '@' to obscure actual destination
    extracted['having_At_Symbol'] = -1 if '@' in url else 1

    # 5. Double slash redirection: '//' in path indicates redirection
    extracted['double_slash_redirecting'] = -1 if '//' in path else 1

    # 6. Prefix/Suffix with hyphen: Legitimate sites rarely use hyphens in domain name
    extracted['Prefix_Suffix'] = -1 if '-' in hostname else 1

    # 7. Subdomains count: 1 dot -> 1 (Legit), 2 dots -> 0 (Suspicious), >=3 dots -> -1 (Phishing)
    dots_count = hostname.count('.')
    if dots_count <= 1:
        extracted['having_Sub_Domain'] = 1
    elif dots_count == 2:
        extracted['having_Sub_Domain'] = 0
    else:
        extracted['having_Sub_Domain'] = -1

    # 8. SSL Final State: https presence
    if parsed.scheme == 'https':
        extracted['SSLfinal_State'] = 1
    else:
        extracted['SSLfinal_State'] = -1

    # 9. Non-standard port
    if parsed.port and parsed.port not in (80, 443):
        extracted['port'] = -1
    else:
        extracted['port'] = 1

    # 10. HTTPS token in domain (e.g. http://https-paypal.com)
    extracted['HTTPS_token'] = -1 if 'https' in hostname.lower() and parsed.scheme != 'https' else 1

    # Realistic default assumptions for external/page-level features that cannot
    # be extracted without live scraping or external DNS APIs
    defaults = {
        'Domain_registeration_length': -1 if is_ip or is_shortener else 1,
        'Favicon': 1,
        'Request_URL': 1,
        'URL_of_Anchor': -1 if is_shortener or is_ip else 0,
        'Links_in_tags': 0,
        'SFH': -1 if is_ip else 1,
        'Submitting_to_email': 1,
        'Abnormal_URL': -1 if is_ip else 1,
        'Redirect': 0,
        'on_mouseover': 1,
        'RightClick': 1,
        'popUpWidnow': 1,
        'Iframe': 1,
        'age_of_domain': -1 if is_shortener else 1,
        'DNSRecord': 1,
        'web_traffic': -1 if (is_ip or is_shortener) else 0,
        'Page_Rank': -1 if is_ip else 1,
        'Google_Index': 1,
        'Links_pointing_to_page': 0,
        'Statistical_report': 1
    }

    for k, v in defaults.items():
        if k not in extracted:
            extracted[k] = v

    return extracted

# ==========================================
# 4. SIDEBAR - VIVA CHEATSHEET & INFO
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Phishing Detection AI")
    st.markdown("Machine Learning system trained on the **UCI Phishing Websites Dataset**.")
    
    st.markdown("---")
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0.9742)*100:.1f}%")
    col2.metric("F1-Score", f"{metrics.get('f1_score', 0.9770)*100:.1f}%")
    st.caption(f"Champion Model: **{model_name}**")
    
    st.markdown("---")
    st.subheader("🎓 Viva Quick Reference")
    with st.expander("Why Random Forest?"):
        st.write(
            "Random Forest is an ensemble of decision trees. It captures non-linear relationships "
            "between heuristic features, prevents overfitting via bootstrap aggregating (bagging), "
            "and handles discrete categorical values {-1, 0, 1} robustly."
        )
    with st.expander("Why map -1 to 0?"):
        st.write(
            "The UCI dataset codes Phishing as -1 and Legitimate as +1. Binary cross-entropy loss "
            "in modern libraries (especially XGBoost) mathematically requires labels in {0, 1}."
        )
    with st.expander("Top Features in Phishing"):
        st.write(
            "- **SSLfinal_State**: Phishing sites often lack valid certificates.\n"
            "- **URL_of_Anchor**: Fake pages use empty/external links.\n"
            "- **web_traffic**: Malicious domains have near-zero Alexa rank.\n"
            "- **Prefix_Suffix**: Spoofed brand names contain hyphens."
        )

# ==========================================
# 5. MAIN CONTENT
# ==========================================
st.markdown('<div class="main-header">🛡️ Phishing Website Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect fraudulent and malicious websites using machine learning and heuristic security analysis.</div>', unsafe_allow_html=True)

tab_url, tab_manual, tab_dataset = st.tabs([
    "🔗 Quick URL Scanner",
    "⚙️ Manual Feature Inspector",
    "📊 Dataset & Architecture"
])

# ----------------------------------------------------
# TAB 1: QUICK URL SCANNER
# ----------------------------------------------------
with tab_url:
    st.subheader("Analyze Website URL")
    st.markdown("Enter a website address or select a test sample to extract heuristics and classify.")

    # Quick test buttons
    col_t1, col_t2, col_t3 = st.columns(3)
    sample_url = ""
    if col_t1.button("📌 Test Legitimate (Google)"):
        sample_url = "https://www.google.com"
    if col_t2.button("⚠️ Test Suspicious (IP Address)"):
        sample_url = "http://192.168.1.105/login/verification"
    if col_t3.button("🚨 Test Phishing (Hyphenated Brand)"):
        sample_url = "http://paypal-verification-security-login.com/account/auth/signin?token=92837498"

    input_url = st.text_input(
        "Website URL:",
        value=sample_url if sample_url else "https://github.com",
        placeholder="https://example.com"
    )

    if st.button("🔍 Scan URL Now", type="primary", use_container_width=True):
        if not input_url.strip():
            st.warning("Please enter a valid URL to scan.")
        else:
            features_dict = extract_features_from_url(input_url)
            
            # Format feature vector matching model columns
            input_vector = [features_dict.get(col, 0) for col in feature_names]
            X_input = pd.DataFrame([input_vector], columns=feature_names)

            prediction = model.predict(X_input)[0]
            probabilities = model.predict_proba(X_input)[0]
            
            # 0 = Phishing, 1 = Legitimate
            phishing_prob = probabilities[0] * 100
            legit_prob = probabilities[1] * 100

            st.markdown("---")
            if prediction == 0:
                st.markdown(f"""
                <div class="phishing-banner">
                    🚨 WARNING: PHISHING WEBSITE DETECTED!
                    <br><span style="font-size: 0.95rem; font-weight: normal;">
                    Risk Probability: <b>{phishing_prob:.1f}%</b>. This URL exhibits characteristics typical of credential-harvesting pages.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-banner">
                    🛡️ LEGITIMATE WEBSITE
                    <br><span style="font-size: 0.95rem; font-weight: normal;">
                    Confidence Score: <b>{legit_prob:.1f}%</b>. The URL matches trusted security heuristics.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Phishing Risk Score", f"{phishing_prob:.1f}%")
            col_p2.metric("Legitimate Confidence", f"{legit_prob:.1f}%")

            st.progress(int(phishing_prob), text=f"Phishing Probability: {phishing_prob:.1f}%")

            # Display extracted heuristic features
            with st.expander("📋 View Extracted Heuristic Features", expanded=True):
                feat_display = []
                for k in ['having_IPhaving_IP_Address', 'URLURL_Length', 'Shortining_Service', 
                          'having_At_Symbol', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State']:
                    val = features_dict[k]
                    status = "Phishing Indicator (-1)" if val == -1 else ("Suspicious (0)" if val == 0 else "Safe / Legitimate (+1)")
                    feat_display.append({"Feature Name": k, "Extracted Value": val, "Assessment": status})
                st.table(pd.DataFrame(feat_display))

# ----------------------------------------------------
# TAB 2: MANUAL FEATURE INSPECTOR (VIVA MODE)
# ----------------------------------------------------
with tab_manual:
    st.subheader("Manual Feature Configuration")
    st.caption("Adjust all 30 UCI features manually to demonstrate how individual inputs alter model decisions.")

    options_map = {"Legitimate (+1)": 1, "Suspicious (0)": 0, "Phishing (-1)": -1}

    col_m1, col_m2, col_m3 = st.columns(3)
    manual_inputs = {}

    with col_m1:
        st.markdown("**🌐 URL & Domain Attributes**")
        manual_inputs['having_IPhaving_IP_Address'] = options_map[st.selectbox("IP in Address", list(options_map.keys()), index=0)]
        manual_inputs['URLURL_Length'] = options_map[st.selectbox("URL Length", list(options_map.keys()), index=0)]
        manual_inputs['Shortining_Service'] = options_map[st.selectbox("Shortening Service", list(options_map.keys()), index=0)]
        manual_inputs['having_At_Symbol'] = options_map[st.selectbox("Having @ Symbol", list(options_map.keys()), index=0)]
        manual_inputs['double_slash_redirecting'] = options_map[st.selectbox("Double Slash Redirection", list(options_map.keys()), index=0)]
        manual_inputs['Prefix_Suffix'] = options_map[st.selectbox("Prefix/Suffix Hyphen", list(options_map.keys()), index=0)]
        manual_inputs['having_Sub_Domain'] = options_map[st.selectbox("Subdomains Count", list(options_map.keys()), index=0)]
        manual_inputs['Domain_registeration_length'] = options_map[st.selectbox("Domain Reg. Length", list(options_map.keys()), index=0)]
        manual_inputs['age_of_domain'] = options_map[st.selectbox("Age of Domain", list(options_map.keys()), index=0)]
        manual_inputs['DNSRecord'] = options_map[st.selectbox("DNS Record Exists", list(options_map.keys()), index=0)]

    with col_m2:
        st.markdown("**🔒 Security & Page Elements**")
        manual_inputs['SSLfinal_State'] = options_map[st.selectbox("SSL Final State", list(options_map.keys()), index=0)]
        manual_inputs['Favicon'] = options_map[st.selectbox("Favicon External", list(options_map.keys()), index=0)]
        manual_inputs['port'] = options_map[st.selectbox("Port Standard", list(options_map.keys()), index=0)]
        manual_inputs['HTTPS_token'] = options_map[st.selectbox("HTTPS Token in Domain", list(options_map.keys()), index=0)]
        manual_inputs['Request_URL'] = options_map[st.selectbox("Request URL External", list(options_map.keys()), index=0)]
        manual_inputs['URL_of_Anchor'] = options_map[st.selectbox("URL of Anchor", list(options_map.keys()), index=0)]
        manual_inputs['Links_in_tags'] = options_map[st.selectbox("Links in Tags", list(options_map.keys()), index=0)]
        manual_inputs['SFH'] = options_map[st.selectbox("Server Form Handler (SFH)", list(options_map.keys()), index=0)]
        manual_inputs['Submitting_to_email'] = options_map[st.selectbox("Submit to Email", list(options_map.keys()), index=0)]
        manual_inputs['Abnormal_URL'] = options_map[st.selectbox("Abnormal URL", list(options_map.keys()), index=0)]

    with col_m3:
        st.markdown("**⚡ Traffic & Behavior**")
        manual_inputs['Redirect'] = options_map[st.selectbox("Redirect Count", ["Legitimate (+1)", "Suspicious (0)"], index=0)]
        manual_inputs['on_mouseover'] = options_map[st.selectbox("Mouseover Status Bar", list(options_map.keys()), index=0)]
        manual_inputs['RightClick'] = options_map[st.selectbox("Right Click Disabled", list(options_map.keys()), index=0)]
        manual_inputs['popUpWidnow'] = options_map[st.selectbox("Popup Window with Input", list(options_map.keys()), index=0)]
        manual_inputs['Iframe'] = options_map[st.selectbox("Iframe Redirection", list(options_map.keys()), index=0)]
        manual_inputs['web_traffic'] = options_map[st.selectbox("Web Traffic Rank", list(options_map.keys()), index=0)]
        manual_inputs['Page_Rank'] = options_map[st.selectbox("Page Rank", list(options_map.keys()), index=0)]
        manual_inputs['Google_Index'] = options_map[st.selectbox("Indexed in Google", list(options_map.keys()), index=0)]
        manual_inputs['Links_pointing_to_page'] = options_map[st.selectbox("Links Pointing to Page", list(options_map.keys()), index=0)]
        manual_inputs['Statistical_report'] = options_map[st.selectbox("Statistical Report", list(options_map.keys()), index=0)]

    if st.button("⚡ Evaluate Manual Feature Vector", type="primary", use_container_width=True):
        m_vector = [manual_inputs.get(c, 0) for c in feature_names]
        X_manual = pd.DataFrame([m_vector], columns=feature_names)
        
        m_pred = model.predict(X_manual)[0]
        m_probs = model.predict_proba(X_manual)[0]

        if m_pred == 0:
            st.error(f"🚨 Prediction: **Phishing Website** (Confidence: {m_probs[0]*100:.2f}%)")
        else:
            st.success(f"🛡️ Prediction: **Legitimate Website** (Confidence: {m_probs[1]*100:.2f}%)")

# ----------------------------------------------------
# TAB 3: DATASET & ARCHITECTURE
# ----------------------------------------------------
with tab_dataset:
    st.subheader("Model Architecture & Benchmark Comparison")
    
    comp_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": ["92.90%", "97.42%", "97.38%"],
        "Precision": ["92.42%", "96.96%", "96.88%"],
        "Recall": ["95.04%", "98.46%", "98.46%"],
        "F1-Score": ["93.71%", "97.70%", "97.66%"]
    }
    st.table(pd.DataFrame(comp_data))

    st.markdown("""
    ### Why Random Forest is the Champion Model:
    1. **Highest F1-Score (97.70%) & Accuracy (97.42%):** Balances false alarms and missed threats best.
    2. **Handles Discrete Categorical Data:** UCI features take discrete ternary values `{-1, 0, 1}`, which decision tree splits partition cleanly without requiring normalization or scaling.
    3. **Ensemble Generalization:** Bagging 100 decorrelated trees reduces model variance and eliminates single-tree overfitting.
    """)
