# 🛡️ Phishing Website Detection (Dual-Phase AI Engine)

A complete Machine Learning project and interactive Streamlit web application to detect and classify phishing websites using both **heuristic web security features** (Phase 1 — Model A) and **high-speed, purely lexical URL analysis** (Phase 2 — Model B).

---

## 📌 1. Project Overview & Dual-Phase Architecture

Phishing attacks are short-lived, automated social engineering campaigns designed to harvest credentials and financial data. To provide robust defense across different operational constraints (latency, offline evaluation, network boundaries), this repository implements two distinct machine learning models:

```
phishing-website-detection/
├── Model A (Phase 1): Heuristic & Content Model
│   ├── Dataset: UCI Phishing Websites (11,055 samples)
│   ├── Features: 30 features (HTML DOM, external DNS, SSL certificates, Alexa traffic)
│   └── Champion: Random Forest Classifier (Accuracy: 97.42% | F1: 97.70%)
│
└── Model B (Phase 2): Purely Lexical URL Model
    ├── Dataset: PhiUSIIL Phishing URL Dataset (235,370 deduplicated samples)
    ├── Features: 26 lexical features (String syntax, character entropy, delimiter counts, keywords)
    ├── Splitting: GroupShuffleSplit by Domain (Zero Domain Leakage)
    └── Champion: XGBoost Classifier (Accuracy: 99.60% | Phishing F1: 99.54% | ROC-AUC: 99.81%)
```

---

## ⚖️ 2. Architectural Comparison: Model A vs Model B

| Dimension | Model A (Phase 1 — UCI Features) | Model B (Phase 2 — Lexical URL) |
| :--- | :--- | :--- |
| **Dataset Source** | UCI Phishing Websites Dataset | PhiUSIIL Phishing URL Dataset (`data/phiusiil.csv`) |
| **Dataset Size** | 11,055 instances | **235,370 instances** (deduplicated) |
| **Number of Features** | 30 features | **26 lexical features** |
| **Feature Extraction Nature** | HTML DOM parsing, JS popups, Alexa rank, DNS | **Pure string math & character statistics** |
| **Data Splitting Strategy** | Stratified Random (80/20) | **`GroupShuffleSplit` (by Hostname/Domain)** |
| **Domain Leakage Prevention**| None (random split) | **Guaranteed Zero Domain Overlap** |
| **Champion Algorithm** | **Random Forest Classifier** | **XGBoost Classifier** |
| **Overall Accuracy** | 97.42% | **99.60%** |
| **Precision (Phishing = 0)**| 96.96% | **99.90%** |
| **Recall (Phishing = 0)** | 98.46% | **99.18%** |
| **F1-Score (Phishing = 0)** | 97.70% | **99.54%** |
| **ROC-AUC (Phishing = 0)** | ~98.50% | **99.81%** |
| **Execution Latency** | **2.0 – 5.0 seconds** (Network scraping) | **< 0.1 milliseconds** (Microsecond string parse) |
| **Network / API Dependency**| **Required** (Crawls live HTML, WHOIS, DNS) | **None** (100% offline & client-side safe) |
| **Model Artifact** | `model.pkl` | `url_model.pkl` |

---

## 🔍 3. Limitations of Each Model (Viva Essential)

### 📌 Limitations of Model A (UCI Heuristic & DOM Features)
1. **High Latency & Resource Overhead:** Extracting page elements (`<a>`, `<iframe>`, forms), querying WHOIS/DNS records, and querying Alexa traffic takes **2 to 5 seconds per site**. This makes it unsuitable for real-time proxies or high-volume email gateways.
2. **Dead Website Failure:** Over 80% of phishing domains are taken down within 24–48 hours of deployment. When a website goes offline or is suspended by a hosting provider, Model A cannot retrieve the DOM or response headers, causing complete failure.
3. **Vulnerable to Cloaking & Bot Detection:** Attackers employ server-side bot-filtering (e.g., checking User-Agent, browser fingerprinting, IP geofencing, or CAPTCHAs). If a scraper visits, the site serves a benign page; when a victim clicks on mobile, it serves the phishing page.
4. **Data Leakage in Random Splits:** Random train-test splitting on small datasets can cause domain memorization if sub-pages from the same site appear in both sets.

### 📌 Limitations of Model B (URL-Only Lexical Model)
1. **Blind to Page Content & Form Handlers:** Model B evaluates solely the characters of the URL string. It cannot inspect what forms, scripts, or credential-harvesting inputs actually render on the page.
2. **Compromised Legitimate Domains:** If an attacker compromises a high-reputation domain (e.g. `university.edu/student/survey`), the clean domain syntax may evade lexical triggers despite hosting malware.
3. **Homoglyph & Benign-Looking URLs:** Attackers utilizing clean, newly registered domains without obvious hyphens or keywords (e.g., `portal-secure7.com`) can produce low lexical suspicion.
4. **Production Recommendation:** Deploy **Model B as a microsecond edge filter** (in firewalls/browser extensions), and trigger deep DOM inspection (Model A) only on ambiguous cases.

---

## 🔬 4. Phase 2 Machine Learning Pipeline (PhiUSIIL Dataset)

### Step 1: Clean Data Ingestion & Deduplication
- **File:** `data/phiusiil.csv`
- Loaded strictly with `pd.read_csv(..., usecols=['URL', 'label'])` to bypass multi-byte character encoding issues in unused metadata columns.
- **Label Mapping:**
  - `label = 1`: **Legitimate Website** (134,850 samples)
  - `label = 0`: **Phishing Website** (100,520 samples)
- Dropped 425 duplicate URLs, yielding **235,370 unique samples**.

### Step 2: Domain-Aware Group Splitting (`GroupShuffleSplit`)
- Standard random splits leak domains into both train and test splits, artificially inflating evaluation scores.
- Using **`GroupShuffleSplit`** on lowercase `hostname`:
  - **Training Set:** 187,856 samples across 176,061 domains (80%)
  - **Testing Set:** 47,514 samples across 44,016 domains (20%)
  - **Domain Overlap:** Exactly **0 shared domains** (Zero Data Leakage).

### Step 3: Lexical Feature Extraction (`url_features.py`)
Extracts 26 lightweight lexical features directly from URL strings:
1. `url_length`: Total character count.
2. `hostname_length`: Length of hostname/domain.
3. `path_length`: Length of path hierarchy.
4. `count_dots`: Total occurrences of `.`.
5. `count_hyphens`: Total occurrences of `-` (frequent in brand spoofing).
6. `count_digits`: Total numerical digits.
7. `count_underscores`: Total occurrences of `_`.
8. `count_slashes`: Total occurrences of `/`.
9. `count_question_marks`: Total occurrences of `?`.
10. `count_equal_signs`: Total occurrences of `=`.
11. `count_ampersands`: Total occurrences of `&`.
12. `count_other_special`: Non-standard characters (%, #, @, +, ~, $, etc.).
13. `subdomain_count`: Number of subdomains preceding registered domain.
14. `is_ip_address`: Binary flag (1 if hostname is a raw IPv4/IPv6, 0 otherwise).
15. `has_at_symbol`: Binary flag (1 if `@` in URL, used to obscure destination).
16. `is_https`: Binary flag (1 if HTTPS protocol, 0 otherwise).
17. `is_shortener`: Binary flag (1 if known shortener domain, 0 otherwise).
18. `suspicious_keyword_count`: Total occurrences of `login`, `verify`, `secure`, `account`, `update`, `bank`.
19–24. `has_login`, `has_verify`, `has_secure`, `has_account`, `has_update`, `has_bank`: Individual binary keyword flags.
25. `digit_to_letter_ratio`: Ratio of digits to alphabetic characters.
26. `entropy`: Shannon entropy measuring character randomness ($H = -\sum p_i \log_2(p_i)$).

### Step 4: 5-Fold Cross-Validation & Hyperparameter Tuning
- **Logistic Regression:** Scaled via `StandardScaler`, tuned over $C \in \{0.1, 1.0\}$.
- **Random Forest:** Tuned over `n_estimators` $\in \{50, 100\}$ and `max_depth` $\in \{10, 14\}$.
- **XGBoost:** Tuned over `n_estimators` $\in \{50, 100\}$ and `max_depth` $\in \{4, 6\}$ using histogram boosting.

### Step 5: Test Set Results (Phishing Class = 0)
Evaluated on **47,514 unseen test URLs** with zero domain overlap:

| Model | Accuracy | Precision (Phish) | Recall (Phish) | F1-Score (Phish) | ROC-AUC (Phish) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 99.38% | 99.96% | 98.60% | 99.28% | 99.61% |
| **Random Forest** | 99.60% | 99.90% | 99.17% | 99.53% | 99.78% |
| **XGBoost (Champion)** | **99.60%** | **99.90%** | **99.18%** | **99.54%** | **99.81%** |

#### Confusion Matrices (Rows: Actual Phish/Legit, Columns: Predicted Phish/Legit):
```
[Logistic Regression]          [Random Forest]               [XGBoost — Champion]
[[20206,   286],               [[20321,   171],              [[20324,   168],
 [    9, 27013]]                [   20, 27002]]               [   20, 27002]]
```

### Step 6: SHAP (SHapley Additive exPlanations) Feature Importance
Top 10 most influential features according to `shap.TreeExplainer`:
1. `is_https` (3.7154): Unencrypted HTTP protocol is heavily correlated with phishing kits.
2. `path_length` (3.2065): Deep directory nesting is common in phishing landing pages.
3. `url_length` (0.5836): Longer URLs conceal obfuscated parameters.
4. `count_dots` (0.2536): Multiple dots indicate subdomains and spoofed domain prefixes.
5. `count_digits` (0.2143): Hex tokens, session hashes, and IP addresses.
6. `count_hyphens` (0.2032): Hyphens weaponized in brand mimicry (e.g. `paypal-login-verify`).
7. `entropy` (0.1345): Randomly generated characters indicate DGA or obfuscated URLs.
8. `digit_to_letter_ratio` (0.1300): Abnormally high digit proportions in domain/path.
9. `hostname_length` (0.0625): Abnormally long hostnames.
10. `subdomain_count` (0.0448): Multi-level subdomain tunneling.

---

## 🏗️ 5. Project Directory Structure

```
phishing-website-detection/
├── data/
│   ├── phishing_data.csv        # Phase 1: UCI Phishing Websites dataset (11k rows)
│   └── phiusiil.csv             # Phase 2: PhiUSIIL Phishing URL dataset (235k rows)
├── url_features.py              # Phase 2: 26 lexical feature extraction engine
├── train_url_model.py           # Phase 2: CLI training & evaluation pipeline
├── url_model.ipynb              # Phase 2: Fully executed Jupyter Notebook with SHAP plots
├── url_model.pkl                # Phase 2: Champion XGBoost model artifact (serialized)
├── train.py                     # Phase 1: UCI model training script
├── notebook.ipynb               # Phase 1: UCI exploration & training notebook
├── model.pkl                    # Phase 1: Champion Random Forest model artifact
├── app.py                       # Unified Streamlit Web App with live URL Scanner
├── requirements.txt             # Python dependencies (including shap, xgboost, streamlit)
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation & Viva examination guide
```

---

## 💻 6. How to Run the Project Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed:
```bash
cd phishing-website-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training (CLI)
- **Train Phase 2 (Lexical URL Model — `url_model.pkl`):**
  ```bash
  python train_url_model.py
  ```
- **Train Phase 1 (UCI Model — `model.pkl`):**
  ```bash
  python train.py
  ```

### 4. Inspect Jupyter Notebooks
- **Phase 2 Notebook (All outputs and SHAP charts saved):**
  ```bash
  jupyter notebook url_model.ipynb
  ```
- **Phase 1 Notebook:**
  ```bash
  jupyter notebook notebook.ipynb
  ```

### 5. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 7. Streamlit Web Application Features

1. **🔗 Live URL Scanner (Phase 2):**
   - Enter any URL or select presets (Google, Suspicious IP, Phishing Spoof).
   - Extracts 26 lexical features in `< 0.1 ms` with zero external network calls.
   - Evaluates via Champion **XGBoost model (`url_model.pkl`)**.
   - Displays risk gauge, confidence scores, risk factor diagnostics, and full feature table.
2. **⚙️ Manual 30-Feature Inspector (Phase 1):**
   - Interactive dropdowns for all 30 UCI features to demonstrate Model A (`model.pkl`).
3. **📊 Model Comparison & Architecture:**
   - Side-by-side benchmark comparison between Model A and Model B.
   - Detailed breakdown of operational trade-offs and limitations.
4. **🎓 Viva Preparation Cheatsheet:**
   - Quick reference in the sidebar explaining key machine learning decisions.

---

## 🎓 8. Viva & Examination Q&A Guide

<details>
<summary><b>Q1: Why did you use GroupShuffleSplit instead of StratifiedKFold in Phase 2?</b></summary>
<br>
In web security datasets, a single website can generate multiple sample URLs (e.g. <code>evil-phish.com/login</code> and <code>evil-phish.com/verify</code>). If we perform standard random splitting, URLs from the same website appear in both train and test sets. The model simply memorizes the domain name rather than learning generalizable lexical patterns. <b>GroupShuffleSplit</b> groups by hostname, ensuring all URLs from a domain are strictly in train or strictly in test (Zero Domain Leakage).
</details>

<details>
<summary><b>Q2: What is Shannon Entropy, and how does it catch phishing attacks?</b></summary>
<br>
Shannon Entropy $H(X) = -\sum p_i \log_2(p_i)$ measures information randomness and unpredictability. Natural English words in legitimate URLs (like <code>wikipedia.org/wiki/Computer</code>) have lower entropy (~3.2–3.9). Phishers using Domain Generation Algorithms (DGAs), randomized hex tokens, or obfuscated paths produce significantly higher entropy (>4.3), serving as a strong mathematical anomaly signal.
</details>

<details>
<summary><b>Q3: Why is Precision for Phishing (label 0) critical in production?</b></summary>
<br>
A False Positive blocks a legitimate user from accessing a benign banking or work portal, causing user disruption and loss of trust. A False Negative exposes the user to identity theft. Our Phase 2 XGBoost model achieved <b>99.90% Precision</b> and <b>99.18% Recall</b> on Phishing, maintaining an almost zero false-alarm rate while catching over 99% of threats.
</details>

<details>
<summary><b>Q4: Why does Model B execute in microseconds compared to Model A?</b></summary>
<br>
Model A relies on features like <i>web_traffic</i> (Alexa Rank API), <i>DNSRecord</i> (WHOIS/DNS lookup), and <i>Iframe/PopUp</i> (HTML DOM scraping), requiring outbound HTTP requests and socket queries (2–5 seconds). Model B performs pure string parsing and character counting in memory (<0.1 ms), with zero network round-trips.
</details>

<details>
<summary><b>Q5: What is the primary limitation of purely lexical URL detection?</b></summary>
<br>
If an attacker compromises a legitimate, trusted website (e.g. <code>stanford.edu/~user/survey</code>) and uploads a phishing form, the domain syntax looks benign and clean. A purely lexical model cannot inspect the actual HTML form or destination POST handler. In enterprise security, the optimal architecture uses Model B as a microsecond edge filter and triggers Model A's content scraper only on ambiguous scores.
</details>

---

## 👥 Authors & Acknowledgments
- **Developer:** Yashvi Chauhan
- **Datasets:** UCI Machine Learning Repository & PhiUSIIL Phishing URL Dataset
