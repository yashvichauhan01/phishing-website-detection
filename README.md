# 🛡️ Phishing Website Detection (Dual-Phase AI Engine)

A complete Machine Learning security system and interactive Streamlit web application to detect and classify phishing websites using both **heuristic web security features** (Phase 1 — Model A) and **high-speed, purely lexical URL analysis** (Phase 2 — Model B).

---

## 📌 1. Project Overview & Dual-Phase Architecture

Phishing attacks are short-lived, automated social engineering campaigns designed to harvest credentials and financial data. To provide robust defense across different operational constraints (latency, offline evaluation, network boundaries), this repository implements two distinct machine learning models:

```
phishing-website-detection/
├── Model A (Phase 1): Heuristic & Content Model
│   ├── Dataset: UCI Phishing Websites (5,849 deduplicated samples)
│   ├── Features: 30 features (HTML DOM, external DNS, SSL certificates, Alexa traffic)
│   ├── Splitting: Stratified 80/20 Split (Post-deduplication)
│   ├── Model Selection: 5-Fold Stratified Cross-Validation on Training Data
│   └── Champion: XGBoost Classifier (Accuracy: 95.21% | F1: 95.04% | CV F1: 94.85%)
│
└── Model B (Phase 2): Purely Lexical URL Model
    ├── Dataset: PhiUSIIL Phishing URL Dataset (235,370 deduplicated samples)
    ├── Features: 26 lexical features (String syntax, character entropy, delimiter counts, keywords)
    ├── Splitting: GroupShuffleSplit by Domain (Zero Domain Leakage)
    ├── Model Selection: 5-Fold GroupKFold Cross-Validation on Training Folds
    └── Champion: Random Forest Classifier (Accuracy: 99.60% | Phishing F1: 99.53% | ROC-AUC: 99.78%)
        (Reference XGBoost: Accuracy: 99.60% | Phishing F1: 99.54% | ROC-AUC: 99.81%)
```

---

## ⚖️ 2. Architectural Comparison: Model A vs Model B

| Dimension | Model A (Phase 1 — UCI Features) | Model B (Phase 2 — Lexical URL) |
| :--- | :--- | :--- |
| **Dataset Source** | UCI Phishing Websites Dataset | PhiUSIIL Phishing URL Dataset (`data/phiusiil.csv`) |
| **Dataset Size** | **5,849 instances** (5,206 duplicates dropped) | **235,370 instances** (425 duplicates dropped) |
| **Number of Features** | 30 features | **26 lexical features** |
| **Feature Extraction Nature** | HTML DOM parsing, JS popups, Alexa rank, DNS | **Pure string math & character statistics** |
| **Data Splitting Strategy** | Stratified Random 80/20 | **`GroupShuffleSplit` (by Hostname/Domain)** |
| **Domain Leakage Prevention**| Deduplication prior to split | **Guaranteed Zero Domain Overlap** |
| **Model Selection Criterion**| **5-Fold Stratified CV on Training Data** | **5-Fold `GroupKFold` CV on Training Groups** |
| **Champion Algorithm** | **XGBoost Classifier** | **Random Forest Classifier** |
| **Overall Accuracy** | **95.21%** (Pre-dedup: 96.92%) | **99.60%** |
| **Precision (Phishing = 0)**| 95.07% | **99.90%** |
| **Recall (Phishing = 0)** | 95.70% | **99.17%** |
| **F1-Score (Phishing = 0)** | 95.38% (Legitimate F1: 95.04%) | **99.53%** |
| **ROC-AUC (Phishing = 0)** | **99.33%** | **99.78%** |
| **Execution Latency** | **2.0 – 5.0 seconds** (Network scraping) | **< 0.1 milliseconds** (Microsecond string parse) |
| **Network / API Dependency**| **Required** (Crawls live HTML, WHOIS, DNS) | **None** (100% offline & client-side safe) |
| **Model Artifact** | `model.pkl` | `url_model.pkl` |

---

## 🔬 3. Phase 1 Machine Learning Pipeline (UCI Dataset)

### Step 1: Duplicate Removal & Data Hygiene
- **Raw Dataset:** 11,055 rows across 32 columns.
- **Duplicate Audit:** Found **5,206 duplicate rows** (excluding the identifier `index` column).
- **Leakage Prevention:** Dropped all 5,206 duplicate rows **prior** to splitting, leaving **5,849 unique instances** (3,019 Phishing, 2,830 Legitimate).

### Step 2: 5-Fold Stratified Cross-Validation
- Stratified 80/20 split on deduplicated data: Train = 4,679, Test = 1,170.
- Model selection performed strictly via 5-Fold Stratified CV on the training set:
  - Logistic Regression: CV F1 = 0.9200 (+/- 0.0135)
  - Random Forest: CV F1 = 0.9447 (+/- 0.0075)
  - **XGBoost (Champion):** **CV F1 = 0.9485 (+/- 0.0058)**

### Step 3: Test Set Evaluation & Impact of Duplicate Removal
| Model | Pre-Dedup Test Accuracy | Post-Dedup Test Accuracy | Shift | Test F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 92.58% | 91.28% | -1.30% | 91.10% |
| **Random Forest** | 97.42% | 94.19% | -3.23% | 94.01% |
| **XGBoost (Champion)** | 96.92% | **95.21%** | -1.71% | **95.04%** |

---

## 🔬 4. Phase 2 Machine Learning Pipeline (PhiUSIIL Dataset)

### Step 1: Clean Data Ingestion & Deduplication
- Loaded strictly `usecols=['URL', 'label']` from `data/phiusiil.csv` (235,795 raw rows).
- Dropped 425 duplicate URLs, yielding **235,370 unique samples** (100,520 Phishing, 134,850 Legitimate).

### Step 2: Domain-Aware Group Splitting (`GroupShuffleSplit`)
- **Training Set:** 187,856 samples across 176,061 domains (80%).
- **Testing Set:** 47,514 samples across 44,016 domains (20%).
- **Domain Overlap:** **Exactly 0 shared domains** (Zero Domain Leakage).

### Step 3: Lexical Feature Extraction (`url_features.py`)
Extracts 26 lightweight lexical features directly from URL strings:
1. `url_length`: Total character count.
2. `hostname_length`: Length of hostname/domain.
3. `path_length`: Length of path hierarchy.
4. `count_dots`, `count_hyphens`, `count_digits`, `count_underscores`, `count_slashes`, `count_question_marks`, `count_equal_signs`, `count_ampersands`, `count_other_special`.
5. `subdomain_count`: Number of subdomains preceding registered domain.
6. `is_ip_address`: Binary flag (1 if hostname is bare IPv4/IPv6, 0 otherwise).
7. `has_at_symbol`: Binary flag (1 if `@` in URL, credential masking tactic).
8. `is_https`: Binary flag (1 if HTTPS protocol, 0 otherwise).
9. `is_shortener`: Binary flag (1 if known shortening domain like bit.ly, tinyurl).
10. `suspicious_keyword_count`, `has_login`, `has_verify`, `has_secure`, `has_account`, `has_update`, `has_bank`.
11. `digit_to_letter_ratio`: Ratio of numeric digits to alphabetic characters.
12. `entropy`: Shannon entropy measuring character distribution unpredictability ($H = -\sum p_i \log_2(p_i)$).

### Step 4: 5-Fold Domain-Aware CV (`GroupKFold`) Tuning & Model Selection
Champion chosen strictly via `best_score_` across 5-fold domain-grouped CV on training folds:
- **Logistic Regression:** Scaled via `StandardScaler`, tuned over $C \in \{0.1, 1.0\}$. Best: $C=1.0$ (CV F1: 99.40%).
- **Random Forest (Champion):** Tuned over `n_estimators` $\in \{50, 100\}$, `max_depth` $\in \{10, 14\}$. Best: `max_depth=14, n_estimators=100` (**CV F1: 99.61%**).
- **XGBoost:** Tuned over `n_estimators` $\in \{50, 100\}$, `max_depth` $\in \{4, 6\}$. Best: `max_depth=4, n_estimators=100` (**CV F1: 99.61%**).

### Step 5: Test Set Results (Phishing Class = 0)
Evaluated on **47,514 unseen test URLs** with zero domain overlap:

| Model | Accuracy | Precision (Phish) | Recall (Phish) | F1-Score (Phish) | ROC-AUC (Phish) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 99.38% | 99.96% | 98.60% | 99.28% | 99.61% |
| **Random Forest (Champion)**| **99.60%** | **99.90%** | **99.17%** | **99.53%** | **99.78%** |
| **XGBoost (Reference)** | **99.60%** | **99.91%** | **99.17%** | **99.54%** | **99.81%** |

#### Confusion Matrix (Champion Random Forest):
```
Actual Phish [0]:  [[20,321  (TP),     171 (FN - Missed Phish)],
Actual Legit [1]:   [    20  (FP),  27,002 (TN - Caught Legit)]]
```

---

## 🔍 5. Limitations

1. **Dataset Shortcut Artifacts (`is_https` & `path_length`):**
   - In the PhiUSIIL dataset, legitimate URLs were collected as top-level domain landing pages (all have `path_length = 0`), whereas phishing URLs frequently contain subpaths.
   - Consequently, the model relies heavily on `path_length` and `is_https`. An ablation study retraining the model without `is_https` and `path_length` saw F1-Score drop from **99.53% to 82.88%**, demonstrating significant shortcut reliance.
2. **Real-World / Out-of-Distribution Generalization Drop:**
   - When evaluating `url_model.pkl` on a live feed of **300 active zero-day phishing URLs from OpenPhish** alongside 300 top-site pages with full path endpoints, the model flagged all URLs with paths as phishing. Real-world URL filters require threshold calibration and continuous domain whitelisting.
3. **Phase 1 Application Dependency:**
   - Model A requires live web crawling (HTML DOM parsing, JS event listeners, WHOIS records, and Alexa traffic APIs), taking 2–5 seconds per URL and completely failing on suspended or offline phishing sites (80% of phishing kits go dark in under 48 hours). For safety and determinism, the Streamlit app offers manual feature slider inspection for Model A rather than executing live uncontrolled web scraping.
4. **Blindness of Purely Lexical Analysis:**
   - Model B cannot inspect rendered page forms, credential POST destinations, or hidden iframes. If an attacker compromises a high-reputation domain (e.g. `university.edu/~student/survey`), the clean lexical syntax evades detection.

---

## 🔮 6. Future Work

1. **Two-Tier Hybrid Security Pipeline:**
   - Deploy Model B as a microsecond edge filter. URLs with unambiguous confidence scores (>95%) are filtered instantly; borderline scores (40%–80%) automatically trigger a sandboxed headless browser to extract Model A content features.
2. **Contextual Deep Learning (Char-BERT / RoBERTa):**
   - Implement character-level Transformers to capture brand name mimicry, homoglyph substitutions, and semantic typo-squatting without relying on explicit dictionary keyword lists.
3. **Integration with Passive DNS & Domain Age APIs:**
   - Incorporate lightweight metadata (domain registration age from RDAP/WHOIS and certificate transparency logs) to neutralize newly registered domains using free Let's Encrypt certificates.
4. **Browser Extension & Edge Proxy Packaging:**
   - Export `url_model.pkl` to ONNX Runtime / WebAssembly for client-side local evaluation in Chrome/Firefox extensions with zero data sent to external servers.

---

## 🏗️ 7. Project Directory Structure

```
phishing-website-detection/
├── data/
│   ├── phishing_data.csv        # Phase 1: UCI Phishing Websites dataset (5,849 clean rows)
│   └── phiusiil.csv             # Phase 2: PhiUSIIL Phishing URL dataset (235k rows; git-ignored)
├── results/                     # Audit, error analysis, robustness reports, and presentation charts
│   ├── error_analysis.md        # Detailed breakdown of False Negatives and False Positives
│   ├── robustness.md            # Shortcut ablation and OpenPhish OOD benchmark report
│   ├── false_negatives.csv      # 171 missed phishing URLs with all 26 features
│   ├── false_positives.csv      # 20 false alarm URLs with all 26 features
│   └── final_summary.md         # Comprehensive project summary and viva defense guide
├── tests/
│   └── test_url_features.py     # Automated pytest suite (11 unit tests covering all edge cases)
├── scripts/
│   ├── check_model_a.py         # Verification script for Model A evaluation
│   └── print_results.py         # Model benchmark inspection helper
├── url_features.py              # Phase 2: 26 lexical feature extraction engine
├── train_url_model.py           # Phase 2: CLI training & GroupKFold CV pipeline
├── url_model.ipynb              # Phase 2: Fully executed Jupyter Notebook with SHAP plots
├── url_model.pkl                # Phase 2: Champion Random Forest model artifact
├── train.py                     # Phase 1: UCI model training & Stratified CV script
├── notebook.ipynb               # Phase 1: UCI exploration & training notebook
├── model.pkl                    # Phase 1: Champion XGBoost model artifact
├── app.py                       # Unified Streamlit Web App with live URL Scanner
├── requirements.txt             # Python dependencies (including shap, xgboost, streamlit, pytest)
├── .gitignore                   # Git exclusion rules (pycache, checkpoints, backups, phiusiil.csv)
└── README.md                    # Project documentation & examination guide
```

---

## 💾 8. Dataset

- **Phase 1 (UCI Phishing Websites):** Located directly in `data/phishing_data.csv`.
- **Phase 2 (PhiUSIIL Phishing URL Dataset):**
  - Download Link: [UCI Machine Learning Repository - PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (or Kaggle mirror)
  - Exact Placement Path: Place the CSV file exactly at `data/phiusiil.csv`.

---

## 💻 9. How to Run

Follow these commands to set up the environment, train the models, run tests, and start the application:

```bash
python -m venv venv ; venv\Scripts\activate
pip install -r requirements.txt
python train.py
python train_url_model.py
python -m pytest tests/
streamlit run app.py
```

---

## 👥 Authors & Acknowledgments
- **Developer:** Yashvi Chauhan
- **Datasets:** UCI Machine Learning Repository & PhiUSIIL Phishing URL Dataset
