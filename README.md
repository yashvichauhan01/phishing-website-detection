# 🛡️ Phishing Website Detection

A complete, beginner-friendly Machine Learning project and interactive web application to detect and classify phishing websites using heuristic web security features and ensemble learning.

---

## 📌 1. Problem Statement
Phishing is one of the most widespread social engineering cyberattacks, wherein fraudulent actors construct deceptive websites mimicking legitimate institutional portals (such as banks, social networks, and payment gateways) to harvest sensitive credentials, credit card details, and personal identities.

Because phishing campaigns are short-lived and rapidly mutate their domains, static blacklists fail to detect zero-day attacks. This project utilizes **Machine Learning** trained on heuristic characteristics (URL syntax, domain metadata, and HTML structure) to classify websites as **Phishing** or **Legitimate** in real time.

---

## 📂 2. Dataset Overview
- **Dataset Source:** UCI Machine Learning Repository — Phishing Websites Dataset
- **Local Path:** `data/phishing_data.csv`
- **Total Samples:** 11,055 instances
- **Total Features:** 30 heuristic security features (excluding `index` and `Result`)
- **Missing Values:** 0 (Dataset is clean and complete)

### Class Distribution:
| Class | UCI Raw Label | Mapped Target | Sample Count | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Phishing** | `-1` | `0` | 4,898 | 44.30% |
| **Legitimate** | `+1` | `1` | 6,157 | 55.70% |

> **Viva Note:** The dataset is well-balanced (~44% vs ~56%), eliminating the need for synthetic oversampling (such as SMOTE) while ensuring realistic evaluation across Precision, Recall, and F1-score.

---

## 🏗️ 3. Project Directory Structure
```
phishing-website-detection/
├── data/
│   └── phishing_data.csv        # UCI Phishing Websites dataset
├── notebook.ipynb               # Full Jupyter Notebook (EDA, charts, training, evaluations)
├── train.py                     # Standalone CLI training pipeline & model exporter
├── app.py                       # Streamlit web application with live URL scanner
├── model.pkl                    # Champion trained model (Random Forest) saved via joblib
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files (caches, checkpoints)
└── README.md                    # Project documentation & Viva preparation guide
```

---

## 🔬 4. Machine Learning Pipeline

### Step 1: Exploratory Data Analysis (EDA)
- Inspected dataset shape (`11055, 32`) and verified datatypes.
- Verified missing values (`0` across all features).
- Analyzed class distribution and visualized class proportions.
- Generated a Pearson correlation matrix and an annotated correlation heatmap for the top features correlated with the target `Result`.

### Step 2: Data Preprocessing
- **Index Removal:** Dropped `index` column as it is an arbitrary row identifier with no predictive generalization.
- **Target Remapping:** Mapped `-1` $\rightarrow$ `0` (Phishing) and `+1` $\rightarrow$ `1` (Legitimate).
  - *Why?* XGBoost and standard cross-entropy loss functions require targets strictly in $\{0, 1\}$.
- **Train-Test Split:** 80% training (8,844 samples) and 20% test (2,211 samples) using stratified sampling (`stratify=y`) to preserve exact class proportions in both splits.

### Step 3: Model Benchmarking & Results
Three models representing linear, bagging, and boosting paradigms were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 92.90% | 92.42% | 95.04% | 93.71% |
| **Random Forest (Champion)** | **97.42%** | **96.96%** | **98.46%** | **97.70%** |
| **XGBoost** | 97.38% | 96.88% | 98.46% | 97.66% |

- **Champion Model:** **Random Forest Classifier** achieved the top F1-Score (97.70%) and Accuracy (97.42%), closely matched by XGBoost.
- Saved to `model.pkl` using `joblib`.

### Step 4: Top Predictive Features
Feature importance derived from Random Forest (Gini impurity reduction):
1. **`SSLfinal_State`:** Legitimate portals mandate valid, trusted SSL certificates; phishers often lack valid HTTPS or use untrusted certificates.
2. **`URL_of_Anchor`:** Fraudulent websites frequently have anchor tags `<a>` pointing to empty anchors (`#`), javascript links, or external domains.
3. **`web_traffic`:** Legitimate enterprises have high global traffic ranks, while phishing domains are ephemeral and have minimal traffic.
4. **`Prefix_Suffix`:** Phishers frequently add hyphens (e.g., `paypal-verify.com`) to mimic brand names.

---

## 💻 5. How to Run the Project Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone or navigate to the project directory:
```bash
cd phishing-website-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training (CLI)
To retrain the models and save `model.pkl`:
```bash
python train.py
```

### 4. Open the Jupyter Notebook
To review the interactive EDA, visualizations, and comparative tables:
```bash
jupyter notebook notebook.ipynb
# or open directly in VS Code
```

### 5. Launch the Streamlit Web App
To start the interactive web application:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 6. Streamlit Web App Features
- **🔗 Quick URL Scanner:** Paste any URL or click preconfigured sample buttons (Legitimate, Suspicious, Phishing) to parse and extract heuristics in real time.
- **⚙️ Manual 30-Feature Inspector:** Adjust all 30 UCI features using interactive dropdowns to demonstrate feature influence during presentations.
- **📊 Probability & Risk Gauge:** Displays prediction badges and class probability percentages.
- **🎓 Viva Cheatsheet Sidebar:** Embedded viva notes for quick reference during evaluation.

---

## 🎓 7. Viva & Interview Questions Guide

<details>
<summary><b>Q1: What is the difference between phishing detection via blacklists vs. machine learning?</b></summary>
<br>
Blacklists rely on static databases of reported malicious URLs (e.g., Google Safe Browsing). While accurate, blacklists suffer from zero-day lag because new phishing domains remain active for hours before being reported. Machine learning analyzes intrinsic heuristic characteristics (URL syntax, domain age, page structure) to detect unknown phishing sites on first visit.
</details>

<details>
<summary><b>Q2: Why did Random Forest outperform Logistic Regression?</b></summary>
<br>
Logistic Regression is a linear classifier that assumes linear decision boundaries in the feature space. The heuristic security features exhibit non-linear interactions (e.g., a short URL is not dangerous on its own, but combined with an IP address and absent SSL, risk multiplies non-linearly). Random Forest creates 100 decorrelated decision trees that partition non-linear subspaces and robustly handle discrete categorical values <code>{-1, 0, 1}</code>.
</details>

<details>
<summary><b>Q3: In phishing detection, is Precision or Recall more important?</b></summary>
<br>
Both are critical, which is why <b>F1-score</b> is our primary metric.
<ul>
  <li><b>High Recall:</b> Ensures we catch as many malicious sites as possible (minimizing False Negatives, where a user falls for a scam).</li>
  <li><b>High Precision:</b> Ensures legitimate websites are not falsely blocked (minimizing False Positives, which annoys users and disrupts business operations).</li>
</ul>
Our Random Forest achieved <b>98.46% Recall</b> and <b>96.96% Precision</b>.
</details>

<details>
<summary><b>Q4: Why was the target column remapped from {-1, 1} to {0, 1}?</b></summary>
<br>
The original UCI dataset used <code>-1</code> for Phishing and <code>+1</code> for Legitimate. Modern gradient boosting frameworks like XGBoost calculate binary cross-entropy loss $\mathcal{L} = - [y \log(p) + (1-y)\log(1-p)]$, which mathematically requires targets in $\{0, 1\}$.
</details>

<details>
<summary><b>Q5: How does the URL Scanner extract features from a raw URL?</b></summary>
<br>
Using Python's <code>urllib.parse</code> and regular expressions, the app inspects:
<ul>
  <li><b>IP address:</b> RegEx checks for dotted IPv4 structure in hostname.</li>
  <li><b>URL Length:</b> Character count (<54: safe, 54-75: suspicious, >75: phishing).</li>
  <li><b>Shortener:</b> Matching known shortening domains (bit.ly, tinyurl, etc.).</li>
  <li><b>Special characters:</b> Checking for <code>@</code> and <code>//</code> redirection.</li>
  <li><b>Subdomains:</b> Counting dot separators in the domain part.</li>
  <li><b>SSL:</b> Inspecting whether protocol scheme is HTTPS.</li>
</ul>
</details>

---

## 👥 Authors & Acknowledgments
- **Project:** Phishing Website Detection AI
- **Dataset:** UCI Machine Learning Repository — Phishing Websites Dataset
