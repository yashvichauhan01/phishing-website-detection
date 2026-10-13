"""
Phishing Website Detection - Presentation Asset Generator
=========================================================
File: scripts/generate_presentation_assets.py
Purpose: Generates high-resolution PNG plots (dpi=200, polished layout) in results/
         for presentation slides, project report, and thesis defense.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import shap

from url_features import extract_url_features_df, extract_hostname, FEATURE_NAMES

def generate_all_assets():
    print("=" * 80)
    print(" GENERATING HIGH-RESOLUTION PRESENTATION ASSETS IN results/")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 15
    })

    # -------------------------------------------------------------------------
    # 1. Class Distribution Plot
    # -------------------------------------------------------------------------
    print("\n[1/7] Generating class_distribution.png...")
    # Load UCI dataset
    df_uci = pd.read_csv("data/phishing_data.csv")
    df_uci_clean = df_uci.drop(columns=['index'], errors='ignore').drop_duplicates()
    uci_counts = df_uci_clean['Result'].map({-1: 'Phishing', 1: 'Legitimate'}).value_counts()

    # Load PhiUSIIL dataset
    df_phi = pd.read_csv("data/phiusiil.csv", usecols=['URL', 'label'])
    df_phi_clean = df_phi.drop_duplicates(subset=['URL'])
    phi_counts = df_phi_clean['label'].map({0: 'Phishing', 1: 'Legitimate'}).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    palette = {'Phishing': '#ef4444', 'Legitimate': '#10b981'}

    # UCI plot
    bars1 = axes[0].bar(uci_counts.index, uci_counts.values, color=[palette[k] for k in uci_counts.index], width=0.55, edgecolor='black', linewidth=1)
    axes[0].set_title("Phase 1: UCI Dataset Distribution\n(5,849 Deduplicated Instances)", fontweight="bold")
    axes[0].set_ylabel("Sample Count")
    axes[0].set_ylim(0, max(uci_counts.values) * 1.15)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,} ({yval/len(df_uci_clean)*100:.1f}%)", ha='center', va='bottom', fontweight='bold')

    # PhiUSIIL plot
    bars2 = axes[1].bar(phi_counts.index, phi_counts.values, color=[palette[k] for k in phi_counts.index], width=0.55, edgecolor='black', linewidth=1)
    axes[1].set_title("Phase 2: PhiUSIIL Dataset Distribution\n(235,370 Deduplicated Samples)", fontweight="bold")
    axes[1].set_ylabel("Sample Count")
    axes[1].set_ylim(0, max(phi_counts.values) * 1.15)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 2000, f"{yval:,} ({yval/len(df_phi_clean)*100:.1f}%)", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=200)
    plt.close()
    print("      Saved results/class_distribution.png")

    # -------------------------------------------------------------------------
    # 2. Model Comparison Bar Chart
    # -------------------------------------------------------------------------
    print("\n[2/7] Generating model_comparison.png...")
    # Phase 1 metrics (post-dedup)
    p1_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost (Champion)'],
        'Accuracy': [91.28, 94.19, 95.21],
        'Precision': [90.00, 93.68, 95.37],
        'Recall': [92.23, 94.35, 94.70],
        'F1-Score': [91.10, 94.01, 95.04]
    }
    df_p1 = pd.DataFrame(p1_data).melt(id_vars='Model', var_name='Metric', value_name='Score')

    # Phase 2 metrics (Phishing class = 0)
    p2_data = {
        'Model': ['Logistic Regression', 'Random Forest (Champion)', 'XGBoost'],
        'Accuracy': [99.38, 99.60, 99.60],
        'Precision': [99.96, 99.90, 99.91],
        'Recall': [98.60, 99.17, 99.17],
        'F1-Score': [99.28, 99.53, 99.54]
    }
    df_p2 = pd.DataFrame(p2_data).melt(id_vars='Model', var_name='Metric', value_name='Score')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=200)

    sns.barplot(data=df_p1, x='Model', y='Score', hue='Metric', ax=axes[0], palette='Blues_r')
    axes[0].set_title("Phase 1: UCI Heuristic Model Benchmarks", fontweight='bold')
    axes[0].set_ylim(80, 100)
    axes[0].set_ylabel("Score (%)")
    axes[0].legend(loc='lower right')

    sns.barplot(data=df_p2, x='Model', y='Score', hue='Metric', ax=axes[1], palette='Purples_r')
    axes[1].set_title("Phase 2: Lexical URL Model Benchmarks (Phishing Class)", fontweight='bold')
    axes[1].set_ylim(95, 100.5)
    axes[1].set_ylabel("Score (%)")
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=200)
    plt.close()
    print("      Saved results/model_comparison.png")

    # -------------------------------------------------------------------------
    # 3. Confusion Matrices of Each Champion Model
    # -------------------------------------------------------------------------
    print("\n[3/7] Generating confusion_matrices.png...")
    # Load Phase 1 Champion (XGBoost)
    m1_payload = joblib.load('model.pkl')
    m1_model = m1_payload['model']

    X_u = df_uci_clean.drop(columns=['Result'])
    y_u = df_uci_clean['Result'].map({-1: 0, 1: 1})
    _, X_u_test, _, y_u_test = train_test_split(X_u, y_u, test_size=0.20, random_state=42, stratify=y_u)
    y_u_pred = m1_model.predict(X_u_test)
    cm_p1 = confusion_matrix(y_u_test, y_u_pred)

    # Load Phase 2 Champion (Random Forest)
    m2_payload = joblib.load('url_model.pkl')
    m2_model = m2_payload['model']

    df_phi_clean['hostname'] = df_phi_clean['URL'].apply(extract_hostname)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(gss.split(df_phi_clean, groups=df_phi_clean['hostname']))
    test_phi = df_phi_clean.iloc[test_idx].reset_index(drop=True)
    X_phi_test = extract_url_features_df(test_phi['URL'])
    y_phi_test = test_phi['label'].values
    y_phi_pred = m2_model.predict(X_phi_test)
    cm_p2 = confusion_matrix(y_phi_test, y_phi_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    sns.heatmap(cm_p1, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Pred Phish (0)', 'Pred Legit (1)'],
                yticklabels=['Actual Phish (0)', 'Actual Legit (1)'], annot_kws={'size': 13, 'weight': 'bold'})
    axes[0].set_title(f"Phase 1 Champion: {m1_payload['model_name']}\nTest Set Confusion Matrix (N = 1,170)", fontweight='bold')

    sns.heatmap(cm_p2, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
                xticklabels=['Pred Phish (0)', 'Pred Legit (1)'],
                yticklabels=['Actual Phish (0)', 'Actual Legit (1)'], annot_kws={'size': 13, 'weight': 'bold'})
    axes[1].set_title(f"Phase 2 Champion: {m2_payload['model_name']}\nTest Set Confusion Matrix (N = 47,514)", fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=200)
    plt.close()
    print("      Saved results/confusion_matrices.png")

    # -------------------------------------------------------------------------
    # 4. ROC Curves of Each Champion Model
    # -------------------------------------------------------------------------
    print("\n[4/7] Generating roc_curves.png...")
    y_u_prob = m1_model.predict_proba(X_u_test)[:, 1]
    fpr_u, tpr_u, _ = roc_curve(y_u_test, y_u_prob)
    auc_u = roc_auc_score(y_u_test, y_u_prob)

    y_phi_prob_phish = m2_model.predict_proba(X_phi_test)[:, 0]
    fpr_phi, tpr_phi, _ = roc_curve((y_phi_test == 0).astype(int), y_phi_prob_phish)
    auc_phi = roc_auc_score((y_phi_test == 0).astype(int), y_phi_prob_phish)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    axes[0].plot(fpr_u, tpr_u, color='#2563eb', lw=2.5, label=f"{m1_payload['model_name']} (AUC = {auc_u*100:.2f}%)")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    axes[0].set_title(f"Phase 1 Champion ROC Curve ({m1_payload['model_name']})", fontweight='bold')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc='lower right', frameon=True)

    axes[1].plot(fpr_phi, tpr_phi, color='#16a34a', lw=2.5, label=f"{m2_payload['model_name']} (AUC = {auc_phi*100:.2f}%)")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    axes[1].set_title(f"Phase 2 Champion ROC Curve (Phishing Class = 0)", fontweight='bold')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=200)
    plt.close()
    print("      Saved results/roc_curves.png")

    # -------------------------------------------------------------------------
    # 5. Feature Importance for Phase 1 Model
    # -------------------------------------------------------------------------
    print("\n[5/7] Generating feature_importance_phase1.png...")
    feat_names_u = list(X_u.columns)
    importances_u = m1_model.feature_importances_
    sorted_u = np.argsort(importances_u)[::-1]

    plt.figure(figsize=(10, 6.5), dpi=200)
    sns.barplot(x=importances_u[sorted_u][:15], y=[feat_names_u[i] for i in sorted_u][:15], palette='Blues_r')
    plt.title(f"Top 15 Most Influential Features — Phase 1 ({m1_payload['model_name']})", fontweight='bold')
    plt.xlabel("Gini / Gain Feature Importance")
    plt.ylabel("Heuristic Feature")
    plt.tight_layout()
    plt.savefig("results/feature_importance_phase1.png", dpi=200)
    plt.close()
    print("      Saved results/feature_importance_phase1.png")

    # -------------------------------------------------------------------------
    # 6. SHAP Summary Plot for Phase 2 Model
    # -------------------------------------------------------------------------
    print("\n[6/7] Generating shap_summary.png...")
    shap_sample = X_phi_test.sample(n=min(1000, len(X_phi_test)), random_state=42)
    explainer = shap.TreeExplainer(m2_model)
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        shap_vals_matrix = shap_values[0]
    elif len(shap_values.shape) == 3:
        shap_vals_matrix = shap_values[:, :, 0]
    else:
        shap_vals_matrix = shap_values

    mean_shap = np.abs(shap_vals_matrix).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Mean_SHAP': mean_shap}).sort_values(by='Mean_SHAP', ascending=False)

    plt.figure(figsize=(10, 6), dpi=200)
    sns.barplot(data=shap_df.head(12), x='Mean_SHAP', y='Feature', palette='mako')
    plt.title(f"SHAP Global Feature Importance — Phase 2 ({m2_payload['model_name']})", fontweight='bold')
    plt.xlabel("Mean |SHAP Value| (Impact on Phishing Detection)")
    plt.ylabel("Lexical Feature")
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=200)
    plt.close()
    print("      Saved results/shap_summary.png")

    # -------------------------------------------------------------------------
    # 7. System Architecture Diagram (Matplotlib)
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating system_architecture.png...")
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    ax.axis('off')

    box_kw = dict(boxstyle="round,pad=0.5", linewidth=1.5)
    
    # Components
    ax.text(0.12, 0.75, "User / Browser / Gateway\n(Raw URL Input)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=2), fontsize=11, fontweight='bold')
    ax.annotate("", xy=(0.28, 0.75), xytext=(0.22, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#0f172a"))

    ax.text(0.42, 0.75, "Phase 2 Edge Engine\n(url_features.py)\nExtracts 26 Lexical Features\nLatency < 0.1 ms", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor="#fef3c7", edgecolor="#d97706", linewidth=2), fontsize=10, fontweight='bold')
    ax.annotate("", xy=(0.58, 0.75), xytext=(0.54, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#0f172a"))

    ax.text(0.72, 0.75, "Model B: Random Forest\n(url_model.pkl)\nFast Threat Classifier", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=2), fontsize=10, fontweight='bold')

    # Confidence branch
    ax.annotate("", xy=(0.72, 0.52), xytext=(0.72, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color="#d97706"))
    ax.text(0.73, 0.58, "Confidence: 40% - 80%\n(Borderline / Ambiguous)", fontsize=9, color="#b45309", fontweight='bold')

    ax.annotate("", xy=(0.87, 0.75), xytext=(0.82, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#16a34a"))
    ax.text(0.92, 0.75, "Instant Verdict\n(> 95% Conf)\nPhish / Safe", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.5), fontsize=10, fontweight='bold')

    # Phase 1 deep scraper branch
    ax.text(0.72, 0.35, "Phase 1 Deep Engine (Model A)\nLive HTML Scraping + WHOIS/DNS\nEvaluated via XGBoost (model.pkl)\nLatency: 2.0 - 5.0 s", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", facecolor="#fee2e2", edgecolor="#dc2626", linewidth=2), fontsize=10, fontweight='bold')
    ax.annotate("", xy=(0.87, 0.35), xytext=(0.82, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color="#dc2626"))

    ax.text(0.92, 0.35, "Deep Defense\nFinal Security\nVerdict", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.5), fontsize=10, fontweight='bold')

    plt.title("Dual-Phase Hybrid Phishing Detection Architecture", fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("results/system_architecture.png", dpi=200)
    plt.close()
    print("      Saved results/system_architecture.png")

    print("\n" + "=" * 80)
    print(" ALL PRESENTATION ASSETS GENERATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    generate_all_assets()
