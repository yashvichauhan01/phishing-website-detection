"""
================================================================================
Phishing Website Detection - Phase 2: URL-Only Lexical Model Pipeline
================================================================================
File: train_url_model.py
Purpose: Standalone CLI pipeline that trains and evaluates Model B:
         1. Loads 'data/phiusiil.csv' (using only 'URL' and 'label' columns).
         2. Drops duplicate URLs.
         3. Extracts hostnames and performs GroupShuffleSplit by domain to eliminate data leakage.
         4. Computes 26 purely lexical features using url_features.py.
         5. Tunes and benchmarks Logistic Regression, Random Forest, and XGBoost via 5-fold CV.
         6. Evaluates test set metrics strictly focused on Phishing (label 0):
            Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
         7. Generates confusion matrices and SHAP feature importance.
         8. Exports the champion model to `url_model.pkl`.

Author: Yashvi Chauhan
Project: Phishing Website Detection (Phase 2)
================================================================================
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import shap

from url_features import (
    extract_url_features_df,
    extract_hostname,
    FEATURE_NAMES
)


def run_phase2_training():
    print("=" * 80)
    print(" PHISHING WEBSITE DETECTION - PHASE 2: URL-ONLY LEXICAL MODEL PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    data_path = os.path.join("data", "phiusiil.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PhiUSIIL dataset not found at '{data_path}'")

    print(f"\n[1/7] Loading dataset from '{data_path}' (URL & label columns only)...")
    # Load strictly URL and label columns to bypass any encoding issues in unused columns
    df_raw = pd.read_csv(data_path, usecols=["URL", "label"])
    print(f"      Raw sample count: {len(df_raw):,}")

    # Deduplicate by URL
    print("      Dropping duplicate URLs...")
    df = df_raw.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    print(f"      Deduplicated sample count: {len(df):,}")
    print(f"      Class Breakdown -> Phishing (0): {(df['label'] == 0).sum():,} "
          f"| Legitimate (1): {(df['label'] == 1).sum():,}")

    # -------------------------------------------------------------------------
    # 2. Domain-Aware Group Splitting (Viva Core: GroupShuffleSplit)
    # -------------------------------------------------------------------------
    print("\n[2/7] Extracting hostnames and performing domain-aware GroupShuffleSplit...")
    df["hostname"] = df["URL"].apply(extract_hostname)

    # GroupShuffleSplit ensures that ALL URLs belonging to the same registered
    # domain/hostname are placed either entirely in the training split or entirely
    # in the test split. This prevents synthetic over-performance from domain leakage.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["hostname"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"      Training samples : {len(train_df):,} (Groups: {train_df['hostname'].nunique():,})")
    print(f"      Testing samples  : {len(test_df):,} (Groups: {test_df['hostname'].nunique():,})")

    # Safety check: Zero hostname overlap
    overlap = set(train_df["hostname"]).intersection(set(test_df["hostname"]))
    print(f"      Hostname overlap verification: {len(overlap)} shared domains (Zero leakage confirmed).")

    # -------------------------------------------------------------------------
    # 3. Lexical Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[3/7] Extracting 26 lexical features using url_features.py...")
    t0 = time.time()
    X_train = extract_url_features_df(train_df["URL"])
    y_train = train_df["label"].values

    X_test = extract_url_features_df(test_df["URL"])
    y_test = test_df["label"].values
    print(f"      Feature extraction finished in {time.time() - t0:.2f}s")
    print(f"      Features shape -> Train: {X_train.shape}, Test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # 4. 5-Fold Cross-Validation and GridSearchCV Tuning
    # -------------------------------------------------------------------------
    print("\n[4/7] Hyperparameter tuning via 5-Fold Cross-Validation & GridSearchCV...")

    # A. Logistic Regression (Linear Baseline with standard scaling)
    print("      [A/C] Tuning Logistic Regression...")
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, random_state=42))
    ])
    grid_lr = GridSearchCV(
        pipe_lr,
        param_grid={"lr__C": [0.1, 1.0]},
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    t0 = time.time()
    grid_lr.fit(X_train, y_train)
    print(f"            Tuned in {time.time() - t0:.2f}s | Best Params: {grid_lr.best_params_} | CV F1: {grid_lr.best_score_:.4f}")

    # B. Random Forest (Bagging Ensemble)
    print("      [B/C] Tuning Random Forest...")
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_rf = GridSearchCV(
        rf_base,
        param_grid={"n_estimators": [50, 100], "max_depth": [10, 14]},
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    t0 = time.time()
    grid_rf.fit(X_train, y_train)
    print(f"            Tuned in {time.time() - t0:.2f}s | Best Params: {grid_rf.best_params_} | CV F1: {grid_rf.best_score_:.4f}")

    # C. XGBoost (Gradient Boosting)
    print("      [C/C] Tuning XGBoost...")
    xgb_base = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    grid_xgb = GridSearchCV(
        xgb_base,
        param_grid={"n_estimators": [50, 100], "max_depth": [4, 6]},
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    t0 = time.time()
    grid_xgb.fit(X_train, y_train)
    print(f"            Tuned in {time.time() - t0:.2f}s | Best Params: {grid_xgb.best_params_} | CV F1: {grid_xgb.best_score_:.4f}")

    # -------------------------------------------------------------------------
    # 5. Comprehensive Test Set Evaluation (Focus: Phishing Class = 0)
    # -------------------------------------------------------------------------
    print("\n[5/7] Evaluating tuned models on unseen test split (Target: Phishing Class = 0)...")
    models = {
        "Logistic Regression": grid_lr.best_estimator_,
        "Random Forest": grid_rf.best_estimator_,
        "XGBoost": grid_xgb.best_estimator_
    }

    eval_results = []
    print("-" * 92)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Prec (Phish)':<12} | {'Rec (Phish)':<12} | {'F1 (Phish)':<10} | {'ROC-AUC':<8}")
    print("-" * 92)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        # Specifically targeting label 0 (Phishing)
        prec_phish = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec_phish = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_phish = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        # ROC-AUC treating label 0 as the positive class
        auc_phish = roc_auc_score((y_test == 0).astype(int), y_proba[:, 0])
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        eval_results.append({
            "model_name": name,
            "model_obj": model,
            "accuracy": acc,
            "precision_phishing": prec_phish,
            "recall_phishing": rec_phish,
            "f1_phishing": f1_phish,
            "roc_auc_phishing": auc_phish,
            "confusion_matrix": cm
        })

        print(f"{name:<22} | {acc*100:6.2f}%   | {prec_phish*100:6.2f}%      | {rec_phish*100:6.2f}%      | {f1_phish*100:6.2f}%   | {auc_phish*100:6.2f}%")
    print("-" * 92)

    # Identify Champion Model based on F1-Score for Phishing & ROC-AUC
    champion_info = max(eval_results, key=lambda x: (x["f1_phishing"], x["roc_auc_phishing"]))
    print(f"\n      >>> Champion Selected: {champion_info['model_name']} "
          f"(F1 Phishing: {champion_info['f1_phishing']*100:.2f}%, Accuracy: {champion_info['accuracy']*100:.2f}%)")

    # Display Confusion Matrices
    print("\n      Confusion Matrices ([Row: Actual 0/1, Col: Predicted 0/1]):")
    for r in eval_results:
        print(f"      [{r['model_name']}]\n{r['confusion_matrix']}")

    # -------------------------------------------------------------------------
    # 6. SHAP Feature Importance Analysis
    # -------------------------------------------------------------------------
    print("\n[6/7] Computing SHAP feature importance for Champion Model...")
    champion_model = champion_info["model_obj"]
    
    # Subsample 1,000 test points for fast and representative SHAP calculation
    shap_sample = X_test.iloc[:1000]
    try:
        explainer = shap.TreeExplainer(champion_model)
        shap_values = explainer.shap_values(shap_sample)
        if isinstance(shap_values, list):
            # Binary classifier: index 0 is Phishing, index 1 is Legitimate
            shap_matrix = shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 0]
        else:
            shap_matrix = shap_values
            
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        print("      Top 10 Most Important Features according to SHAP:")
        for idx, row in shap_importance_df.head(10).iterrows():
            print(f"        {idx+1:2d}. {row['Feature']:<25} : {row['Mean_Abs_SHAP']:.4f}")
    except Exception as e:
        print(f"      SHAP calculation notice: {e}")
        shap_importance_df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 7. Save Best Model to url_model.pkl
    # -------------------------------------------------------------------------
    print("\n[7/7] Serializing best model as 'url_model.pkl' via joblib...")
    payload = {
        "model": champion_model,
        "model_name": champion_info["model_name"],
        "feature_names": FEATURE_NAMES,
        "classes": {0: "Phishing", 1: "Legitimate"},
        "metrics": {
            "accuracy": champion_info["accuracy"],
            "precision_phishing": champion_info["precision_phishing"],
            "recall_phishing": champion_info["recall_phishing"],
            "f1_phishing": champion_info["f1_phishing"],
            "roc_auc_phishing": champion_info["roc_auc_phishing"]
        },
        "all_models_summary": [
            {
                "model": r["model_name"],
                "accuracy": r["accuracy"],
                "precision_phishing": r["precision_phishing"],
                "recall_phishing": r["recall_phishing"],
                "f1_phishing": r["f1_phishing"],
                "roc_auc_phishing": r["roc_auc_phishing"],
                "confusion_matrix": r["confusion_matrix"].tolist()
            }
            for r in eval_results
        ]
    }
    joblib.dump(payload, "url_model.pkl")
    print("      Saved 'url_model.pkl' successfully.")

    # Verification load
    verify_load = joblib.load("url_model.pkl")
    print(f"      Verification: successfully reloaded '{verify_load['model_name']}' with {len(verify_load['feature_names'])} features.")
    print("\n" + "=" * 80)
    print(" PHASE 2 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_training()
