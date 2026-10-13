"""
Phishing Website Detection - Model Training Pipeline (Phase 1)
==============================================================
This script trains and evaluates three machine learning models:
1. Logistic Regression (Linear baseline)
2. Random Forest Classifier (Ensemble Bagging)
3. XGBoost Classifier (Gradient Boosting)

Dataset: UCI Phishing Websites Dataset (data/phishing_data.csv)
Saves the champion model chosen via 5-fold Stratified Cross-Validation to `model.pkl`.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_training():
    print("=" * 65)
    print("  PHISHING WEBSITE DETECTION - MODEL TRAINING PIPELINE (PHASE 1)")
    print("=" * 65)

    # 1. Load Dataset
    data_path = os.path.join('data', 'phishing_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"\n[1/6] Loading dataset from '{data_path}'...")
    df_raw = pd.read_csv(data_path)
    raw_rows, raw_cols = df_raw.shape
    print(f"      Raw Rows: {raw_rows}, Columns: {raw_cols}")

    # Drop identifier column 'index' if present
    feature_df = df_raw.drop(columns=['index'], errors='ignore')

    # Count duplicates excluding 'index'
    n_duplicates = feature_df.duplicated().sum()
    print(f"      Duplicates detected (excluding 'index'): {n_duplicates:,}")

    # Drop duplicates BEFORE train/test split to prevent duplicate data leakage
    df_clean = feature_df.drop_duplicates().reset_index(drop=True)
    clean_rows = len(df_clean)
    print(f"      Cleaned dataset rows: {clean_rows:,} (dropped {n_duplicates:,} duplicate rows)")

    # 2. Preprocess Data
    print("\n[2/6] Preprocessing data...")
    X = df_clean.drop(columns=['Result'])
    
    # Target mapping:
    # UCI Dataset original labels: -1 = Phishing, 1 = Legitimate
    # We map -1 -> 0 (Phishing) and 1 -> 1 (Legitimate)
    # for compatibility with standard binary classifiers (especially XGBoost).
    y = df_clean['Result'].map({-1: 0, 1: 1})

    feature_names = list(X.columns)
    print(f"      Features count: {len(feature_names)}")
    print(f"      Class distribution -> Phishing (0): {(y == 0).sum():,}, Legitimate (1): {(y == 1).sum():,}")

    # 3. Train-Test Split (80/20 stratified split with random_state=42)
    print("\n[3/6] Splitting data into 80% train and 20% test sets (stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"      Training samples: {X_train.shape[0]:,}, Test samples: {X_test.shape[0]:,}")

    # 4. 5-Fold Stratified Cross-Validation on Training Set (Fixes Model Selection Leakage)
    print("\n[4/6] Running 5-Fold Stratified Cross-Validation on Training Set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    }

    cv_scores = {}
    print("-" * 65)
    print(f"{'Model':<22} | {'5-Fold CV F1':<15} | {'Std Dev':<10}")
    print("-" * 65)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"{name:<22} | {scores.mean():.4f}          | +/- {scores.std():.4f}")
    print("-" * 65)

    # Select champion model strictly based on CV performance on training data
    champion_name = max(cv_scores, key=cv_scores.get)
    print(f"\n      >>> Champion Selected via 5-Fold CV: {champion_name} (CV F1: {cv_scores[champion_name]:.4f})")

    # 5. Fit on Train Set & Evaluate on Test Set
    print("\n[5/6] Fitting on Training Set and evaluating on unseen Test Set...")
    results = []
    trained_models = {}

    print("-" * 75)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        trained_models[name] = model
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Confusion Matrix': cm,
            'CV_F1': cv_scores[name]
        })
        print(f"{name:<22} | {acc*100:6.2f}%    | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%")
    print("-" * 75)

    # Print comparison with pre-deduplication (raw data) baseline
    print("\n      [Audit] Comparison: Pre-Deduplication vs Post-Deduplication Test Accuracy:")
    old_accuracies = {
        'Logistic Regression': 0.9258,
        'Random Forest': 0.9742,
        'XGBoost': 0.9692
    }
    for r in results:
        m_name = r['Model']
        old_acc = old_accuracies.get(m_name, 0.0)
        new_acc = r['Accuracy']
        diff = (new_acc - old_acc) * 100
        print(f"        • {m_name:<20}: Old = {old_acc*100:5.2f}% | New = {new_acc*100:5.2f}% | Shift = {diff:+5.2f}%")

    champion_result = next(r for r in results if r['Model'] == champion_name)
    champion_model = trained_models[champion_name]

    # 6. Save Champion Model
    print(f"\n[6/6] Saving champion model as 'model.pkl' using joblib...")
    model_payload = {
        'model': champion_model,
        'model_name': champion_name,
        'feature_names': feature_names,
        'classes': {0: 'Phishing', 1: 'Legitimate'},
        'metrics': {
            'accuracy': champion_result['Accuracy'],
            'precision': champion_result['Precision'],
            'recall': champion_result['Recall'],
            'f1_score': champion_result['F1-Score']
        }
    }
    joblib.dump(model_payload, 'model.pkl')
    print("      Model successfully saved to 'model.pkl'!")

    # Verify loading
    loaded_data = joblib.load('model.pkl')
    print(f"      Verification check: model.pkl successfully reloaded with {loaded_data['model_name']}.")
    print("\nPhase 1 Training completed successfully!")

if __name__ == '__main__':
    run_training()
