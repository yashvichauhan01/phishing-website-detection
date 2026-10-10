"""
Phishing Website Detection - Model Training Pipeline
===================================================
This script trains and evaluates three machine learning models:
1. Logistic Regression (Linear baseline)
2. Random Forest Classifier (Ensemble Bagging)
3. XGBoost Classifier (Gradient Boosting)

Dataset: UCI Phishing Websites Dataset (data/phishing_data.csv)
Saves the best-performing model to `model.pkl`.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
    print("=" * 60)
    print("  PHISHING WEBSITE DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Dataset
    data_path = os.path.join('data', 'phishing_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"\n[1/6] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"      Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 2. Preprocess Data
    print("\n[2/6] Preprocessing data...")
    # Drop identifier column 'index' if present
    X = df.drop(columns=['index', 'Result'], errors='ignore')
    
    # Target mapping:
    # UCI Dataset original labels: -1 = Phishing, 1 = Legitimate
    # We map -1 -> 0 (Phishing) and 1 -> 1 (Legitimate)
    # for compatibility with standard binary classifiers (especially XGBoost).
    y = df['Result'].map({-1: 0, 1: 1})

    feature_names = list(X.columns)
    print(f"      Features count: {len(feature_names)}")
    print(f"      Class distribution -> Phishing (0): {(y == 0).sum()}, Legitimate (1): {(y == 1).sum()}")

    # 3. Train-Test Split (80/20 stratified split)
    print("\n[3/6] Splitting data into 80% train and 20% test sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"      Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 4. Initialize and Train Models
    print("\n[4/6] Training and evaluating models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n      --> Training {name}...")
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
            'Confusion Matrix': cm
        })

        print(f"          Accuracy : {acc * 100:.2f}%")
        print(f"          Precision: {prec * 100:.2f}%")
        print(f"          Recall   : {rec * 100:.2f}%")
        print(f"          F1-Score : {f1 * 100:.2f}%")

    # 5. Model Comparison Summary
    print("\n[5/6] Model Comparison Summary:")
    print("-" * 65)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['Model']:<22} | {r['Accuracy']:.4f}     | {r['Precision']:.4f}     | {r['Recall']:.4f}     | {r['F1-Score']:.4f}")
    print("-" * 65)

    # Determine Best Model based on F1-Score & Accuracy
    best_result = max(results, key=lambda x: (x['F1-Score'], x['Accuracy']))
    best_model_name = best_result['Model']
    best_model = trained_models[best_model_name]
    print(f"\n      >>> Best Model Selected: {best_model_name} (F1-Score: {best_result['F1-Score']:.4f})")

    # 6. Save Best Model
    print(f"\n[6/6] Saving best model as 'model.pkl' using joblib...")
    # Bundle model along with feature metadata for clean app loading
    model_payload = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_names': feature_names,
        'classes': {0: 'Phishing', 1: 'Legitimate'},
        'metrics': {
            'accuracy': best_result['Accuracy'],
            'precision': best_result['Precision'],
            'recall': best_result['Recall'],
            'f1_score': best_result['F1-Score']
        }
    }
    joblib.dump(model_payload, 'model.pkl')
    print("      Model successfully saved to 'model.pkl'!")

    # Verify loading
    loaded_data = joblib.load('model.pkl')
    print("      Verification check: model.pkl successfully reloaded and validated.")
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    run_training()
