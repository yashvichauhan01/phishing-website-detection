import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('data/phishing_data.csv')
X = df.drop(columns=['index', 'Result'], errors='ignore')
y = df['Result'].map({-1: 0, 1: 1})
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
m = joblib.load('model.pkl')['model']
print(classification_report(y_test, m.predict(X_test), target_names=['Phishing (0)', 'Legitimate (1)'], digits=4))
print('ROC-AUC phishing:', round(roc_auc_score((y_test == 0).astype(int), m.predict_proba(X_test)[:, 0]), 4))   