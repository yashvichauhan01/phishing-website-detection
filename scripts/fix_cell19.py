import nbformat

nb = nbformat.read('url_model.ipynb', as_version=4)

s19 = """model_output_path = "url_model.pkl"
print(f"Serializing champion model to '{model_output_path}'...")

champion_res = next(r for r in results if r['Model'] == champion_name)

model_payload = {
    'model': champion_model,
    'model_name': champion_name,
    'feature_names': FEATURE_NAMES,
    'classes': {0: 'Phishing', 1: 'Legitimate'},
    'metrics': {
        'accuracy': champion_res['Accuracy'],
        'precision_phishing': champion_res['Precision (Phishing)'],
        'recall_phishing': champion_res['Recall (Phishing)'],
        'f1_phishing': champion_res['F1-Score (Phishing)'],
        'roc_auc_phishing': champion_res['ROC-AUC (Phishing)']
    },
    'all_models_summary': [
        {
            'model': r['Model'],
            'accuracy': r['Accuracy'],
            'precision_phishing': r['Precision (Phishing)'],
            'recall_phishing': r['Recall (Phishing)'],
            'f1_phishing': r['F1-Score (Phishing)'],
            'roc_auc_phishing': r['ROC-AUC (Phishing)']
        }
        for r in results
    ]
}

joblib.dump(model_payload, model_output_path)
print(f"Champion model successfully saved to '{model_output_path}'!")

# Verification test
verified = joblib.load(model_output_path)
print(f"Verification successful: Loaded '{verified['model_name']}' with {len(verified['feature_names'])} features.")
print(f"Metrics: Accuracy={verified['metrics']['accuracy']*100:.2f}%, F1={verified['metrics']['f1_phishing']*100:.2f}%")
"""

nb.cells[19].source = s19
with open('url_model.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Updated cell 19 successfully.')
