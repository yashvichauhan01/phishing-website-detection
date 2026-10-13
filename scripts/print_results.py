import joblib

p = joblib.load('url_model.pkl')
print('Champion:', p['model_name'])
print('Keys saved:', list(p.keys()))
for m in p['all_models_summary']:
    print('=====', m['model'])
    for k, v in m.items():
        if k == 'model':
            continue
        if isinstance(v, float):
            v = round(v * 100, 2)
        print(' ', k, ':', v)