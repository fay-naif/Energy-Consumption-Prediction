"""Reproduce the notebook's Extra Trees random-split result and save the app bundle."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from features import make_features
ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / 'data' / 'energydata_complete.csv')
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values('date').reset_index(drop=True)
df = make_features(raw)
X = df.drop(columns=['Appliances'])
y = df['Appliances']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
def build_model():
    return ExtraTreesRegressor(n_estimators=300, max_features=.7, random_state=42, n_jobs=-1)
model = build_model()
model.fit(X_train, y_train)
pred = model.predict(X_test)
def metrics(actual, predicted):
    return {'r2':float(r2_score(actual,predicted)), 'mae':float(mean_absolute_error(actual,predicted)), 'rmse':float(mean_squared_error(actual,predicted)**.5)}
result = {'model':'Extra Trees', 'evaluation':'random 80/20 split', **metrics(y_test,pred), 'n_train':len(X_train), 'n_test':len(X_test)}
(ROOT/'models').mkdir(exist_ok=True)
joblib.dump({'model':model,'features':list(X.columns),'defaults':X_train.median().to_dict(),'model_name':'Extra Trees'}, ROOT/'models'/'energy_model.joblib',compress=3)
(ROOT/'reports'/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('Random split:',result,flush=True)
# Separate chronological holdout, not used to select or train the deployed model.
split = int(len(X)*.8)
chronological_model = build_model()
chronological_model.fit(X.iloc[:split], y.iloc[:split])
chrono_pred = chronological_model.predict(X.iloc[split:])
chrono = {'model':'Extra Trees', 'evaluation':'chronological 80/20 holdout', **metrics(y.iloc[split:],chrono_pred), 'n_train':split, 'n_test':len(X)-split}
(ROOT/'reports'/'chronological_metrics.json').write_text(json.dumps(chrono,ensure_ascii=False,indent=2),encoding='utf8')
print('Chronological holdout:',chrono,flush=True)
