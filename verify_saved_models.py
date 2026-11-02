import os,sys,json,hashlib
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd,numpy as np,joblib
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from churn_core import metrics
import zipfile,io,h5py
d=pd.read_csv(root/'data/customer_risk.csv');meta=json.loads((root/'models/model_metadata.json').read_text())
x=d[meta['features']];y=d['실제이탈']
pre=joblib.load(root/'models/MLP_Preprocessor.pkl');z=pre.transform(x)
if hasattr(z,'toarray'):z=z.toarray()
archive=zipfile.ZipFile(root/'models/MLP_B_Comparison.keras')
config=json.loads(archive.read('config.json'))
h=h5py.File(io.BytesIO(archive.read('model.weights.h5')))
z=np.asarray(z,dtype='float32');dense_index=0
for layer in config['config']['layers']:
 kind=layer['class_name']
 if kind in ['InputLayer','Dropout']:continue
 if kind!='Dense':raise ValueError('지원하지 않는 신경망 계층: '+kind)
 group='dense' if dense_index==0 else f'dense_{dense_index}'
 weights=np.array(h[f'layers/{group}/vars/0']);bias=np.array(h[f'layers/{group}/vars/1'])
 z=z@weights+bias;activation=layer['config']['activation']
 if activation=='relu':z=np.maximum(z,0)
 elif activation=='sigmoid':z=1/(1+np.exp(-np.clip(z,-80,80)))
 else:raise ValueError('지원하지 않는 활성화 함수')
 dense_index+=1
p=z.ravel()
assert len(p)==len(d) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
rows=[]
for label,path,prob,t in [('XGBoost 最終',root/'models/XGBoost_Final.pkl',joblib.load(root/'models/XGBoost_Final.pkl').predict_proba(x)[:,1],.52),('MLP-B 深層学習',root/'models/MLP_B_Comparison.keras',p,.5)]:
 rows.append(dict(model='XGBoost 최종' if label.startswith('XG') else 'MLP-B 신경망',threshold=t,customers=len(d),**{k:float(v) for k,v in metrics(y,prob,t).items()},model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='동일 과거 고객 재계산 · 독립 평가 여부 미확인'))
(root/'reports').mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(root/'reports/verified_models.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'실제이탈':y,'MLP확률':p}).to_csv(root/'reports/mlp_replay.csv',index=False,encoding='utf-8-sig')

print(pd.DataFrame(rows)[['model','accuracy','precision','recall','f1_score']].to_string(index=False))
