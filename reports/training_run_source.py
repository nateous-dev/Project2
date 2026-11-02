import os,sys,json,shutil,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
source=Path(r'C:\python\work\Bank_Customer_Churn_v003_제출용');root=source.parent/'Bank_Customer_Churn_v004_제출용'
if root.exists():raise RuntimeError('v004 already exists')
shutil.copytree(source,root,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
sys.path.insert(0,str(root))
from project_model import ProjectModel
from churn_core import metrics
adapter=ProjectModel(root);test=adapter.reference;raw=pd.read_csv(root/'Churn_Modelling.csv')
raw=raw.loc[raw.Exited.isin([0,1]) & raw.Age.between(21,60)].copy()
x=adapter.transform(raw)
test_keys=pd.MultiIndex.from_frame(test[adapter.names]);all_keys=pd.MultiIndex.from_frame(x)
pool=raw.loc[~all_keys.isin(test_keys)].copy()
assert len(pool)+len(test)==len(raw),'ambiguous test match'
train,val=train_test_split(pool,test_size=.2,random_state=42,stratify=pool.Exited)
xt,xv,xe=[adapter.transform(d) for d in [train,val,test]]
rows=[];models={};probs={}
estimators={'XGBoost':clone(adapter.pipeline.named_steps['model']),
 'MLP 신경망':MLPClassifier(hidden_layer_sizes=(64,32,16),activation='relu',early_stopping=True,validation_fraction=.15,n_iter_no_change=20,max_iter=300,random_state=42)}
for name,estimator in estimators.items():
 print('TRAIN',name,flush=True)
 pipe=Pipeline([('scaler',StandardScaler()),('model',estimator)])
 pipe.fit(xt,train.Exited.astype(int));vp=pipe.predict_proba(xv)[:,1]
 thresholds=np.arange(.1,.81,.01)
 scores=[metrics(val.Exited,vp,t)['f1_score'] for t in thresholds]
 t=float(round(thresholds[int(np.argmax(scores))],2))
 p=pipe.predict_proba(xe)[:,1]
 rows.append(dict(model=name,threshold=t,customers=len(test),validation_f1=max(scores),**metrics(test['실제이탈'],p,t),scope='공통 학습·검증 분리 / 기존 평가 고객 재사용'))
 models[name]=pipe;probs[name]=p
 joblib.dump(pipe,root/'models'/('MLP_New.pkl' if name.startswith('MLP') else 'XGBoost_New.pkl'))
winner=max(rows,key=lambda r:r['validation_f1']);name=winner['model']
# Selection uses validation F1 only, never evaluation F1.
joblib.dump(models[name],root/'models/Selected_Model.pkl')
meta=adapter.metadata;meta.update(final_model=name,model_file='Selected_Model.pkl',threshold=winner['threshold'],test_metrics={k:float(winner[k]) for k in ['accuracy','precision','recall','f1_score','roc_auc']})
(root/'models/model_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
test['이탈예측확률']=probs[name];test['이탈예측']=(probs[name]>=winner['threshold']).astype(int)
test.to_csv(root/'data/customer_risk.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(root/'reports/trained_comparison.csv',index=False,encoding='utf-8-sig')
audit={'selected_model':name,'criterion':'validation F1 only','train_customers':len(train),'validation_customers':len(val),'evaluation_customers':len(test),
       'training_ids':train.CustomerId.astype(str).tolist(),'validation_ids':val.CustomerId.astype(str).tolist(),
       'architecture':[64,32,16],'note':'同じ過去評価対象の再利用。将来の新規データでの検証ではありません。'}
audit['note']='過去評価対象の再利用'.replace('過去評価対象の再利用','기존 평가 고객 재사용: 미래 고객 검증은 아님')
(root/'reports/training_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows)[['model','threshold','validation_f1','accuracy','precision','recall','f1_score']].to_string(index=False),flush=True)
print('SELECTED',name,flush=True)
