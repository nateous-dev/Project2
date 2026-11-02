"""Adapter for this project's saved StandardScaler + XGBoost pipeline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from churn_core import NUMERIC, validate_features


class ProjectModel:
    def __init__(self, root):
        import joblib
        self.root = Path(root)
        self.metadata = json.loads((self.root/'models/model_metadata.json').read_text(encoding='utf-8-sig'))
        self.pipeline = joblib.load(self.root/'models'/self.metadata.get('model_file','XGBoost_Final.pkl'))
        self.names = list(self.pipeline.feature_names_in_)
        if self.names != self.metadata['features']:
            raise ValueError('모델의 입력 순서와 모델 설명 파일이 다릅니다.')
        if list(self.pipeline.classes_) != [0, 1]:
            raise ValueError('유지=0, 이탈=1인 분류 모델이 필요합니다.')
        self.reference = pd.read_csv(self.root/'data/customer_risk.csv')
        # The available project artifact verifies decade bands for ages 21..60.
        age = self.reference['Age']
        if not self.reference['Age_step'].eq((age-1)//10).all():
            raise ValueError('저장된 연령대 변환 규칙을 확인하지 못했습니다.')
        self.age_min, self.age_max = int(age.min()), int(age.max())
        self.replayed = self.predict(self.reference)
        self.max_difference = float(np.max(np.abs(self.replayed - self.reference['이탈예측확률'])))
        if self.max_difference > 1e-5:
            raise ValueError('현재 모델이 저장된 고객 예측 결과를 재현하지 못했습니다. 모델과 데이터의 버전을 확인하세요.')

    def transform(self, data):
        if set(self.names).issubset(data.columns):
            x = data[self.names].apply(pd.to_numeric, errors='coerce')
            if not np.isfinite(x.to_numpy(float)).all():
                raise ValueError('모델 입력 값에 빈 값 또는 숫자가 아닌 값이 있습니다.')
            if not x.Age.between(self.age_min, self.age_max).all():
                raise ValueError(f'현재 모델에서 확인한 연령 범위는 {self.age_min}~{self.age_max}세입니다.')
            if not x.Age_step.eq((x.Age-1)//10).all():
                raise ValueError('Age_step이 기존 모델의 연령대 규칙과 다릅니다.')
            for cols in [['Geography_France', 'Geography_Germany', 'Geography_Spain'], ['Gender_Female', 'Gender_Male']]:
                if not x[cols].isin([0, 1]).all().all() or not x[cols].sum(axis=1).eq(1).all():
                    raise ValueError('국가와 성별은 각각 한 범주만 1이어야 합니다.')
            if not x[['HasCrCard','IsActiveMember']].isin([0,1]).all().all() or x[NUMERIC].lt(0).any().any():
                raise ValueError('고객 입력 값 범위를 확인하세요.')
            return x
        numbers = validate_features(data)
        if numbers.isna().any().any():
            raise ValueError('신규 고객의 숫자 항목은 모두 입력해야 합니다.')
        if not numbers.Age.between(self.age_min, self.age_max).all():
            raise ValueError(f'현재 확인된 모델 입력 범위는 {self.age_min}~{self.age_max}세입니다.')
        x = numbers.copy()
        x['Age_step'] = ((numbers.Age-1)//10).astype(int)
        for name, cats in [('Geography', ['France','Germany','Spain']), ('Gender', ['Female','Male'])]:
            s = data[name].astype('string').str.strip()
            if not s.isin(cats).all():
                raise ValueError(name + ': ' + ', '.join(cats) + ' 중에서 선택하세요.')
            for value in cats:
                x[f'{name}_{value}'] = s.eq(value).astype(int)
        return x[self.names]

    def predict(self, data):
        return self.pipeline.predict_proba(self.transform(data))[:, 1]

    def contributions(self, data):
        import xgboost as xgb
        x = self.transform(data)
        scaled = self.pipeline.named_steps['scaler'].transform(x)
        booster = self.pipeline.named_steps['model'].get_booster()
        names = booster.feature_names
        dm = xgb.DMatrix(scaled, feature_names=names)
        values = booster.predict(dm, pred_contribs=True)[0]
        return pd.DataFrame({'변수':self.names, '기여도':values[:-1]}), float(values[-1])
