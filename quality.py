"""Customer identity recovery and auditable application status."""
import hashlib
import numpy as np
import pandas as pd
from churn_core import NUMERIC


def link_customer_ids(scored, original):
    data = scored.copy()
    if any(c in data for c in ['CustomerId', 'customer_id', '고객ID']):
        data['고객연결상태'] = '입력 파일의 고객 번호'
        return data
    data['고객ID'] = [f'ROW-{i+1:05d}' for i in range(len(data))]
    data['고객연결상태'] = '원본 고객 확인 필요'
    if original is None or 'CustomerId' not in original:
        return data
    reference = original.copy()
    for col, categories in [('Geography',['France','Germany','Spain']),('Gender',['Female','Male'])]:
        if col not in reference:return data
        for category in categories:
            reference[f'{col}_{category}'] = reference[col].eq(category).astype(int)
    keys = NUMERIC + ['Geography_France','Geography_Germany','Geography_Spain','Gender_Female','Gender_Male']
    if not set(keys).issubset(data) or not set(keys).issubset(reference):return data
    # Never infer identity by row order, approximate matching, or target labels.
    unique = reference.loc[~reference.duplicated(keys,keep=False),keys+['CustomerId']].copy()
    unique = unique.loc[unique.CustomerId.notna() & ~unique.CustomerId.duplicated(keep=False)]
    unique = unique.rename(columns={'CustomerId':'_matched_id'})
    merged = data.merge(unique,on=keys,how='left',sort=False,validate='many_to_one')
    matched = merged['_matched_id'].notna()
    merged.loc[matched,'고객ID'] = merged.loc[matched,'_matched_id'].map(
        lambda value: str(int(value)) if isinstance(value,(int,float,np.integer,np.floating)) and float(value).is_integer() else str(value))
    merged.loc[matched,'고객연결상태'] = '원본 특성 일치 · 고객 번호 복원'
    return merged.drop(columns='_matched_id')


def model_fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_comparison(df, thresholds):
    rows=[]
    for threshold in thresholds:
        p=pd.to_numeric(df['이탈예측확률'],errors='coerce')
        valid=p.notna()
        y=pd.to_numeric(df['실제이탈'],errors='coerce') if '실제이탈' in df else pd.Series(np.nan,index=df.index)
        known=valid & y.isin([0,1]); high=p.ge(threshold)
        tp=int((known & high & y.eq(1)).sum()); fp=int((known & high & y.eq(0)).sum())
        fn=int((known & ~high & y.eq(1)).sum())
        rows.append({'집중관리 기준':threshold,'분류 고객':int((valid & high).sum()),
                     '평가 가능한 대상':int((known & high).sum()),'실제 이탈 발견':tp,'이탈 놓침':fn,
                     '실제 유지 포함':fp,'대상 중 실제 이탈 비율':tp/(tp+fp) if tp+fp else np.nan,
                     '이탈 고객 발견 비율':tp/(tp+fn) if tp+fn else np.nan})
    return pd.DataFrame(rows)
