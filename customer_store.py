"""Atomic customer registration; predictions never become outcome labels."""
from pathlib import Path
from datetime import datetime,timezone
import os,shutil,uuid
import pandas as pd
from churn_core import FEATURES,read_csv


def attach_names(data,original):
    result=data.copy();result['고객명']='이름 미확인'
    if original is None or 'CustomerId' not in original:return result
    reference=original.copy()
    name=next((c for c in ['name','Name','고객명','Surname'] if c in reference),None)
    if name is None:return result
    names=reference[name].astype('string')
    if 'Surname' in reference:names=names.fillna(reference['Surname'].astype('string'))
    keys=reference.CustomerId.map(lambda v:str(int(v)) if isinstance(v,(int,float)) and pd.notna(v) and float(v).is_integer() else str(v))
    counts=keys.value_counts();mapping=dict(zip(keys[counts.reindex(keys).to_numpy()==1],names[counts.reindex(keys).to_numpy()==1]))
    result['고객명']=result['고객ID'].astype(str).map(mapping).fillna('이름 미확인')
    return result


def register_customer(root,values,name,token):
    if not name.strip():raise ValueError('고객명을 입력해 주세요.')
    path=Path(root)/'Churn_Modelling.csv';lock=path.with_suffix('.registration.lock')
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise ValueError('다른 고객 정보를 저장 중입니다. 잠시 후 다시 시도하세요.')
    temp=None
    try:
        os.close(fd);data=read_csv(path.read_bytes())
        if 'RegistrationToken' in data:
            same=data.loc[data.RegistrationToken.eq(token)]
            if len(same):return str(int(same.CustomerId.iloc[0]))
        nums=pd.to_numeric(data.CustomerId,errors='coerce');cid=int(nums.max())+1
        record={c:values[c] for c in FEATURES}
        record.update(CustomerId=cid,RowNumber=len(data)+1,Surname=name.strip(),name=name.strip(),Exited=None,
                      RegisteredAt=datetime.now(timezone.utc).isoformat(),RegistrationToken=token)
        result=pd.concat([data,pd.DataFrame([record])],ignore_index=True)
        backups=Path(root)/'backups';backups.mkdir(exist_ok=True)
        shutil.copy2(path,backups/f'customers_{uuid.uuid4().hex}.csv')
        temp=path.with_name('customers_'+uuid.uuid4().hex+'.tmp')
        result.to_csv(temp,index=False,encoding='utf-8-sig');os.replace(temp,path)
        return str(cid)
    finally:
        if temp is not None and temp.exists():temp.unlink()
        lock.unlink(missing_ok=True)


def register_batch(root,frame,token):
    """Register a reviewed batch atomically, with one backup and retry protection."""
    if frame.empty:raise ValueError('등록할 고객이 없습니다.')
    name_col=next((c for c in ['name','Name','고객명','Surname'] if c in frame),None)
    if name_col is None:raise ValueError('고객명 컬럼(name 또는 고객명)을 포함해 주세요.')
    names=frame[name_col].astype('string').str.strip()
    if names.isna().any() or names.eq('').any():raise ValueError('모든 고객의 이름을 입력해 주세요.')
    if not set(FEATURES).issubset(frame):raise ValueError('예측에 필요한 고객 정보가 부족합니다.')
    path=Path(root)/'Churn_Modelling.csv';lock=path.with_suffix('.registration.lock');temp=None
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise ValueError('다른 고객 정보를 저장 중입니다. 잠시 후 다시 시도하세요.')
    try:
        os.close(fd);data=read_csv(path.read_bytes())
        if 'RegistrationToken' in data:
            same=data.loc[data.RegistrationToken.astype(str).str.startswith(token+':')]
            if len(same):return same.CustomerId.astype(str).tolist()
        first=int(pd.to_numeric(data.CustomerId,errors='coerce').max())+1
        added=frame[FEATURES].copy().reset_index(drop=True)
        added['CustomerId']=range(first,first+len(added));added['RowNumber']=range(len(data)+1,len(data)+len(added)+1)
        added['name']=names.to_numpy();added['Surname']=names.to_numpy();added['Exited']=None
        added['RegisteredAt']=datetime.now(timezone.utc).isoformat()
        added['RegistrationToken']=[token+':'+str(i) for i in range(len(added))]
        result=pd.concat([data,added],ignore_index=True)
        backups=Path(root)/'backups';backups.mkdir(exist_ok=True)
        shutil.copy2(path,backups/f'customers_{uuid.uuid4().hex}.csv')
        temp=path.with_name('customers_'+uuid.uuid4().hex+'.tmp')
        result.to_csv(temp,index=False,encoding='utf-8-sig');os.replace(temp,path)
        return added.CustomerId.astype(str).tolist()
    finally:
        if temp is not None and temp.exists():temp.unlink()
        lock.unlink(missing_ok=True)
