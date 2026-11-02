"""Run: python -m streamlit run app.py"""
from html import escape
import os
import uuid
import json
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from churn_core import FEATURES, GRADES, COLORS, classify, filter_customers, metrics, normalize, read_csv
from project_model import ProjectModel
from navigation import sidebar_navigation, remember_page_scroll
from quality import link_customer_ids, model_fingerprint, threshold_comparison
from customer_store import attach_names, register_customer, register_batch
from advisor_page import advisor_page

st.set_page_config(page_title='은행 고객 이탈 상담 지원 모델', page_icon='◈', layout='wide')
alt.theme.enable('default')
PALETTE = alt.Scale(domain=GRADES + ['미산정'], range=COLORS + ['#a6b0c0'])
LABELS = {'CreditScore': '신용점수', 'Age': '연령', 'Tenure': '거래기간', 'Balance': '잔액',
          'NumOfProducts': '상품 수', 'HasCrCard': '카드 보유', 'IsActiveMember': '활성 여부',
          'EstimatedSalary': '추정 급여', 'Geography': '국가', 'Gender': '성별'}

st.markdown('''<style>
:root{--ink:#18314a;--muted:#6d8192;--teal:#148a80}
.stApp{background:#f4f7fa;color:var(--ink)}
html,body,[class*="css"],button,input{font-family:'Noto Sans KR',sans-serif}
.block-container{max-width:1490px;padding:2rem 2.6rem 4rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e3eaf0}
[data-testid="stSidebar"] .block-container{padding:2rem 1.2rem}
h1,h2,h3{color:#18314a;letter-spacing:-.04em}
h3{font-size:1.15rem!important} .stCaption{color:#425b70;font-size:14px!important;line-height:1.8}
[data-testid="stMetric"]{background:white;border:1px solid #e3eaf0;border-radius:16px;padding:20px 23px}
[data-testid="stMetricLabel"]{color:#6d8192;font-size:13px}
[data-testid="stMetricValue"]{color:#18314a;font-size:30px;font-weight:700}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:18px!important}
.hero{position:relative;overflow:hidden;background:linear-gradient(115deg,#102c43,#155366);border-radius:22px;padding:32px 36px;margin:4px 0 24px;color:white;min-height:205px}
.hero h1{color:white;font-size:30px;max-width:70%;margin:7px 0 12px;line-height:1.4}
.hero p{color:#bfd5df;font-size:14px;max-width:68%;line-height:1.8;margin:0}
.eyebrow{font-size:23px;line-height:1.5;font-weight:700;letter-spacing:.2em;color:#72d3c2}
.hero-art{position:absolute;right:20px;top:4px;width:265px;height:210px;opacity:.95}
.brand{font-size:22px;line-height:1.45;letter-spacing:-.03em;font-weight:800;color:#17384e;margin-bottom:3px}
.brand small{color:#16a390}.brand-sub{font-size:11px;color:#8798a7;letter-spacing:.12em;margin-bottom:22px}
.section-kicker{font-size:23px;line-height:1.5;color:#148a80;font-weight:800;letter-spacing:.18em;margin:26px 0 4px}
.grade{background:white;border:1px solid #e1e9ee;border-radius:15px;padding:21px;margin:5px 0 12px;border-top:3px solid var(--accent)}
.grade .name{font-size:14px;font-weight:700;color:var(--accent)}.grade strong{display:block;font-size:26px;margin:10px 0;color:#18314a}
.grade p{color:#425b70;font-size:14px;margin:7px 0;line-height:1.65}
.signal{background:#edf7f5;border:1px solid #cee8e2;border-radius:14px;padding:16px 20px;color:#235d58;font-size:14px;margin:14px 0}
.profile{background:#fff;border-radius:18px;padding:25px;border:1px solid #e3eaf0}
.profile .score{font-size:48px;font-weight:800;color:#17384e;line-height:1.2;margin:15px 0}
.profile p{font-size:13px;color:#6d8192;line-height:1.7}
.footer{font-size:11px;color:#8a9aa8;margin-top:35px;border-top:1px solid #e1e8ee;padding-top:17px}
button[kind="primary"]{background:#147f77;border-color:#147f77;border-radius:10px}
@media(max-width:800px){.block-container{padding:1.2rem}.hero{padding:24px}.hero-art{display:none}.hero h1,.hero p{max-width:100%}.hero h1{font-size:24px}}

/* Editable fields: blue fill + visible border. Read-only results: white cards. */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] [data-baseweb="select"]>div,
[data-testid="stMultiSelect"] [data-baseweb="select"]>div{
 background:#eaf4ff!important;border:1.5px solid #77a6d4!important;border-radius:9px!important;color:#163650!important}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:transparent!important;color:#163650!important}
[data-testid="stTextInput"] input::placeholder,[data-testid="stTextArea"] textarea::placeholder{color:#526f89!important;opacity:1}
[data-testid="stTextArea"] textarea:focus,[data-baseweb="input"]:focus-within{outline:2px solid #2378bb!important;outline-offset:2px}
[data-testid="stText"]{background:#fff;border:1px solid #d7e1ea;border-left:4px solid #148a80;border-radius:10px;padding:18px 20px;line-height:1.8}
[data-testid="stForm"]{background:#fff;border:1px solid #c8d8e7!important;border-radius:14px;padding:20px!important}
.flow-strip{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0 24px}
.flow-step{flex:1;min-width:180px;background:#fff;border:1px solid #cfdeea;border-top:4px solid #148a80;border-radius:12px;padding:16px;color:#18314a}
.flow-step strong{display:block;font-size:17px;margin-bottom:7px}.flow-step span{font-size:14px;line-height:1.7}
</style>''', unsafe_allow_html=True)


def hero(title, subtitle, kicker='고객 이탈 분석'):
    st.markdown(f'''<div class="hero" translate="no"><div class="eyebrow">{escape(kicker)}</div>
    <h1>{escape(title)}</h1><p>{escape(subtitle)}</p>
    <svg class="hero-art" viewBox="0 0 270 220" fill="none" aria-hidden="true">
    <rect x="14" y="38" width="151" height="136" rx="16" fill="#214e61" stroke="#4c8490"/>
    <circle cx="55" cy="78" r="17" fill="#86d9c6"/>
    <path d="M28 132c2-20 12-31 27-31s25 11 27 31" fill="#86d9c6"/>
    <path d="M103 82h37M103 96h29M103 110h37" stroke="#83b8b9" stroke-width="5" stroke-linecap="round"/>
    <rect x="99" y="128" width="47" height="25" rx="6" fill="#153c4d"/>
    <path d="M107 146l8-7 7 3 7-9 9 3" stroke="#65d6bc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M151 33h87a15 15 0 0 1 15 15v60a15 15 0 0 1-15 15h-45l-19 16v-16h-23a15 15 0 0 1-15-15V48a15 15 0 0 1 15-15z" fill="#65d6bc" stroke="#b3f5df" stroke-width="2"/>
    <rect x="166" y="57" width="57" height="41" rx="10" fill="#19566a"/>
    <circle cx="183" cy="76" r="4" fill="#b3f5df"/><circle cx="206" cy="76" r="4" fill="#b3f5df"/>
    <path d="M184 89h21M194 57V49" stroke="#b3f5df" stroke-width="3" stroke-linecap="round"/>
    <circle cx="194" cy="47" r="3" fill="#f3bd69"/>
    <circle cx="225" cy="158" r="6" fill="#f3bd69"/><circle cx="194" cy="177" r="3" fill="#7ed8c4"/>
    </svg></div>''', unsafe_allow_html=True)


def section(kicker, title):
    st.markdown(f'<div class="section-kicker" translate="no">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.subheader(title)


def chart(spec, height=260):
    st.altair_chart(spec.properties(height=height).configure_view(strokeWidth=0).configure_axis(
        labelColor='#718493', titleColor='#718493', gridColor='#edf1f5', domain=False,
        labelFontSize=11, titleFontSize=11).configure_legend(title=None, labelColor='#516779'), width='stretch')


def grade_cards(df, t, w):
    descriptions = [('우선 상담 대상', f'{t:.0%} 이상', '이용 불편 및 이탈 의향 확인'),
                    ('관찰 대상', f'{w:.0%} 이상 ~ {t:.0%} 미만', '이용 현황 점검 및 활성화 지원'),
                    ('정기 관리 대상', f'{w:.0%} 미만', '이용 현황 및 고객 만족도 모니터링')]
    for col, name, color, (action, rule, desc) in zip(st.columns(3), GRADES, COLORS, descriptions):
        with col:
            count = int(df['위험등급'].eq(name).sum())
            st.markdown(f'<div class="grade" style="--accent:{color}"><div class="name">● {name} · {action}</div>'
                        f'<strong>{count:,}<span style="font-size:13px;font-weight:400"> 명</span></strong>'
                        f'<p>예측 확률 {rule}</p><p>{desc}</p></div>', unsafe_allow_html=True)


def data_table(df):
    wanted = ['고객명', '고객ID', '위험등급', '이탈예측확률', '권장조치', 'Age', 'Balance',
              'NumOfProducts', 'IsActiveMember', '실제이탈']
    display = df[[c for c in wanted if c in df]].copy()
    display['이탈예측확률'] *= 100
    st.dataframe(display.rename(columns={**LABELS,'고객ID':'고객 번호','고객명':'고객명(원본 성)'}), hide_index=True, width='stretch',
                 column_config={'이탈예측확률': st.column_config.ProgressColumn(
                     '이탈 예측 확률 (%)', min_value=0, max_value=100, format='%.1f%%'),
                     '실제이탈': st.column_config.NumberColumn('관측 이탈 (0/1)', format='%d')}, height=330)


ROOT = Path(os.environ.get('BANK_PROJECT_DIR', Path(__file__).resolve().parent))


@st.cache_resource(show_spinner=False)
def load_model(root, signature):
    return ProjectModel(root)


def load_csv(relative):
    path = ROOT / relative
    return read_csv(path.read_bytes()) if path.exists() else None


metadata = {}; model = None; connection_error = None; df = None
try:
    mp = ROOT / 'models/model_metadata.json'
    if mp.exists():
        metadata = json.loads(mp.read_text(encoding='utf-8-sig'))
    saved = load_csv('data/customer_risk.csv')
    original = load_csv('Churn_Modelling.csv')
    importance = load_csv('data/permutation_importance.csv')
    paths = [ROOT/'models'/metadata.get('model_file','XGBoost_Final.pkl'), mp, ROOT/'data/customer_risk.csv']
    if all(path.exists() for path in paths):
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        try:
            model = load_model(str(ROOT), signature)
        except Exception as exc:
            connection_error = str(exc)
except Exception as exc:
    saved = original = importance = None
    connection_error = str(exc)


def reset_filters():
    st.session_state.grades = GRADES + ['미산정']
    st.session_state.prob_range = (0, 100)
    st.session_state.search = ''
    st.session_state.eligible = False


def toggle_grade(grade):
    selected = list(st.session_state.get('grades', GRADES + ['미산정']))
    if grade in selected:
        selected.remove(grade)
    else:
        selected.append(grade)
    st.session_state.grades = [g for g in GRADES + ['미산정'] if g in selected]


def select_grades(select_all):
    st.session_state.grades = GRADES + ['미산정'] if select_all else []


st.markdown('<style>\n[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0.85rem}\n[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:8px}\n[data-testid="stSidebar"] .brand{font-size:19px;margin:0 0 5px;line-height:1.5}\n[data-testid="stSidebar"] .brand-sub{font-size:12px;margin-bottom:16px}\n[data-testid="stSidebar"] h3{font-size:17px!important;padding:6px 0!important;margin:0!important;line-height:1.6!important}\n[data-testid="stSidebar"] hr{margin:15px 0!important}\n[data-testid="stSidebar"] [class*="st-key-nav_"] button{min-height:43px!important;padding:10px 12px!important}\n[data-testid="stSidebar"] p{font-size:14px;line-height:1.75;margin:0}\n</style>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand" translate="no">은행 고객 이탈 상담 지원 모델</div><div class="brand-sub">고객 이탈 예측 · 관리 지원</div>', unsafe_allow_html=True)
    sidebar_navigation()
    page = st.session_state.get('retain_page', 'home')
    st.divider()
    st.subheader('관리 기준')
    try:
        default_t=float(metadata.get('threshold',.52))
        if not np.isfinite(default_t) or not .02<=default_t<=.99:default_t=.52
    except (ValueError,TypeError):default_t=.52
    st.caption('리스크 임계값 설정')
    t_pct=st.slider('집중관리 시작 확률 (%)',2,99,int(round(default_t*100)),key='high_threshold')
    w_pct=st.slider('관찰 시작 확률 (%)',1,t_pct-1,max(1,t_pct//2),key=f'watch_{t_pct}')
    threshold,watch=t_pct/100,w_pct/100
    st.write(f'집중관리 **{threshold:.0%} 이상** · 관찰 **{watch:.0%}~{threshold:.0%} 미만** · 일반관리 **{watch:.0%} 미만**')
    st.divider()
    st.subheader('모델 서비스 상태')
    if model:
        st.success('선정 모델 · XGBoost')
        selection_threshold=float(metadata.get('threshold',.52))
        selection=metrics(model.reference['실제이탈'],model.replayed,selection_threshold)
        st.markdown(f"""<div style="font-size:14px;line-height:2.1;padding:5px 0 10px">
        <div>검증 선정 기준 <strong>{selection_threshold:.0%}</strong></div>
        <div>정확도 <strong>{selection['accuracy']:.1%}</strong> · 적중률 <strong>{selection['precision']:.1%}</strong></div>
        <div>탐지율 <strong>{selection['recall']:.1%}</strong> · 균형 점수 <strong>{selection['f1_score']:.3f}</strong></div>
        </div>""",unsafe_allow_html=True)
    elif saved is not None:
        st.warning('모델 미연결 · 저장 결과 조회')
    else:
        st.warning('모델·자료 미연결')
    with st.expander('자료 및 예측 모델'):
        for path in ['models/XGBoost_Final.pkl','models/model_metadata.json','data/customer_risk.csv','Churn_Modelling.csv']:
            st.write(('✓ ' if (ROOT/path).exists() else '○ ') + path)
        st.write('프로젝트 폴더의 파일을 자동으로 읽습니다. 매번 업로드할 필요가 없습니다.')

if connection_error:
    st.warning('학습 모델을 연결하지 못했습니다. 아래 오류를 확인해 주세요.')
    with st.expander('연결 오류 및 실행 환경'):
        st.write('예측 실행 환경 또는 모델 파일 연결을 확인해야 합니다.')
        st.write('이 PC에서는 agent 환경에 기존 모델 실행에 필요한 패키지가 설치되어 있습니다. 프로젝트 폴더에서 다음 명령으로 실행하세요.')
        st.code(r'& "C:\Users\S5_KKM\anaconda3\envs\agent\python.exe" -m streamlit run app.py', language='powershell')

if saved is not None:
    try:
        current = link_customer_ids(saved, original)
        if model:
            current['이탈예측확률'] = model.replayed
        df = classify(current, threshold, watch)
        if model and original is not None and 'RegisteredAt' in original:
            added=original.loc[original.RegisteredAt.notna()].copy()
            if not added.empty:
                encoded=model.transform(added)
                for column in encoded:added[column]=encoded[column].to_numpy()
                added['이탈예측확률']=model.predict(added)
                added['고객ID']=added.CustomerId.map(lambda v:str(int(v)))
                added['실제이탈']=np.nan
                added['고객연결상태']='신규 등록'
                df=pd.concat([df,classify(added,threshold,watch)],ignore_index=True)
        df=attach_names(df,original)
    except Exception as exc:
        st.error('고객 예측 데이터를 읽지 못했습니다: ' + str(exc))


def empty_state():
    st.info('표시할 실제 고객 데이터가 없습니다. 프로젝트의 data 폴더에 customer_risk.csv가 있는지 확인하세요.')


def evaluate_data():
    if df is None or '실제이탈' not in df:
        return None
    valid = df.dropna(subset=['실제이탈', '이탈예측확률'])
    if valid.empty:
        return None
    return valid, metrics(valid['실제이탈'], valid['이탈예측확률'], threshold)


def overview():
    hero('은행 고객 이탈 상담 지원 모델',
         'XGBoost 기반 이탈 예측 · 리스크 세분화 · 고객 유지 전략')
    if df is None or df.empty:
        empty_state(); return
    with st.expander('자료 출처 및 모델 확인 범위'):
        st.write('모델 실행과 예측 결과의 일치를 확인했습니다. 이것이 새로운 고객에서도 같은 성능을 보장한다는 뜻은 아닙니다.')
        st.write('현재 결과는 과거 고객 평가·탐색용입니다. 이탈 확률의 기간 정의와 별도 시험 데이터 여부는 제공된 파일만으로 확인되지 않았습니다.')
        st.write('원본에는 프랑스·독일·스페인 고객 정보가 있습니다. 국내 은행 고객에게 동일하게 적용할 수 있는지는 추가 검증이 필요합니다.')
        st.write('현재 확인된 연령대 변환 범위는 21~60세입니다. 학습 노트북에서 제외된 연령대가 있어 전체 연령에 대한 모델로 해석하지 않습니다.')
        linked=df['고객연결상태'].str.startswith('원본 특성').sum() if '고객연결상태' in df else 0
        st.write(f'원본 특성으로 고객 번호를 유일하게 연결한 고객: {linked:,}명. 일치하지 않거나 여러 후보가 있으면 ROW 번호를 유지합니다.')
        if model:
            st.write('모델 파일과 기존 예측 결과의 일치 확인 완료')
            st.write('같은 모델 파일인지 확인할 수 있는 식별값입니다. 모델의 학습 품질을 보증하는 수치는 아닙니다.')
    if model:
        st.success(f'실제 모델 실행 확인 · 고객 {len(df):,}명의 예측을 다시 계산했고 기존 결과와 일치합니다.')
    else:
        st.info('아래 숫자는 저장된 고객 예측 결과입니다. 현재 모델의 실시간 실행은 확인되지 않았습니다.')
    high = df.loc[df['위험등급'].eq(GRADES[0])]
    evaluation = evaluate_data()
    a,b,c,d = st.columns(4)
    a.metric('분석 대상 고객', f'{len(df):,}명'); a.write('예측 결과 데이터 기준')
    b.metric('고위험 고객', f'{len(high):,}명'); b.write(f'모델이 이탈 가능성을 {threshold:.0%} 이상으로 본 고객입니다.')
    if evaluation:
        _, m = evaluation
        c.metric('이탈 예측 적중률', f'{m["precision"]:.1%}' if m['tp']+m['fp'] else '대상 없음')
        c.write(f'집중관리로 본 {m["tp"]+m["fp"]:,}명 중 {m["tp"]:,}명이 실제 이탈했습니다.' if m['tp']+m['fp'] else '현재 기준에서 집중관리로 분류된 고객이 없습니다.')
        d.metric('이탈 고객 탐지율', f'{m["recall"]:.1%}' if m['tp']+m['fn'] else '확인할 이탈 없음')
        d.write(f'실제 이탈 {m["tp"]+m["fn"]:,}명 중 {m["tp"]:,}명을 찾아냈습니다.')
    else:
        c.metric('평균 이탈 확률', f'{df["이탈예측확률"].mean():.1%}' if df['이탈예측확률'].notna().any() else '계산 자료 없음')
        c.write('조회 고객의 모델 예측 확률을 평균한 값입니다.')
        d.metric('미산정 고객', f'{df["이탈예측확률"].isna().sum():,}명')
        d.write('예측 정보가 없어 별도 확인이 필요한 고객입니다.')
    st.write('이 화면은 과거 고객의 예측 결과로 관리 기준을 살펴보는 화면입니다. 현재 연락할 고객은 가입 상태와 상담 동의를 확인한 뒤 정하세요.')
    section('관리 우선순위', '리스크 등급별 고객 세그먼트')
    grade_cards(df, threshold, watch)
    a,b = st.columns([1.3,1])
    with a, st.container(border=True):
        st.subheader('이탈 예측 확률 분포')
        st.write('막대가 높을수록 그 확률대에 속한 고객이 많습니다. 오른쪽일수록 모델이 이탈 가능성을 높게 봅니다.')
        bars = alt.Chart(df[['이탈예측확률']].dropna()).mark_bar(color='#269b8e', cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('이탈예측확률:Q', bin=alt.Bin(step=.05,extent=[0,1]),title='이탈 예측 확률',axis=alt.Axis(format='.0%'),scale=alt.Scale(domain=[0,1])),
            y=alt.Y('count():Q',title='고객 수'),tooltip=[alt.Tooltip('count():Q',title='고객 수')])
        lines = alt.Chart(pd.DataFrame({'기준':[watch,threshold]})).mark_rule(color='#e2ad44',strokeDash=[5,4]).encode(x='기준:Q')
        chart(bars+lines)
        st.write(f'점선은 관찰 시작 {watch:.0%}와 집중관리 시작 {threshold:.0%}입니다.')
    with b, st.container(border=True):
        st.subheader('등급별 관측 이탈률')
        st.write('각 등급에서 실제로 이탈한 고객의 비율입니다. 높은 등급일수록 이 비율도 높은지 확인하세요.')
        if '실제이탈' in df:
            summary = df.groupby('위험등급').agg(고객수=('위험등급','size'),확인고객=('실제이탈','count'),이탈고객=('실제이탈','sum')).reindex(GRADES).fillna(0).reset_index()
            summary['관측이탈률'] = summary['이탈고객']/summary['확인고객'].replace(0,np.nan)
            chart(alt.Chart(summary.dropna(subset=['관측이탈률'])).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(
                x=alt.X('위험등급:N',sort=GRADES,title=None),y=alt.Y('관측이탈률:Q',title='실제로 이탈한 비율',axis=alt.Axis(format='.0%'),scale=alt.Scale(domain=[0,1])),
                color=alt.Color('위험등급:N',scale=PALETTE,legend=None),tooltip=['위험등급','확인고객','이탈고객',alt.Tooltip('관측이탈률:Q',format='.1%')]))
        else:
            st.info('실제 이탈 여부가 없어 등급별 결과를 비교할 수 없습니다.')
    section('주요 이탈 신호', '주요 예측 변수 중요도')
    if isinstance(importance,pd.DataFrame) and {'Feature','Importance_Mean'}.issubset(importance):
        imp = importance.copy(); imp['Importance_Mean']=pd.to_numeric(imp['Importance_Mean'],errors='coerce')
        imp=imp.dropna(subset=['Importance_Mean']).nlargest(5,'Importance_Mean')
        friendly = {**LABELS,'Geography_Germany':'독일 거주 여부','Age_step':'연령대'}
        imp['정보']=imp.Feature.map(lambda f:friendly.get(f,f))
        # Rank is useful for a manager; raw permutation values are technical diagnostics.
        imp['영향순위'] = np.arange(len(imp),0,-1)
        chart(alt.Chart(imp).mark_bar(color='#2e9f92',cornerRadiusEnd=4).encode(
            y=alt.Y('정보:N',sort=imp['정보'].tolist(),title=None),
            x=alt.X('영향순위:Q',title=None,axis=None),tooltip=['정보']),220)
        st.write('기존 분석에서 중요한 순서로 뽑힌 정보입니다. 막대는 순위만 나타냅니다. 예를 들어 연령이 중요하다는 것은 연령별 차이를 참고했다는 뜻이며, 나이가 많으면 반드시 이탈한다는 뜻은 아닙니다.')
    section('상담 준비', '고위험 고객 우선순위')
    st.write('이미 이탈한 것으로 확인된 고객을 제외한 상위 5명입니다. 이탈 여부가 아직 없는 고객도 포함되므로 현재 가입 상태를 다시 확인하세요.')
    queue = filter_customers(df,[GRADES[0]],eligible=True).head(5)
    if queue.empty:
        st.info('현재 기준에서 이 조건에 맞는 고객이 없습니다.')
    else:
        data_table(queue)


def explorer():
    hero('개별 고객 이탈 진단', '고객 검색 · 이탈 가능성 · 주요 영향 요인')
    if df is None or df.empty:
        empty_state(); return
    with st.container(border=True):
        a,b=st.columns([1.3,1])
        with a:
            st.session_state.setdefault('grades',GRADES+['미산정'])
            st.write('**관리 등급 선택**')
            st.markdown('''<style>
            [class*="st-key-grade_filter_"] button {
                background:#ffffff!important;color:#344054!important;
                border:1px solid #cbd2d9!important;border-radius:8px;
                min-height:46px;box-shadow:none!important;
            }
            [class*="st-key-grade_filter_"] button[kind="primary"] {
                background:#d5d9df!important;color:#182230!important;
                border:2px solid #7b8491!important;font-weight:700;
            }
            [class*="st-key-grade_filter_"] button:hover {border-color:#475467!important;}
            </style>''',unsafe_allow_html=True)
            for start in [0,2]:
                for column,index in zip(st.columns(2),range(start,start+2)):
                    grade=(GRADES+['미산정'])[index]
                    active=grade in st.session_state.grades
                    count=int(df['위험등급'].eq(grade).sum())
                    column.button(f'{"✓ " if active else ""}{grade} · {count:,}명',
                                  key=f'grade_filter_{index}',type='primary' if active else 'secondary',
                                  width='stretch',on_click=toggle_grade,args=(grade,))
            left,right=st.columns(2)
            left.button('전체 선택',key='grade_filter_all',width='stretch',on_click=select_grades,args=(True,))
            right.button('전체 해제',key='grade_filter_none',width='stretch',on_click=select_grades,args=(False,))
            st.caption('회색 + ✓: 필터에 포함 · 흰색: 제외. 버튼을 다시 누르면 선택이 바뀝니다. 0명인 등급도 선택할 수 있습니다.')
            grades=list(st.session_state.grades)
        with b:
            st.session_state.setdefault('prob_range',(0,100))
            bounds=st.slider('이탈 가능성 범위 (%)',0,100,key='prob_range')
        a,b,c=st.columns([1,1.6,.7])
        query=a.text_input('고객명 또는 고객 번호 검색',key='search')
        with b:
            eligible=st.checkbox('이미 이탈한 고객 제외',key='eligible')
        c.button('전체 조건 초기화',on_click=reset_filters,width='stretch')
    selected=filter_customers(df,grades,(bounds[0]/100,bounds[1]/100),'',eligible)
    if query:
        selected=selected.loc[selected['고객명'].str.contains(query,regex=False,na=False)|selected['고객ID'].str.contains(query,regex=False,na=False)]
    st.subheader(f'선택한 고객 {len(selected):,}명')
    st.write('확률이 높은 순서입니다. 원본 고객 정보와 유일하게 일치한 경우 실제 고객 번호를 연결했습니다. ROW로 시작하는 번호는 확인이 필요한 파일 행 번호입니다.')
    if selected.empty:
        st.info('조건에 맞는 고객이 없습니다. 등급을 다시 선택하거나 전체 조건을 초기화하세요.'); return
    data_table(selected)
    st.download_button('선택 고객 목록 저장',selected.to_csv(index=False).encode('utf-8-sig'),'관리대상고객.csv','text/csv',type='primary')
    section('고객 상세', '고객 정보 및 주요 영향 요인')
    idx=st.selectbox('고객 선택',selected.index.tolist(),format_func=lambda i:f'{selected.at[i,"고객명"]} · 고객 번호 {selected.at[i,"고객ID"]} · {selected.at[i,"위험등급"]}')
    row=selected.loc[idx]
    show_prediction(selected.loc[[idx]],row)
    if st.button('고객 상담 지원으로 이동',key='open_advisor'):
        st.session_state.advisor_customer_id=str(row['고객ID'])
        st.session_state.pop('advisor_choice',None)
        st.session_state.advisor_search=''
        st.session_state.advisor_search_applied=''
        st.session_state.retain_page='advisor'
        st.session_state.retain_navigation+=1
        st.rerun()



def show_prediction(frame,row):
    p=row['이탈예측확률']
    a,b=st.columns([.75,1.5])
    with a:
        st.metric('이탈 예측 확률',f'{p:.1%}' if pd.notna(p) else '예측 정보 없음')
        st.write(f'**{row["위험등급"]}** · {row["권장조치"]}')
        if row.get('실제이탈')==1:
            st.warning('과거에 실제 이탈한 고객입니다. 현재 유지 상담 대상과 구분하세요.')
        st.write('확률이 70%라면 모델이 이탈 가능성을 높게 본다는 뜻입니다. 이 고객이 반드시 이탈한다는 의미는 아닙니다.')
    with b,st.container(border=True):
        st.subheader('이탈 예측의 주요 영향 요인')
        if model and pd.notna(p):
            try:
                contrib,bias=model.contributions(frame)
                material=contrib.loc[contrib['기여도'].abs().ge(.01)].copy()
                omitted=len(contrib)-len(material)
                material=material.loc[material['기여도'].abs().nlargest(6).index]
                friendly={**LABELS,'Age_step':'연령대','Geography_Germany':'독일 거주 여부','Geography_France':'프랑스 거주 여부','Geography_Spain':'스페인 거주 여부','Gender_Female':'여성 여부','Gender_Male':'남성 여부'}
                material['정보']=material['변수'].map(lambda x:friendly.get(x,x))
                if material.empty:
                    st.write('이 고객은 특정 입력 정보 하나가 점수를 크게 바꾸지 않았습니다.')
                else:
                    chart(alt.Chart(material).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X('기여도:Q',axis=None,title='← 이탈 가능성을 낮춘 쪽 | 높인 쪽 →'),
                        y=alt.Y('정보:N',sort='-x',title=None),color=alt.condition('datum.기여도>0',alt.value(COLORS[0]),alt.value(COLORS[2])),
                        tooltip=['정보']),230)
                st.write('주황색은 점수를 높인 정보, 초록색은 점수를 낮춘 정보입니다. 막대가 길수록 이번 예측에 더 크게 작용했습니다. 고객이 떠나는 이유를 확정한 것은 아닙니다.')
                if omitted:
                    st.write(f'점수를 거의 바꾸지 않은 {omitted}개 항목은 숨겼습니다.')
                with st.expander('영향 점수 상세'):
                    st.write('아래는 각 고객 정보가 예측에 미친 영향입니다. 확률을 몇 % 바꿨다는 수치가 아니라 모델 내부 점수의 변화량입니다.')
                    details=contrib.copy()
                    details['변수']=details['변수'].map(lambda value:friendly.get(value,value))
                    st.dataframe(details.rename(columns={'변수':'고객 정보','기여도':'영향 점수'}),hide_index=True,column_config={'영향 점수':st.column_config.NumberColumn(format='%.4f')})
            except Exception as exc:
                st.info('이 고객의 영향 정보를 계산하지 못했습니다. 입력 항목을 확인하세요.')
                with st.expander('상세 오류'):
                    st.write('입력 항목과 모델 연결 상태를 확인해 주세요.')
        else:
            st.write('실행 중인 모델이 없어 이 고객의 예측 근거를 계산할 수 없습니다.')
    with st.container(border=True):
        st.subheader('상담 사전 점검 항목')
        if row.get('IsActiveMember')==0:
            st.write('• 비활성 고객으로 기록되어 있습니다. 최근 거래가 줄었는지, 이용에 불편이 있었는지 먼저 확인하세요.')
        if row.get('NumOfProducts')==1:
            st.write('• 한 가지 상품을 이용합니다. 현재 상품이 고객의 목적에 맞는지 확인하세요.')
        st.write('• 최근 거래·민원·상담 기록을 확인하고, 고객의 동의를 받아 연락하세요.')
        st.write('• 상담 결과를 남겨 다음 관리와 모델 개선에 활용하세요. 나이·성별 등 한 가지 속성만으로 고객을 판단하지 마세요.')


def predict_page():
    hero('신규 고객 이탈 예측', '학습 모델 기반 예측 · 고객별 이탈 확률 · 예측 기여도 분석')
    if model is None:
        st.warning('모델이 연결되어야 새 고객을 예측할 수 있습니다. 왼쪽 연결 상태와 오류 안내를 확인하세요.');return
    st.markdown('<div class="flow-strip"><div class="flow-step"><strong>1 · 고객 정보 입력</strong><span>파란 입력란에 고객명과 거래 정보를 입력</span></div><div class="flow-step"><strong>2 · 이탈 가능성 확인</strong><span>현재 선정 모델로 확률·관리 등급 산출</span></div><div class="flow-step"><strong>3 · 결과 검토</strong><span>등록할 고객 정보와 예측 결과 확인</span></div><div class="flow-step"><strong>4 · 원본 자료에 등록</strong><span>등록 버튼으로 고객 추가·관리 시작</span></div></div>',unsafe_allow_html=True)
    st.caption('연한 파란색 테두리 = 입력·선택 영역 / 흰색 카드 = 결과·안내 영역. 예측만으로 고객이 등록되지는 않습니다.')
    one,bulk=st.tabs(['고객 한 명 예측','여러 고객 파일 예측'])
    with one:
        st.subheader('1 · 신규 고객 정보 입력')
        with st.form('new_customer'):
            customer_name=st.text_input('고객명 (등록 시 필수)')
            a,b,c=st.columns(3);values={}
            with a:
                values['CreditScore']=st.number_input('신용점수',300,850,650)
                values['Age']=st.number_input('나이',model.age_min,model.age_max,40)
                values['Tenure']=st.number_input('은행 거래기간 (년)',0,10,5)
                values['Geography']=st.selectbox('거주 국가',['France','Germany','Spain'],format_func=lambda v:{'France':'프랑스','Germany':'독일','Spain':'스페인'}[v])
            with b:
                values['Balance']=st.number_input('계좌 잔액',0.,value=50000.,step=1000.)
                values['NumOfProducts']=st.number_input('보유 상품 수',1,4,2)
                values['EstimatedSalary']=st.number_input('추정 연간 급여',0.,value=70000.,step=1000.)
                st.write('잔액과 급여는 학습 데이터와 같은 통화 단위로 입력하세요. 원본 파일에는 통화명이 없습니다.')
            with c:
                values['Gender']=st.selectbox('성별',['Female','Male'],format_func=lambda v:'여성' if v=='Female' else '남성')
                values['HasCrCard']=st.selectbox('신용카드',[1,0],format_func=lambda v:'보유' if v else '미보유')
                values['IsActiveMember']=st.selectbox('은행 이용 상태',[1,0],format_func=lambda v:'활성' if v else '비활성')
                st.write(f'연령대 변환이 확인된 {model.age_min}~{model.age_max}세 고객을 지원합니다. 이 범위 밖의 고객은 모델 검토가 먼저 필요합니다.')
            submit=st.form_submit_button('이 고객의 이탈 가능성 확인',type='primary')
        if submit:
            try:
                frame=pd.DataFrame([values]);frame['이탈예측확률']=model.predict(frame)
                frame['고객명']=customer_name.strip() or '미등록 고객'
                st.session_state.new_customer_prediction=frame
                st.session_state.registration_values=values.copy()
                st.session_state.registration_name=customer_name.strip()
                st.session_state.registration_token=uuid.uuid4().hex
                st.session_state.pop('registered_customer',None)
            except Exception as exc:
                st.error(str(exc))
        st.subheader('2 · 예측 결과 검토 및 고객 등록')
        ready='new_customer_prediction' in st.session_state
        registered=st.session_state.get('registered_customer')
        if ready:
            frame=classify(st.session_state.new_customer_prediction,threshold,watch)
            result=frame.iloc[0]
            with st.container(border=True):
                st.write('예측 고객: '+st.session_state.registration_name if st.session_state.registration_name else '예측 고객: 이름 미입력 · 고객명을 입력하고 다시 예측하세요.')
                x,y=st.columns(2)
                x.metric('이탈 예측 확률',f'{result["이탈예측확률"]:.1%}')
                y.metric('관리 등급',result['위험등급'])
                st.caption('아래 등록 버튼은 마지막으로 예측한 정보를 저장합니다. 입력을 수정했다면 이탈 가능성 확인을 다시 누르세요.')
                st.dataframe(pd.DataFrame([st.session_state.registration_values]).rename(columns=LABELS),hide_index=True,width='stretch')
        else:
            st.info('아직 예측하지 않았습니다. 위의 고객 정보를 입력하고 이 고객의 이탈 가능성 확인을 누르면 확률·관리 등급·예측 요인이 나타나며 등록 버튼이 활성화됩니다.')
        st.write('등록 시 고객 정보가 원본 고객 자료에 추가됩니다. 실제 이탈 여부는 미확인으로 저장하며, 추가 고객은 고객 목록에서 관리합니다. 모델 자동 재학습은 수행하지 않습니다.')
        can_register=ready and bool(st.session_state.get('registration_name')) and not registered
        if st.button('예측 고객을 원본 자료에 등록',key='register_customer',type='primary',disabled=not can_register):
            try:
                cid=register_customer(ROOT,st.session_state.registration_values,st.session_state.registration_name,st.session_state.registration_token)
                st.session_state.registered_customer=cid
                st.rerun()
            except Exception as exc:st.error(str(exc))
        if registered:
            st.success('원본 자료 등록 완료 · 고객 번호 '+registered+' · 고객 목록에서 확인할 수 있습니다.')
            if st.button('등록 고객 관리로 이동',key='open_registered_customer'):
                st.session_state.retain_page='customers'
                st.session_state.retain_navigation=st.session_state.get('retain_navigation',0)+1
                st.rerun()
        if ready:
            st.subheader('3 · 예측 요인 상세')
            show_prediction(frame,frame.iloc[0])
    with bulk:
        st.caption('파일 업로드 → 전체 예측 결과 확인 → 등록 확인 체크 → 원본 자료에 일괄 등록')
        st.write('고객 특성 10개가 들어 있는 고객 자료 파일를 선택하세요. 실제 이탈 여부는 필요하지 않습니다.')
        template=pd.DataFrame(columns=['name']+FEATURES).to_csv(index=False).encode('utf-8-sig')
        st.download_button('고객 입력 양식 받기',template,'신규고객_입력양식.csv','text/csv')
        upload=st.file_uploader('새 고객 자료 파일',type='csv')
        if upload:
            try:
                frame=read_csv(upload.getvalue())
                if frame.empty:raise ValueError('고객 정보가 비어 있습니다.')
                frame['이탈예측확률']=model.predict(frame)
                results=classify(frame,threshold,watch)
                st.success(f'실제 XGBoost 모델로 {len(results):,}명의 예측을 완료했습니다.')
                data_table(results)
                st.download_button('예측 결과 저장',results.to_csv(index=False).encode('utf-8-sig'),'신규고객_예측결과.csv','text/csv')
                import hashlib
                batch_key=hashlib.sha256(upload.getvalue()).hexdigest()
                if st.session_state.get('batch_file')!=batch_key:
                    st.session_state.batch_file=batch_key
                    st.session_state.batch_token=uuid.uuid4().hex
                    st.session_state.pop('batch_registered',None)
                    st.session_state.pop('batch_confirm',None)
                st.write('등록하면 업로드한 고객 정보가 원본 자료에 추가됩니다. 고객 번호는 새로 부여하며 실제 이탈 여부는 미확인으로 저장합니다. 기존 고객 수정·중복 고객 판별은 수행하지 않습니다.')
                confirmed=st.checkbox('예측 결과와 고객명을 확인했으며 신규 고객으로 일괄 추가합니다.',key='batch_confirm')
                if st.button('예측 고객 전체를 원본 자료에 등록',key='register_batch',type='primary',disabled=not confirmed or bool(st.session_state.get('batch_registered'))):
                    st.session_state.batch_registered=register_batch(ROOT,frame,st.session_state.batch_token)
                    st.rerun()
                if st.session_state.get('batch_registered'):st.success(f'신규 고객 {len(st.session_state.batch_registered):,}명 원본 등록 완료 · 고객 목록에서 확인하세요.')
            except Exception as exc:
                st.error(str(exc))


def validation_page():
    hero('모델 성능 및 운영 기준 검증', '예측 성능 · 분류 기준 비교 · 확률 적합성 · 자료 품질')
    section('최종 모델 선정', '머신러닝 · 딥러닝 학습 및 검증 결과')
    comparison=pd.read_csv(ROOT/'reports/trained_comparison.csv')
    audit=json.loads((ROOT/'reports/training_audit.json').read_text(encoding='utf-8'))
    st.success(f"최종 선정 모델: {audit['selected_model']} · 선정 기준: 검증 자료의 균형 점수")
    st.write(f"학습 {audit['train_customers']:,}명 · 기준 선택용 검증 {audit['validation_customers']:,}명 · 기존 평가 고객 {audit['evaluation_customers']:,}명. 두 모델은 동일한 분할을 사용했고 전처리는 학습 고객에서만 계산했습니다.")
    st.dataframe(comparison[['model','threshold','validation_f1','accuracy','precision','recall','f1_score']].rename(columns={
        'model':'모델','threshold':'선정 기준','validation_f1':'선정용 검증 균형 점수','accuracy':'평가 정확도','precision':'평가 적중률','recall':'평가 탐지율','f1_score':'평가 균형 점수'}),hide_index=True,width='stretch',
        column_config={k:st.column_config.NumberColumn(format='percent') for k in ['선정 기준','평가 정확도','평가 적중률','평가 탐지율']})
    st.write('신규 딥러닝: 64·32·16개 노드의 다층 신경망을 실제 학습했습니다. 내부 검증 결과가 개선되지 않으면 학습을 조기 종료합니다. 외부 검증 자료에서 기준 확률을 선택한 뒤 기존 평가 고객에게 적용했습니다.')
    st.write('최종 선택은 평가 자료가 아닌 검증 자료의 균형 점수로 결정했습니다. 고객 번호는 학습 입력에서 제외했습니다. 기존 평가 고객은 이전 실험에도 사용되어 완전히 새로운 최종 시험을 대신하지 않습니다.')
    with st.expander('과거 모델 비교 기록'):
        st.dataframe(pd.read_csv(ROOT/'reports/historical_models.csv').rename(columns={'Accuracy':'정확도','Precision':'적중률','Recall':'탐지율','F1-score':'균형 점수'}),hide_index=True,width='stretch')
        st.write('과거 실험 기록입니다. 위 표의 신규 학습 결과와 구분합니다.')
    evaluation=evaluate_data()
    if evaluation is None:
        st.info('실제 이탈 여부와 예측 확률이 모두 있어야 성능을 계산할 수 있습니다.');return
    valid,m=evaluation
    st.write(f'실제 이탈 여부가 있는 고객 {len(valid):,}명과 현재 관리 기준 {threshold:.0%}로 다시 계산한 결과입니다. 이 파일이 학습에 쓰이지 않은 별도 시험 데이터인지 여부는 저장된 파일만으로 확인할 수 없습니다.')
    a,b,c=st.columns(3)
    a.metric('전체 예측 정확도',f'{m["accuracy"]:.1%}')
    a.write(f'유지·이탈을 포함해 {len(valid):,}명 중 {m["tp"]+m["tn"]:,}명의 결과를 맞혔습니다.')
    b.metric('이탈 예측 적중률',f'{m["precision"]:.1%}' if m['tp']+m['fp'] else '대상 없음')
    b.write(f'이탈할 것으로 본 {m["tp"]+m["fp"]:,}명 중 실제 이탈은 {m["tp"]:,}명입니다.')
    c.metric('이탈 고객 탐지율',f'{m["recall"]:.1%}' if m['tp']+m['fn'] else '평가할 이탈 없음')
    c.write(f'실제 이탈 {m["tp"]+m["fn"]:,}명 중 {m["tp"]:,}명을 미리 찾아냈습니다.')
    section('관리 인원과 기준 비교', '분류 임계값별 성능 및 대상 규모')
    comparison=threshold_comparison(valid,sorted(set([.3,.4,.52,.6,.7,threshold])))
    st.write('같은 과거 고객 파일에 여러 기준을 적용한 비교입니다. 가장 잘 나온 기준을 새로운 검증 결과로 취급하지 않습니다. 기준을 확정하려면 별도 검증 데이터와 상담 비용을 함께 확인해야 합니다.')
    st.dataframe(comparison,hide_index=True,width='stretch',column_config={
        '집중관리 기준':st.column_config.NumberColumn(format='percent'),
        '대상 중 실제 이탈 비율':st.column_config.NumberColumn(format='percent'),
        '이탈 고객 발견 비율':st.column_config.NumberColumn(format='percent')})
    report={'model':metadata.get('final_model','XGBoost'),'threshold':threshold,'watch_threshold':watch,
            'evaluated_customers':len(valid),'metrics':{k:float(v) if np.isfinite(v) else None for k,v in m.items()},
            'independent_test_verified':False,'evaluation_scope':'provided historical customer file',
            'model_sha256':model_fingerprint(ROOT/'models'/metadata.get('model_file','XGBoost_Final.pkl')) if model else None}
    report_text=f"평가 고객: {len(valid):,}명\n집중관리 기준: {threshold:.0%}\n정확도: {m['accuracy']:.1%}\n이탈 예측 적중률: {m['precision']:.1%}\n이탈 탐지율: {m['recall']:.1%}\n평가 범위: 과거 고객 자료 기준, 독립 평가 여부 미확인"
    st.download_button('평가 결과 보고서 저장',report_text,'고객이탈_평가보고서.txt','text/plain')
    section('결과를 고객 수로 확인', '혼동행렬 구성 지표')
    a,b,c,d=st.columns(4)
    for col,title,value,desc in [(a,'이탈을 찾아낸 고객',m['tp'],'모델도 이탈로 봤고, 실제로 이탈했습니다.'),
                                  (b,'이탈을 놓친 고객',m['fn'],'유지할 것으로 봤지만, 실제로 이탈했습니다.'),
                                  (c,'이탈 예상과 달리 유지한 고객',m['fp'],'이탈할 것으로 봤지만, 실제로는 유지했습니다.'),
                                  (d,'유지를 맞힌 고객',m['tn'],'유지할 것으로 봤고, 실제로 유지했습니다.')]:
        col.metric(title,f'{value:,}명');col.write(desc)
    if m['tp']==0:
        st.warning('현재 기준에서는 실제 이탈 고객을 한 명도 찾아내지 못했습니다. 0은 자료가 없다는 뜻이 아니라 탐지에 실패했다는 뜻입니다. 기준과 입력 데이터를 확인하세요.')
    if not m['tp']+m['fp']:
        st.info('이탈로 예측한 고객이 없으므로 집중관리 대상의 적중 비율을 계산할 수 없습니다. 의미 없는 0.000 대신 대상 없음으로 표시합니다.')
    st.write('기준을 낮추면 더 많은 이탈 고객을 찾을 수 있지만, 유지할 고객에게도 연락할 수 있습니다. 상담 가능 인원과 고객 불편을 함께 고려해 기준을 정하세요.')
    section('확률의 의미 확인', '예측 확률과 관측 이탈률 비교')
    cal=valid[['이탈예측확률','실제이탈']].copy()
    cal['확률구간']=pd.cut(cal['이탈예측확률'],[0,.2,.4,.6,.8,1],include_lowest=True,labels=['0~20%','20~40%','40~60%','60~80%','80~100%'])
    summary=cal.groupby('확률구간',observed=True).agg(고객수=('실제이탈','size'),예측평균=('이탈예측확률','mean'),실제이탈률=('실제이탈','mean')).reset_index()
    long=summary.melt(id_vars=['확률구간','고객수'],value_vars=['예측평균','실제이탈률'],var_name='비교',value_name='비율')
    long['확률구간']=long['확률구간'].astype(str)
    chart(alt.Chart(long).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
        x=alt.X('확률구간:N',sort=['0~20%','20~40%','40~60%','60~80%','80~100%'],title='모델이 예측한 확률 구간'),
        xOffset='비교:N',y=alt.Y('비율:Q',axis=alt.Axis(format='.0%'),scale=alt.Scale(domain=[0,1]),title='비율'),
        color=alt.Color('비교:N',scale=alt.Scale(domain=['예측평균','실제이탈률'],range=['#2d9b8d','#e9b44c'])),
        tooltip=['확률구간','고객수','비교',alt.Tooltip('비율:Q',format='.1%')]))
    st.write('같은 구간의 두 막대가 비슷할수록 예측 확률이 실제 결과와 가깝습니다. 차이가 크면 고객 우선순위를 정하는 데 활용하되, 숫자 그대로의 발생률로 해석하는 데 주의해야 합니다.')
    with st.expander('평가 항목 및 기존 분석 기록'):
        st.write(f'F1: {m["f1_score"]:.3f} / ROC-AUC: {m["roc_auc"]:.3f} / 확률 오차(Brier): {m["brier"]:.3f}')
        st.write('F1은 적중률과 탐지율의 균형, ROC-AUC는 이탈 고객에 더 높은 점수를 주는 정도, Brier는 확률과 실제 결과의 차이를 봅니다.')
        recorded=metadata.get('test_metrics',{})
        if recorded:
            st.dataframe(pd.DataFrame([{'평가 항목': {'accuracy':'정확도','precision':'이탈 예측 적중률','recall':'이탈 탐지율','f1_score':'균형 점수','roc_auc':'구분 능력','pr_auc':'이탈 탐지 종합 지표'}.get(k,k),'기록값':v} for k,v in recorded.items()]),hide_index=True)
            if 'accuracy' in recorded:
                st.write(f'기존 기록의 정확도는 {recorded["accuracy"]:.1%}, 현재 고객 자료 파일 재계산은 {m["accuracy"]:.1%}입니다. 현재 화면은 고객 자료 파일에서 확인한 값을 사용합니다.')
    section('원본 데이터 확인', '분석 데이터 구성')
    if isinstance(original,pd.DataFrame):
        st.write(f'원본 고객 파일: {len(original):,}행. 현재 화면의 예측 결과 파일: {len(df):,}행. 전체 원본과 예측 결과의 고객 수는 다를 수 있습니다.')
        with st.expander('데이터 품질 상세'):
            missing=int(original.isna().sum().sum())
            st.write(f'빈 항목 {missing:,}개 · 완전히 같은 행 {original.duplicated().sum():,}개. 빈 항목이 0개면 입력 정보가 모두 채워져 있다는 뜻입니다.')
            preview=original.drop(columns=['RegistrationToken'],errors='ignore').head(20)
            st.dataframe(preview.rename(columns={**LABELS,'CustomerId':'고객 번호','Surname':'원본 성','name':'등록 고객명','Exited':'실제 이탈 여부','RowNumber':'자료 순번','RegisteredAt':'등록 시각'}),hide_index=True,width='stretch')
    st.write('모델은 앞으로의 이탈을 보장하거나 막아주는 장치가 아닙니다. 상담 결과를 기록하고, 이후 고객에서도 같은 수준으로 맞히는지 계속 확인해야 합니다.')


if page=='home':overview()
elif page=='customers':explorer()
elif page=='predict':predict_page()
elif page=='validation':validation_page()
elif page=='advisor':advisor_page(ROOT,df,hero)
else:
    st.error('유효하지 않은 메뉴입니다. 새로고침 후 다시 선택하세요.')
st.markdown('<div class="footer">◈ RETAIN · 학습된 모델의 예측과 실제 고객 정보를 함께 확인하는 고객 관리 화면</div>',unsafe_allow_html=True)
remember_page_scroll()


