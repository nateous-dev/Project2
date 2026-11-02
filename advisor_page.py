import json,uuid,subprocess,os
from pathlib import Path
import streamlit as st
from chat_backend import settings,Logs,probe,answer
from rag_engine import KnowledgeIndex,find_corpus,anonymize


@st.cache_resource
def knowledge(path,signature):return KnowledgeIndex(path)


def advisor_page(root,df,hero):
    hero('고객 상담 지원 AI','고객 정보 확인 · 근거 기반 상담 초안 · 상담 이력 관리')
    st.caption('연한 파란색 테두리의 칸에 입력하세요. 흰색 상자는 생성 결과이며 직접 수정하는 입력란이 아닙니다.')
    st.write('고객 정보 확인 → 사유 선택 → 근거 검색 → 상담 초안 → 이어지는 질문 → 고객별 기록')
    if df is None or df.empty:st.info('고객 자료가 필요합니다.');return
    config=settings(root);logs=Logs(root)
    pool=df
    preferred=st.session_state.get('advisor_customer_id')
    options=pool.index.tolist()
    default=next((i for i,k in enumerate(options) if str(pool.at[k,'고객ID'])==preferred),0)
    idx=st.selectbox('상담 고객',options,index=default,key='advisor_choice',format_func=lambda i:f'{pool.at[i,"고객명"]} · {pool.at[i,"고객ID"]}')
    row=pool.loc[idx];cid=str(row['고객ID']);session_key='chat_session_'+cid
    st.session_state.setdefault(session_key,uuid.uuid4().hex)
    a,b,c=st.columns(3);a.metric('이탈 예측 확률',f'{row["이탈예측확률"]:.1%}');b.metric('관리 등급',row['위험등급']);c.metric('보유 상품',str(row.get('NumOfProducts','미확인')))
    if row.get('실제이탈')==1:st.warning('과거 이탈이 확인된 고객입니다. 신규 이탈 예방 상담과 구분해 과거 사유·재거래 의향을 확인하세요.')
    st.write(f'고객 정보: 연령 {row.get("Age","미확인")}세 · 이용 상태 {"비활성" if row.get("IsActiveMember")==0 else "활성"} · 거래기간 {row.get("Tenure","미확인")}년')
    st.caption('모델의 위험 신호는 실제 이탈 이유의 확정 근거가 아닙니다. 고객의 설명을 먼저 확인하세요.')
    from rag_engine import REASON_QUERIES
    pending_key='pending_counsel_'+cid
    pending=st.session_state.get(pending_key)
    with st.container(border=True):
        reason=st.selectbox('고객이 확인한 사유 또는 현재 확인 상태',list(REASON_QUERIES),index=5,key='reason_'+cid)
        statement=st.text_area('고객 설명 · 상담 메모',key='statement_'+cid,placeholder='개인정보를 제외하고 불편한 점과 원하는 지원을 기록하세요.')
        initial=st.button('고객 정보 기반 1차 상담 제안 생성',key='initial_'+cid,disabled=bool(pending),type='primary')
        consent=st.checkbox('외부 모델에 개인정보를 제외한 상담 내용을 전송하는 것을 확인했습니다. 기록은 생성 후 보관 여부를 선택합니다.',key='consent_'+cid)
    try:
        path=find_corpus(root);index=knowledge(str(path),(path.stat().st_size,path.stat().st_mtime_ns))
        st.caption(f'상담 지식 자료 {len(index.docs)}개 연결 · 문서 출처와 페이지 확인 가능')
    except Exception as exc:st.warning(str(exc));return
    records=logs.get(cid)
    with st.expander(f'과거 상담 기록 · {len(records)}건',expanded=False):
        needle=st.text_input('기록 내용 검색',key='history_search_'+cid)
        filtered=[r for r in records if not needle or needle.lower() in r['content'].lower()]
        for r in filtered[-30:]:
            st.caption(r['created']+' · '+('상담자' if r['role']=='user' else '상담 지원'))
            st.text(r['content'])
        if st.button('과거 상담 요약',key='summary_action_'+cid):
            questions=[r['content'] for r in records if r['role']=='user']
            summary=('상담 기록 없음' if not questions else f'질문·요청 {len(questions)}건\n최근 기록:\n'+'\n'.join('• '+q[:220] for q in questions[-5:]))
            st.session_state['summary_'+cid]=summary
        if st.button('AI 상담 요약',key='ai_summary_'+cid,disabled=not config['LLM_PROVIDER'] or not records):
            if not consent:
                st.warning('기록 저장·전송 확인을 먼저 체크하세요.')
            else:
                try:
                    evidence=index.search(reason,'상담 기록 요약 후속 조치')
                    history=[{'role':r['role'],'content':anonymize(r['content'])} for r in records[-12:]]
                    with st.spinner('상담 기록 요약 중…'):
                        result,method=answer(config,'제공된 과거 대화만으로 고객이 직접 말한 사유, 확인된 사실, 미해결 사항, 다음 상담 확인 항목을 요약하세요. 모델 안내를 고객의 발언으로 바꾸지 마세요. 과거 기록과 참고 문서의 권고를 구분하세요.',history,evidence,{},'과거 상담 요약')
                    st.session_state['ai_result_'+cid]=result
                except Exception as exc:st.error(str(exc))
        if st.session_state.get('ai_result_'+cid):
            st.caption('AI 요약 · 최근 대화 12건 기준 · 상담자 검토 필요')
            st.text(st.session_state['ai_result_'+cid])
        if st.session_state.get('summary_'+cid):
            st.caption('기록 발췌 요약 · AI 추론 요약 아님');st.text(st.session_state['summary_'+cid])
        st.download_button('고객 상담 기록 내려받기','\n\n'.join(r['created']+' '+r['role']+'\n'+r['content'] for r in records),'상담기록.txt','text/plain',key='export_'+cid)
    st.subheader('상담 대화')
    for r in records[-12:]:
        with st.chat_message(r['role']):
            st.text(r['content'])
            if r['method']:st.caption(r['method'])
            for e in json.loads(r['evidence'] or '[]'):
                with st.expander(f'[{e["ref"]}] {e["title"]} · '+(f'{e["page"]}쪽' if e.get('page') else '웹 본문')):
                    st.markdown(f'[원문]({e["url"]})');st.text(e['text'])
    if pending:
        st.subheader('보관 전 상담 초안')
        st.text(pending['text'])
        st.caption('이 초안은 아직 기록에 저장되지 않았습니다. 보관하면 다음 질문에서 이 고객의 상담 이력으로 참고합니다. 모델 자체를 재학습하지는 않습니다.')
        for e in pending['evidence']:
            with st.expander(f'[{e["ref"]}] {e["title"]} · '+(f'{e["page"]}쪽' if e.get('page') else '웹 본문')):
                st.markdown(f'[원문]({e["url"]})');st.text(e['text'])
        save_col,discard_col=st.columns(2)
        if save_col.button('상담 내용 보관 · 다음 질문에 활용',key='keep_'+cid):
            logs.add_pair(cid,st.session_state[session_key],pending['prompt'],pending['text'],pending['method'],pending['evidence'])
            del st.session_state[pending_key]
            st.session_state['saved_notice_'+cid]=True
            st.rerun()
        if discard_col.button('보관하지 않기',key='discard_'+cid):
            del st.session_state[pending_key]
            st.rerun()
    if st.session_state.pop('saved_notice_'+cid,False):st.success('해당 고객의 상담 기록에 보관했습니다. 이어서 질문하면 보관된 이력을 함께 참고합니다.')
    with st.form('followup_'+cid,clear_on_submit=True):
        question=st.text_input('이어서 질문',placeholder='예: 고객이 수수료 부담을 말하면 무엇부터 확인할까요?')
        send=st.form_submit_button('질문 전송',disabled=bool(pending))
    if initial or send:
        if not consent:st.warning('상담 내용 전송 확인을 먼저 체크하세요.');return
        if not initial and not question.strip():st.warning('질문을 입력하세요.');return
        prompt='고객의 입력 정보를 바탕으로 상담자가 확인할 질문과 1차 대응 및 상담 초안을 제안해 주세요.' if initial else question.strip()
        evidence=index.search(reason,statement+' '+prompt)
        profile={'이탈확률':float(row['이탈예측확률']),'등급':row['위험등급'],'연령':int(row['Age']),'상품수':int(row['NumOfProducts']),'활성여부':int(row['IsActiveMember']),'과거이탈':bool(row.get('실제이탈')==1)}
        # Previous questions and answers stay customer-scoped and are masked before external use.
        history=[{'role':r['role'],'content':anonymize(r['content'])} for r in records[-12:]]
        try:
            with st.spinner('근거 검색 및 답변 준비 중…'):
                text,method=answer(config,prompt,history,evidence,profile,reason+' / '+anonymize(statement))
            st.session_state[pending_key]={'prompt':prompt+'\n확인 사유: '+reason+'\n고객 설명: '+statement,'text':text,'method':method,'evidence':evidence}
            st.rerun()
        except Exception as exc:st.error(str(exc))
