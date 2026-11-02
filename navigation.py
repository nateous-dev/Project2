"""Sidebar buttons and per-page scroll memory for the current browser tab."""
import json
import streamlit as st
import streamlit.components.v1 as components

PAGES = [
    ('home', '이탈 리스크 대시보드'),
    ('customers', '개별 고객 진단'),
    ('predict', '신규 고객 예측'),
    ('validation', '모델 성능 검증'),
    ('advisor', '고객 상담 지원 AI'),
]


def _select_page(page_id):
    st.session_state.retain_page = page_id
    st.session_state.retain_navigation += 1


def sidebar_navigation():
    st.session_state.setdefault('retain_page', 'home')
    st.session_state.setdefault('retain_navigation', 0)
    active = st.session_state.retain_page
    st.markdown('''<style>
    [data-testid="stSidebar"] [class*="st-key-nav_"] button {
        width:100%;min-height:49px;border-radius:8px;padding:12px 15px;
        justify-content:flex-start;border:1px solid #dce6ed;transition:background .15s;
        background:#f8fafc;color:#304e64;box-shadow:none;
    }
    [data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {
        background:#e9f4f1;border-color:#9cccc1;color:#126c64;
    }
    [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] {
        background:#147f77;border-color:#147f77;color:white;font-weight:700;
    }
    [data-testid="stSidebar"] [class*="st-key-nav_"] button p {
        font-size:14px;white-space:normal;text-align:left;
    }
    </style>''', unsafe_allow_html=True)
    for page_id, title in PAGES:
        st.button(title, key=f'nav_{page_id}', width='stretch',
                  type='primary' if page_id == active else 'secondary',
                  on_click=_select_page, args=(page_id,))
    return active


SCROLL_SCRIPT = r'''
(() => {
  const config = __CONFIG__;
  const host = window.parent;
  const doc = host.document;
  // Preserve the application's reviewed terminology against automatic translation.
  doc.documentElement.setAttribute('translate', 'no');
  doc.documentElement.classList.add('notranslate');
  if (!doc.querySelector('meta[name="google"][content="notranslate"]')) {
    const meta = doc.createElement('meta');
    meta.name = 'google'; meta.content = 'notranslate'; doc.head.appendChild(meta);
  }
  const storageKey = 'retain.page-scroll.v1:' + host.location.pathname;
  let state = host.__retainScrollMemory;
  if (!state) {
    let positions = {};
    try { positions = JSON.parse(host.sessionStorage.getItem(storageKey) || '{}'); }
    catch (_) { positions = {}; }
    if (!positions || typeof positions !== 'object' || Array.isArray(positions)) positions = {};
    state = host.__retainScrollMemory = {
      page: null, navigation: null, positions, restoring: false,
      pending: false, token: 0, cleanup: null,
    };
    state.main = () => doc.querySelector('[data-testid="stMain"]') ||
      doc.querySelector('[data-testid="stAppViewContainer"]') || doc.scrollingElement;
    state.position = () => {
      const el = state.main();
      return Math.max(0, el ? el.scrollTop : (host.scrollY || 0));
    };
    state.persist = () => {
      try { host.sessionStorage.setItem(storageKey, JSON.stringify(state.positions)); } catch (_) {}
    };
    state.save = () => {
      if (!state.page || state.restoring || state.pending) return;
      state.positions[state.page] = state.position();
      state.persist();
    };
    // Capture outgoing position before Streamlit begins its Python rerun.
    doc.addEventListener('click', event => {
      const target = event.target && event.target.closest ? event.target : null;
      if (target && target.closest('[data-testid="stSidebar"] [class*="st-key-nav_"] button')) {
        state.save();
        state.pending = true;
        if (state.pendingTimeout) host.clearTimeout(state.pendingTimeout);
        state.pendingTimeout = host.setTimeout(() => { state.pending = false; }, 10000);
      }
    }, true);
    doc.addEventListener('scroll', event => {
      const el = state.main();
      if (event.target === el || event.target === doc || event.target === doc.scrollingElement) state.save();
    }, true);
    host.addEventListener('scroll', state.save, {passive:true});
    host.addEventListener('pagehide', state.save);
  }

  // Filters, forms and chart updates on the same page must not reset scrolling.
  if (state.page === config.page && state.navigation === config.navigation) {
    state.pending = false;
    if (state.pendingTimeout) host.clearTimeout(state.pendingTimeout);
    return;
  }
  if (state.cleanup) state.cleanup();
  if (!state.pending) state.save();
  state.page = config.page;
  state.navigation = config.navigation;
  state.pending = false;
  if (state.pendingTimeout) host.clearTimeout(state.pendingTimeout);
  const desired = 0;
  state.restoring = true;
  const token = ++state.token;
  const start = host.performance.now();
  let stableSince = null;
  let timer = null;
  let stopped = false;
  const cancelKeys = new Set(['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' ']);
  const finish = () => {
    if (stopped) return;
    stopped = true;
    if (timer !== null) host.clearTimeout(timer);
    doc.removeEventListener('wheel', cancel, true);
    doc.removeEventListener('touchstart', cancel, true);
    doc.removeEventListener('pointerdown', cancel, true);
    doc.removeEventListener('keydown', cancel, true);
    state.cleanup = null;
    state.restoring = false;
    // Keep the requested position if a short layout has not finished expanding.
  };
  function cancel(event) {
    if (event.type === 'keydown' && !cancelKeys.has(event.key)) return;
    finish();
    // User interaction takes priority over any restoration in progress.
    host.requestAnimationFrame(state.save);
  }
  state.cleanup = finish;
  for (const type of ['wheel','touchstart','pointerdown','keydown']) doc.addEventListener(type,cancel,true);
  function restore() {
    if (stopped || token !== state.token) return;
    const el = state.main();
    if (el) {
      const max = Math.max(0, el.scrollHeight-el.clientHeight);
      el.scrollTo({top:Math.min(desired,max),behavior:'instant'});
      // Streamlit normally scrolls stMain, but older builds may scroll the window.
      if (el === doc.scrollingElement) host.scrollTo({top:desired,behavior:'instant'});
    }
    const reached = Math.abs(state.position()-desired) < 3;
    const now = host.performance.now();
    if (reached && stableSince === null) stableSince = now;
    if (!reached) stableSince = null;
    // Wait for charts/tables to settle; never keep pulling after user input.
    if (now-start >= 3500 || (stableSince !== null && now-stableSince >= 750)) {
      finish();
      return;
    }
    timer = host.setTimeout(restore,50);
  }
  host.requestAnimationFrame(restore);
})();
'''


def remember_page_scroll():
    config = json.dumps({'page': st.session_state.retain_page,
                         'navigation': st.session_state.retain_navigation})
    html = '<script>' + SCROLL_SCRIPT.replace('__CONFIG__', config) + '</script>'
    if hasattr(st, 'iframe'):
        st.iframe(html, height=1, tab_index=-1)
    else:
        components.html(html, height=0, scrolling=False)
