# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM
from datetime import datetime, timezone
from zoneinfo import ZoneInfo   # ← Replacement for pytz (built-in)
# ---------------------------------------------------------
# 0. PAGE CONFIG + GLOBAL CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="SecureBank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --navy: #0B1A3B;
    --navy-mid: #112255;
    --blue: #1A3A8F;
    --accent: #00D4AA;
    --accent2: #FFB700;
    --red: #FF4D4D;
    --text: #E8EDF8;
    --text-muted: #8A9BC2;
    --glass: rgba(255,255,255,0.07);
    --glass-b: rgba(255,255,255,0.12);
    --radius: 16px;
}
* { font-family: 'DM Sans', sans-serif !important; }
h1,h2,h3 { font-family: 'Playfair Display', serif !important; }
/* ---- full-page background ---- */
.stApp {
    background: linear-gradient(135deg, #0B1A3B 0%, #112255 50%, #1A3A8F 100%) !important;
    background-attachment: fixed !important;
    color: var(--text) !important;
}
/* ---- hide default streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebar"] { display: none !important; }
/* ---- buttons ---- */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.25s !important;
    border: none !important;
}
.stButton > button:hover { transform: translateY(-2px); }
/* ---- primary (green) ---- */
div[data-testid="stButton-primary"] > button,
.btn-primary-custom > .stButton > button {
    background: linear-gradient(135deg, #00D4AA, #007A62) !important;
    color: #000 !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.3) !important;
}
/* ---- ghost ---- */
.btn-ghost-custom > .stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--glass-b) !important;
    color: var(--text) !important;
}
.btn-ghost-custom > .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
/* ---- danger (red) ---- */
.btn-danger-custom > .stButton > button {
    background: linear-gradient(135deg, #FF4D4D, #CC0000) !important;
    color: white !important;
}
/* ---- glass card ---- */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(14px);
    border: 1.5px solid var(--glass-b);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin: 12px 0;
    transition: all 0.3s;
}
.glass-card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(0,0,0,0.4); }
/* ---- hero ---- */
.hero-wrap {
    min-height: 92vh;
    display: flex; flex-direction: column; justify-content: center;
    padding: 60px 40px;
    position: relative; overflow: hidden;
}
.hero-grid-bg {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 60px 60px;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.3);
    padding: 6px 18px; border-radius: 100px;
    font-size: 0.78rem; color: var(--accent);
    font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 24px; width: fit-content;
}
.pulse-dot {
    width: 8px; height: 8px;
    background: var(--accent); border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.4)} }
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.8rem, 4.5vw, 4.5rem) !important;
    font-weight: 900 !important; line-height: 1.1 !important;
    margin-bottom: 20px !important;
}
.hero-title em { font-style: normal; color: var(--accent); }
.hero-sub {
    color: var(--text-muted); font-size: 1.05rem; line-height: 1.75;
    max-width: 560px; margin-bottom: 36px;
}
/* ---- stat cards ---- */
.stat-card {
    background: var(--glass);
    border: 1px solid var(--glass-b);
    backdrop-filter: blur(20px);
    border-radius: var(--radius); padding: 24px 20px;
    text-align: center;
    animation: floatCard 4s ease-in-out infinite;
}
.stat-card:nth-child(2){animation-delay:-2s}
.stat-card:nth-child(3){animation-delay:-1s}
@keyframes floatCard { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
.stat-num {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: var(--accent); line-height: 1;
}
.stat-lbl { color: var(--text-muted); font-size: 0.78rem; margin-top: 4px; }
/* ---- section ---- */
.section-tag {
    font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase;
    color: var(--accent); font-weight: 700; margin-bottom: 12px;
}
.section-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(1.8rem,3vw,2.8rem) !important;
    font-weight: 700 !important; margin-bottom: 12px !important;
    line-height: 1.2 !important;
}
.section-sub { color: var(--text-muted); font-size: 0.95rem; line-height: 1.7; max-width: 520px; }
/* ---- feature card ---- */
.feat-card {
    background: var(--glass);
    border: 1px solid var(--glass-b);
    border-radius: var(--radius); padding: 28px;
    transition: all 0.3s; height: 100%;
}
.feat-card:hover {
    border-color: rgba(0,212,170,0.4);
    background: rgba(0,212,170,0.04);
    transform: translateY(-4px);
}
.feat-icon {
    width: 48px; height: 48px;
    background: rgba(0,212,170,0.12);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin-bottom: 16px;
}
.feat-title { font-size: 1.0rem; font-weight: 600; margin-bottom: 8px; }
.feat-body { color: var(--text-muted); font-size: 0.87rem; line-height: 1.6; }
/* ---- step ---- */
.step-num {
    width: 56px; height: 56px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), #007A62);
    color: #000; font-weight: 700; font-size: 1.3rem;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px;
}
.step-title { font-size: 1rem; font-weight: 600; margin-bottom: 8px; text-align: center; }
.step-body { color: var(--text-muted); font-size: 0.85rem; line-height: 1.6; text-align: center; }
/* ---- plan card ---- */
.plan-card {
    background: var(--glass);
    border: 1px solid var(--glass-b);
    border-radius: var(--radius); padding: 32px 28px;
    transition: all 0.3s; height: 100%;
}
.plan-card.featured { border-color: var(--accent); background: rgba(0,212,170,0.05); }
.plan-name { color: var(--text-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.plan-price {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem !important; font-weight: 700 !important;
    margin: 12px 0 4px;
}
.plan-price sup { font-size: 1rem; font-family: 'DM Sans', sans-serif !important; }
.plan-price sub { font-size: 0.85rem; color: var(--text-muted); font-family: 'DM Sans', sans-serif !important; }
.plan-feat-item {
    padding: 7px 0; font-size: 0.88rem; color: var(--text-muted);
    border-bottom: 1px solid var(--glass-b);
    display: flex; align-items: center; gap: 10px;
}
.plan-feat-item:last-child { border: none; }
.plan-feat-check { color: var(--accent); font-weight: 700; }
/* ---- auth forms ---- */
.auth-wrap {
    max-width: 420px; margin: 0 auto; padding: 48px 0;
}
.auth-logo { text-align: center; margin-bottom: 32px; }
.auth-logo h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important; font-weight: 700 !important;
    margin: 12px 0 4px;
}
.auth-logo p { color: var(--text-muted); font-size: 0.85rem; }
.auth-card {
    background: linear-gradient(135deg, #0F2060, #0B1A3B);
    border: 1px solid var(--glass-b);
    border-radius: 24px; padding: 40px 36px;
}
.auth-tabs {
    display: flex; background: rgba(0,0,0,0.3);
    border-radius: 12px; padding: 4px; margin-bottom: 28px; gap: 4px;
}
.auth-tab-btn {
    flex: 1; padding: 9px; border-radius: 9px;
    background: none; border: none; cursor: pointer; transition: all 0.2s;
    font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 0.88rem;
    color: var(--text-muted);
}
.auth-tab-btn.active { background: var(--accent); color: #000; }
/* ---- inputs ---- */
.stTextInput input {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid var(--glass-b) !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    background: rgba(0,212,170,0.05) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}
.stTextInput label { color: var(--text-muted) !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600 !important; }
/* ---- messages ---- */
.msg-error {
    background: rgba(255,77,77,0.15); border: 1px solid rgba(255,77,77,0.4);
    color: #FF8080; border-radius: 10px; padding: 12px 16px;
    font-size: 0.88rem; margin: 8px 0;
}
.msg-success {
    background: rgba(0,212,170,0.12); border: 1px solid rgba(0,212,170,0.4);
    color: var(--accent); border-radius: 10px; padding: 12px 16px;
    font-size: 0.88rem; margin: 8px 0;
}
/* ---- dashboard ---- */
.dash-greeting {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
}
.dash-sub { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 28px; }
.metric-card {
    background: var(--glass);
    border: 1px solid var(--glass-b);
    border-radius: var(--radius); padding: 24px;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem !important; font-weight: 700 !important; line-height: 1;
}
.metric-lbl { color: var(--text-muted); font-size: 0.78rem; margin-top: 6px; }
/* ---- result indicator ---- */
.result-safe {
    background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.3);
    border-radius: 14px; padding: 28px; text-align: center;
}
.result-fraud {
    background: rgba(255,77,77,0.1); border: 1px solid rgba(255,77,77,0.3);
    border-radius: 14px; padding: 28px; text-align: center;
}
.traffic-light {
    width: 90px; height: 90px; border-radius: 50%;
    margin: 0 auto 16px;
    animation: glowLight 2s infinite alternate;
}
.light-safe { background: radial-gradient(circle, #00D4AA, #007A62); box-shadow: 0 0 30px #00D4AA; }
.light-fraud { background: radial-gradient(circle, #FF4D4D, #CC0000); box-shadow: 0 0 30px #FF4D4D; }
@keyframes glowLight {
    from { box-shadow: 0 0 25px currentColor; }
    to { box-shadow: 0 0 55px currentColor, 0 0 80px rgba(0,0,0,0.3); }
}
.result-status-safe { font-family:"Playfair Display",serif!important; font-size:1.8rem!important; font-weight:700!important; color:var(--accent); }
.result-status-fraud { font-family:"Playfair Display",serif!important; font-size:1.8rem!important; font-weight:700!important; color:var(--red); }
/* ---- admin table ---- */
.users-table { width: 100%; border-collapse: collapse; }
.users-table th {
    text-align: left; padding: 12px 14px;
    font-size: 0.75rem; font-weight: 700;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid var(--glass-b);
}
.users-table td {
    padding: 13px 14px; font-size: 0.88rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.badge-active { color: var(--accent); background: rgba(0,212,170,0.12); padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; }
.badge-blocked { color: var(--red); background: rgba(255,77,77,0.12); padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; }
/* ---- blocked screen ---- */
.blocked-wrap { text-align: center; padding: 100px 40px; }
.blocked-icon { font-size: 5rem; margin-bottom: 16px; }
/* ---- footer ---- */
.site-footer {
    background: rgba(5,12,30,0.7);
    border-top: 1px solid var(--glass-b);
    padding: 40px; text-align: center;
    color: var(--text-muted); font-size: 0.85rem;
    margin-top: 40px;
}
/* ---- divider ---- */
.divider-line {
    border: none; border-top: 1px solid var(--glass-b); margin: 24px 0;
}
/* stSlider */
.stSlider > div { color: var(--text-muted) !important; }
/* ---- title bar ---- */
.titlebar {
    position: sticky; top: 0; z-index: 200;
    background: rgba(11,26,59,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.09);
    padding: 0;
    margin-bottom: 0;
}
.titlebar-inner {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 14px 32px;
    gap: 16px;
}
.tb-left { display: flex; align-items: center; gap: 12px; }
.tb-center { display: flex; align-items: center; gap: 24px; }
.tb-right { display: flex; align-items: center; gap: 10px; }
.tb-logo {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem; font-weight: 700; color: white;
    text-decoration: none; white-space: nowrap;
}
.tb-logo-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg,#00D4AA,#007A62);
    border-radius: 9px; display: flex;
    align-items: center; justify-content: center; font-size: 1rem;
    flex-shrink: 0;
}
.tb-sep {
    width: 1px; height: 22px;
    background: rgba(255,255,255,0.15); flex-shrink: 0;
}
.tb-breadcrumb {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.82rem; color: var(--text-muted);
}
.tb-breadcrumb .crumb { color: var(--text-muted); }
.tb-breadcrumb .crumb-sep { color: rgba(255,255,255,0.2); }
.tb-breadcrumb .crumb-active{ color: white; font-weight: 600; }
.tb-navlink {
    color: var(--text-muted); font-size: 0.88rem; font-weight: 500;
    text-decoration: none; padding: 6px 4px; transition: color 0.2s;
    white-space: nowrap;
}
.tb-navlink:hover { color: var(--accent); }
.tb-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.25);
    color: var(--accent); border-radius: 100px;
    padding: 4px 12px; font-size: 0.75rem; font-weight: 700;
    white-space: nowrap;
}
.tb-email {
    color: var(--text-muted); font-size: 0.82rem; white-space: nowrap;
    max-width: 180px; overflow: hidden; text-overflow: ellipsis;
}
/* back button in titlebar */
.tb-back-btn button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    color: var(--text-muted) !important;
    border-radius: 9px !important;
    padding: 6px 14px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    white-space: nowrap;
}
.tb-back-btn button:hover {
    background: rgba(255,255,255,0.13) !important;
    border-color: rgba(255,255,255,0.28) !important;
    color: white !important;
    transform: translateX(-2px) !important;
}
/* log-out button */
.tb-logout-btn button {
    background: rgba(255,77,77,0.1) !important;
    border: 1px solid rgba(255,77,77,0.25) !important;
    color: #FF8080 !important;
    border-radius: 9px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.tb-logout-btn button:hover {
    background: rgba(255,77,77,0.22) !important;
    color: white !important;
}
/* login/register buttons in titlebar */
.tb-login-btn button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: white !important;
    border-radius: 9px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
.tb-register-btn button {
    background: linear-gradient(135deg,#00D4AA,#007A62) !important;
    border: none !important;
    color: #000 !important;
    border-radius: 9px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 12px rgba(0,212,170,0.3) !important;
}
/* zero top padding on the block container so bar sits flush */
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. FIREBASE & MODEL
# ---------------------------------------------------------
def init_firebase():
    if not firebase_admin._apps:
        cred_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    if "db" not in st.session_state:
        st.session_state.db = firestore.client()
init_firebase()
MODEL, SCALER, THRESHOLD = load_model_and_assets()
if MODEL is None:
    st.stop()
db = st.session_state.db
ADMIN_EMAIL = "admin@securebank.com"

# ---------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------
def fb_get_user(email: str):
    try:
        return auth.get_user_by_email(email)
    except Exception:
        return None

def fb_create_user(email: str, password: str):
    try:
        user = auth.create_user(email=email, password=password)
        return user, None
    except Exception as e:
        return None, str(e)

def fb_send_password_reset(email: str):
    return fb_get_user(email) is not None

def is_blocked(uid: str) -> bool:
    try:
        doc = db.collection("blocked_users").document(uid).get()
        return doc.exists and doc.to_dict().get("blocked", False)
    except Exception:
        return False

def set_blocked(uid: str, blocked: bool):
    db.collection("blocked_users").document(uid).set(
        {"blocked": blocked, "updated": firestore.SERVER_TIMESTAMP}
    )

# ==================== FIXED: save_transaction (using zoneinfo) ====================
def save_transaction(uid: str, amount: float, error: float, fraud: bool):
    performed_by = st.session_state.get("email", uid)
    try:
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            d = user_doc.to_dict()
            name = d.get("name", "").strip()
            email = d.get("email", "").strip()
            if name and email:
                performed_by = f"{name} ({email})"
            elif name:
                performed_by = name
            elif email:
                performed_by = email
    except Exception:
        pass

    try:
        nairobi_tz = ZoneInfo("Africa/Nairobi")
        local_time = datetime.now(nairobi_tz)

        db.collection("transactions").add({
            "uid": str(uid),
            "performed_by": performed_by,
            "amount": float(amount),
            "error": float(error),
            "fraud": bool(fraud),
            "timestamp": firestore.SERVER_TIMESTAMP,
            "local_time": local_time,
        })
    except Exception as e:
        st.error(f"Could not save transaction: {e}")

# ---------------------------------------------------------
# 3. SESSION DEFAULTS
# ---------------------------------------------------------
defaults = {
    "logged_in": False,
    "uid": None,
    "email": None,
    "is_admin": False,
    "page": "landing",
    "page_history": [],
    "auth_error": "",
    "auth_success": "",
    "last_result": None,
    "threshold": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.threshold is None:
    try:
        thr_doc = db.collection("settings").document("model").get()
        if thr_doc.exists:
            st.session_state.threshold = float(thr_doc.to_dict().get("threshold", THRESHOLD))
        else:
            st.session_state.threshold = float(THRESHOLD)
    except Exception:
        st.session_state.threshold = float(THRESHOLD)

def go(page: str, push_history: bool = True):
    if push_history and st.session_state.page not in ("blocked",):
        history = st.session_state.get("page_history", [])
        if not history or history[-1] != st.session_state.page:
            history.append(st.session_state.page)
        st.session_state.page_history = history
    st.session_state.page = page
    st.session_state.auth_error = ""
    st.session_state.auth_success = ""
    st.rerun()

def go_back():
    history = st.session_state.get("page_history", [])
    if history:
        prev = history.pop()
        st.session_state.page_history = history
        st.session_state.page = prev
        st.session_state.auth_error = ""
        st.session_state.auth_success = ""
        st.rerun()
    else:
        go("landing", push_history=False)

def logout():
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

# ---------------------------------------------------------
# 4. NAVIGATION TITLE BAR
# ---------------------------------------------------------
PAGE_META = {
    "landing": {"label": "Home", "parent": None},
    "login": {"label": "Sign In", "parent": "landing"},
    "register": {"label": "Create Account", "parent": "landing"},
    "reset": {"label": "Reset Password", "parent": "login"},
    "dashboard": {"label": "Dashboard", "parent": "landing"},
    "blocked": {"label": "Account Suspended", "parent": None},
}

def _breadcrumbs(current_page: str) -> list:
    chain = []
    p = current_page
    while p:
        meta = PAGE_META.get(p, {"label": p.title(), "parent": None})
        chain.insert(0, (p, meta["label"]))
        p = meta.get("parent")
    return chain

def render_nav():
    page = st.session_state.page
    history = st.session_state.get("page_history", [])
    can_go_back = len(history) > 0

    crumbs = _breadcrumbs(page)
    crumb_html = ""
    for i, (pg, label) in enumerate(crumbs):
        is_last = (i == len(crumbs) - 1)
        if is_last:
            crumb_html += f'<span class="crumb-active">{label}</span>'
        else:
            crumb_html += f'<span class="crumb">{label}</span>'
            crumb_html += '<span class="crumb-sep">›</span>'

    if not st.session_state.logged_in:
        mid_html = """
            <a href="#features" class="tb-navlink">Features</a>
            <a href="#how-it-works" class="tb-navlink">How It Works</a>
        """
    else:
        role = "Admin" if st.session_state.is_admin else "Member"
        email = st.session_state.email or ""
        mid_html = f"""
            <span class="tb-pill">
                <span style="width:7px;height:7px;background:#00D4AA;border-radius:50%;
                             display:inline-block;animation:pulse 2s infinite;"></span>
                {role}
            </span>
            <span class="tb-email" title="{email}">{email}</span>
        """

    st.markdown(f"""
    <div class="titlebar">
      <div class="titlebar-inner">
        <div class="tb-left">
          <div class="tb-logo">
            <div class="tb-logo-icon">🏦</div>
            SecureBank
          </div>
          <div class="tb-sep"></div>
          <div class="tb-breadcrumb">{crumb_html}</div>
        </div>
        <div class="tb-center">{mid_html}</div>
        <div class="tb-right" id="tb-buttons-placeholder"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, btn_area = st.columns([6, 1.6])
    with btn_area:
        btn_cols_count = (1 if can_go_back else 0) + (2 if not st.session_state.logged_in else 1)
        if btn_cols_count == 0:
            btn_cols_count = 1
        bcols = st.columns(btn_cols_count)
        col_idx = 0
        if can_go_back:
            with bcols[col_idx]:
                st.markdown('<div class="tb-back-btn">', unsafe_allow_html=True)
                if st.button("← Back", key=f"nav_back_{page}", use_container_width=True):
                    go_back()
                st.markdown('</div>', unsafe_allow_html=True)
            col_idx += 1
        if not st.session_state.logged_in:
            with bcols[col_idx]:
                st.markdown('<div class="tb-login-btn">', unsafe_allow_html=True)
                if st.button("Log In", key=f"nav_login_{page}", use_container_width=True):
                    go("login")
                st.markdown('</div>', unsafe_allow_html=True)
            col_idx += 1
            with bcols[col_idx]:
                st.markdown('<div class="tb-register-btn">', unsafe_allow_html=True)
                if st.button("Get Started", key=f"nav_register_{page}", use_container_width=True):
                    go("register")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with bcols[col_idx]:
                st.markdown('<div class="tb-logout-btn">', unsafe_allow_html=True)
                if st.button("Log Out", key=f"nav_logout_{page}", use_container_width=True):
                    logout()
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5–7. LANDING, AUTH & BLOCKED PAGES (unchanged)
# ---------------------------------------------------------
# (All your original render_landing, render_login, render_register, render_reset, render_blocked functions remain exactly the same)
# For brevity, they are kept identical to your original code.

def render_landing():
    # ... [Your full original render_landing() code here - unchanged] ...
    st.markdown('<div class="hero-grid-bg"></div>', unsafe_allow_html=True)
    col_hero, col_stats = st.columns([3, 1])
    with col_hero:
        st.markdown("""
        <div class="hero-wrap">
            <div class="hero-badge"><span class="pulse-dot"></span>&nbsp;AI-Powered Security</div>
            <h1 class="hero-title">Banking Built on <em>Trust.</em></h1>
            <p class="hero-sub">
                Our machine-learning engine monitors every transaction in real time — catching fraud
                before it impacts you, with 86% detection accuracy and near-zero false alerts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.4, 1.4, 2])
        with c1:
            if st.button("🚀 Open an Account", use_container_width=True, type="primary", key="hero_register"):
                go("register")
        with c2:
            if st.button("Sign In →", use_container_width=True, key="hero_login"):
                go("login")
    with col_stats:
        st.markdown("""
        <div style="padding-top:80px;display:flex;flex-direction:column;gap:16px;">
            <div class="stat-card"><div class="stat-num">86%</div><div class="stat-lbl">Fraud Detection Rate</div></div>
            <div class="stat-card"><div class="stat-num">&lt;4%</div><div class="stat-lbl">False Alert Rate</div></div>
            <div class="stat-card"><div class="stat-num">2ms</div><div class="stat-lbl">Scan Latency</div></div>
        </div>
        """, unsafe_allow_html=True)
    # features and how it works sections (unchanged - copy from your original)
    # ... [paste the rest of your original render_landing() if needed] ...

# (To keep the response reasonable in length, the full auth and blocked functions are identical to your original.
# Just copy-paste your original `render_login`, `render_register`, `render_reset`, `render_blocked` here.)

# ---------------------------------------------------------
# 8. USER DASHBOARD - Fixed with ZoneInfo
# ---------------------------------------------------------
def render_user_dashboard():
    nairobi_tz = ZoneInfo("Africa/Nairobi")
    now = datetime.now(nairobi_tz)
    hour = now.hour
    greet = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    try:
        doc = db.collection("users").document(st.session_state.uid).get()
        display_name = doc.to_dict().get("name", st.session_state.email).split()[0] if doc.exists else st.session_state.email
    except Exception:
        display_name = st.session_state.email

    # Rest of your original render_user_dashboard() code goes here unchanged...
    # (Only the greeting part was updated)

    # ... [Paste the rest of your original render_user_dashboard() from the previous version] ...

# ---------------------------------------------------------
# 9. ADMIN DASHBOARD - Fixed Timestamp Display
# ---------------------------------------------------------
def render_admin_dashboard():
    # ... your original metrics, pie chart, user management (unchanged) ...

    # Transaction history - Fixed timezone handling
    st.markdown("""
    <div style="margin-top:40px;">
        <div class="section-tag">Audit Trail</div>
        <h3 style="font-family:&quot;Playfair Display&quot;,serif;font-size:1.4rem;margin-bottom:20px;">All Transactions</h3>
    </div>
    """, unsafe_allow_html=True)

    try:
        # ... your uid_to_name logic (unchanged) ...

        tx_docs = (
            db.collection("transactions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(100)
            .stream()
        )
        tx_data = []
        nairobi_tz = ZoneInfo("Africa/Nairobi")

        for doc in tx_docs:
            d = doc.to_dict()
            ts = d.get("timestamp") or d.get("local_time")
            uid = (d.get("uid") or "").strip()
            performed_by = (d.get("performed_by") or "") or uid[:14] + "…" if len(uid) > 14 else uid or "—"

            if ts:
                if hasattr(ts, 'astimezone'):
                    ts_local = ts.astimezone(nairobi_tz)
                else:
                    ts_local = datetime.fromtimestamp(ts.timestamp(), tz=nairobi_tz) if hasattr(ts, 'timestamp') else ts
                ts_str = ts_local.strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = "—"

            tx_data.append({
                "Timestamp": ts_str,
                "Performed By": performed_by,
                "Amount": f"${float(d.get('amount', 0)):,.2f}",
                "Status": "🔴 Blocked" if d.get("fraud") else "🟢 Approved",
                "Risk": f"{int(float(d.get('error', 0)) * 100)}%",
            })

        if tx_data:
            df_tx = pd.DataFrame(tx_data)
            st.dataframe(df_tx, hide_index=True, use_container_width=True)
        else:
            st.info("No transactions recorded yet.")
    except Exception as e:
        st.error(f"Firestore error: {e}")

    # ... rest of your original admin dashboard (threshold slider etc.) unchanged ...

# ---------------------------------------------------------
# 10. ROUTER
# ---------------------------------------------------------
page = st.session_state.page
if page == "dashboard":
    if not st.session_state.logged_in:
        st.session_state.page = "login"
        st.rerun()
    elif is_blocked(st.session_state.uid):
        st.session_state.page = "blocked"
        st.rerun()
elif page not in ("landing", "login", "register", "reset", "blocked", "dashboard"):
    st.session_state.page = "landing"
    st.rerun()

page = st.session_state.page
render_nav()

if page == "landing":
    render_landing()
elif page == "login":
    render_login()
elif page == "register":
    render_register()
elif page == "reset":
    render_reset()
elif page == "blocked":
    render_blocked()
elif page == "dashboard":
    if st.session_state.is_admin:
        render_admin_dashboard()
    else:
        render_user_dashboard()
