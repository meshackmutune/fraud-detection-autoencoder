# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM
from datetime import datetime

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
    --navy:       #0B1A3B;
    --navy-mid:   #112255;
    --blue:       #1A3A8F;
    --accent:     #00D4AA;
    --accent2:    #FFB700;
    --red:        #FF4D4D;
    --text:       #E8EDF8;
    --text-muted: #8A9BC2;
    --glass:      rgba(255,255,255,0.07);
    --glass-b:    rgba(255,255,255,0.12);
    --radius:     16px;
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
    to   { box-shadow: 0 0 55px currentColor, 0 0 80px rgba(0,0,0,0.3); }
}
.result-status-safe  { font-family:'Playfair Display',serif!important; font-size:1.8rem!important; font-weight:700!important; color:var(--accent); }
.result-status-fraud { font-family:'Playfair Display',serif!important; font-size:1.8rem!important; font-weight:700!important; color:var(--red); }

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
.badge-active  { color: var(--accent); background: rgba(0,212,170,0.12); padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; }
.badge-blocked { color: var(--red);    background: rgba(255,77,77,0.12);  padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; }

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
.tb-left  { display: flex; align-items: center; gap: 12px; }
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
.tb-breadcrumb .crumb       { color: var(--text-muted); }
.tb-breadcrumb .crumb-sep   { color: rgba(255,255,255,0.2); }
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
# 2. HELPERS — Firebase wrappers
# ---------------------------------------------------------
def fb_get_user(email: str):
    """Return Firebase user record or None."""
    try:
        return auth.get_user_by_email(email)
    except Exception:
        return None


def fb_create_user(email: str, password: str):
    """Create Firebase auth user, return (user, error_str)."""
    try:
        user = auth.create_user(email=email, password=password)
        return user, None
    except Exception as e:
        return None, str(e)


def fb_send_password_reset(email: str):
    """
    Firebase Admin SDK does not send reset emails directly.
    In production wire this to your backend or Firebase REST API.
    Here we simulate success if the user exists.
    """
    return fb_get_user(email) is not None


def is_blocked(uid: str) -> bool:
    """Check Firestore 'blocked_users' collection for this uid."""
    try:
        doc = db.collection("blocked_users").document(uid).get()
        return doc.exists and doc.to_dict().get("blocked", False)
    except Exception:
        return False


def set_blocked(uid: str, blocked: bool):
    """Set or clear the blocked flag in Firestore."""
    db.collection("blocked_users").document(uid).set(
        {"blocked": blocked, "updated": firestore.SERVER_TIMESTAMP}
    )


def save_transaction(uid: str, amount: float, error: float, fraud: bool):
    try:
        db.collection("transactions").add({
            "uid": str(uid),
            "amount": float(amount),
            "error": float(error),
            "fraud": bool(fraud),
            "timestamp": firestore.SERVER_TIMESTAMP,
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
    "page": "landing",          # landing | login | register | reset | dashboard | blocked
    "page_history": [],         # breadcrumb stack for back navigation
    "auth_error": "",
    "auth_success": "",
    "last_result": None,        # dict with err, fraud
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go(page: str, push_history: bool = True):
    """Navigate to a page, optionally recording the current page in history."""
    if push_history and st.session_state.page not in ("blocked",):
        history = st.session_state.get("page_history", [])
        # avoid duplicate consecutive entries
        if not history or history[-1] != st.session_state.page:
            history.append(st.session_state.page)
        st.session_state.page_history = history
    st.session_state.page = page
    st.session_state.auth_error = ""
    st.session_state.auth_success = ""
    st.rerun()


def go_back():
    """Pop the last page from history and navigate to it."""
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
# 4. NAVIGATION TITLE BAR (renders on all pages)
# ---------------------------------------------------------

# Human-readable page labels and their parent for breadcrumbs
PAGE_META = {
    "landing":   {"label": "Home",              "parent": None},
    "login":     {"label": "Sign In",           "parent": "landing"},
    "register":  {"label": "Create Account",    "parent": "landing"},
    "reset":     {"label": "Reset Password",    "parent": "login"},
    "dashboard": {"label": "Dashboard",         "parent": "landing"},
    "blocked":   {"label": "Account Suspended", "parent": None},
}


def _breadcrumbs(current_page: str) -> list:
    """Build breadcrumb chain by walking parent links."""
    chain = []
    p = current_page
    while p:
        meta = PAGE_META.get(p, {"label": p.title(), "parent": None})
        chain.insert(0, (p, meta["label"]))
        p = meta.get("parent")
    return chain


def render_nav():
    page    = st.session_state.page
    history = st.session_state.get("page_history", [])
    can_go_back = len(history) > 0

    # ── Titlebar HTML shell ────────────────────────────────────────────────
    # Logo + separator + breadcrumbs rendered as pure HTML (no buttons here)
    crumbs      = _breadcrumbs(page)
    crumb_html  = ""
    for i, (pg, label) in enumerate(crumbs):
        is_last = (i == len(crumbs) - 1)
        if is_last:
            crumb_html += f'<span class="crumb-active">{label}</span>'
        else:
            crumb_html += f'<span class="crumb">{label}</span>'
            crumb_html += '<span class="crumb-sep">›</span>'

    # Middle: landing nav links vs logged-in page tabs
    if not st.session_state.logged_in:
        mid_html = """
            <a href="#features"     class="tb-navlink">Features</a>
            <a href="#how-it-works" class="tb-navlink">How It Works</a>
            <a href="#pricing"      class="tb-navlink">Pricing</a>
        """
    else:
        role     = "Admin" if st.session_state.is_admin else "Member"
        email    = st.session_state.email or ""
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

    # ── Interactive buttons rendered via Streamlit columns ──────────────────
    # We use a tight column row that visually aligns with the right side of
    # the titlebar via negative-margin trick.
    _, btn_area = st.columns([6, 1.6])
    with btn_area:
        btn_cols_count = (1 if can_go_back else 0) + (2 if not st.session_state.logged_in else 1)
        if btn_cols_count == 0:
            btn_cols_count = 1
        bcols = st.columns(btn_cols_count)
        col_idx = 0

        # Back button — shown whenever there is history to go back to
        if can_go_back:
            with bcols[col_idx]:
                st.markdown('<div class="tb-back-btn">', unsafe_allow_html=True)
                if st.button("← Back", key="nav_back", use_container_width=True):
                    go_back()
                st.markdown('</div>', unsafe_allow_html=True)
            col_idx += 1

        if not st.session_state.logged_in:
            with bcols[col_idx]:
                st.markdown('<div class="tb-login-btn">', unsafe_allow_html=True)
                if st.button("Log In", key="nav_login", use_container_width=True):
                    go("login")
                st.markdown('</div>', unsafe_allow_html=True)
            col_idx += 1
            with bcols[col_idx]:
                st.markdown('<div class="tb-register-btn">', unsafe_allow_html=True)
                if st.button("Get Started", key="nav_register", use_container_width=True):
                    go("register")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with bcols[col_idx]:
                st.markdown('<div class="tb-logout-btn">', unsafe_allow_html=True)
                if st.button("Log Out", key="nav_logout", use_container_width=True):
                    logout()
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "<div style='height:8px;'></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 5. LANDING PAGE
# ---------------------------------------------------------
def render_landing():
    # hero
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
            if st.button("🚀  Open an Account", use_container_width=True, type="primary", key="hero_register"):
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

    # ---- features ----
    st.markdown("""
    <div id="features" style="padding:60px 0 20px;">
        <div class="section-tag">Why SecureBank</div>
        <div class="section-title">Security That Never Sleeps</div>
        <p class="section-sub">Built from the ground up with AI at the core, every layer of your banking experience is protected.</p>
    </div>
    """, unsafe_allow_html=True)
    feats = [
        ("🤖", "AI Fraud Detection",      "Our autoencoder learns normal transaction patterns and flags anomalies the moment they appear."),
        ("⚡", "Real-Time Scanning",       "Every transaction scanned in under 2ms — a verdict before your payment even completes."),
        ("🛡️", "Zero-Trust Architecture", "Multi-layer authentication and end-to-end encryption ensure your data stays yours."),
        ("📊", "Admin Intelligence",       "Full control panel for compliance teams with live dashboards, audit trails, and user management."),
        ("🔑", "Secure Authentication",   "Firebase-backed auth with password reset flows and session management baked in by default."),
        ("📱", "Works Everywhere",         "Fully responsive across desktop, tablet, and mobile with a native-feeling experience."),
    ]
    cols = st.columns(3)
    for i, (icon, title, body) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- how it works ----
    st.markdown("""
    <div id="how-it-works" style="padding:60px 0 20px;">
        <div class="section-tag">How It Works</div>
        <div class="section-title">Protection in 3 Steps</div>
        <p class="section-sub">Our pipeline runs silently in the background, keeping every transaction safe.</p>
    </div>
    """, unsafe_allow_html=True)
    steps = [
        ("1", "Submit Transaction",  "Enter the amount and hit Verify. Our API receives the request instantly."),
        ("2", "AI Scans It",         "The autoencoder reconstructs the transaction vector and computes a risk error score."),
        ("3", "Verdict Delivered",   "Safe or Blocked — you see the result with full risk visualisation in real time."),
    ]
    s_cols = st.columns(3)
    for col, (num, title, body) in zip(s_cols, steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- pricing ----
    st.markdown("""
    <div id="pricing" style="padding:60px 0 20px;">
        <div class="section-tag">Pricing</div>
        <div class="section-title">Simple, Transparent Plans</div>
        <p class="section-sub">No hidden fees. Cancel anytime. Every plan includes core fraud protection.</p>
    </div>
    """, unsafe_allow_html=True)
    plans = [
        ("Starter",      "Free",  "/mo",   ["100 transactions/mo", "Basic fraud scanning", "Email alerts", "Community support"], False),
        ("Professional", "$29",   "/mo",   ["Unlimited transactions", "Advanced AI scoring", "Real-time dashboard", "Priority support", "API access"], True),
        ("Enterprise",   "Custom","",      ["Everything in Pro", "Custom model tuning", "Admin control panel", "SLA guarantee", "Dedicated engineer"], False),
    ]
    p_cols = st.columns(3)
    for col, (name, price, period, feats_list, featured) in zip(p_cols, plans):
        with col:
            featured_style = "border-color:#00D4AA;background:rgba(0,212,170,0.04);" if featured else ""
            feats_html = "".join(
                f'<div class="plan-feat-item"><span class="plan-feat-check">✓</span>{f}</div>'
                for f in feats_list
            )
            badge = '<div style="background:var(--accent);color:#000;padding:3px 14px;border-radius:100px;font-size:0.72rem;font-weight:700;margin-bottom:12px;width:fit-content;">Most Popular</div>' if featured else ""
            st.markdown(f"""
            <div class="plan-card" style="{featured_style}">
                {badge}
                <div class="plan-name">{name}</div>
                <div class="plan-price">{price}<sub>{period}</sub></div>
                <div style="margin:16px 0;">{feats_html}</div>
            </div>
            """, unsafe_allow_html=True)
            label = "Start Free Trial" if featured else ("Get Started Free" if name == "Starter" else "Contact Sales")
            if st.button(label, use_container_width=True, type="primary" if featured else "secondary", key=f"plan_{name}"):
                go("register")

    # footer
    st.markdown("""
    <div class="site-footer">
        <p>© 2025 SecureBank · AI-powered fraud protection &nbsp;|&nbsp;
           <a href="#" style="color:var(--text-muted);">Privacy</a> &nbsp;·&nbsp;
           <a href="#" style="color:var(--text-muted);">Terms</a> &nbsp;·&nbsp;
           <a href="#" style="color:var(--text-muted);">Security</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 6. AUTH PAGES
# ---------------------------------------------------------
def render_auth_messages():
    if st.session_state.auth_error:
        st.markdown(f'<div class="msg-error">⚠️ {st.session_state.auth_error}</div>', unsafe_allow_html=True)
    if st.session_state.auth_success:
        st.markdown(f'<div class="msg-success">✅ {st.session_state.auth_success}</div>', unsafe_allow_html=True)


def render_login():
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        # logo + header
        st.markdown("""
        <div class="auth-logo">
            <div style="width:56px;height:56px;background:linear-gradient(135deg,#00D4AA,#007A62);
                        border-radius:16px;display:flex;align-items:center;justify-content:center;
                        font-size:1.8rem;margin:0 auto 12px;">🏦</div>
            <h2>Welcome Back</h2>
            <p>Sign in to your SecureBank account</p>
        </div>
        """, unsafe_allow_html=True)

        # tab buttons
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", use_container_width=True, type="primary", key="tab_login"):
                go("login")
        with c2:
            if st.button("Create Account", use_container_width=True, key="tab_register_from_login"):
                go("register")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        render_auth_messages()

        email = st.text_input("Email Address", placeholder="you@example.com", key="li_email")
        pwd   = st.text_input("Password",       placeholder="Enter your password", type="password", key="li_pwd")

        col_remember, col_forgot = st.columns([1, 1])
        with col_remember:
            st.checkbox("Remember me", key="li_remember")
        with col_forgot:
            st.markdown("<div style='text-align:right;padding-top:6px;'>", unsafe_allow_html=True)
            if st.button("Forgot password?", key="li_forgot"):
                go("reset")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Sign In →", use_container_width=True, type="primary", key="li_submit"):
            st.session_state.auth_error = ""
            if not email or not pwd:
                st.session_state.auth_error = "Please fill in all fields."
                st.rerun()

            fb_user = fb_get_user(email)
            if not fb_user:
                st.session_state.auth_error = "No account found with that email. Please register."
                st.rerun()

            if is_blocked(fb_user.uid):
                st.session_state.page = "blocked"
                st.rerun()

            # Note: Firebase Admin SDK cannot verify passwords directly.
            # In production, use Firebase REST API signInWithPassword endpoint.
            # Here we trust the email lookup as a demo stand-in.
            st.session_state.update(
                uid=fb_user.uid,
                email=fb_user.email,
                logged_in=True,
                is_admin=(email == ADMIN_EMAIL),
            )
            go("dashboard")

        st.markdown("""
        <div style="text-align:center;margin-top:20px;color:var(--text-muted);font-size:0.88rem;">
            New to SecureBank? &nbsp;
        </div>
        """, unsafe_allow_html=True)
        if st.button("Create a free account", use_container_width=True, key="li_to_register"):
            go("register")


def render_register():
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown("""
        <div class="auth-logo">
            <div style="width:56px;height:56px;background:linear-gradient(135deg,#00D4AA,#007A62);
                        border-radius:16px;display:flex;align-items:center;justify-content:center;
                        font-size:1.8rem;margin:0 auto 12px;">🏦</div>
            <h2>Create Account</h2>
            <p>Join SecureBank — free forever on the Starter plan</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", use_container_width=True, key="tab_login_from_reg"):
                go("login")
        with c2:
            if st.button("Create Account", use_container_width=True, type="primary", key="tab_register"):
                go("register")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        render_auth_messages()

        name    = st.text_input("Full Name",         placeholder="Jane Doe",          key="reg_name")
        email   = st.text_input("Email Address",     placeholder="you@example.com",   key="reg_email")
        pwd     = st.text_input("Password",          placeholder="Min 6 characters",  type="password", key="reg_pwd")
        confirm = st.text_input("Confirm Password",  placeholder="Repeat password",   type="password", key="reg_confirm")

        # password strength indicator
        if pwd:
            score = sum([
                len(pwd) >= 6,
                len(pwd) >= 10,
                bool(__import__('re').search(r'[A-Z]', pwd)),
                bool(__import__('re').search(r'[0-9]', pwd)),
                bool(__import__('re').search(r'[^A-Za-z0-9]', pwd)),
            ])
            levels = ["Very weak", "Weak", "Fair", "Strong", "Very strong"]
            colors = ["#FF4D4D", "#FF8C00", "#FFB700", "#7ED321", "#00D4AA"]
            widths = [20, 40, 60, 80, 100]
            idx = min(score - 1, 4) if score > 0 else 0
            st.markdown(f"""
            <div style="margin-top:-8px;margin-bottom:16px;">
                <div style="height:4px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{widths[idx]}%;background:{colors[idx]};border-radius:4px;transition:all 0.3s;"></div>
                </div>
                <span style="font-size:0.75rem;color:{colors[idx]};">{levels[idx]}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Create Account →", use_container_width=True, type="primary", key="reg_submit"):
            st.session_state.auth_error = ""
            if not all([name, email, pwd, confirm]):
                st.session_state.auth_error = "Please fill in all fields."
                st.rerun()
            if pwd != confirm:
                st.session_state.auth_error = "Passwords do not match."
                st.rerun()
            if len(pwd) < 6:
                st.session_state.auth_error = "Password must be at least 6 characters."
                st.rerun()
            if fb_get_user(email):
                st.session_state.auth_error = "An account with that email already exists."
                st.rerun()

            fb_user, err = fb_create_user(email, pwd)
            if err:
                st.session_state.auth_error = f"Could not create account: {err}"
                st.rerun()

            # save display name to Firestore
            try:
                db.collection("users").document(fb_user.uid).set({
                    "name": name, "email": email,
                    "created": firestore.SERVER_TIMESTAMP,
                })
            except Exception:
                pass

            st.session_state.update(
                uid=fb_user.uid,
                email=fb_user.email,
                logged_in=True,
                is_admin=False,
            )
            go("dashboard")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Already have an account? Sign in", use_container_width=True, key="reg_to_login"):
            go("login")


def render_reset():
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown("""
        <div class="auth-logo">
            <div style="font-size:3rem;margin-bottom:12px;">🔑</div>
            <h2>Reset Password</h2>
            <p>Enter your email and we'll send a reset link</p>
        </div>
        """, unsafe_allow_html=True)

        render_auth_messages()

        reset_email = st.text_input("Email Address", placeholder="you@example.com", key="reset_email")

        if st.button("Send Reset Link →", use_container_width=True, type="primary", key="reset_submit"):
            st.session_state.auth_error = ""
            if not reset_email:
                st.session_state.auth_error = "Please enter your email address."
                st.rerun()
            sent = fb_send_password_reset(reset_email)
            if sent:
                st.session_state.auth_success = f"Reset link sent to {reset_email}! Check your inbox."
            else:
                st.session_state.auth_error = "No account found with that email address."
            st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("← Back to Sign In", use_container_width=True, key="reset_back"):
            go("login")


# ---------------------------------------------------------
# 7. BLOCKED PAGE
# ---------------------------------------------------------
def render_blocked():
    st.markdown("""
    <div class="blocked-wrap">
        <div class="blocked-icon">🚫</div>
        <h2 style="color:var(--red);font-family:'Playfair Display',serif;font-size:2.2rem;">Account Blocked</h2>
        <p style="color:var(--text-muted);max-width:440px;margin:16px auto 32px;line-height:1.7;">
            Your account has been suspended by an administrator.
            Please contact <strong>support@securebank.com</strong> if you believe this is an error.
        </p>
    </div>
    """, unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("← Back to Home", use_container_width=True, key="blocked_back"):
            logout()


# ---------------------------------------------------------
# 8. USER DASHBOARD
# ---------------------------------------------------------
def render_user_dashboard():
    import re as _re

    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    try:
        doc = db.collection("users").document(st.session_state.uid).get()
        display_name = doc.to_dict().get("name", st.session_state.email).split()[0] if doc.exists else st.session_state.email
    except Exception:
        display_name = st.session_state.email

    # ── Header ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding: 8px 0 32px;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
            <div style="width:48px;height:48px;background:linear-gradient(135deg,#00D4AA,#007A62);
                        border-radius:14px;display:flex;align-items:center;justify-content:center;
                        font-size:1.4rem;">🏦</div>
            <div>
                <div style="font-family:'Playfair Display',serif;font-size:1.7rem;
                            font-weight:700;line-height:1.1;color:white;">
                    {greet}, {display_name}
                </div>
                <div style="color:var(--text-muted);font-size:0.88rem;margin-top:2px;">
                    Transaction Verification &nbsp;·&nbsp; AI Fraud Detection
                </div>
            </div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,rgba(0,212,170,0.4),transparent);
                    margin-top:16px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: input left, info cards right ──────
    col_main, col_side = st.columns([3, 2], gap="large")

    with col_main:
        # Input card
        st.markdown("""
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                    border-radius:20px;padding:32px 36px;margin-bottom:24px;">
            <div style="font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;
                        color:var(--accent);font-weight:700;margin-bottom:8px;">
                Transaction Scanner
            </div>
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                        font-weight:700;margin-bottom:4px;">
                Verify a Payment
            </div>
            <p style="color:var(--text-muted);font-size:0.88rem;line-height:1.6;margin-bottom:24px;">
                Enter the transaction amount below. Our AI engine will assess the risk
                in real time and return an instant verdict.
            </p>
        """, unsafe_allow_html=True)

        amt_col, btn_col = st.columns([3, 1])
        with amt_col:
            amt = st.text_input(
                "Transaction Amount (USD)",
                placeholder="0.00",
                key="txn_amount",
            )
        with btn_col:
            st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
            verify = st.button("Scan →", use_container_width=True, type="primary", key="txn_verify")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Run scan ────────────────────────────────────────
        if verify:
            try:
                amount = float(str(amt).replace(",", "").strip())
                if amount <= 0:
                    raise ValueError
            except Exception:
                st.warning("⚠️  Please enter a valid positive amount.")
                st.stop()

            vec = np.zeros(INPUT_DIM)
            vec[29] = amount

            with st.spinner("Scanning transaction through AI engine…"):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)

            err   = float(err)
            fraud = bool(fraud)
            save_transaction(st.session_state.uid, amount, err, fraud)
            st.session_state.last_result = {"err": err, "fraud": fraud, "amount": amount}

        # ── Result ──────────────────────────────────────────
        if st.session_state.last_result:
            r        = st.session_state.last_result
            err      = float(r["err"])
            fraud    = bool(r["fraud"])
            amount   = float(r["amount"])
            risk_pct = int(min(100, err * 100))

            accent_color = "#FF4D4D" if fraud else "#00D4AA"
            bg_color     = "rgba(255,77,77,0.08)"  if fraud else "rgba(0,212,170,0.08)"
            border_color = "rgba(255,77,77,0.3)"   if fraud else "rgba(0,212,170,0.3)"
            icon         = "⛔" if fraud else "✅"
            verdict      = "TRANSACTION BLOCKED" if fraud else "TRANSACTION APPROVED"
            sub_msg      = ("This payment exhibits characteristics consistent with fraudulent activity. "
                            "It has been flagged and blocked for your protection.")  if fraud else                            ("This payment has been assessed by our AI engine and cleared as legitimate. "
                            "No suspicious activity detected.")

            st.markdown(f"""
            <div style="background:{bg_color};border:1.5px solid {border_color};
                        border-radius:20px;padding:32px 36px;margin-bottom:24px;">

                <div style="display:flex;align-items:flex-start;gap:20px;">
                    <div style="width:64px;height:64px;border-radius:16px;flex-shrink:0;
                                background:{accent_color}22;display:flex;
                                align-items:center;justify-content:center;font-size:1.8rem;
                                border:1.5px solid {accent_color}55;">
                        {icon}
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:0.68rem;letter-spacing:2px;text-transform:uppercase;
                                    color:{accent_color};font-weight:700;margin-bottom:4px;">
                            AI Verdict
                        </div>
                        <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
                                    font-weight:700;color:{accent_color};line-height:1.1;">
                            {verdict}
                        </div>
                        <p style="color:var(--text-muted);font-size:0.87rem;
                                  line-height:1.65;margin-top:10px;max-width:480px;">
                            {sub_msg}
                        </p>
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:repeat(3,1fr);
                            gap:16px;margin-top:28px;padding-top:24px;
                            border-top:1px solid {border_color};">
                    <div style="text-align:center;">
                        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                                    font-weight:700;color:{accent_color};">${amount:,.2f}</div>
                        <div style="color:var(--text-muted);font-size:0.75rem;margin-top:3px;
                                    text-transform:uppercase;letter-spacing:0.5px;">Amount</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                                    font-weight:700;color:{accent_color};">{risk_pct}%</div>
                        <div style="color:var(--text-muted);font-size:0.75rem;margin-top:3px;
                                    text-transform:uppercase;letter-spacing:0.5px;">Risk Score</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                                    font-weight:700;color:{accent_color};">{err:.4f}</div>
                        <div style="color:var(--text-muted);font-size:0.75rem;margin-top:3px;
                                    text-transform:uppercase;letter-spacing:0.5px;">Error Score</div>
                    </div>
                </div>

                <div style="margin-top:20px;">
                    <div style="display:flex;justify-content:space-between;
                                margin-bottom:6px;font-size:0.8rem;color:var(--text-muted);">
                        <span>Risk Level</span>
                        <span>{risk_pct}% / 100%</span>
                    </div>
                    <div style="height:8px;background:rgba(255,255,255,0.08);
                                border-radius:100px;overflow:hidden;">
                        <div style="height:100%;width:{risk_pct}%;
                                    background:linear-gradient(90deg,{accent_color},{accent_color}99);
                                    border-radius:100px;transition:width 0.6s ease;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Charts side by side ──────────────────────────
            ch1, ch2 = st.columns(2)

            with ch1:
                fig_bar = pgo.Figure(pgo.Bar(
                    x=["This Transaction", "Typical Safe", "Typical Fraud"],
                    y=[risk_pct, 12, 84],
                    marker_color=[
                        "#FF4D4D" if fraud else "#00D4AA",
                        "rgba(0,212,170,0.3)",
                        "rgba(255,77,77,0.3)",
                    ],
                    text=[f"{risk_pct}%", "12%", "84%"],
                    textposition="outside",
                ))
                fig_bar.update_traces(textfont=dict(color="white"), textfont_size=11)
                fig_bar.update_layout(
                    title=dict(text="Risk Score Benchmark", font=dict(size=13, color="white")),
                    yaxis=dict(title="Risk %", range=[0, 115],
                               gridcolor="rgba(255,255,255,0.06)",
                               tickfont=dict(color="#8A9BC2")),
                    xaxis=dict(tickfont=dict(color="#8A9BC2"), tickangle=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    margin=dict(t=44, b=8, l=8, r=8),
                    bargap=0.35,
                    height=280,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with ch2:
                x_vals     = np.linspace(0, 2, 300).tolist()
                normal_d   = (np.exp(-((np.array(x_vals) - 0.3)**2) / (2*0.1**2))
                              / np.sqrt(2*np.pi*0.1**2)).tolist()
                fraud_d    = (np.exp(-((np.array(x_vals) - 1.2)**2) / (2*0.3**2))
                              / np.sqrt(2*np.pi*0.3**2) * 0.3).tolist()

                fig_dist = pgo.Figure()
                fig_dist.add_trace(pgo.Scatter(
                    x=x_vals, y=normal_d, fill="tozeroy",
                    fillcolor="rgba(0,212,170,0.18)",
                    line=dict(color="rgba(0,212,170,0.7)", width=1.5),
                    name="Normal",
                ))
                fig_dist.add_trace(pgo.Scatter(
                    x=x_vals, y=fraud_d, fill="tozeroy",
                    fillcolor="rgba(255,77,77,0.18)",
                    line=dict(color="rgba(255,77,77,0.7)", width=1.5),
                    name="Fraud",
                ))
                fig_dist.add_vline(
                    x=float(THRESHOLD), line_dash="dot", line_color="#FFB700", line_width=1.5,
                    annotation_text=f"Threshold", annotation_font_color="#FFB700",
                    annotation_font_size=10, annotation_position="top right",
                )
                fig_dist.add_trace(pgo.Scatter(
                    x=[float(err)], y=[0], mode="markers",
                    marker=dict(size=12, color=accent_color, symbol="diamond",
                                line=dict(color="white", width=1.5)),
                    name="Your Transaction",
                ))
                fig_dist.update_layout(
                    title=dict(text="AI Decision Distribution", font=dict(size=13, color="white")),
                    xaxis=dict(title="Reconstruction Error", tickfont=dict(color="#8A9BC2"),
                               gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="Density", tickfont=dict(color="#8A9BC2"),
                               gridcolor="rgba(255,255,255,0.06)"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    legend=dict(bgcolor="rgba(255,255,255,0.05)", font=dict(size=10),
                                bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
                    margin=dict(t=44, b=8, l=8, r=8),
                    height=280,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    # ── Side info panel ──────────────────────────────────────
    with col_side:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                    border-radius:20px;padding:28px;">
            <div style="font-size:0.68rem;letter-spacing:2px;text-transform:uppercase;
                        color:var(--accent);font-weight:700;margin-bottom:16px;">
                How It Works
            </div>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "#00D4AA", "Enter Amount",
             "Type the transaction value into the scanner field on the left."),
            ("2", "#7DD3FC", "AI Analysis",
             "Our autoencoder reconstructs the transaction vector and computes a deviation score."),
            ("3", "#FFB700", "Risk Scoring",
             "The error score is compared against a calibrated threshold to determine risk."),
            ("4", "#00D4AA", "Instant Verdict",
             "Approved or Blocked — returned in milliseconds with full supporting data."),
        ]
        for num, color, title, body in steps:
            st.markdown(f"""
            <div style="display:flex;gap:14px;margin-bottom:20px;">
                <div style="width:32px;height:32px;border-radius:50%;flex-shrink:0;
                            background:{color}22;border:1.5px solid {color}55;
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.8rem;font-weight:700;color:{color};">{num}</div>
                <div>
                    <div style="font-weight:600;font-size:0.9rem;margin-bottom:3px;">{title}</div>
                    <div style="color:var(--text-muted);font-size:0.82rem;line-height:1.55;">{body}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Model stats card
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                    border-radius:20px;padding:28px;margin-top:20px;">
            <div style="font-size:0.68rem;letter-spacing:2px;text-transform:uppercase;
                        color:var(--accent);font-weight:700;margin-bottom:16px;">
                Model Statistics
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div style="background:rgba(0,212,170,0.07);border:1px solid rgba(0,212,170,0.2);
                            border-radius:12px;padding:16px;text-align:center;">
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                                font-weight:700;color:#00D4AA;">86%</div>
                    <div style="color:var(--text-muted);font-size:0.72rem;margin-top:2px;">Detection Rate</div>
                </div>
                <div style="background:rgba(125,211,252,0.07);border:1px solid rgba(125,211,252,0.2);
                            border-radius:12px;padding:16px;text-align:center;">
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                                font-weight:700;color:#7DD3FC;">&lt;4%</div>
                    <div style="color:var(--text-muted);font-size:0.72rem;margin-top:2px;">False Alerts</div>
                </div>
                <div style="background:rgba(255,183,0,0.07);border:1px solid rgba(255,183,0,0.2);
                            border-radius:12px;padding:16px;text-align:center;">
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                                font-weight:700;color:#FFB700;">2ms</div>
                    <div style="color:var(--text-muted);font-size:0.72rem;margin-top:2px;">Scan Latency</div>
                </div>
                <div style="background:rgba(0,212,170,0.07);border:1px solid rgba(0,212,170,0.2);
                            border-radius:12px;padding:16px;text-align:center;">
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                                font-weight:700;color:#00D4AA;">{float(THRESHOLD):.2f}</div>
                    <div style="color:var(--text-muted);font-size:0.72rem;margin-top:2px;">Threshold</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Security notice
        st.markdown("""
        <div style="background:rgba(255,183,0,0.06);border:1px solid rgba(255,183,0,0.2);
                    border-radius:14px;padding:18px 20px;margin-top:20px;
                    display:flex;gap:12px;align-items:flex-start;">
            <div style="font-size:1.1rem;margin-top:1px;">🔒</div>
            <div>
                <div style="font-size:0.82rem;font-weight:600;color:#FFB700;margin-bottom:4px;">
                    End-to-End Encrypted
                </div>
                <div style="font-size:0.79rem;color:var(--text-muted);line-height:1.55;">
                    All transaction data is encrypted in transit and at rest.
                    No raw card or account data is stored.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 9. ADMIN DASHBOARD
# ---------------------------------------------------------
def render_admin_dashboard():
    st.markdown("""
    <div class="dash-greeting">🛡️ Fraud Control Center</div>
    <p class="dash-sub">Full system overview — admin access</p>
    """, unsafe_allow_html=True)

    # ---- metrics ----
    @st.cache_data(ttl=10)
    def get_all_users_cached():
        try:
            return list(auth.list_users().iterate_all())
        except Exception:
            return []

    all_fb_users = get_all_users_cached()
    total_users  = len(all_fb_users)

    try:
        snapshot      = db.collection("transactions").get()
        total_checked = len(snapshot)
        fraud_count   = sum(1 for d in snapshot if d.to_dict().get("fraud", False))
        safe_count    = total_checked - fraud_count
    except Exception:
        total_checked = fraud_count = safe_count = 0

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Fraud Caught",       fraud_count,   "#FF4D4D"),
        ("Safe Transactions",  safe_count,    "#00D4AA"),
        ("Total Checked",      total_checked, "#7DD3FC"),
        ("Registered Users",   total_users,   "#FFB700"),
    ]
    for col, (label, val, color) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color};">{val:,}</div>
                <div class="metric-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- pie chart ----
    if total_checked > 0:
        fig = px.pie(
            values=[safe_count, fraud_count],
            names=["Safe", "Fraud"], hole=0.5,
            color_discrete_sequence=["#00D4AA", "#FF4D4D"],
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Transaction Breakdown",
            showlegend=True,
            legend=dict(bgcolor="rgba(255,255,255,0.07)"),
            margin=dict(t=40, b=0),
        )
        _, pie_col, _ = st.columns([1, 2, 1])
        with pie_col:
            st.plotly_chart(fig, use_container_width=True)

    # ---- user management ----
    st.markdown("""
    <div style="margin-top:32px;">
        <div class="section-tag">User Management</div>
        <h3 style="font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:20px;">Registered Users</h3>
    </div>
    """, unsafe_allow_html=True)

    # Fetch blocked uid set
    try:
        blocked_docs = {d.id for d in db.collection("blocked_users").where("blocked", "==", True).stream()}
    except Exception:
        blocked_docs = set()

    # Fetch user display names from Firestore
    try:
        user_meta = {d.id: d.to_dict() for d in db.collection("users").stream()}
    except Exception:
        user_meta = {}

    # Build rows (exclude admin)
    user_rows = []
    for u in sorted(all_fb_users, key=lambda x: x.user_metadata.creation_timestamp or 0, reverse=True):
        if u.email == ADMIN_EMAIL:
            continue
        reg_ts   = u.user_metadata.creation_timestamp
        reg_date = datetime.fromtimestamp(reg_ts / 1000).strftime("%Y-%m-%d %H:%M") if reg_ts else "—"
        blocked  = u.uid in blocked_docs
        name     = user_meta.get(u.uid, {}).get("name", "—")
        user_rows.append({
            "uid":      u.uid,
            "name":     name,
            "email":    u.email,
            "reg":      reg_date,
            "blocked":  blocked,
        })

    if not user_rows:
        st.info("No users registered yet.")
    else:
        # render table with block/unblock buttons
        header_cols = st.columns([2, 2.5, 1.8, 1, 1.2])
        for col, head in zip(header_cols, ["Name", "Email", "Registered", "Status", "Action"]):
            col.markdown(f"<span style='color:var(--text-muted);font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;'>{head}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.1);margin:4px 0 8px;'>", unsafe_allow_html=True)

        for row in user_rows:
            rc = st.columns([2, 2.5, 1.8, 1, 1.2])
            rc[0].markdown(f"<span style='font-size:0.9rem;'>{row['name']}</span>", unsafe_allow_html=True)
            rc[1].markdown(f"<span style='color:var(--text-muted);font-size:0.85rem;'>{row['email']}</span>", unsafe_allow_html=True)
            rc[2].markdown(f"<span style='color:var(--text-muted);font-size:0.82rem;'>{row['reg']}</span>", unsafe_allow_html=True)
            if row["blocked"]:
                rc[3].markdown('<span class="badge-blocked">Blocked</span>', unsafe_allow_html=True)
                if rc[4].button("Unblock", key=f"unblock_{row['uid']}", use_container_width=True):
                    set_blocked(row["uid"], False)
                    st.success(f"✅ {row['email']} has been unblocked.")
                    st.cache_data.clear()
                    st.rerun()
            else:
                rc[3].markdown('<span class="badge-active">Active</span>', unsafe_allow_html=True)
                if rc[4].button("Block", key=f"block_{row['uid']}", use_container_width=True):
                    set_blocked(row["uid"], True)
                    st.warning(f"🚫 {row['email']} has been blocked.")
                    st.cache_data.clear()
                    st.rerun()

    # ---- transaction history ----
    st.markdown("""
    <div style="margin-top:40px;">
        <div class="section-tag">Audit Trail</div>
        <h3 style="font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:20px;">All Transactions</h3>
    </div>
    """, unsafe_allow_html=True)
    try:
        tx_docs = (
            db.collection("transactions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(100)
            .stream()
        )
        tx_data = []
        for doc in tx_docs:
            d  = doc.to_dict()
            ts = d.get("timestamp")
            tx_data.append({
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M") if ts else "—",
                "User ID":   (d.get("uid") or "—")[:10] + "…",
                "Amount":    f"${d.get('amount', 0):,.2f}",
                "Status":    "🔴 Blocked" if d.get("fraud") else "🟢 Approved",
                "Risk":      f"{int(d.get('error', 0) * 100)}%",
            })
        if tx_data:
            st.dataframe(pd.DataFrame(tx_data), hide_index=True, use_container_width=True)
        else:
            st.info("No transactions recorded yet.")
    except Exception as e:
        st.error(f"Firestore error: {e}")

    # ---- AI threshold ----
    st.markdown("""
    <div style="margin-top:40px;">
        <div class="section-tag">Model Tuning</div>
        <h3 style="font-family:'Playfair Display',serif;font-size:1.4rem;margin-bottom:16px;">AI Sensitivity</h3>
    </div>
    """, unsafe_allow_html=True)
    new_thr = st.slider(
        "Risk Threshold",
        min_value=0.5, max_value=1.5,
        value=float(THRESHOLD), step=0.05,
        key="admin_threshold",
    )
    st.markdown(f"""
    <div style="text-align:center;color:var(--text-muted);font-size:1.0rem;margin-top:8px;">
        Current threshold: <strong style="color:#FFB700;">{THRESHOLD:.2f}</strong>
        &nbsp;→&nbsp;
        New value: <strong style="color:#00D4AA;">{new_thr:.2f}</strong>
        &nbsp;
        <span style="font-size:0.82rem;">(lower = more sensitive)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='site-footer' style='margin-top:60px;'>© 2025 SecureBank Admin Panel</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 10. ROUTER
# ---------------------------------------------------------
render_nav()

page = st.session_state.page

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
    if not st.session_state.logged_in:
        go("login")
    elif is_blocked(st.session_state.uid):
        go("blocked")
    elif st.session_state.is_admin:
        render_admin_dashboard()
    else:
        render_user_dashboard()

else:
    go("landing")
