# app.py - CUSTOMER SAVES → ADMIN SEES INSTANTLY
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, auth, firestore
from model_utils import load_model_and_assets, predict_transaction, INPUT_DIM
from datetime import datetime

# ---------------------------------------------------------
# 0. PAGE CONFIG + THEME
# ---------------------------------------------------------
st.set_page_config(page_title="Secure Bank", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #1E3A8A, #1E40AF, #1D4ED8, #2563EB);
        background-size: 300% 300%;
        animation: gradient 12s ease infinite;
        color: white;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 24px;
        margin: 16px 0;
        border: 1.5px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }

    .stButton > button {
        background: linear-gradient(45deg, #10B981, #14B8A6, #0EA5E9);
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.7);
    }

    .traffic-light {
        width: 100px; height: 100px;
        border-radius: 50%;
        margin: 0 auto 16px;
        box-shadow: 0 0 30px currentColor, 0 0 60px rgba(0,0,0,0.4);
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 30px currentColor, 0 0 60px rgba(0,0,0,0.4); }
        to { box-shadow: 0 0 50px currentColor, 0 0 80px rgba(0,0,0,0.5); }
    }
    .green-light { background: #10B981; }
    .red-light { background: #EF4444; }

    .big-number {
        font-size: 52px; font-weight: 700; margin: 0; color: white;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .label { font-size: 17px; color: #E0E7FF; margin-top: 6px; font-weight: 500; }

    .transaction-table table {
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 14px;
    }
    .transaction-table th {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        text-align: center !important;
    }
    .transaction-table td {
        padding: 10px !important;
        text-align: center !important;
        color: black !important;
        font-weight: bold !important;
    }
    .blocked-row { background-color: #FCA5A5 !important; }
    .approved-row { background-color: #86EFAC !important; }
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

# ---------------------------------------------------------
# 2. AUTH
# ---------------------------------------------------------
def login(email, pwd):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.update(uid=user.uid, logged_in=True,
                                is_admin=(email == "admin@securebank.com"))
        st.success("Logged in!")
    except:
        st.error("Login failed.")

def register(email, pwd):
    try:
        user = auth.create_user(email=email, password=pwd)
        st.session_state.update(uid=user.uid, logged_in=True, is_admin=False)
        st.success("Welcome!")
    except:
        st.error("Register failed.")

def logout():
    for k in ["uid", "logged_in", "is_admin"]:
        st.session_state.pop(k, None)
    st.success("Logged out.")

# ---------------------------------------------------------
# 3. LOGIN PAGE
# ---------------------------------------------------------
if not st.session_state.get("logged_in", False):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/fluency/256/bank-building.png", width=220)
    with col2:
        st.markdown("""
        <h1 style='color: white; font-weight: 700;'>Secure Bank</h1>
        <p style='color: #C7D2FE; font-size: 1.3rem;'><b>AI-Powered Fraud Protection</b></p>
        <p style='color: #A5B4FC;'>86% fraud caught • Only 4% false alerts</p>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px; background: rgba(255,255,255,0.15); border-radius: 16px; margin-bottom: 20px;">
            <img src="https://img.icons8.com/fluency/96/bank-building.png" width="70">
            <h3 style="color: white; margin: 12px 0 0; font-weight: 700;">Secure Bank Portal</h3>
        </div>
        """, unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="you@securebank.com")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"): login(email, pwd); st.rerun()
        if st.button("Register"): register(email, pwd); st.rerun()
    st.stop()

# ---------------------------------------------------------
# 4. LOGGED IN
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding:20px; background: rgba(255,255,255,0.15); border-radius: 16px; margin-bottom: 20px;">
    <img src="https://img.icons8.com/fluency/96/bank-building.png" width="70">
    <h3 style="color: white; margin: 12px 0 0; font-weight: 700;">Secure Bank Portal</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background: rgba(255,255,255,0.2); padding: 16px; border-radius: 14px; text-align: center; margin-bottom: 20px;">
    <p style="color: white; margin: 0; font-weight: 600;">User: {st.session_state.uid[:8]}...</p>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Logout"): logout(); st.rerun()

pages = ["Check Transaction"]
if st.session_state.get("is_admin"):
    pages.insert(0, "Admin Dashboard")
page = st.sidebar.radio("Menu", pages)

# ---------------------------------------------------------
# 5. CUSTOMER: SAVE EVERY TRANSACTION
# ---------------------------------------------------------
if page == "Check Transaction":
    st.markdown("<h1 style='color: white; text-align: center; font-weight: 700;'>Check Your Transaction</h1>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 3])
    with colA:
        amt = st.text_input("Amount (USD)", "100.00", help="Enter any amount")
        try: amount = float(amt)
        except: amount = 0.0; st.warning("Invalid")

        if st.button("Verify Now", use_container_width=True):
            vec = np.zeros(INPUT_DIM); vec[29] = amount
            with st.spinner("AI is scanning..."):
                err, fraud = predict_transaction(MODEL, SCALER, THRESHOLD, vec)

            # === SAVE FOR ALL USERS ===
            try:
                db.collection("transactions").add({
                    "uid": st.session_state.uid,
                    "amount": amount,
                    "error": float(err),
                    "fraud": fraud,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
                st.success("Transaction saved to history!")
            except Exception as e:
                st.error(f"Save failed: {e}")

            st.session_state.last_err = err
            st.session_state.last_fraud = fraud

    with colB:
        if 'last_err' in st.session_state:
            fraud = st.session_state.last_fraud
            status = "BLOCKED" if fraud else "SAFE"
            light_class = "red-light" if fraud else "green-light"

            st.markdown(f"""
            <div class="glass-card">
                <div class="traffic-light {light_class}"></div>
                <p class="big-number" style="color: {'#EF4444' if fraud else '#10B981'};">{status}</p>
                <p class="label">Transaction is <b>{status}</b></p>
            </div>
            """, unsafe_allow_html=True)

            your_risk = min(100, int(err * 100))
            avg_risk = 15
            fig = go.Figure(go.Bar(
                x=['Your Risk', 'Average'],
                y=[your_risk, avg_risk],
                marker_color=['#EF4444' if fraud else '#10B981', '#64748B'],
                text=[f"{your_risk}%", f"{avg_risk}%"],
                textposition='outside'
            ))
            fig.update_layout(title="Risk Level", yaxis_title="Risk %", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            # Threshold Graph
            st.markdown("### How the AI Decides")
            x = np.linspace(0, 2, 200)
            normal_errors = np.exp(-((x - 0.3)**2) / (2 * 0.1**2)) / np.sqrt(2 * np.pi * 0.1**2)
            fraud_errors = np.exp(-((x - 1.2)**2) / (2 * 0.3**2)) / np.sqrt(2 * np.pi * 0.3**2) * 0.3

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=normal_errors, fill='tozeroy', fillcolor='rgba(16,185,129,0.3)', line_color='rgba(0,0,0,0)', name='Normal'))
            fig2.add_trace(go.Scatter(x=x, y=fraud_errors, fill='tozeroy', fillcolor='rgba(239,68,68,0.3)', line_color='rgba(0,0,0,0)', name='Fraud'))
            fig2.add_vline(x=THRESHOLD, line_dash="dash", line_color="#F59E0B", annotation_text=f"Threshold: {THRESHOLD:.2f}", annotation_position="top")
            fig2.add_scatter(x=[err], y=[0], mode='markers', marker=dict(size=16, color='#EF4444' if fraud else '#10B981', symbol='star'), name="Your Transaction")
            fig2.update_layout(title="AI Decision Engine", xaxis_title="Reconstruction Error", yaxis_title="Density", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white", legend=dict(y=0.99, x=0.01, bgcolor='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# 6. ADMIN DASHBOARD: REAL-TIME STATS + HISTORY
# ---------------------------------------------------------
elif page == "Admin Dashboard":
    st.markdown("<h1 style='color: white; text-align: center; font-weight: 700;'>Fraud Control Center</h1>", unsafe_allow_html=True)

    # === USER COUNT ===
    try:
        total_users = len(list(auth.list_users().iterate_all()))
    except:
        total_users = 0

    # === TRANSACTION STATS ===
    try:
        snapshot = db.collection("transactions").get()
        total_checked = len(snapshot)
        fraud_count = sum(1 for doc in snapshot if doc.to_dict().get("fraud", False))
        safe_count = total_checked - fraud_count
    except:
        total_checked = fraud_count = safe_count = 0

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Fraud Caught", fraud_count, "#EF4444"),
        ("False Alerts", 0, "#F59E0B"),
        ("Total Checked", total_checked, "#0EA5E9"),
        ("Registered Users", total_users, "#8B5CF6")
    ]
    for col, (label, value, color) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <p class="big-number" style="color: {color};">{value:,}</p>
                <p class="label">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    # === PIE CHART ===
    st.markdown("### Transaction Breakdown")
    if total_checked > 0:
        fig = px.pie(values=[safe_count, fraud_count], names=['Safe', 'Fraud'], hole=0.5,
                     color_discrete_sequence=['#10B981', '#EF4444'])
        fig.update_traces(textinfo='percent+label', textfont_size=15)
        fig.update_layout(font_color="white", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions yet. Customers will save them here.")

    # === HISTORY TABLE ===
    st.markdown("### Transaction History")
    try:
        docs = db.collection("transactions")\
                 .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                 .limit(50).stream()
        data = []
        for doc in docs:
            d = doc.to_dict()
            ts = d.get("timestamp")
            timestamp_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "—"
            data.append({
                "Timestamp": timestamp_str,
                "User ID": d.get("uid", "—")[:8] + "...",
                "Amount": f"${d.get('amount', 0):,.2f}",
                "Status": "Blocked" if d.get("fraud") else "Approved",
                "Risk": f"{int(d.get('error', 0) * 100)}%"
            })
        if data:
            df = pd.DataFrame(data)

            def make_row(row):
                cls = "blocked-row" if row["Status"] == "Blocked" else "approved-row"
                return f"""
                <tr class="{cls}">
                    <td>{row["Timestamp"]}</td>
                    <td>{row["User ID"]}</td>
                    <td>{row["Amount"]}</td>
                    <td>{row["Status"]}</td>
                    <td>{row["Risk"]}</td>
                </tr>
                """
            rows = "".join(df.apply(make_row, axis=1))
            table = f"""
            <div class="transaction-table">
            <table>
                <thead><tr>
                    <th>Timestamp</th><th>User ID</th><th>Amount</th><th>Status</th><th>Risk</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            </div>
            """
            st.markdown(table, unsafe_allow_html=True)
        else:
            st.info("No transaction history yet.")
    except Exception as e:
        st.error(f"Firestore error: {e}")

    # === AI SENSITIVITY ===
    st.markdown("### AI Sensitivity")
    new_thr = st.slider("Risk Threshold", 0.5, 1.5, THRESHOLD, 0.05)
    st.markdown(f"""
    <div style="text-align: center; color: #E0E7FF; font-size: 1.1rem;">
        Current: <b style="color: #F59E0B;">{THRESHOLD:.2f}</b> → 
        New: <b style="color: #0EA5E9;">{new_thr:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
