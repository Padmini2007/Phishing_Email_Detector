"""
app.py
----------------------------------------------------------------------
Phishing Email Detector - Streamlit Web App
----------------------------------------------------------------------
A visual interface for the trained phishing detection model.

Run:
    python -m streamlit run app.py

Requires that you've already run phishing_detector.py once, so that
phishing_model.joblib, tfidf_vectorizer.joblib and feature_scaler.joblib
exist in this folder.

Developed by PADMINI J
----------------------------------------------------------------------
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.sparse import hstack, csr_matrix
from feature_engineering import build_feature_dataframe, extract_features

MODEL_PATH = "phishing_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
SCALER_PATH = "feature_scaler.joblib"

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Global styling — colorful, animated, glassy theme
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 45%, #24243e 100%);
        background-attachment: fixed;
    }

    /* ---- Animated hero header ---- */
    .hero {
        text-align: center;
        padding: 2.2rem 1.5rem 1.6rem 1.5rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        background: linear-gradient(120deg, rgba(0,255,170,0.18), rgba(0,194,255,0.18), rgba(255,75,75,0.15));
        background-size: 300% 300%;
        animation: heroflow 10s ease infinite;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    @keyframes heroflow {
        0%   {background-position: 0% 50%;}
        50%  {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .big-title {
        font-size: 3.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00FFAA, #00C2FF 40%, #FF4FD8 75%, #FFD23F);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        animation: shine 6s linear infinite;
        letter-spacing: 1px;
        text-shadow: 0 0 40px rgba(0,194,255,0.35);
        line-height: 1.15;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #c9cdd6;
        font-size: 1.02rem;
        margin-top: 0;
        font-weight: 400;
    }

    .badge-row { text-align:center; margin-top: 0.9rem; }
    .badge {
        display:inline-block;
        padding: 0.32rem 0.95rem;
        margin: 0 0.3rem;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-purple { background: rgba(124,77,255,0.22); color:#c7b3ff; border:1px solid rgba(124,77,255,0.45);}
    .badge-teal   { background: rgba(0,255,170,0.15); color:#7fffd4; border:1px solid rgba(0,255,170,0.4);}
    .badge-pink   { background: rgba(255,79,216,0.18); color:#ff9fe9; border:1px solid rgba(255,79,216,0.4);}

    /* ---- Section headers ---- */
    .section-label {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f3f8;
        margin: 1.4rem 0 0.6rem 0;
        display:flex; align-items:center; gap:0.5rem;
    }

    /* ---- Top toolbar / Deploy button ---- */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #1a1438, #241b4d) !important;
        border-bottom: 1px solid rgba(0,194,255,0.25);
    }
    div[data-testid="stToolbar"] button,
    div[data-testid="stToolbarActions"] button {
        background: linear-gradient(90deg, #FF4F81, #FF7849) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    header[data-testid="stHeader"] * {
        color: #e6e8ee;
    }

    /* ---- Selectbox (dropdown) ---- */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #1a1438 !important;
        border: 1.5px solid rgba(0,194,255,0.35) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
    }
    div[data-testid="stSelectbox"] svg {
        fill: #00FFAA !important;
    }
    ul[data-baseweb="menu"] {
        background-color: #1a1438 !important;
        border: 1px solid rgba(0,194,255,0.35) !important;
    }
    ul[data-baseweb="menu"] li {
        color: #e6e8ee !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: rgba(0,255,170,0.15) !important;
    }

    /* ---- Input box / cards ---- */
    div[data-testid="stTextArea"] textarea {
        background-color: #1a1438 !important;
        border: 1.5px solid rgba(0,194,255,0.35) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        caret-color: #00FFAA !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #9aa0a6 !important;
        opacity: 1 !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border: 1.5px solid #00FFAA !important;
        box-shadow: 0 0 0 3px rgba(0,255,170,0.15) !important;
    }

    /* ---- Buttons ---- */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #FF4F81, #FF7849);
        border: none;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.65rem 0;
        font-size: 1rem;
        box-shadow: 0 4px 18px rgba(255,79,129,0.4);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 24px rgba(255,79,129,0.55);
    }
    div.stButton > button:not([kind="primary"]) {
        background: rgba(255,255,255,0.06);
        border: 1.5px solid rgba(255,255,255,0.18);
        border-radius: 14px;
        font-weight: 600;
        color: #e6e8ee;
        padding: 0.65rem 0;
    }
    div.stButton > button:not([kind="primary"]):hover {
        border: 1.5px solid #00C2FF;
        color: #00C2FF;
    }

    /* ---- Result cards ---- */
    .result-card {
        border-radius: 20px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 0.6rem;
        position: relative;
        overflow: hidden;
        animation: popin 0.45s ease;
    }
    @keyframes popin {
        0% { opacity:0; transform: scale(0.95) translateY(8px); }
        100% { opacity:1; transform: scale(1) translateY(0); }
    }
    .result-phish {
        background: linear-gradient(135deg, rgba(255,75,75,0.22), rgba(255,79,129,0.12));
        border: 1.5px solid rgba(255,75,75,0.55);
        box-shadow: 0 0 30px rgba(255,75,75,0.25);
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(0,255,170,0.20), rgba(0,194,255,0.10));
        border: 1.5px solid rgba(0,255,170,0.5);
        box-shadow: 0 0 30px rgba(0,255,170,0.22);
    }
    .result-icon { font-size: 2.6rem; margin-bottom: 0.2rem; }
    .result-label {
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0.2rem 0 0.3rem 0;
        letter-spacing: 0.5px;
    }
    .result-phish .result-label { color: #ff8a8a; }
    .result-safe .result-label { color: #7fffd4; }
    .result-conf {
        font-size: 1rem;
        font-weight: 600;
        color: #e6e8ee;
        opacity: 0.9;
    }

    /* ---- Feature pills ---- */
    .feature-pill {
        display: inline-block;
        background: linear-gradient(120deg, rgba(124,77,255,0.18), rgba(0,194,255,0.15));
        border-radius: 24px;
        padding: 0.42rem 1.1rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #e6e8ee;
        border: 1px solid rgba(124,77,255,0.4);
    }
    .feature-pill.warn {
        background: linear-gradient(120deg, rgba(255,75,75,0.25), rgba(255,140,60,0.18));
        border: 1px solid rgba(255,75,75,0.5);
        color: #ffd6d6;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1438, #241b4d);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #e6e8ee; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        background: linear-gradient(90deg, #00FFAA, #00C2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        margin-top: 2.4rem;
        padding: 1.2rem 0 0.6rem 0;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #9aa0a6;
        font-size: 0.88rem;
    }
    .footer-credit {
        font-weight: 700;
        background: linear-gradient(90deg, #00FFAA, #00C2FF, #FF4FD8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <p class="big-title">🛡️ Phishing Email Detector</p>
        <p class="subtitle">AI-powered email security — paste an email and let the model decide</p>
        <div class="badge-row">
            <span class="badge badge-purple">🌲 RandomForest</span>
            <span class="badge badge-teal">📝 TF-IDF</span>
            <span class="badge badge-pink">🔗 URL Analysis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Load model artifacts (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [p for p in (MODEL_PATH, VECTORIZER_PATH, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        return None, None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, vectorizer, scaler


model, vectorizer, scaler = load_artifacts()

if model is None:
    st.error(
        "⚠️ No trained model found. Please run `python phishing_detector.py` "
        "first to train and save the model, then restart this app."
    )
    st.stop()


# ----------------------------------------------------------------------
# Prediction helper
# ----------------------------------------------------------------------
def predict(text: str):
    df = pd.DataFrame({"text": [text]})
    tfidf_matrix = vectorizer.transform(df["text"])
    hand_feats = build_feature_dataframe(df["text"])
    hand_scaled = scaler.transform(hand_feats)
    X = hstack([tfidf_matrix, csr_matrix(hand_scaled)])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = "Phishing" if pred == 1 else "Safe"
    confidence = proba[pred] * 100
    return label, confidence, proba, hand_feats.iloc[0].to_dict()


# ----------------------------------------------------------------------
# Sidebar — sample emails + info
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Try a sample")
    sample_choice = st.selectbox(
        "Load an example email",
        [
            "-- choose --",
            "Phishing example",
            "Safe example",
        ],
    )

    samples = {
        "Phishing example": (
            "Subject: Urgent: Verify Your Account Now\n"
            "Dear customer, we noticed suspicious activity on your account. "
            "Click here http://secure-paypa1-verify.com/login to verify your "
            "identity within 24 hours or your account will be permanently "
            "suspended."
        ),
        "Safe example": (
            "Subject: Meeting Reminder for Tomorrow\n"
            "Hi team, just a quick reminder that our meeting is scheduled for "
            "tomorrow at 10 AM. Please review the attached agenda before "
            "joining."
        ),
    }

    st.divider()
    st.markdown("## ℹ️ About")
    st.write(
        "This app uses a RandomForest model trained on TF-IDF text "
        "features plus hand-crafted signals like URL count, IP-based "
        "links, and suspicious keywords."
    )

    st.divider()
    st.markdown(
        """
        <div style="text-align:center; padding: 0.6rem 0;">
            <div style="font-size:0.78rem; color:#9aa0a6;">Crafted with 💜</div>
            <div style="font-weight:700; font-size:1rem;
                background: linear-gradient(90deg, #00FFAA, #00C2FF, #FF4FD8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PADMINI J
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

default_text = samples.get(sample_choice, "")

# ----------------------------------------------------------------------
# Main input area
# ----------------------------------------------------------------------
st.markdown('<div class="section-label">📧 Email content</div>', unsafe_allow_html=True)

email_text = st.text_area(
    "Email content",
    value=default_text,
    height=220,
    placeholder="Paste the full email text here (subject + body)...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("🔍 Analyze Email", use_container_width=True, type="primary")
with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if analyze_clicked:
    if not email_text.strip():
        st.warning("Please paste some email text first.")
    else:
        label, confidence, proba, feats = predict(email_text)

        if label == "Phishing":
            st.markdown(
                f"""
                <div class="result-card result-phish">
                    <div class="result-icon">🚨</div>
                    <div class="result-label">PHISHING DETECTED</div>
                    <div class="result-conf">Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-safe">
                    <div class="result-icon">✅</div>
                    <div class="result-label">THIS EMAIL LOOKS SAFE</div>
                    <div class="result-conf">Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- Probability bar chart ---
        st.markdown('<div class="section-label">📊 Probability breakdown</div>', unsafe_allow_html=True)
        fig = go.Figure(
            go.Bar(
                x=[proba[0] * 100, proba[1] * 100],
                y=["Safe", "Phishing"],
                orientation="h",
                marker=dict(
                    color=["#00FFAA", "#FF4F81"],
                    line=dict(color="rgba(255,255,255,0.15)", width=1),
                ),
                text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                textposition="auto",
                textfont=dict(size=14, color="#0f0c29", family="Poppins"),
            )
        )
        fig.update_layout(
            xaxis_range=[0, 100],
            height=190,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6e8ee", family="Poppins"),
            xaxis=dict(showgrid=False, color="#9aa0a6"),
            yaxis=dict(color="#e6e8ee"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Extracted feature pills ---
        st.markdown('<div class="section-label">🔬 Extracted signals</div>', unsafe_allow_html=True)
        pill_defs = [
            (f"🔗 URLs found: {feats['num_urls']}", feats['num_urls'] > 0),
            (f"⚠️ Suspicious keywords: {feats['num_suspicious_words']}", feats['num_suspicious_words'] > 2),
            (f"🌐 IP-based URL: {'Yes' if feats['has_ip_url'] else 'No'}", bool(feats['has_ip_url'])),
            (f"🪤 '@' in URL: {'Yes' if feats['url_has_at_symbol'] else 'No'}", bool(feats['url_has_at_symbol'])),
            (f"❗ Exclamation marks: {feats['num_exclamations']}", feats['num_exclamations'] > 2),
            (f"🔠 ALL-CAPS words: {feats['num_uppercase_words']}", feats['num_uppercase_words'] > 2),
            (f"💰 Money/refund terms: {'Yes' if feats['has_money_symbol'] else 'No'}", bool(feats['has_money_symbol'])),
            (f"📏 Text length: {feats['text_length']} chars", False),
        ]
        pills_html = "".join(
            f'<span class="feature-pill{" warn" if is_warn else ""}">{p}</span>'
            for p, is_warn in pill_defs
        )
        st.markdown(pills_html, unsafe_allow_html=True)

        with st.expander("📜 View raw email text analyzed"):
            st.text(email_text)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Built with Scikit-learn + Streamlit | Phishing Email Detection Model<br>
        <span class="footer-credit">Developed by PADMINI J</span>
    </div>
    """,
    unsafe_allow_html=True,
)
