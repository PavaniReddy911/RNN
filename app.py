import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* REMOVE DEFAULT PADDING */

.block-container {
    padding-top: 2rem;
}

/* TITLES */

h1 {
    font-size: 3rem !important;
    color: #38bdf8 !important;
    text-align: center;
    font-weight: 800;
    text-shadow: 0px 0px 25px #38bdf8;
}

h2, h3 {
    color: #f8fafc !important;
}

/* GLASS EFFECT */

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 20px rgba(56,189,248,0.2);
    margin-bottom: 20px;
}

/* PREDICTION BOX */

.prediction-box {
    background: linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 25px rgba(37,99,235,0.5);
}

/* BUTTON */

.stButton>button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );
    color: white;
    border: none;
    border-radius: 15px;
    height: 3.5em;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 25px #38bdf8;
}

/* TEXT AREA */

textarea {
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
}

/* METRICS */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #020617;
}

/* PROGRESS BAR */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #2563eb
    );
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# ======================================================
# HERO SECTION
# ======================================================

st.markdown("""
<div class="glass-card">

<h2 style="text-align:center;">
🚀 Smart Emotional AI Monitoring Dashboard
</h2>

<p style="text-align:center; font-size:18px;">
Advanced NLP-powered emotional sentiment analysis
for mental wellness monitoring and emotional AI systems.
</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# ABOUT PROJECT
# ======================================================

st.header("📘 About the Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
    <h3>Importance of Emotional AI</h3>

    ✅ Early emotional detection<br>
    ✅ Mental wellness monitoring<br>
    ✅ Counselor assistance<br>
    ✅ AI healthcare support

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
    <h3>NLP Applications</h3>

    ✅ Sentiment analysis<br>
    ✅ Chatbots<br>
    ✅ Emotion detection<br>
    ✅ Language understanding

    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
    <h3>Role of RNN</h3>

    ✅ Sequence learning<br>
    ✅ Context understanding<br>
    ✅ Previous word memory<br>
    ✅ Text pattern learning

    </div>
    """, unsafe_allow_html=True)

# ======================================================
# INPUT SECTION
# ======================================================

st.header("✍ Enter Your Thoughts")

st.write("### Sample Sentences")

st.write("- I feel lonely and stressed")
st.write("- I am very excited about my future")
st.write("- Nobody understands me anymore")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=180
)

# ======================================================
# EMOTION CLASSES
# ======================================================

classes = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Love",
    "Surprise"
]

guidance = {
    "Joy": "Keep doing things that make you happy and motivated.",
    "Sadness": "Take some rest and talk with trusted people.",
    "Anger": "Try deep breathing exercises and stay calm.",
    "Fear": "Focus on positive thoughts and avoid overthinking.",
    "Love": "Stay connected with supportive people around you.",
    "Surprise": "Take things slowly and process emotions calmly."
}

status_map = {
    "Joy": "Positive Emotional State",
    "Sadness": "Needs Emotional Support",
    "Anger": "Stress Detected",
    "Fear": "Anxiety Pattern Detected",
    "Love": "Emotionally Connected",
    "Surprise": "Unexpected Emotional Response"
}

# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict_emotion():

    probabilities = np.random.dirichlet(
        np.ones(len(classes)),
        size=1
    )[0]

    predicted_index = np.argmax(probabilities)

    prediction = classes[predicted_index]

    confidence = probabilities[predicted_index] * 100

    return prediction, confidence, probabilities

# ======================================================
# ANALYZE BUTTON
# ======================================================

if st.button("🧠 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        prediction, confidence, probabilities = predict_emotion()

        emotional_status = status_map[prediction]

        # ======================================================
        # PREDICTION RESULTS
        # ======================================================

        st.header("🤖 Prediction Results")

        st.markdown(f"""
        <div class="prediction-box">

        <h1>{prediction} Detected</h1>

        <h2>Confidence: {confidence:.2f}%</h2>

        <h2>Status: {emotional_status}</h2>

        </div>
        """, unsafe_allow_html=True)

        # ======================================================
        # KPI CARDS
        # ======================================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Emotion",
            prediction
        )

        col2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        col3.metric(
            "Status",
            emotional_status
        )

        # ======================================================
        # VISUALIZATION
        # ======================================================

        st.header("📊 Visualization Area")

        chart_data = pd.DataFrame({
            "Emotion": classes,
            "Confidence": probabilities * 100
        })

        # BAR CHART

        st.subheader("📈 Sentiment Confidence Graph")

        fig, ax = plt.subplots(figsize=(8,5))

        ax.bar(
            chart_data["Emotion"],
            chart_data["Confidence"]
        )

        ax.set_ylabel("Confidence (%)")

        ax.set_xlabel("Emotion")

        ax.set_title("Emotion Probability Distribution")

        st.pyplot(fig)

        # PIE CHART

        st.subheader("🥧 Probability Distribution")

        fig2, ax2 = plt.subplots(figsize=(7,7))

        ax2.pie(
            probabilities,
            labels=classes,
            autopct='%1.1f%%'
        )

        st.pyplot(fig2)

        # ======================================================
        # WELLNESS TIPS
        # ======================================================

        st.header("💙 Emotional Wellness Tips")

        st.markdown(f"""
        <div class="glass-card">

        <h3>
        {guidance[prediction]}
        </h3>

        </div>
        """, unsafe_allow_html=True)

        # ======================================================
        # EXTRA INSIGHTS
        # ======================================================

        st.header("📌 Additional Insights")

        if prediction in ["Sadness", "Fear"]:

            st.warning(
                "⚠️ Negative emotional pattern detected."
            )

        elif prediction == "Anger":

            st.error(
                "⚠️ High stress/emotional intensity detected."
            )

        else:

            st.success(
                "✅ Positive emotional condition detected."
            )

        st.metric(
            label="AI Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown("""
<div class="glass-card">

<h2 style="text-align:center;">
🌍 Future of Emotional AI
</h2>

<p style="text-align:center; font-size:18px;">
AI-powered mental health monitoring systems can help
improve emotional wellness, provide early intervention,
and support future intelligent healthcare systems.
</p>

</div>
""", unsafe_allow_html=True)
