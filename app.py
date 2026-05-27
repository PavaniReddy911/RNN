import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

.block-container {
    padding-top: 2rem;
}

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

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 20px rgba(56,189,248,0.2);
    margin-bottom: 20px;
}

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

textarea {
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

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
for emotional wellness and intelligent healthcare systems.
</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# ABOUT SECTION
# ======================================================

st.header("📘 About the Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
    <h3>Importance of Emotional AI</h3>

    ✅ Mental wellness monitoring<br>
    ✅ Early emotional detection<br>
    ✅ AI healthcare systems<br>
    ✅ Emotional support analysis

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
    <h3>NLP Applications</h3>

    ✅ Sentiment analysis<br>
    ✅ Chatbots<br>
    ✅ Language understanding<br>
    ✅ Emotional intelligence

    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
    <h3>Role of RNN</h3>

    ✅ Sequence learning<br>
    ✅ Word memory<br>
    ✅ Context understanding<br>
    ✅ Pattern recognition

    </div>
    """, unsafe_allow_html=True)

# ======================================================
# USER INPUT
# ======================================================

st.header("✍ User Text Input")

st.write("### Sample Sentences")

st.write("- I feel lonely and stressed")
st.write("- I am very happy today")
st.write("- Nobody understands me anymore")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=180
)

# ======================================================
# CLASSES
# ======================================================

classes = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Love",
    "Surprise"
]

# ======================================================
# GUIDANCE
# ======================================================

guidance = {

    "Joy":
    "Keep doing activities that make you feel positive and motivated.",

    "Sadness":
    "Take some rest and talk with trusted people around you.",

    "Anger":
    "Try deep breathing and relaxation techniques.",

    "Fear":
    "Focus on positive thoughts and avoid overthinking.",

    "Love":
    "Stay connected with supportive and caring people.",

    "Surprise":
    "Take things calmly and process your emotions slowly."
}

# ======================================================
# STATUS MAP
# ======================================================

status_map = {

    "Joy":
    "Positive Emotional State",

    "Sadness":
    "Needs Emotional Support",

    "Anger":
    "Stress Detected",

    "Fear":
    "Anxiety Pattern Detected",

    "Love":
    "Emotionally Connected",

    "Surprise":
    "Unexpected Emotional Response"
}

# ======================================================
# KEYWORD-BASED PREDICTION
# ======================================================

def predict_emotion(text):

    text = text.lower()

    emotion_keywords = {

        "Joy": [
            "happy", "excited", "great",
            "awesome", "good", "amazing",
            "wonderful", "fantastic"
        ],

        "Sadness": [
            "sad", "depressed", "cry",
            "hopeless", "lonely", "tired"
        ],

        "Anger": [
            "angry", "hate", "mad",
            "annoyed", "frustrated"
        ],

        "Fear": [
            "fear", "afraid", "anxious",
            "worried", "stress", "scared"
        ],

        "Love": [
            "love", "care", "affection",
            "relationship"
        ],

        "Surprise": [
            "surprised", "shocked",
            "unexpected"
        ]
    }

    scores = []

    for emotion in classes:

        score = 0

        for word in emotion_keywords[emotion]:

            if word in text:

                score += 1

        scores.append(score)

    if max(scores) == 0:

        probabilities = np.ones(len(classes)) / len(classes)

        prediction = "Joy"

        confidence = 50

    else:

        predicted_index = np.argmax(scores)

        prediction = classes[predicted_index]

        confidence = (
            scores[predicted_index]
            / sum(scores)
        ) * 100

        probabilities = np.array(scores) / sum(scores)

    return prediction, confidence, probabilities

# ======================================================
# ANALYZE BUTTON
# ======================================================

if st.button("🧠 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        prediction, confidence, probabilities = predict_emotion(user_input)

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

        # ======================================================
        # PIE CHART
        # ======================================================

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
                "⚠️ High emotional stress detected."
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
AI-powered emotional monitoring systems can support
mental wellness, provide early intervention,
and improve intelligent healthcare technologies.
</p>

</div>
""", unsafe_allow_html=True)
