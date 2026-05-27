import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import string

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL & FILES
# -----------------------------------

model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# -----------------------------------
# PARAMETERS
# -----------------------------------

max_length = 50

# -----------------------------------
# TEXT PREPROCESSING
# -----------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    return text

# -----------------------------------
# WELLNESS TIPS
# -----------------------------------

guidance = {

    "sadness":
    "Take a short break and talk with someone you trust.",

    "anger":
    "Try deep breathing exercises and relax for a while.",

    "fear":
    "Stay calm and focus on positive thoughts.",

    "joy":
    "Keep doing activities that make you feel happy.",

    "love":
    "Spend time with supportive people around you.",

    "surprise":
    "Take things slowly and process your emotions calmly."
}

# -----------------------------------
# HEADER
# -----------------------------------

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# -----------------------------------
# ABOUT SECTION
# -----------------------------------

st.markdown("## 📌 About the Project")

st.write("""
This project uses Artificial Intelligence and Natural Language Processing (NLP)
to analyze emotional sentiment from text messages.

The system uses a Simple Recurrent Neural Network (RNN)
to understand sequence patterns and detect emotional states.

Applications include:
- emotional wellness monitoring
- NLP-based sentiment analysis
- counselor assistance systems
- early emotional trend detection
""")

# -----------------------------------
# USER INPUT
# -----------------------------------

st.markdown("## ✍ User Text Input")

st.write("### Sample Sentences")

st.write("- I feel lonely and exhausted")
st.write("- I am excited about my future")
st.write("- Nobody understands me anymore")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=150
)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Clean text
        cleaned_text = clean_text(user_input)

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        # Padding
        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_length,
            padding='post'
        )

        # Prediction
        prediction = model.predict(padded_sequence)

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction)

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # -----------------------------------
        # OUTPUT SECTION
        # -----------------------------------

        st.markdown("## 📊 Prediction Output")

        st.success(
            f"Emotion Detected: {predicted_emotion}"
        )

        st.info(
            f"Confidence Score: {confidence*100:.2f}%"
        )

        # Emotional status
        if confidence > 0.80:
            st.write(
                "Emotional Status: Strong Emotional Pattern Detected"
            )
        else:
            st.write(
                "Emotional Status: Moderate Emotional Pattern Detected"
            )

        # -----------------------------------
        # VISUALIZATION
        # -----------------------------------

        st.markdown("## 📈 Sentiment Confidence Graph")

        emotions = label_encoder.classes_

        probs = prediction[0]

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(emotions, probs)

        ax.set_xlabel("Emotion")

        ax.set_ylabel("Confidence")

        ax.set_title("Emotion Probability Distribution")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        # -----------------------------------
        # WELLNESS TIPS
        # -----------------------------------

        st.markdown("## 💙 Emotional Wellness Tips")

        if predicted_emotion in guidance:

            st.write(guidance[predicted_emotion])

        else:

            st.write(
                "Take care of yourself and maintain a healthy routine."
            )
