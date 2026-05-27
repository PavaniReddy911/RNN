import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import joblib

from keras.preprocessing.sequence import pad_sequences

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

# -----------------------------------------
# LOAD MODEL & FILES
# -----------------------------------------

model = joblib.load("mental_health_rnn_model.pkl")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# -----------------------------------------
# PARAMETERS
# -----------------------------------------

max_length = 50

# -----------------------------------------
# TEXT PREPROCESSING
# -----------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    return text

# -----------------------------------------
# EMOTIONAL GUIDANCE
# -----------------------------------------

guidance = {

    "sadness":
    "Take a short break and talk with someone you trust.",

    "anger":
    "Try deep breathing exercises and relax for a while.",

    "fear":
    "Stay calm and focus on positive thoughts.",

    "joy":
    "Keep enjoying activities that make you happy.",

    "love":
    "Spend time with supportive people around you.",

    "surprise":
    "Take things slowly and process your emotions calmly."
}

# -----------------------------------------
# HEADER SECTION
# -----------------------------------------

st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

# -----------------------------------------
# ABOUT PROJECT
# -----------------------------------------

st.markdown("## 📌 About the Project")

st.write("""
This application uses Artificial Intelligence and Natural Language Processing (NLP)
to analyze emotional sentiment from user text messages.

The system uses a Simple Recurrent Neural Network (RNN)
to identify emotional patterns in text sequences.
""")

# -----------------------------------------
# USER INPUT AREA
# -----------------------------------------

st.markdown("## ✍ User Text Input")

st.write("### Sample Sentences")
st.write("- I feel lonely and stressed")
st.write("- I am excited about my future")
st.write("- Nobody understands me anymore")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=150
)

# -----------------------------------------
# PREDICTION BUTTON
# -----------------------------------------

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Clean Text
        cleaned_text = clean_text(user_input)

        # Convert to Sequence
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

        # -----------------------------------------
        # OUTPUT SECTION
        # -----------------------------------------

        st.markdown("## 📊 Prediction Output")

        st.success(
            f"Emotion Detected: {predicted_emotion}"
        )

        st.info(
            f"Confidence Score: {confidence*100:.2f}%"
        )

        # Emotional Status
        if confidence > 0.80:
            st.write("Emotional Status: Strong Emotional Pattern Detected")
        else:
            st.write("Emotional Status: Moderate Emotional Pattern Detected")

        # -----------------------------------------
        # VISUALIZATION
        # -----------------------------------------

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

        # -----------------------------------------
        # GUIDANCE SECTION
        # -----------------------------------------

        st.markdown("## 💙 Emotional Wellness Tips")

        if predicted_emotion in guidance:
            st.write(guidance[predicted_emotion])

        else:
            st.write(
                "Maintain a healthy routine and take care of yourself."
            )
