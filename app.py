import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 Spam Detection App")
st.write("Enter a message to check if it's Spam or Not Spam")

# User input
message = st.text_area("Enter Message")

# Prediction button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        # Transform text using vectorizer
        message_vectorized = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vectorized)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
