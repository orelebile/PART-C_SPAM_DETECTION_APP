import streamlit as st
import pickle
import re
import string

# Define the text cleaning function (copied from previous cells for self-containment)
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove extra whitespace
    return re.sub(r"\s+", " ", text).strip()

# Load trained model and vectorizer
try:
    model = pickle.load(open('Spam Prediction Model.pkl', 'rb'))
    vectorizer = pickle.load(open('Tfidf Vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or vectorizer files not found. Please ensure 'Spam Prediction Model.pkl' and 'Tfidf Vectorizer.pkl' are in the correct directory.")
    st.stop() # Stop the app if files are missing

# Define the prediction function 
def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "NotSpam"

st.title("Spam Email Classification App")

st.write("Enter an email message to classify it as Spam or Not Spam.")

# Text input for the email message
email_message = st.text_area("Email Message", height=150)

# Prediction button
if st.button("Classify Email"):
    if email_message:
        result = predict_sentiment(email_message)
        if result == "Spam":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter an email message to classify.")
