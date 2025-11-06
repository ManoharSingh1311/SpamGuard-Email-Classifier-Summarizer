import streamlit as st
import requests
import time
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Load Environment Variables ---
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    st.error("⚠️ Hugging Face API key not found! Please check your .env file.")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="SpamGuard: Smart Email Classifier & Summarizer",
    layout="centered",
    page_icon="🛡️"
)

st.title("**SpamGuard: Smart Email Classifier & Summarizer**")
st.markdown("#### Enter the body of the email below and we will check if it's spam or valid.")

st.sidebar.header("**How it works**")
st.sidebar.markdown(
    "1. **Spam Detection**: Our model checks whether the email is spam or not.\n"
    "2. **Summary Generation**: If the email is valid, we generate a summary using an advanced NLP model."
)

# --- Input Text Area ---
question = st.text_area(
    "Email Body",
    placeholder="Type or paste the body of your email here...",
    height=300
)

# --- Load and Prepare Dataset ---
raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# --- Feature Extraction and Model Training ---
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)

# --- Spam Checker Function ---
def checkSpam(text):
    input_mail = [text]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)
    return 1 if prediction[0] == 0 else 0  # 1 for spam, 0 for non-spam

# --- Hugging Face Summarizer ---
def summarize_text(text):
    API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return {"error": f"Request failed ({response.status_code}): {response.text}"}
        try:
            return response.json()
        except Exception as e:
            return {"error": f"JSON decode failed: {str(e)}"}

    output = query({
        "inputs": text,
        "parameters": {"min_length": 30, "max_length": 100}
    })

    if "error" in output:
        return f"[Error from API]: {output['error']}"
    elif isinstance(output, list) and len(output) > 0 and "summary_text" in output[0]:
        return output[0]["summary_text"]
    else:
        return "[Error]: Unexpected response format from summarization API."

# --- Button Click Handler ---
if st.button("Check Spam"):
    if question.strip():
        with st.spinner("Checking the email..."):
            is_spam = checkSpam(question)

            st.header("Result")
            if is_spam == 1:
                st.subheader("This is a **Spam Email** 🛑")
                st.markdown("No summary will be generated for spam emails.")
            else:
                st.subheader("This is a **Valid Email** ✅")
                st.markdown("Generating a summary...")

                summary = summarize_text(question)

                st.markdown("### **Summary of the Email**")
                st.write(summary)
    else:
        st.warning("Please enter the body of an email to check.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "Created by [Manohar Singh](https://github.com/ManoharSingh1311) | "
    "[Mail](mailto:burathimannu@gmail.com) | [Contact](tel:+916399121342)"
)
