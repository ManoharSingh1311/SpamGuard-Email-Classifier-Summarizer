
import streamlit as st
import requests
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up the title, logo, and some initial styling
st.set_page_config(page_title="SpamGuard: Smart Email Classifier & Summarizer", layout="centered", page_icon="🛡️")  # Adding a page icon (can also add a logo here)

st.title("**SpamGuard: Smart Email Classifier & Summarizer**")
st.markdown("#### Enter the body of the email below and we will check if it's spam or valid.")

st.sidebar.header("**How it works**")
st.sidebar.markdown(
    "1. **Spam Detection**: Our model checks whether the email is spam or not."
    "\n2. **Summary Generation**: If the email is valid, we generate a summary using an advanced NLP model."
)

# Input text area for user to enter email body (increased size for better visibility)
question = st.text_area("Email Body", placeholder="Type or paste the body of your email here...", height=300)  # Adjust the height for larger input area

# Spam detection model setup
raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction and model setup
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
model = LogisticRegression()
model.fit(X_train_features, Y_train)

def checkSpam():
    input_mail = [question]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)
    return 1 if prediction[0] == 0 else 0  # Return 1 for spam, 0 for non-spam

# Button for checking spam status
if st.button("Check Spam"):
    if question:
        with st.spinner("Checking the email..."):
            ans = checkSpam()
            st.header("Answer")
            if ans == 1:
                st.subheader("This is a **Spam Email** 🛑")
                st.markdown("No summary will be generated for spam emails.")
            else:
                st.subheader("This is a **Valid Email** ✅")
                st.markdown("Generating a summary...")

                # API call to Hugging Face for summarization
                API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
                headers = {"Authorization": "....................."}

                data = question
                def query(payload):
                    response = requests.post(API_URL, headers=headers, json=payload)
                    return response.json()

                output = query({
                    "inputs": data,
                    "parameters": {"min_length": 30, "max_length": 100}
                })[0]

                # Display summary
                st.markdown("### **Summary of the Email**")
                st.write(output["summary_text"])

    else:
        st.warning("Please enter the body of an email to check.")

# Optional: Add footer with additional information or links
st.markdown("---")
st.markdown("Created by [Manohar Singh](https://github.com/ManoharSingh1311) | [Mail](burathimannu@gmail.com) | [Contact](+91 6399121342)")

