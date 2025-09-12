
import streamlit as st
import requests
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="SpamGuard: Smart Email Classifier & Summarizer",
    layout="centered",
    page_icon="🛡️"
)

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🛡️ SpamGuard</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center;'>Smart Email Classifier & Summarizer</p>", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/906/906334.png", width=100)
st.sidebar.header("✨ How it works")
st.sidebar.markdown(
    """
    1. **Spam Detection** 🕵️: Our ML model detects spam emails.  
    2. **Summary Generation** 📝: For valid emails, we generate a concise summary.  
    """
)

# ----------------- INPUT AREA -----------------
question = st.text_area(
    "📧 Paste Email Body",
    placeholder="Type or paste the body of your email here...",
    height=250
)

# ----------------- LOAD & TRAIN MODEL -----------------
raw_mail_data = pd.read_csv("mail_data.csv")
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), "")

mail_data.loc[mail_data["Category"] == "spam", "Category"] = 0
mail_data.loc[mail_data["Category"] == "ham", "Category"] = 1

X = mail_data["Message"]
Y = mail_data["Category"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

model = LogisticRegression()
model.fit(X_train_features, Y_train)

# ----------------- SPAM CHECK FUNCTION -----------------
def checkSpam(text):
    input_data_features = feature_extraction.transform([text])
    prediction = model.predict_proba(input_data_features)[0]
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

# ----------------- HUGGING FACE SUMMARIZER -----------------
def summarize_text(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    hf_token = st.secrets.get("HF_TOKEN", None)  # safer than hardcoding
    if not hf_token:
        return "[Error]: Hugging Face Token not found. Add it in st.secrets."

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text, "parameters": {"min_length": 30, "max_length": 100}}

    for _ in range(5):  # retry loop if model is still loading
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        if "error" in result and "loading" in result["error"].lower():
            time.sleep(5)
            continue
        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]
        if "error" in result:
            return f"[Error from API]: {result['error']}"
        break

    return "[Error]: Could not generate summary."

# ----------------- BUTTON ACTION -----------------
if st.button("🚀 Analyze Email"):
    if question.strip():
        with st.spinner("🔍 Analyzing the email..."):
            pred_class, probs = checkSpam(question)

            st.markdown("## 📊 Result")

            spam_prob = probs[0] * 100
            ham_prob = probs[1] * 100

            st.progress(int(ham_prob))
            st.markdown(
                f"✅ **Valid Email Probability:** {ham_prob:.2f}%"
                f"<br>🛑 **Spam Email Probability:** {spam_prob:.2f}%",
                unsafe_allow_html=True
            )

            if pred_class == 0:  # spam
                st.error("🚨 This email looks like **Spam**. No summary will be generated.")
            else:
                st.success("🎉 This email looks **Valid**.")
                st.markdown("### 📝 Email Summary")

                summary = summarize_text(question)
                st.info(summary)

    else:
        st.warning("⚠️ Please enter the body of an email to check.")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Created by "
    "<a href='https://github.com/ManoharSingh1311'>Manohar Singh</a> | "
    "<a href='mailto:burathimannu@gmail.com'>Mail</a> | "
    "<a href='tel:+916399121342'>Contact</a></p>",
    unsafe_allow_html=True
)
