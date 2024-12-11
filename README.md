# **SpamGuard: Smart Email Classifier & Summarizer** 🛡️

## **Overview**
The **SpamGuard** is a web application built with Streamlit that detects whether an email is spam or valid. Additionally, if the email is valid, it generates a summary using an advanced NLP model. This application leverages machine learning for spam classification and Hugging Face's transformer models for text summarization.

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [How it Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [API for Summarization](#api-for-summarization)
- [Demo](#demo)
- [Contributing](#contributing)
- [Contact](#contact)
- [Disclaimer](#disclaimer)

## **Features**
- **Spam Detection**: Uses a trained Logistic Regression model to classify an email as spam or valid.
- **Summary Generation**: If the email is valid, a summary of the email's content is generated using Hugging Face's BART transformer model.
- **Interactive UI**: Simple and engaging interface built using Streamlit, allowing users to input the email body and get instant feedback.

## **How it Works**
1. **Spam Detection**:
   - The model processes the email body and predicts whether it's spam or not.
   - Spam emails will not have a summary generated.
   - Valid (non-spam) emails will proceed to the next step of summarization.

2. **Summary Generation**:
   - For valid emails, the BART model from Hugging Face is used to generate a concise summary of the email content.

## **Tech Stack**
- **Frontend**: Streamlit (Python)
- **Backend**: Logistic Regression (Scikit-learn) for spam classification
- **Text Summarization**: Hugging Face's BART model (`facebook/bart-large-cnn`)
- **Data**: Custom dataset for training the spam classifier (`mail_data.csv`)

## **Installation and Setup**

### 1. Install the Dependencies:
Make sure you have Python 3.x installed, then install the required libraries:

```bash
pip install -r requirements.txt
```
## Contact
For any questions or feedback, please reach out to: `burathimannu@gmail.com`

---

This `README.md` provides a comprehensive overview, installation steps, and guidance on using and further developing the project.
