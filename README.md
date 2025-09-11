# **SpamGuard: Smart Email Classifier & Summarizer** 🛡️  

## **Overview**  
**SpamGuard** is a web application built with **Streamlit** that detects whether an email is spam or valid. Additionally, if the email is valid, it generates a summary using advanced NLP models from Hugging Face.  

It combines **machine learning** for spam classification and **transformer models** for text summarization, providing an end-to-end smart email assistant.  

---

## **Table of Contents**  
- [Features](#features)  
- [How it Works](#how-it-works)  
- [Tech Stack](#tech-stack)  
- [Project Structure](#project-structure)  
- [Installation and Setup](#installation-and-setup)  
- [Usage](#usage)  
- [Configuring Secrets](#configuring-secrets)  
- [Model Training](#model-training)  
- [API for Summarization](#api-for-summarization)  
- [Demo](#demo)  
- [Contributing](#contributing)  
- [Contact](#contact)  
- [Disclaimer](#disclaimer)  

---

## **Features**  
- **Spam Detection**: Logistic Regression model classifies emails as spam or valid.  
- **Summarization**: Generates summaries of valid emails using Hugging Face’s **BART** (`facebook/bart-large-cnn`).  
- **Interactive UI**: Streamlit frontend lets you input an email and instantly see results.  
- **Configurable**: Secure token management with `secrets.toml`.  

---

## **How it Works**  

1. **Spam Detection**  
   - Input email body is passed to the trained **Logistic Regression** classifier.  
   - If detected as **spam**, no summary is generated.  
   - If **valid**, it proceeds to summarization.  

2. **Summarization**  
   - Hugging Face’s **BART** model generates a concise summary of the email content.  

---

## **Tech Stack**  
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend**: Logistic Regression (Scikit-learn)  
- **Summarization**: Hugging Face Transformers (`facebook/bart-large-cnn`)  
- **Dataset**: Custom dataset (`mail_data.csv`)  

---

## **Project Structure**  

```
SpamGuard-Email-Classifier-Summarizer/
│── .streamlit/
│ └── secrets.toml # Hugging Face token stored here
│── venv/ # Virtual environment
│── mail_data.csv # Dataset for spam classification
│── main.py # Streamlit application entry point
│── requirements.txt # Dependencies
│── README.md # Documentation
│── Spam_mail_example.txt # Sample spam email
│── Non-spam_example.txt # Sample non-spam email
│── Project_Overview.txt # Notes about the project

```
