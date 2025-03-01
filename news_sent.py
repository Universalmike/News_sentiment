import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Load Naive Bayes model and vectorizer
nb_model = joblib.load("Naive-Bayes-fin-news-predictor.pkl")
tfidf_vectorizer = joblib.load("vectorizer.pkl")

# Load BERT model and tokenizer
model_path = "bert_fin_news_model-20250228T143139Z-001/bert_fin_news_model"
#bert_model_name = "bert-base-uncased"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
bert_model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

def predict_naive_bayes(texts):
    """Predict sentiment using the Naive Bayes model."""
    X_tfidf = tfidf_vectorizer.transform(texts)
    predictions = nb_model.predict(X_tfidf)
    return predictions

def predict_bert(texts):
    """Predict sentiment using the BERT model."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    outputs = bert_model(**inputs)
    predictions = torch.nn.functional.softmax(torch.tensor(outputs.logits), dim=1).argmax(dim=1).numpy()
    return predictions

def evaluate_models(texts, labels):
    """Compare both models and display results."""
    nb_preds = predict_naive_bayes(texts)
    bert_preds = predict_bert(texts)
    
    nb_report = classification_report(labels, nb_preds, target_names=['positive', 'neutral', 'negative'], output_dict=True)
    bert_report = classification_report(labels, bert_preds, target_names=['positive', 'neutral', 'negative'], output_dict=True)
    
    return nb_report, bert_report

st.title("Sentiment Analysis Model Comparison")

# User input
input_type = st.radio("Choose input type:", ("Enter Text", "Upload CSV"))

if input_type == "Enter Text":
    user_text = st.text_area("Enter your text:")
    if st.button("Analyze") and user_text:
        nb_pred = predict_naive_bayes([user_text])[0]
        bert_pred = predict_bert([user_text])[0]
        
        st.write(f"Naive Bayes Prediction: {nb_pred}")
        st.write(f"BERT Prediction: {bert_pred}")

elif input_type == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'text' and 'label' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns and 'label' in df.columns:
            nb_report, bert_report = evaluate_models(df['text'].tolist(), df['label'].tolist())
            
            st.subheader("Naive Bayes Classification Report")
            st.json(nb_report)
            
            st.subheader("BERT Classification Report")
            st.json(bert_report)
        else:
            st.error("CSV must contain 'text' and 'label' columns.")
