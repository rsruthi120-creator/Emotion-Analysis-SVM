import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# 1. Setup NLTK for text cleaning
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 2. Load the models using RELATIVE paths for Cloud Deployment
# This fixes the FileNotFoundError you were seeing
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop_words])

# 3. Streamlit User Interface
st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("📊 Advanced Emotion Analysis")
st.markdown("---")
st.write("This project uses **Linear SVM** and **TF-IDF Bigrams** to detect fine-grained human emotions.")

# Input box
user_input = st.text_input("Type your sentence here:", placeholder="Example: I am so surprised by the results!")

if st.button("Analyze Emotion"):
    if user_input:
        # Process the input
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        
        # Predict
        prediction = model.predict(vec)[0]
        emotion = le.inverse_transform([prediction])[0]
        
        # Display Result
        st.success(f"Detected Emotion: **{emotion.upper()}**")
        st.info("Analysis complete using Bigram TF-IDF features.")
    else:
        st.warning("⚠️ Please enter a sentence first!")

st.markdown("---")
st.caption("Expandable Project: 4th Sem (Binary) -> 6th Sem (Multi-class Emotion)")
