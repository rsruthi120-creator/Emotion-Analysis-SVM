import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load files
model = joblib.load('/content/sentiment_model.pkl')
tfidf = joblib.load('/content/tfidf_vectorizer.pkl')
le = joblib.load('/content/label_encoder.pkl')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop_words])

st.title(" 📊 Sentiment Analysis AI ")

# The 'key' and 'on_change' logic helps the tunnel stay active
user_input = st.text_input("Type here and click Analyze:", key="input_box")

if st.button("Analyze Emotion"):
    if user_input:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        prediction = model.predict(vec)[0]
        label = le.inverse_transform([prediction])[0]
        st.success(f"Result: {label.upper()}")
    else:
        st.error("Please type a sentence first.")
