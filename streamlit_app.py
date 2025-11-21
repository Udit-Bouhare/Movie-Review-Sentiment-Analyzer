import streamlit as st
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    w2v = Word2Vec.load("word2vec_sentiment.model")
    clf = pickle.load(open("sentiment_classifier.pkl", "rb"))
    return w2v, clf

w2v_model, classifier = load_models()

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_text(text):
    # 1. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Lowercase
    text = text.lower()

    # 4. Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # 5. Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # 6. Remove extra spaces
    text = " ".join(text.split())
    
    return text

# -------------------------------
# Avg Word2Vec Function
# -------------------------------
def avg_word2vec(words, model):
    valid = [w for w in words if w in model.wv.index_to_key]

    if len(valid) == 0:
        return np.zeros(model.vector_size)

    return np.mean([model.wv[w] for w in valid], axis=0)


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("This app predicts the **sentiment (Positive/Negative)** of a movie review using **Word2Vec + ML model**.")

review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if review.strip() == "":
        st.error("Please enter a review text.")
    else:
        # Preprocess the text
        cleaned = preprocess_text(review)

        # Tokenize
        tokens = cleaned.split()

        # Word2Vec vector
        vector = avg_word2vec(tokens, w2v_model).reshape(1, -1)

        # Prediction
        prediction = classifier.predict(vector)[0]

        # Sentiment output
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.subheader("Prediction Result")
        st.success(f"Sentiment: {sentiment}")

        # Probability (if available)
        try:
            proba = classifier.predict_proba(vector)[0]
            st.write(f"Confidence: {max(proba) * 100:.2f}%")
        except:
            st.write("Confidence: Not available.")

