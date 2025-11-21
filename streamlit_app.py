import streamlit as st
import numpy as np
import pickle
from load_glove import load_glove

with open("glove_vectors.pkl", "rb") as f:
    glove = pickle.load(f)
model = pickle.load(open("glove_sentiment_model.pkl", "rb"))

def sentence_vector(text, glove, dim=100):
    words = text.lower().split()
    vectors = [glove[w] for w in words if w in glove]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

st.title("🎬 Movie Sentiment Analyzer (GloVe)")

text = st.text_area("Enter your review:")

if st.button("Predict"):
    vec = sentence_vector(text, glove).reshape(1,-1)
    pred = model.predict(vec)[0]
    if pred == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😞")
