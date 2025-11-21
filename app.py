import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Embeddings + Model
# ----------------------------
st.cache_resource()
def load_glove_small():
    with open("small_glove.pkl", "rb") as f:
        return pickle.load(f)

st.cache_resource()
def load_model():
    return pickle.load(open("glove_sentiment_model.pkl", "rb"))

glove = load_glove_small()
model = load_model()

# ----------------------------
# Sentence Vector Function
# ----------------------------
def sentence_vector(text, glove, dim=100):
    words = text.lower().split()
    vectors = [glove[w] for w in words if w in glove]

    if len(vectors) == 0:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Header Section
st.markdown("""
<div style='text-align: center;'>
    <h1>🎬 Movie Sentiment Analyzer</h1>
    <p style='font-size:18px;'>Analyze your movie reviews using GloVe word embeddings + ML 🤖</p>
</div>
""", unsafe_allow_html=True)

# Main Input Box
st.markdown("### ✍️ Enter your movie review below:")
text = st.text_area("", height=150, placeholder="Type your review here...")

# Predict Button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    
    if text.strip() == "":
        st.warning("⚠️ Please write a review before predicting!")
    else:
        vec = sentence_vector(text, glove).reshape(1, -1)
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred] * 100

        # Display Result
        if pred == 1:
            st.success(f"🎉 **Positive Review!** \n\n Confidence: **{prob:.2f}%** 😊")
        else:
            st.error(f"😞 **Negative Review** \n\n Confidence: **{prob:.2f}%**")

        # Stylish Divider
        st.markdown("<hr>", unsafe_allow_html=True)

        # Show Analysis
        st.markdown("### 📊 Review Breakdown")
        st.write(f"""
        - **Review entered:**  
          *{text}*

        - **Predicted Sentiment:**  
          {'😊 Positive' if pred == 1 else '😞 Negative'}

        - **Confidence:**  
          {prob:.2f}%
        """)

# Footer
st.markdown("""
<hr>
<div style='text-align: center; font-size: 14px; color: gray;'>
    Built with ❤️ using Python, GloVe & Streamlit
</div>
""", unsafe_allow_html=True)
