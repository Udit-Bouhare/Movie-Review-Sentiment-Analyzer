import streamlit as st 
import requests 

API_URL = "http://127.0.0.1:5001/predict"

st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="📊"
)


st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to predict its sentiment using Word2Vec + ML model.")

review = st.text_area("Enter your movie review: ") 

if st.button("Predict"):
    if not review.strip(): 
        st.error("Please enter a review.")
    else: 
        response = requests.post(API_URL, json={"text" : review})

        if response.status_code == 200: 
            result = response.json()
             
            st.subheader("Prediction Result")
            st.write(f"**Sentiment:** {result['sentiment']}")
            st.write(f"**Confidence:** {result.get('confidence', 'N/A')}")


            if "probabilities" in result:
                st.write("Class Probabilities:")
                st.json(result["probabilities"])


        else:
            st.error("API Error:")
            st.write(response.text)