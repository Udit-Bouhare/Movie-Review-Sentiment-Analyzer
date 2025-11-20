from flask import Flask, request, jsonify 
from gensim.models import Word2Vec 
import pickle 
import numpy as np 
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

app = Flask(__name__)


w2v_model = Word2Vec.load('word2vec_sentiment.model')

with open('sentiment_classifier.pkl', 'rb') as f: 
    classifier = pickle.load(f)

def avg_word2vec(words, model): 
    valid = [w for w in words if w in model.wv.index_to_key]

    if len(valid) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean([model.wv[w] for w in valid], axis=0)

stop_words = set(stopwords.words("english"))

def preprocess_text(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 4. Remove special characters (keep a–z, 0–9, space)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 5. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # 7. Tokenize
    return text.split()


@app.route('/')
def home(): 
    return jsonify({
        "message" : "Movie Review Sentiment Analysis API",
        "model" : "Word2Vec + Classifier",
        "usage" : { 
            "endpoint" : "/predict",
            "method" : "POST",
            "body" : {"text" : "your review here"}
        }
    })


@app.route('/predict', methods = ['POST'])
def predict(): 
    try: 
        data = request.get_json()

        if not data: 
            return jsonify({
                "status": "error",
                "message": "No data provided. Send JSON with 'text' field."
            }), 400
        
        if 'text' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'text' field in request"
            }), 400
        
        review_text = data['text']

        if not review_text or review_text.strip() == "":
            return jsonify({
                "status": "error",
                "message": "Text cannot be empty"
            }), 400
        
          # Preprocess: tokenize (same as training)
        words = preprocess_text(review_text)
        
        # Convert to vector using your function
        review_vector = avg_word2vec(words, w2v_model)
        
        # Reshape for prediction (classifier expects 2D array)
        review_vector = review_vector.reshape(1, -1)
        
        # Make prediction
        prediction = classifier.predict(review_vector)[0]
        
        # Get probabilities (if your classifier supports it)
        try:
            probabilities = classifier.predict_proba(review_vector)[0]
            has_proba = True
        except:
            has_proba = False
        
        # Format response based on your label encoding
        # Adjust this based on your actual labels
        sentiment = "Positive" if prediction == 1 else "Negative"

         # Build response
        response = {
            "status": "success",
            "review": review_text,
            "sentiment": sentiment,
            "prediction_value": int(prediction)
        }

        if has_proba:
            confidence = float(max(probabilities)) * 100
            response["confidence"] = f"{confidence:.2f}%"
            response["probabilities"] = {
                "negative": f"{probabilities[0]*100:.2f}%",
                "positive": f"{probabilities[1]*100:.2f}%"
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500
    
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    }), 200


if __name__ == "__main__": 
    app.run(debug = True, port = 5001)