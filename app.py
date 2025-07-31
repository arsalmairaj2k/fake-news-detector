from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from preprocessor import preprocess_text  # We'll create this file next

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = joblib.load('fake_news_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def predict_news(text):
    """
    Predict if a news text is fake or real
    """
    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Transform using saved vectorizer
        text_features = vectorizer.transform([processed_text]).toarray()
        
        # Add dummy metadata features (10 features to match training)
        dummy_metadata = np.zeros((1, 10))
        
        # Combine features
        combined_features = np.hstack([text_features, dummy_metadata])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probability = model.predict_proba(combined_features)[0]
        
        return {
            'prediction': 'True' if prediction == 1 else 'False',
            'confidence': float(max(probability)),
            'probabilities': {
                'false': float(probability[0]),
                'true': float(probability[1])
            }
        }
    except Exception as e:
        return {
            'error': str(e),
            'prediction': 'Error',
            'confidence': 0.0,
            'probabilities': {'false': 0.0, 'true': 0.0}
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        if not text:
            return jsonify({'error': 'No text provided'})
        
        result = predict_news(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)