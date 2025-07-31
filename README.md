# Fake News Detector

An advanced machine learning-powered web application that helps users identify potential fake news articles with high accuracy. The project combines natural language processing, machine learning, and a modern web interface to provide real-time fake news detection.

![Fake News Detector](https://img.shields.io/badge/AI-Fake%20News%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![Flask](https://img.shields.io/badge/Flask-2.0.1-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-1.4.2-orange)

<img width="1506" height="732" alt="Screenshot from 2025-07-31 23-15-34" src="https://github.com/user-attachments/assets/267f2314-7bc4-4ec2-a53b-009c097de854" />

<img width="1607" height="800" alt="Screenshot from 2025-07-31 23-16-08" src="https://github.com/user-attachments/assets/8221d4c3-8af2-40a6-914f-581a5e813017" />

<img width="1607" height="800" alt="Screenshot from 2025-07-31 23-16-24" src="https://github.com/user-attachments/assets/a4f9f592-18c7-4846-9d22-2ad0a236063c" />

## Features

### 1. Machine Learning Model
- **XGBoost Classifier**: Trained on the LIAR dataset
- **High Accuracy**: Achieves over 73% accuracy on validation data
- **Balanced Predictions**: Handles both true and fake news effectively
- **Confidence Scores**: Provides probability scores for predictions

### 2. Text Processing
- Advanced NLP pipeline including:
  - Text normalization
  - Special character removal
  - Tokenization
  - Stop word removal
  - Lemmatization
- TF-IDF vectorization for feature extraction
- Metadata feature integration

### 3. Web Interface
- **Modern UI/UX**:
  - Clean, responsive design
  - Glass-morphism effects
  - Intuitive layout
  - Real-time analysis
- **Interactive Features**:
  - Example text suggestions
  - Copy-paste functionality
  - Dynamic loading indicators
  - Animated results display

### 4. Analysis Results
- Binary classification (Real/Fake)
- Confidence percentage
- Detailed probability breakdown
- Visual indicators:
  - Color-coded results
  - Progress bars
  - Verdict badges

## Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd fake-news-detector
```

2. **Create Virtual Environment**
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure
```
fake-news-detector/
├── app.py                 # Flask application
├── preprocessor.py        # Text preprocessing module
├── requirements.txt       # Project dependencies
├── templates/
│   └── index.html        # Web interface template
├── fake_news_model.joblib # Trained ML model
├── tfidf_vectorizer.joblib# Fitted vectorizer
├── fakenewsdetector.ipynb # Model training notebook
└── README.md             # Project documentation
```

## Usage

1. **Start the Server**
```bash
python app.py
```

2. **Access the Application**
- Open your web browser
- Navigate to `http://localhost:5000`
- Enter or paste news text
- Click "Analyze" to get results

## Model Training

The model was trained on the LIAR dataset using:
- 10,240 training samples
- Balanced class distribution
- Feature engineering including:
  - Text features (TF-IDF)
  - Metadata features
  - Credibility scores

### Model Performance
```
Classification Report:
              precision    recall  f1-score   support
           0       0.72      0.62      0.67       898
           1       0.73      0.81      0.77      1150
accuracy                            0.73      2048
```

## API Endpoints

### 1. Home Page
```
GET /
Returns the main web interface
```

### 2. Prediction API
```
POST /predict
Content-Type: application/json
{
    "text": "News article text"
}

Response:
{
    "prediction": "True/False",
    "confidence": float,
    "probabilities": {
        "true": float,
        "false": float
    }
}
```

## Dependencies

- Flask==2.0.1
- joblib==1.1.0
- nltk==3.6.3
- numpy==1.21.2
- scikit-learn==0.24.2
- xgboost==1.4.2

## Future Improvements

1. **Model Enhancements**
   - Integration of BERT/transformer models
   - Multi-language support
   - Real-time model updates

2. **Feature Additions**
   - URL input support
   - Batch processing
   - Source credibility checking
   - Historical analysis

3. **UI Improvements**
   - User accounts
   - Analysis history
   - Detailed explanation of results
   - Social sharing integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LIAR dataset for training data
- XGBoost team for the classifier
- Flask team for the web framework
- All contributors and maintainers

## Contact

For questions and support, please open an issue in the repository.
