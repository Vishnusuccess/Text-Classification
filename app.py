import re
import string
import spacy
import joblib
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request 
from flask_restx import Api, Resource, fields

"""
This script defines a Flask-based REST API for predicting medical trial categories
using a trained LightGBM model and TF-IDF vectorizer.

Functionality:
- Accepts a medical trial description via POST request
- Preprocesses the text (lowercase, punctuation removal, lemmatization)
- Uses a trained TF-IDF + LightGBM pipeline to classify the input
- Returns the predicted label as JSON

Key features:
- Built with Flask and Flask-RESTX (Swagger UI at /docs)
- Custom text preprocessing with SpaCy and NLTK stopwords
- Model and vectorizer loaded from joblib pickle files
- Supports prediction for: ALS, Dementia, OCD, Parkinson’s Disease, and Scoliosis

Example JSON input:
{
    "description": "Amyotrophic lateral sclerosis (ALS) is a neurodegenerative disease affecting motor neurons."
}

Example JSON output:
{
    "predicted_label": "ALS"
}

"""


# Load SpaCy model and stopwords
print("Loading SpaCy model and stopwords...")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Load the saved models
print("Loading saved LightGBM model and TF-IDF vectorizer...")
lgbm_model = joblib.load("lgbm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['ALS', 'Dementia', 'Obsessive Compulsive Disorder', 'Parkinson’s Disease', 'Scoliosis'])

# Initialize Flask app and Flask-RESTX API
app = Flask(__name__)
api = Api(app, doc='/docs')
ns = api.namespace('Prediction', description='Prediction operations')

# Define input model for Swagger UI
predict_model = api.model('Prediction', {
    'description': fields.String(required=True, description='The description of the medical trial')
})

# Preprocessing function
def preprocess(text):
    print("\n[DEBUG] Raw input:", text)
    text = text.lower()
    text = re.sub(r"\|\|", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_stop]
    cleaned = " ".join(tokens)
    print("[DEBUG] Cleaned text:", cleaned)
    return cleaned

@ns.route('/predict')
class Predict(Resource):
    @api.doc('predict')
    @api.expect(predict_model)
    def post(self):
        try:
            data = request.get_json(force=True)
            description = data.get('description', '')
            print("\n[INFO] Received description:", description)

            cleaned_description = preprocess(description)
            print("[DEBUG] Cleaned text:", cleaned_description)

            prediction = lgbm_model.predict([cleaned_description])
            print("[DEBUG] Prediction:", prediction)

            predicted_label = label_encoder.inverse_transform(prediction)[0]
            print("[INFO] Predicted label:", predicted_label)

            return {'predicted_label': predicted_label}

        except Exception as e:
            print("[ERROR]", e)
            return {'error': str(e)}, 500



# Run the app
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
