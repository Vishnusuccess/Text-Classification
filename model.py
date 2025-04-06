import pandas as pd
import re
import string
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

"""

This script trains text classification models on medical trial descriptions to predict disease categories.
It performs preprocessing (cleaning, lemmatization, stopword removal), encodes labels, and trains two machine learning
models: XGBoost and LightGBM. The trained models and TF-IDF vectorizer are saved as .pkl files for future inference.

Steps performed:
1. Load and preprocess textual data
2. Apply TF-IDF vectorization
3. Encode target labels
4. Train XGBoost and LightGBM classification models
5. Evaluate models and visualize confusion matrices
6. Save models and vectorizer for deployment

"""


# Load SpaCy model and stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Load data
df = pd.read_csv("trials.csv")

# Function to preprocess the text (remove punctuation, tokenize, lemmatize)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\|\|", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_stop]
    return " ".join(tokens)

# Apply preprocessing to the 'description' column
df['clean_description'] = df['description'].apply(preprocess)

# Filter out short descriptions (below 20 words)
df['desc_length'] = df['clean_description'].apply(lambda x: len(x.split()))
df = df[df['desc_length'] > 20] 

# Truncate long descriptions (above 500 words)
MAX_LENGTH = 500
df['clean_description'] = df['clean_description'].apply(lambda x: ' '.join(x.split()[:MAX_LENGTH]))

# Encode the labels (y-values)
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_description'], df['encoded_label'], test_size=0.2, random_state=42, stratify=df['encoded_label']
)

# 1. Train XGBoost Model
print("Training model with XGBoost...")
xgb_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1))
])

# Fit and save the XGBoost model
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgb_model.pkl")
print("XGBoost model saved to xgb_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(xgb_model.named_steps['tfidf'], "tfidf_vectorizer.pkl")
print("TF-IDF vectorizer saved to tfidf_vectorizer.pkl")

# Evaluate the XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))

# Confusion Matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('XGBoost Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 2. Train LightGBM Model
print("Training model with LightGBM...")
lgbm_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1))
])

# Fit and save the LightGBM model
lgbm_model.fit(X_train, y_train)
joblib.dump(lgbm_model, "lgbm_model.pkl")
print("LightGBM model saved to lgbm_model.pkl")

# Evaluate the LightGBM model
y_pred_lgbm = lgbm_model.predict(X_test)
print("\nLightGBM Classification Report:\n")
print(classification_report(y_test, y_pred_lgbm, target_names=label_encoder.classes_))

# Confusion Matrix for LightGBM
cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lgbm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('LightGBM Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

