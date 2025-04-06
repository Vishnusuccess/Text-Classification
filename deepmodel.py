import pandas as pd
import re
import string
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""

This script builds and evaluates a Multi-Layer Perceptron (MLP) model using TensorFlow/Keras
to classify medical trial descriptions into disease categories.

Pipeline overview:
1. Load and preprocess textual descriptions (lowercasing, punctuation removal, lemmatization)
2. Filter short descriptions and truncate overly long ones
3. Encode labels into integers using LabelEncoder
4. Convert text into numerical features using TF-IDF vectorization
5. Build and train a feedforward neural network (MLP) using TensorFlow/Keras
6. Evaluate the model using classification metrics and confusion matrix

Key features:
- Custom text preprocessing using SpaCy and NLTK stopwords
- TF-IDF vectorization (max_features=5000)
- MLP architecture with dropout to reduce overfitting
- Evaluation using accuracy, precision, recall, and confusion matrix
- Visualizations with Seaborn and Matplotlib

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

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

#  Build the MLP model (Feedforward Neural Network)
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(X_train_tfidf.shape[1],)))  
model.add(layers.Dense(256, activation='relu'))  
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))  

# Compile the model with multiple evaluation metrics
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#  Train the model
history = model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_data=(X_test_tfidf, y_test))

# Evaluate the model on the test set
y_pred_probs = model.predict(X_test_tfidf) 
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification metrics
print("\nMLP Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix for MLP
cm_mlp = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('MLP Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy:.4f}")

# Precision and Recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Precision on Test Set: {precision:.4f}")
print(f"Recall on Test Set: {recall:.4f}")
