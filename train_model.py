"""
Model Trainer for Fake News Detection
Downloads Kaggle dataset and trains TF-IDF + Logistic Regression model
"""

import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
print(">>> DOWNLOADING NLTK RESOURCES <<<")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

print(">>> INITIALIZING PREPROCESSOR <<<")
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text for training"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def download_dataset():
    """Download fake news dataset from Kaggle"""
    dataset_url = "https://storage.googleapis.com/kaggle-data-sets/592440/1254840/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240105%2Fus-central1%2Fcompute%2Fgoogleapis_research_kaggle%2Fgo4k&X-Goog-Date=20240105T180000Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0e7e4b6eaa2b1e6a0e3a5c3f5c3b7e7e8a8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8"
    
    # Alternative: Create sample dataset if download fails
    print(">>> CHECKING FOR LOCAL DATASET <<<")
    
    # Check if dataset file exists
    if os.path.exists("fake_news.csv"):
        print(">>> FOUND LOCAL DATASET <<<")
        return pd.read_csv("fake_news.csv")
    
    # Check if kaggle.json exists for API download
    if os.path.exists("kaggle.json"):
        print(">>> KAGGLE API FOUND - DOWNLOADING DATASET <<<")
        os.system("pip install kaggle -q")
        os.system("kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p . --unzip")
        if os.path.exists("Fake.csv") and os.path.exists("True.csv"):
            fake_df = pd.read_csv("Fake.csv")
            real_df = pd.read_csv("True.csv")
            fake_df['label'] = 0  # FAKE = 0
            real_df['label'] = 1  # REAL = 1
            combined_df = pd.concat([fake_df, real_df], ignore_index=True)
            return combined_df
    
    # Create sample dataset for demonstration
    print(">>> CREATING SAMPLE DATASET <<<")
    print("!!! NOTE: FOR PRODUCTION, DOWNLOAD KAGGLE FAKE/REAL NEWS DATASET !!!")
    
    sample_data = {
        'text': [
            # FAKE news samples
            "BREAKING: Scientists discover that water is actually dry and nobody noticed until now. Experts say this changes everything we thought we knew about hydration.",
            "SHOCKING: Local cat becomes mayor of small town, promises more fish and nap breaks for all citizens.",
            "ALERT: Government secretly replaced all broccoli with tiny trees, sources confirm. Health officials refuse to comment.",
            "INCREDIBLE: Man sneezes and accidentally solves complex mathematical equation that stumped mathematicians for decades.",
            "WARNING: Your pillow might be plotting against you while you sleep. Sleep experts warn of pillow uprising.",
            # REAL news samples  
            "The Federal Reserve announced today that it will maintain current interest rates, citing ongoing economic recovery and labor market improvements.",
            "Scientists at MIT have developed a new method for carbon capture that could significantly reduce industrial emissions.",
            "The World Health Organization released new guidelines for pandemic preparedness based on lessons learned from recent global health challenges.",
            "Apple Inc. reported quarterly earnings that exceeded analyst expectations, with strong iPhone and services revenue growth.",
            "NASA successfully launched its latest satellite to monitor climate patterns and weather changes across the globe.",
        ] * 50,  # Repeat for substantial dataset
        'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 50
    }
    
    return pd.DataFrame(sample_data)

def train_model():
    print("=== FAKE NEWS DETECTION MODEL TRAINER ===")
    print(">>> STEP 1: LOADING DATASET <<<")
    
    # Download or load dataset
    df = download_dataset()
    
    print(f">>> DATASET SIZE: {len(df)} samples <<<")
    print(f">>> COLUMNS: {df.columns.tolist()} <<<")
    
    # Handle missing values
    df = df.dropna(subset=['text'])
    df = df.drop_duplicates()
    
    print(f">>> AFTER CLEANING: {len(df)} samples <<<")
    
    # Use 'text' column
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist() if 'label' in df.columns else df['label_num'].tolist()
    
    print(">>> STEP 2: PREPROCESSING TEXT <<<")
    print(">>> This may take a moment... <<<")
    
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Filter out empty processed texts
    valid_pairs = [(text, label) for text, label in zip(processed_texts, labels) if text.strip()]
    processed_texts = [p[0] for p in valid_pairs]
    labels = [p[1] for p in valid_pairs]
    
    print(f">>> VALID SAMPLES: {len(processed_texts)} <<<")
    
    print(">>> STEP 3: SPLITTING DATA <<<")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f">>> TRAINING SET: {len(X_train)} <<<")
    print(f">>> TEST SET: {len(X_test)} <<<")
    
    print(">>> STEP 4: CREATING TF-IDF FEATURES <<<")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f">>> FEATURE DIMENSION: {X_train_tfidf.shape[1]} <<<")
    
    print(">>> STEP 5: TRAINING LOGISTIC REGRESSION <<<")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_tfidf, y_train)
    
    print(">>> STEP 6: EVALUATING MODEL <<<")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f">>> ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%) <<<")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    
    print(">>> STEP 7: SAVING MODEL <<<")
    joblib.dump(model, "fake_news_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    
    print("\n=== TRAINING COMPLETE ===")
    print(">>> MODEL SAVED TO: fake_news_model.joblib <<<")
    print(">>> VECTORIZER SAVED TO: tfidf_vectorizer.joblib <<<")
    print("\n>>> YOU CAN NOW RUN: uvicorn main:app --reload <<<")
    
    return model, vectorizer, accuracy

if __name__ == "__main__":
    train_model()
