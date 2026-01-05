# Fake News Detection System - Matrix Edition

A full-stack web application for detecting fake news using machine learning, featuring a dark Matrix-style terminal UI.

## 🌿 Branch Information
- **Branch**: `feat-fake-news-ml-matrix-ui`
- **Status**: Ready for development

## 🎨 Features

- **Matrix Terminal UI**: Dark hacker-style interface with neon green glowing text
- **Matrix Rain Animation**: Falling code background effect
- **Real-time Analysis**: Fast API predictions with loading states
- **Confidence Display**: Visual confidence bars with glowing effects
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture

```
/home/engine/project
├── index.html          # Frontend UI
├── styles.css          # Matrix terminal styling
├── script.js           # Matrix animation + API calls
├── main.py             # FastAPI backend
├── train_model.py      # ML model trainer
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Download/create a sample dataset
- Preprocess text using NLTK (tokenization, stemming, stopword removal)
- Create TF-IDF features (5000 features, unigrams + bigrams)
- Train Logistic Regression classifier
- Save model as `fake_news_model.joblib`
- Save vectorizer as `tfidf_vectorizer.joblib`

### 3. Start the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 4. Open the Frontend

Open `index.html` in your browser, or serve it with a static file server:

```bash
# Using Python
python -m http.server 8080

# Then open http://localhost:8080
```

## 📡 API Endpoints

### GET `/`
Returns system status and version info.

### GET `/status`
Returns model loading status.

**Response:**
```json
{
  "model_loaded": true,
  "status": "READY"
}
```

### POST `/predict`
Analyze news text for authenticity.

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.9234,
  "probabilities": {
    "FAKE": 0.9234,
    "REAL": 0.0766
  }
}
```

## 🎯 ML Pipeline

1. **Text Preprocessing**:
   - Lowercase conversion
   - Remove special characters and digits
   - Tokenization
   - Stopword removal (NLTK English stopwords)
   - Stemming (Porter Stemmer)

2. **Feature Extraction**:
   - TF-IDF Vectorization
   - Max 5000 features
   - Unigrams and bigrams
   - Min document frequency: 2
   - Max document frequency: 95%

3. **Model**:
   - Logistic Regression
   - L-BFGS optimizer
   - Max 1000 iterations
   - C = 1.0 (regularization)

## 🎨 UI Features

- **Matrix Rain**: Canvas-based falling code animation
- **Scanlines**: CRT monitor effect overlay
- **Glowing Text**: Neon green text-shadow effects
- **Terminal Input**: Monospace font input box
- **Animated Loading**: Progress bar with shimmer effect
- **Result Display**: Glowing badges for FAKE/REAL results
- **Confidence Bars**: Animated progress indicators
- **Keyboard Shortcuts**: Ctrl+Enter to submit

## 🔧 Configuration

### Changing Model Parameters

Edit `train_model.py`:
```python
# TF-IDF settings
vectorizer = TfidfVectorizer(
    max_features=5000,  # Number of features
    ngram_range=(1, 2), # Uni + bigrams
    min_df=2,
    max_df=0.95
)

# Logistic Regression settings
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver='lbfgs'
)
```

### Using Real Kaggle Dataset

For production use, download the Kaggle Fake and Real News Dataset:
1. Go to [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download and extract to get `Fake.csv` and `True.csv`
3. Place them in the project root
4. Run `python train_model.py`

## 📦 Dependencies

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **joblib**: Model serialization
- **scikit-learn**: ML algorithms
- **nltk**: Natural language processing
- **pandas**: Data handling
- **numpy**: Numerical operations

## 🎮 Usage

1. Enter news article text in the input box
2. Click ">> ANALYZE_NEURAL_NETWORK" or press Ctrl+Enter
3. Wait for the neural network analysis
4. View the FAKE/REAL result with confidence score

## ⚠️ Notes

- The default training uses a small sample dataset for demonstration
- For accurate predictions, train with the full Kaggle Fake/Real News dataset
- Ensure the API server is running before using the frontend
- The model may take a few seconds to respond on first prediction

## 📄 License

MIT License - Feel free to use and modify!
