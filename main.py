"""
Fake News Detection API - Matrix Terminal Style
FastAPI backend with ML prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from model_inference import SimpleInference

# Set NLTK data path to local directory
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)

# Download required NLTK data (will skip if already in nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

app = FastAPI(
    title="FAKE NEWS DETECTION TERMINAL",
    description="Matrix Neural Network Analysis System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize preprocessor
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model
MODEL_DATA_PATH = "model_data.json"

def preprocess_text(text):
    """Preprocess text like the training pipeline"""
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

try:
    if os.path.exists(MODEL_DATA_PATH):
        model_inference = SimpleInference(MODEL_DATA_PATH)
        MODEL_LOADED = True
        print(">>> MODEL DATA LOADED SUCCESSFULLY <<<")
    else:
        MODEL_LOADED = False
        print("!!! MODEL DATA NOT FOUND - Run export_model.py first !!!")
except Exception as e:
    MODEL_LOADED = False
    print(f"!!! ERROR LOADING MODEL DATA: {e} !!!")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

@app.get("/")
async def root():
    return {
        "status": "ONLINE",
        "system": "FAKE NEWS DETECTION MATRIX",
        "version": "1.0.0",
        "endpoint": "/predict"
    }

@app.get("/status")
async def status():
    return {
        "model_loaded": MODEL_LOADED,
        "status": "READY" if MODEL_LOADED else "MODEL_MISSING"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: PredictionRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train and export the model first.")
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    # Preprocess
    processed_text = preprocess_text(request.text)
    
    if not processed_text:
        raise HTTPException(status_code=400, detail="Invalid text after preprocessing")
    
    # Predict using SimpleInference
    prediction, probabilities = model_inference.transform_and_predict(processed_text)
    
    # Map prediction
    label = "REAL" if prediction == 1 else "FAKE"
    
    # Get confidence
    confidence = float(max(probabilities))
    
    return PredictionResponse(
        prediction=label,
        confidence=confidence,
        probabilities={
            "FAKE": float(probabilities[0]),
            "REAL": float(probabilities[1])
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
