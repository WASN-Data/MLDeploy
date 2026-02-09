"""
FastAPI Serving API for Music Genre Classification.
Provides endpoints for:
- /predict: Upload audio file → get genre prediction
- /feedback: Submit feedback for model retraining
- /health: Health check
"""

import os
import io
import pickle
import tempfile
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTIFACTS_DIR = "/artifacts"
DATA_DIR = "/data"
RETRAIN_THRESHOLD = 10  # Retrain every k feedback samples

# Feature columns (must match training)
FEATURE_COLUMNS = [
    'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean',
    'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean',
    'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean',
    'perceptr_var', 'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var',
    'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
    'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var',
    'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var',
    'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var',
    'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var',
    'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
]

GENRE_LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                'jazz', 'metal', 'pop', 'reggae', 'rock']

# ============================================================================
# GLOBAL MODEL VARIABLES (loaded at startup)
# ============================================================================

model = None
scaler = None

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Music Genre Classification API",
    description="Upload an audio file to classify its music genre",
    version="1.0.0"
)

# Enable CORS for webapp communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STARTUP: Load model artifacts
# ============================================================================

@app.on_event("startup")
async def load_artifacts():
    """Load model and scaler at API startup."""
    global model, scaler
    
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    
    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Loading scaler from: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✅ Model and scaler loaded successfully!")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_audio(audio_bytes: bytes) -> dict:
    """
    Extract audio features from bytes (uploaded file).
    Returns dictionary matching GTZAN feature format.
    """
    # Save bytes to temporary file for librosa to read
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Load audio with librosa
        y, sr = librosa.load(tmp_path, sr=22050, duration=30)
        
        features = {}
        
        # Length
        features['length'] = len(y)
        
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = float(np.mean(chroma_stft))
        features['chroma_stft_var'] = float(np.var(chroma_stft))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_var'] = float(np.var(rms))
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_var'] = float(np.var(spectral_centroid))
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_var'] = float(np.var(spectral_bandwidth))
        
        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = float(np.mean(rolloff))
        features['rolloff_var'] = float(np.var(rolloff))
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        features['zero_crossing_rate_var'] = float(np.var(zcr))
        
        # Harmony and Percussive
        harmony, perceptr = librosa.effects.hpss(y)
        features['harmony_mean'] = float(np.mean(harmony))
        features['harmony_var'] = float(np.var(harmony))
        features['perceptr_mean'] = float(np.mean(perceptr))
        features['perceptr_var'] = float(np.var(perceptr))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))
        
        return features
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def features_to_array(features: dict) -> np.ndarray:
    """Convert features dict to numpy array in correct column order."""
    return np.array([[features[col] for col in FEATURE_COLUMNS]])

# ============================================================================
# API ENDPOINTS
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    genre: str
    confidence: float
    probabilities: dict

class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    features: dict
    prediction: str
    actual_label: str

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict music genre from uploaded audio file.
    
    Accepts: .wav, .mp3, .ogg, .flac audio files
    Returns: Predicted genre with confidence and all class probabilities
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read uploaded file
    audio_bytes = await file.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    try:
        # Extract features
        features = extract_features_from_audio(audio_bytes)
        
        # Convert to array and scale
        X = features_to_array(features)
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction_idx = model.predict(X_scaled)[0]
        
        # Convert prediction index to genre string first
        if isinstance(prediction_idx, (int, np.integer)):
            genre_predicted = GENRE_LABELS[int(prediction_idx)]
        else:
            genre_predicted = str(prediction_idx)
        
        # Get probabilities if available (SVC needs probability=True)
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_scaled)[0]
                prob_dict = {genre: float(prob) for genre, prob in zip(GENRE_LABELS, probabilities)}
                confidence = float(max(probabilities))
            except Exception:
                # Model's predict_proba failed - use prediction only
                prob_dict = {genre: (1.0 if genre == genre_predicted else 0.0) for genre in GENRE_LABELS}
                confidence = 1.0
        else:
            # Model doesn't have predict_proba - use prediction only
            prob_dict = {genre: (1.0 if genre == genre_predicted else 0.0) for genre in GENRE_LABELS}
            confidence = 1.0
        
        return PredictionResponse(
            genre=genre_predicted,
            confidence=confidence,
            probabilities=prob_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    prediction: str = Form(...),
    actual_label: str = Form(...)
):
    """
    Submit feedback for model retraining.
    
    The feedback (features + actual label) is saved to prod_data.csv.
    When enough feedback is collected, the model is automatically retrained.
    """
    global model
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate actual_label
    if actual_label not in GENRE_LABELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid label. Must be one of: {GENRE_LABELS}"
        )
    
    # Read uploaded file
    audio_bytes = await file.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    try:
        # Extract features
        features = extract_features_from_audio(audio_bytes)
        
        # Scale features (same as training)
        X = features_to_array(features)
        X_scaled = scaler.transform(X)
        
        # Prepare row for prod_data.csv
        row_data = {col: X_scaled[0][i] for i, col in enumerate(FEATURE_COLUMNS)}
        row_data['label'] = actual_label
        row_data['prediction'] = prediction
        
        # Append to prod_data.csv
        prod_data_path = os.path.join(DATA_DIR, "prod_data.csv")
        
        new_row_df = pd.DataFrame([row_data])
        
        if os.path.exists(prod_data_path):
            prod_df = pd.read_csv(prod_data_path)
            prod_df = pd.concat([prod_df, new_row_df], ignore_index=True)
        else:
            prod_df = new_row_df
        
        prod_df.to_csv(prod_data_path, index=False)
        
        feedback_count = len(prod_df)
        print(f"Feedback received. Total: {feedback_count}")
        
        # Check if retraining should be triggered
        should_retrain = feedback_count % RETRAIN_THRESHOLD == 0
        retrained = False
        
        if should_retrain:
            print(f"🔄 Retraining triggered (threshold: {RETRAIN_THRESHOLD})")
            model = retrain_model()
            retrained = True
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "feedback_count": feedback_count,
            "retrained": retrained
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")


def retrain_model():
    """
    Retrain model using reference and production data.
    Saves versioned checkpoints so you don't lose previous iterations.
    Returns the newly trained model.
    """
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime
    
    ref_data_path = os.path.join(DATA_DIR, "ref_data.csv")
    prod_data_path = os.path.join(DATA_DIR, "prod_data.csv")
    
    print("Loading reference data...")
    ref_df = pd.read_csv(ref_data_path)
    
    print("Loading production data...")
    prod_df = pd.read_csv(prod_data_path)
    
    # Remove 'prediction' column from prod_data if exists (not needed for training)
    if 'prediction' in prod_df.columns:
        prod_df = prod_df.drop(columns=['prediction'])
    
    # Combine datasets
    combined_df = pd.concat([ref_df, prod_df], ignore_index=True)
    print(f"Total training samples: {len(combined_df)}")
    
    # Train new model
    X = combined_df[FEATURE_COLUMNS].values
    y = combined_df['label'].values
    
    new_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    new_model.fit(X, y)
    
    # ===== MODEL VERSIONING =====
    # Create checkpoints directory
    checkpoints_dir = os.path.join(ARTIFACTS_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Save versioned checkpoint with timestamp and sample count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_samples = len(combined_df)
    n_prod = len(prod_df)
    checkpoint_name = f"model_v{timestamp}_n{n_samples}_prod{n_prod}.pkl"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(new_model, f)
    print(f"✅ Checkpoint saved: {checkpoint_name}")
    
    # Save main model (overwrites current)
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(new_model, f)
    
    # Save metadata about this training run
    metadata = {
        "timestamp": timestamp,
        "ref_samples": len(ref_df),
        "prod_samples": len(prod_df),
        "total_samples": n_samples,
        "checkpoint_path": checkpoint_path,
    }
    metadata_path = os.path.join(ARTIFACTS_DIR, "training_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model retrained and saved! (checkpoint: {checkpoint_name})")
    return new_model


@app.get("/genres")
async def get_genres():
    """Return list of supported genres."""
    return {"genres": GENRE_LABELS}


@app.get("/model-info")
async def model_info():
    """Return information about the current model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prod_data_path = os.path.join(DATA_DIR, "prod_data.csv")
    feedback_count = 0
    if os.path.exists(prod_data_path):
        feedback_count = len(pd.read_csv(prod_data_path))
    
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "n_features": len(FEATURE_COLUMNS),
        "n_classes": len(GENRE_LABELS),
        "genres": GENRE_LABELS,
        "feedback_count": feedback_count,
        "retrain_threshold": RETRAIN_THRESHOLD
    }
