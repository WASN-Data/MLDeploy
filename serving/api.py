"""
FastAPI Serving API for Music Genre Classification.
Endpoints:
- /predict: Upload audio file → get genre prediction
- /feedback: Submit feedback for model retraining
- /health: Health check
"""

import os
import pickle
import tempfile
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import librosa


# =============================================================================
# CONFIGURATION
# =============================================================================

ARTIFACTS_DIR = "/artifacts"
DATA_DIR = "/data"
RETRAIN_THRESHOLD = 10  # Retrain every k feedback samples

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


# =============================================================================
# GLOBALS (loaded at startup)
# =============================================================================

model = None
scaler = None
label_encoder = None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Music Genre Classification API",
    description="Upload an audio file to classify its music genre",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# STARTUP: Load artifacts
# =============================================================================

@app.on_event("startup")
async def load_artifacts():
    global model, scaler, label_encoder

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

    # Ensure artifacts exist (better error message than a silent crash)
    for p in [model_path, scaler_path, encoder_path]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing artifact file: {p}")

    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loading scaler from: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"Loading label encoder from: {encoder_path}")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    print("✅ Model, scaler and label encoder loaded successfully!")


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features_from_audio(audio_bytes: bytes) -> dict:
    """
    Extract audio features from bytes (uploaded file).
    Returns dict matching FEATURE_COLUMNS.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, duration=30)

        features = {}
        features['length'] = len(y)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = float(np.mean(chroma_stft))
        features['chroma_stft_var'] = float(np.var(chroma_stft))

        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_var'] = float(np.var(rms))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_var'] = float(np.var(spectral_centroid))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_var'] = float(np.var(spectral_bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = float(np.mean(rolloff))
        features['rolloff_var'] = float(np.var(rolloff))

        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        features['zero_crossing_rate_var'] = float(np.var(zcr))

        harmony, perceptr = librosa.effects.hpss(y)
        features['harmony_mean'] = float(np.mean(harmony))
        features['harmony_var'] = float(np.var(harmony))
        features['perceptr_mean'] = float(np.mean(perceptr))
        features['perceptr_var'] = float(np.var(perceptr))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))

        return features

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def features_to_array(features: dict) -> np.ndarray:
    return np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    probabilities: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", model_loaded=(model is not None))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=(model is not None))


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        features = extract_features_from_audio(audio_bytes)
        X = features_to_array(features)
        X_scaled = scaler.transform(X)

        pred_idx = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]

        classes = list(label_encoder.classes_)  # ['blues', ...]
        prob_dict = {cls: float(p) for cls, p in zip(classes, probabilities)}
        confidence = float(np.max(probabilities))

        genre = label_encoder.inverse_transform([pred_idx])[0]

        return PredictionResponse(
            genre=str(genre),
            confidence=confidence,
            probabilities=prob_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    prediction: str = Form(...),
    actual_label: str = Form(...),
):
    global model

    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if actual_label not in GENRE_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid label. Must be one of: {GENRE_LABELS}")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        features = extract_features_from_audio(audio_bytes)

        # Save RAW features (recommended). Keep consistent with ref_data.csv if it's raw.
        X_raw = features_to_array(features)  # shape (1, n_features)

        row_data = {col: float(X_raw[0][i]) for i, col in enumerate(FEATURE_COLUMNS)}
        row_data["label"] = actual_label
        row_data["prediction"] = prediction

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

        should_retrain = (feedback_count % RETRAIN_THRESHOLD == 0)
        retrained = False

        if should_retrain:
            print(f"🔄 Retraining triggered (threshold: {RETRAIN_THRESHOLD})")
            model = retrain_model()
            retrained = True

        return {
            "status": "success",
            "message": "Feedback recorded",
            "feedback_count": feedback_count,
            "retrained": retrained,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")


def retrain_model():
    """
    Retrain model using reference and production data.
    Uses label_encoder to keep numeric classes consistent with the original training.
    """
    global scaler, label_encoder

    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler

    ref_data_path = os.path.join(DATA_DIR, "ref_data.csv")
    prod_data_path = os.path.join(DATA_DIR, "prod_data.csv")

    if not os.path.exists(ref_data_path):
        raise RuntimeError(f"Missing reference data: {ref_data_path}")
    if not os.path.exists(prod_data_path):
        raise RuntimeError(f"Missing production data: {prod_data_path}")

    ref_df = pd.read_csv(ref_data_path)
    prod_df = pd.read_csv(prod_data_path)

    if "prediction" in prod_df.columns:
        prod_df = prod_df.drop(columns=["prediction"])

    combined_df = pd.concat([ref_df, prod_df], ignore_index=True)

    X = combined_df[FEATURE_COLUMNS].values
    y_str = combined_df["label"].astype(str).values

    # encode labels using the SAME encoder
    y_enc = label_encoder.transform(y_str)

    # refit scaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    new_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),  # keep 10
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    new_model.fit(X_s, y_enc)

    # save updated artifacts
    with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(new_model, f)

    print("✅ XGBoost retrained + scaler updated + artifacts saved!")
    return new_model


@app.get("/genres")
async def get_genres():
    return {"genres": GENRE_LABELS}


@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prod_data_path = os.path.join(DATA_DIR, "prod_data.csv")
    feedback_count = 0
    if os.path.exists(prod_data_path):
        feedback_count = len(pd.read_csv(prod_data_path))

    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "n_features": len(FEATURE_COLUMNS),
        "n_classes": len(GENRE_LABELS),
        "genres": GENRE_LABELS,
        "feedback_count": feedback_count,
        "retrain_threshold": RETRAIN_THRESHOLD,
    }
