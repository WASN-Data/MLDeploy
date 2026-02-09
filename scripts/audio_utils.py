"""
Audio feature extraction utilities for Music Genre Classification.
These functions extract the same features used in GTZAN dataset.
"""

import numpy as np
import librosa


# Feature names matching GTZAN dataset (without filename and label)
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


def extract_features_from_audio(audio_path: str = None, y: np.ndarray = None, sr: int = 22050) -> dict:
    """
    Extract audio features matching the GTZAN dataset format.
    
    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        y: Audio time series (if already loaded)
        sr: Sample rate (default 22050)
    
    Returns:
        Dictionary with feature names and values
    """
    # Load audio if path provided
    if audio_path is not None:
        y, sr = librosa.load(audio_path, sr=sr, duration=30)  # Limit to 30 seconds
    
    if y is None:
        raise ValueError("Must provide either audio_path or y (audio array)")
    
    features = {}
    
    # Length of audio
    features['length'] = len(y)
    
    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)
    
    # RMS (Root Mean Square - Energy)
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)
    
    # Harmony and Perceptr (using harmonic-percussive source separation)
    harmony, perceptr = librosa.effects.hpss(y)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['perceptr_mean'] = np.mean(perceptr)
    features['perceptr_var'] = np.var(perceptr)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
    
    # MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
    
    return features


def features_to_array(features: dict) -> np.ndarray:
    """
    Convert features dictionary to numpy array in correct order for model input.
    
    Args:
        features: Dictionary from extract_features_from_audio()
    
    Returns:
        Numpy array of shape (1, 57) ready for model prediction
    """
    return np.array([[features[col] for col in FEATURE_COLUMNS]])


def extract_features_for_prediction(audio_path: str = None, y: np.ndarray = None, sr: int = 22050) -> np.ndarray:
    """
    Convenience function: extract features and return as array ready for prediction.
    
    Args:
        audio_path: Path to audio file
        y: Audio time series (if already loaded)
        sr: Sample rate
    
    Returns:
        Numpy array of shape (1, 57) ready for model prediction
    """
    features = extract_features_from_audio(audio_path=audio_path, y=y, sr=sr)
    return features_to_array(features)


# Genre labels in the same order as training
GENRE_LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                'jazz', 'metal', 'pop', 'reggae', 'rock']
