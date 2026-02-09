"""
Re-extract audio features from raw GTZAN audio files using current librosa version.

This ensures consistency between training features and prediction features.
Run this script to regenerate features_30_sec.csv with the current librosa version.
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'genres_original')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SAMPLE_RATE = 22050
DURATION = 30  # seconds

# Genre labels
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
           'jazz', 'metal', 'pop', 'reggae', 'rock']


def extract_features(file_path: str) -> dict:
    """
    Extract audio features from a single file.
    Uses the SAME feature extraction as the API for consistency.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
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
        # Handle both old and new librosa API (tempo can be array or scalar)
        if hasattr(tempo, '__len__'):
            features['tempo'] = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            features['tempo'] = float(tempo)
        
        # MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc{i+1}_var'] = float(np.var(mfccs[i]))
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    """Extract features from all audio files and save to CSV."""
    print(f"=" * 60)
    print(f"GTZAN Feature Extraction with librosa {librosa.__version__}")
    print(f"=" * 60)
    
    if not os.path.exists(AUDIO_DIR):
        print(f"ERROR: Audio directory not found: {AUDIO_DIR}")
        print("Please ensure the genres_original folder exists in data/")
        return
    
    all_features = []
    
    # Process each genre
    for genre in GENRES:
        genre_dir = os.path.join(AUDIO_DIR, genre)
        
        if not os.path.exists(genre_dir):
            print(f"WARNING: Genre directory not found: {genre_dir}")
            continue
        
        audio_files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        print(f"\nProcessing {genre}: {len(audio_files)} files")
        
        for audio_file in tqdm(audio_files, desc=genre):
            file_path = os.path.join(genre_dir, audio_file)
            features = extract_features(file_path)
            
            if features is not None:
                features['filename'] = audio_file
                features['label'] = genre
                all_features.append(features)
    
    if not all_features:
        print("ERROR: No features extracted. Check audio files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns to match expected format
    cols = ['filename', 'length', 'chroma_stft_mean', 'chroma_stft_var', 
            'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 
            'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo']
    
    # Add MFCC columns
    for i in range(1, 21):
        cols.extend([f'mfcc{i}_mean', f'mfcc{i}_var'])
    
    cols.append('label')
    
    # Ensure all columns exist
    for col in cols:
        if col not in df.columns:
            print(f"WARNING: Missing column {col}")
    
    df = df[[c for c in cols if c in df.columns]]
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'features_30_sec.csv')
    
    # Backup original if exists
    if os.path.exists(output_path):
        backup_path = os.path.join(OUTPUT_DIR, 'features_30_sec_original.csv')
        if not os.path.exists(backup_path):
            os.rename(output_path, backup_path)
            print(f"\nOriginal features backed up to: {backup_path}")
    
    df.to_csv(output_path, index=False)
    
    print(f"\n" + "=" * 60)
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Features per sample: {len(df.columns) - 2}")  # Exclude filename, label
    print(f"Genres: {df['label'].nunique()}")
    print(f"Librosa version: {librosa.__version__}")
    print(f"Output saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Run the notebook to retrain the model")
    print(f"  2. Rebuild Docker containers")


if __name__ == '__main__':
    main()
