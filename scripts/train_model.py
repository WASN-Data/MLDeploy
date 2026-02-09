"""
Model training script for Music Genre Classification.
Creates ref_data.csv and trains the classification model.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

# Feature columns (57 features - excluding filename and label)
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


def load_and_prepare_data(csv_path: str = None) -> tuple:
    """
    Load GTZAN features CSV and prepare for training.
    
    Returns:
        X: Feature matrix
        y: Labels
        df: Full dataframe
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, 'features_30_sec.csv')
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Genres: {df['label'].unique()}")
    print(f"Samples per genre:\n{df['label'].value_counts()}")
    
    # Extract features and labels
    X = df[FEATURE_COLUMNS].values
    y = df['label'].values
    
    return X, y, df


def create_ref_data(df: pd.DataFrame, scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Create ref_data.csv in the format required by the assignment:
    - Feature columns (scaled)
    - Label column
    
    This format matches what prod_data.csv will have for Evidently comparison.
    """
    # Get features
    X = df[FEATURE_COLUMNS].values
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Create dataframe with scaled features
    ref_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    ref_df['label'] = df['label'].values
    
    return ref_df, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                X_test: np.ndarray = None, y_test: np.ndarray = None) -> RandomForestClassifier:
    """
    Train a RandomForest classifier.
    
    RandomForest is chosen for:
    - Fast training and prediction
    - Good accuracy on tabular data
    - No hyperparameter tuning needed for decent results
    - Easy retraining with new data
    """
    print("\nTraining RandomForest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=200,       # Number of trees
        max_depth=15,           # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate if test data provided
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return model


def save_artifacts(model, scaler: StandardScaler, label_encoder: LabelEncoder = None):
    """Save trained model and preprocessing artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save label encoder if provided
    if label_encoder is not None:
        le_path = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')
        with open(le_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to: {le_path}")


def retrain_model(ref_data_path: str = None, prod_data_path: str = None):
    """
    Retrain model using both reference and production data.
    Used for continuous learning when new feedback is collected.
    
    Args:
        ref_data_path: Path to ref_data.csv
        prod_data_path: Path to prod_data.csv
    """
    if ref_data_path is None:
        ref_data_path = os.path.join(DATA_DIR, 'ref_data.csv')
    if prod_data_path is None:
        prod_data_path = os.path.join(DATA_DIR, 'prod_data.csv')
    
    print("=" * 50)
    print("RETRAINING MODEL WITH NEW DATA")
    print("=" * 50)
    
    # Load reference data
    ref_df = pd.read_csv(ref_data_path)
    print(f"Reference data: {len(ref_df)} samples")
    
    # Load production data if exists
    if os.path.exists(prod_data_path):
        prod_df = pd.read_csv(prod_data_path)
        print(f"Production data: {len(prod_df)} samples")
        
        # Combine datasets
        combined_df = pd.concat([ref_df, prod_df], ignore_index=True)
    else:
        print("No production data found, using reference data only")
        combined_df = ref_df
    
    print(f"Total training samples: {len(combined_df)}")
    
    # Features are already scaled in ref_data.csv
    X = combined_df[FEATURE_COLUMNS].values
    y = combined_df['label'].values
    
    # Train model on all data (no split for retraining to maximize data usage)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Save updated model (scaler remains the same)
    model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model retrained and saved to: {model_path}")
    return model


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("MUSIC GENRE CLASSIFICATION - MODEL TRAINING")
    print("=" * 50)
    
    # Step 1: Load data
    X, y, df = load_and_prepare_data()
    
    # Step 2: Create scaler and scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 3: Create and save ref_data.csv
    ref_df, _ = create_ref_data(df, scaler)
    ref_data_path = os.path.join(DATA_DIR, 'ref_data.csv')
    ref_df.to_csv(ref_data_path, index=False)
    print(f"\nReference data saved to: {ref_data_path}")
    print(f"ref_data.csv shape: {ref_df.shape}")
    
    # Step 4: Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 5: Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Step 6: Save artifacts
    save_artifacts(model, scaler)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nArtifacts saved in: {ARTIFACTS_DIR}")
    print("- model.pkl (RandomForest classifier)")
    print("- scaler.pkl (StandardScaler)")
    print(f"\nReference data: {ref_data_path}")


if __name__ == "__main__":
    main()
