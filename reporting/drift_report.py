"""
Data Drift and Classification Report Generator.

This script generates Evidently reports comparing:
- Reference data (training) vs Production data (feedback)

Usage:
    python drift_report.py

Outputs:
    - drift_report.html: Feature distribution drift analysis
    - classification_report.html: Classification performance metrics
"""

import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths (auto-detect Docker vs local)
if os.path.exists("/data"):
    DATA_DIR = "/data"
else:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

REF_DATA_PATH = os.path.join(DATA_DIR, "ref_data.csv")
PROD_DATA_PATH = os.path.join(DATA_DIR, "prod_data.csv")
OUTPUT_DIR = os.path.dirname(__file__)

# Feature columns (GTZAN features)
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


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Dynamically detect feature columns from dataframe.
    Excludes 'label' and 'prediction' columns.
    """
    exclude = {'label', 'prediction', 'filename', 'track_id'}
    return [col for col in df.columns if col not in exclude]


def generate_drift_report():
    """Generate data drift report comparing ref vs prod data."""
    from evidently import Report, Dataset
    from evidently.presets import DataDriftPreset
    
    print("=" * 60)
    print("GENERATING DATA DRIFT REPORT")
    print("=" * 60)
    
    # Check files exist
    if not os.path.exists(REF_DATA_PATH):
        print(f"❌ Reference data not found: {REF_DATA_PATH}")
        return None
    
    if not os.path.exists(PROD_DATA_PATH) or os.path.getsize(PROD_DATA_PATH) == 0:
        print(f"❌ Production data not found or empty: {PROD_DATA_PATH}")
        print("   Submit feedback through the webapp to generate production data.")
        return None
    
    # Load data
    print(f"Loading reference data from: {REF_DATA_PATH}")
    ref_df = pd.read_csv(REF_DATA_PATH)
    print(f"  → {len(ref_df)} samples")
    
    print(f"Loading production data from: {PROD_DATA_PATH}")
    prod_df = pd.read_csv(PROD_DATA_PATH)
    print(f"  → {len(prod_df)} samples")
    
    # Get feature columns dynamically
    feature_cols = get_feature_columns(ref_df)
    print(f"Using {len(feature_cols)} features for drift analysis")
    
    # Prepare feature-only datasets
    ref_features = ref_df[feature_cols]
    prod_features = prod_df[[c for c in feature_cols if c in prod_df.columns]]
    
    # Align columns
    common_cols = list(set(ref_features.columns) & set(prod_features.columns))
    ref_features = ref_features[common_cols]
    prod_features = prod_features[common_cols]
    
    print(f"Analyzing {len(common_cols)} common features")
    
    # Create Evidently datasets
    ref_dataset = Dataset.from_pandas(ref_features)
    prod_dataset = Dataset.from_pandas(prod_features)
    
    # Generate report
    print("\nGenerating drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_dataset, current_data=prod_dataset)
    
    # Save report
    output_path = os.path.join(OUTPUT_DIR, "drift_report.html")
    report.save_html(output_path)
    print(f"\n✅ Drift report saved: {output_path}")
    
    # Print summary
    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])
    if metrics:
        result = metrics[0].get("result", {})
        drift_detected = result.get("drift_detected", False)
        drift_share = result.get("drift_share", 0)
        
        print("\n" + "=" * 40)
        print("DRIFT SUMMARY")
        print("=" * 40)
        if drift_detected:
            print(f"⚠️  DRIFT DETECTED: {drift_share:.1%} of features drifted")
        else:
            print(f"✅ No significant drift: {drift_share:.1%} of features drifted")
    
    return report


def generate_classification_report():
    """Generate classification performance report."""
    from evidently import Report, Dataset
    from evidently.presets import ClassificationPreset
    
    print("\n" + "=" * 60)
    print("GENERATING CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Check prod data exists and has labels
    if not os.path.exists(PROD_DATA_PATH) or os.path.getsize(PROD_DATA_PATH) == 0:
        print(f"❌ Production data not found or empty: {PROD_DATA_PATH}")
        return None
    
    prod_df = pd.read_csv(PROD_DATA_PATH)
    
    if 'label' not in prod_df.columns or 'prediction' not in prod_df.columns:
        print("❌ Production data missing 'label' or 'prediction' columns")
        return None
    
    print(f"Loaded {len(prod_df)} production samples")
    
    # Prepare data for classification report
    # Create target and prediction columns
    report_df = prod_df[['label', 'prediction']].copy()
    report_df.columns = ['target', 'prediction']
    
    # Create Evidently dataset
    prod_dataset = Dataset.from_pandas(
        report_df,
        target='target',
        prediction='prediction'
    )
    
    # Generate report
    print("\nGenerating classification report...")
    report = Report(metrics=[ClassificationPreset()])
    report.run(current_data=prod_dataset)
    
    # Save report
    output_path = os.path.join(OUTPUT_DIR, "classification_report.html")
    report.save_html(output_path)
    print(f"\n✅ Classification report saved: {output_path}")
    
    # Print summary
    accuracy = (report_df['target'] == report_df['prediction']).mean()
    print("\n" + "=" * 40)
    print("CLASSIFICATION SUMMARY")
    print("=" * 40)
    print(f"Samples: {len(report_df)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Correct: {(report_df['target'] == report_df['prediction']).sum()}")
    print(f"Incorrect: {(report_df['target'] != report_df['prediction']).sum()}")
    
    return report


def main():
    """Generate all reports."""
    print("\n🔍 Evidently Report Generator")
    print("=" * 60)
    print(f"Reference data: {REF_DATA_PATH}")
    print(f"Production data: {PROD_DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Generate drift report
    drift_report = generate_drift_report()
    
    # Generate classification report
    class_report = generate_classification_report()
    
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    
    if drift_report:
        print(f"📊 Drift report: {os.path.join(OUTPUT_DIR, 'drift_report.html')}")
    if class_report:
        print(f"📈 Classification report: {os.path.join(OUTPUT_DIR, 'classification_report.html')}")
    
    print("\nOpen the HTML files in your browser to view the reports.")


if __name__ == "__main__":
    main()
