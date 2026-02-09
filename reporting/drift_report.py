"""
Evidently Reporting for Music Genre Classification.

Features:
1. Data Drift: Compares reference data (training) with production data (feedback)
   to detect feature drift that might affect model performance.
2. Classification Metrics: Tracks model health with F1, precision, recall, 
   balanced accuracy by comparing predictions vs true labels in prod data.
"""

import os
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
)

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


def load_data(data_dir: str = None, keep_prediction: bool = False):
    """
    Load reference and production data.
    
    Args:
        data_dir: Directory containing the CSV files
        keep_prediction: If True, keep the 'prediction' column for classification metrics
    
    Returns:
        tuple: (reference_df, production_df) or (reference_df, None) if no prod data
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    ref_path = os.path.join(data_dir, 'ref_data.csv')
    prod_path = os.path.join(data_dir, 'prod_data.csv')
    
    # Load reference data
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference data not found: {ref_path}")
    
    ref_df = pd.read_csv(ref_path)
    
    # Load production data if exists
    prod_df = None
    if os.path.exists(prod_path):
        prod_df = pd.read_csv(prod_path)
        # Only remove 'prediction' column if not needed for classification metrics
        if not keep_prediction and 'prediction' in prod_df.columns:
            prod_df = prod_df.drop(columns=['prediction'])
    
    return ref_df, prod_df


def get_column_mapping(for_classification: bool = False):
    """
    Define column mapping for Evidently.
    
    Args:
        for_classification: If True, include prediction column mapping
    """
    if for_classification:
        return ColumnMapping(
            target='label',
            prediction='prediction',
            numerical_features=FEATURE_COLUMNS
        )
    return ColumnMapping(
        target='label',
        numerical_features=FEATURE_COLUMNS
    )


def generate_data_drift_report(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> Report:
    """
    Generate a comprehensive data drift report.
    
    Args:
        ref_df: Reference (training) data
        prod_df: Production (feedback) data
        
    Returns:
        Evidently Report object
    """
    column_mapping = get_column_mapping(for_classification=False)
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(
        reference_data=ref_df,
        current_data=prod_df,
        column_mapping=column_mapping
    )
    
    return report


def generate_classification_report(prod_df: pd.DataFrame) -> Report:
    """
    Generate a classification quality report with F1, precision, recall, balanced accuracy.
    
    This compares predictions vs actual labels in production data to track model health.
    
    Args:
        prod_df: Production data with both 'label' (true) and 'prediction' columns
        
    Returns:
        Evidently Report object with classification metrics
    """
    if 'prediction' not in prod_df.columns:
        raise ValueError("Production data must have 'prediction' column for classification metrics")
    
    column_mapping = get_column_mapping(for_classification=True)
    
    report = Report(metrics=[
        ClassificationPreset(),
    ])
    
    report.run(
        reference_data=None,  # No reference needed for classification quality
        current_data=prod_df,
        column_mapping=column_mapping
    )
    
    return report


def get_classification_metrics(prod_df: pd.DataFrame) -> dict:
    """
    Extract key classification metrics (F1, precision, recall, balanced accuracy).
    
    Args:
        prod_df: Production data with 'label' and 'prediction' columns
        
    Returns:
        dict with classification metrics
    """
    if 'prediction' not in prod_df.columns or 'label' not in prod_df.columns:
        return None
    
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, 
        balanced_accuracy_score, accuracy_score
    )
    
    y_true = prod_df['label']
    y_pred = prod_df['prediction']
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'sample_count': len(prod_df),
    }


def generate_data_quality_report(ref_df: pd.DataFrame, prod_df: pd.DataFrame = None) -> Report:
    """
    Generate a data quality report.
    
    Args:
        ref_df: Reference (training) data
        prod_df: Production data (optional)
        
    Returns:
        Evidently Report object
    """
    column_mapping = get_column_mapping(for_classification=False)
    
    report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    report.run(
        reference_data=ref_df,
        current_data=prod_df,
        column_mapping=column_mapping
    )
    
    return report


def generate_target_drift_report(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> Report:
    """
    Generate a target (label) drift report.
    
    Args:
        ref_df: Reference (training) data
        prod_df: Production (feedback) data
        
    Returns:
        Evidently Report object
    """
    column_mapping = get_column_mapping(for_classification=False)
    
    report = Report(metrics=[
        TargetDriftPreset(),
    ])
    
    report.run(
        reference_data=ref_df,
        current_data=prod_df,
        column_mapping=column_mapping
    )
    
    return report


def get_drift_summary(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> dict:
    """
    Get a quick summary of drift status.
    
    Returns:
        dict with drift statistics
    """
    column_mapping = get_column_mapping()
    
    report = Report(metrics=[
        DatasetDriftMetric(),
    ])
    
    report.run(
        reference_data=ref_df,
        current_data=prod_df,
        column_mapping=column_mapping
    )
    
    # Extract results
    result = report.as_dict()
    metrics = result.get('metrics', [])
    
    if metrics:
        drift_metric = metrics[0].get('result', {})
        return {
            'dataset_drift': drift_metric.get('dataset_drift', False),
            'drift_share': drift_metric.get('drift_share', 0.0),
            'number_of_drifted_columns': drift_metric.get('number_of_drifted_columns', 0),
            'number_of_columns': drift_metric.get('number_of_columns', 0),
        }
    
    return {'dataset_drift': None, 'drift_share': 0.0}


def save_report_html(report: Report, output_path: str):
    """Save report as HTML file."""
    report.save_html(output_path)


if __name__ == "__main__":
    # Quick test
    print("Loading data...")
    ref_df, prod_df = load_data(keep_prediction=True)
    
    print(f"Reference data: {len(ref_df)} samples")
    
    if prod_df is not None and len(prod_df) > 0:
        print(f"Production data: {len(prod_df)} samples")
        
        # Drift Analysis
        print("\n" + "="*60)
        print("DATA DRIFT ANALYSIS")
        print("="*60)
        
        # Load without prediction for drift
        ref_df_drift, prod_df_drift = load_data(keep_prediction=False)
        summary = get_drift_summary(ref_df_drift, prod_df_drift)
        print(f"Dataset drift detected: {summary['dataset_drift']}")
        print(f"Drift share: {summary['drift_share']:.2%}")
        print(f"Drifted columns: {summary['number_of_drifted_columns']}/{summary['number_of_columns']}")
        
        # Classification Metrics
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS (Model Health)")
        print("="*60)
        
        if 'prediction' in prod_df.columns:
            metrics = get_classification_metrics(prod_df)
            if metrics:
                print(f"Sample count: {metrics['sample_count']}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
                print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
                print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        else:
            print("No prediction column found - cannot calculate classification metrics")
        
        # Save reports
        print("\nGenerating reports...")
        output_dir = os.path.dirname(__file__)
        
        # Drift report
        report = generate_data_drift_report(ref_df_drift, prod_df_drift)
        output_path = os.path.join(output_dir, 'drift_report.html')
        save_report_html(report, output_path)
        print(f"Drift report saved to: {output_path}")
        
        # Classification report
        if 'prediction' in prod_df.columns:
            class_report = generate_classification_report(prod_df)
            class_output_path = os.path.join(output_dir, 'classification_report.html')
            save_report_html(class_report, class_output_path)
            print(f"Classification report saved to: {class_output_path}")
    else:
        print("No production data yet. Submit some feedback first!")
