"""
Streamlit Web Application for Music Genre Classification.
Features:
- Audio file upload
- Waveform visualization
- Genre prediction with confidence
- Feedback submission for model retraining
- Data drift monitoring with Evidently
- Classification metrics tracking (F1, precision, recall, balanced accuracy)
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import io
import os
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    balanced_accuracy_score, accuracy_score
)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "http://serving-api:8080"  # Docker network URL

# Genre emoji mapping for visual appeal
GENRE_EMOJIS = {
    'blues': '🎸',
    'classical': '🎻',
    'country': '🤠',
    'disco': '🕺',
    'hiphop': '🎤',
    'jazz': '🎷',
    'metal': '🤘',
    'pop': '🎵',
    'reggae': '🌴',
    'rock': '🎸'
}

GENRE_COLORS = {
    'blues': '#1E90FF',
    'classical': '#FFD700',
    'country': '#8B4513',
    'disco': '#FF69B4',
    'hiphop': '#9400D3',
    'jazz': '#FFA500',
    'metal': '#2F4F4F',
    'pop': '#FF1493',
    'reggae': '#00FF00',
    'rock': '#DC143C'
}

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96e6a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .genre-label {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .confidence-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .feedback-section {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_genre(audio_bytes: bytes) -> dict:
    """Send audio to API for prediction."""
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.text}")


def submit_feedback(audio_bytes: bytes, prediction: str, actual_label: str) -> dict:
    """Submit feedback to API."""
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"prediction": prediction, "actual_label": actual_label}
    response = requests.post(f"{API_URL}/feedback", files=files, data=data, timeout=60)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.text}")


def create_waveform_plot(audio_bytes: bytes):
    """Create waveform visualization using librosa."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=30)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Waveform
    axes[0].set_facecolor('#1a1a2e')
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#4ecdc4')
    axes[0].set_title('Waveform', fontsize=14, fontweight='bold', color='white')
    axes[0].set_xlabel('')
    axes[0].tick_params(colors='white')
    
    # Mel spectrogram
    axes[1].set_facecolor('#1a1a2e')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axes[1], cmap='magma')
    axes[1].set_title('Mel Spectrogram', fontsize=14, fontweight='bold', color='white')
    axes[1].tick_params(colors='white')
    
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    
    return fig


def create_probability_chart(probabilities: dict):
    """Create horizontal bar chart for probabilities."""
    import matplotlib.pyplot as plt
    
    # Sort by probability
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    genres = list(sorted_probs.keys())
    probs = list(sorted_probs.values())
    colors = [GENRE_COLORS.get(g, '#808080') for g in genres]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    bars = ax.barh(genres, probs, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Probability', fontsize=12, color='white')
    ax.set_xlim(0, 1)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontsize=10, color='white')
    
    # Add emoji labels
    for i, genre in enumerate(genres):
        emoji = GENRE_EMOJIS.get(genre, '')
        ax.text(-0.02, i, emoji, va='center', ha='right', fontsize=14)
    
    plt.tight_layout()
    return fig


# Feature columns for drift detection
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


def load_drift_data(keep_prediction: bool = False):
    """Load reference and production data for drift analysis."""
    # In Docker, data is mounted at /data
    data_dir = "/data" if os.path.exists("/data") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    ref_path = os.path.join(data_dir, 'ref_data.csv')
    prod_path = os.path.join(data_dir, 'prod_data.csv')
    
    ref_df = None
    prod_df = None
    
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
    
    if os.path.exists(prod_path):
        prod_df = pd.read_csv(prod_path)
        if not keep_prediction and 'prediction' in prod_df.columns:
            prod_df = prod_df.drop(columns=['prediction'])
    
    return ref_df, prod_df


def get_classification_metrics_simple(prod_df: pd.DataFrame) -> dict:
    """Calculate classification metrics from production data."""
    if prod_df is None or 'prediction' not in prod_df.columns or 'label' not in prod_df.columns:
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
        'sample_count': len(prod_df),
    }


def render_drift_page():
    """Render the data drift monitoring and model health page."""
    st.header("📊 Model Monitoring Dashboard")
    
    st.write("""
    This page tracks:
    1. **Data Drift**: When production data differs from training data
    2. **Model Health**: Classification metrics (F1, precision, recall, balanced accuracy)
    """)
    
    # Load data - need prediction for classification metrics
    ref_df, prod_df_with_pred = load_drift_data(keep_prediction=True)
    ref_df, prod_df = load_drift_data(keep_prediction=False)  # For drift analysis
    
    if ref_df is None:
        st.error("❌ Reference data (ref_data.csv) not found!")
        return
    
    st.success(f"✅ Reference data loaded: **{len(ref_df)}** samples")
    
    if prod_df is None or len(prod_df) == 0:
        st.warning("⚠️ No production data yet. Submit some feedback to start monitoring!")
        st.info("Upload audio files, get predictions, and submit feedback to collect production data.")
        return
    
    st.success(f"✅ Production data loaded: **{len(prod_df)}** samples")
    
    # ==================== CLASSIFICATION METRICS ====================
    st.divider()
    st.subheader("🎯 Model Health - Classification Metrics")
    
    class_metrics = get_classification_metrics_simple(prod_df_with_pred)
    
    if class_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("F1 Score", f"{class_metrics['f1_weighted']:.3f}",
                      help="Harmonic mean of precision and recall (weighted)")
        with col2:
            st.metric("Balanced Accuracy", f"{class_metrics['balanced_accuracy']:.3f}",
                      help="Average recall across all classes")
        with col3:
            st.metric("Precision", f"{class_metrics['precision_weighted']:.3f}",
                      help="True positives / (true + false positives)")
        with col4:
            st.metric("Recall", f"{class_metrics['recall_weighted']:.3f}",
                      help="True positives / (true + false negatives)")
        
        # Show accuracy trend warning if applicable
        if class_metrics['accuracy'] < 0.7:
            st.warning(f"⚠️ Model accuracy ({class_metrics['accuracy']:.1%}) is below 70%. Consider retraining.")
        elif class_metrics['accuracy'] < 0.8:
            st.info(f"ℹ️ Model accuracy: {class_metrics['accuracy']:.1%}")
        else:
            st.success(f"✅ Model accuracy: {class_metrics['accuracy']:.1%}")
    else:
        st.info("Classification metrics will appear after feedback is submitted.")
    
    # ==================== DATA DRIFT ====================
    st.divider()
    
    # Try to use Evidently if available
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset
        from evidently import ColumnMapping
        from evidently.metrics import DatasetDriftMetric
        
        # Column mapping for drift
        column_mapping = ColumnMapping(
            target='label',
            numerical_features=FEATURE_COLUMNS
        )
        
        # Quick drift summary
        st.subheader("🔍 Data Drift Summary")
        
        with st.spinner("Analyzing drift..."):
            quick_report = Report(metrics=[DatasetDriftMetric()])
            quick_report.run(
                reference_data=ref_df,
                current_data=prod_df,
                column_mapping=column_mapping
            )
            
            result = quick_report.as_dict()
            metrics = result.get('metrics', [])
            
            if metrics:
                drift_result = metrics[0].get('result', {})
                drift_detected = drift_result.get('dataset_drift', False)
                drift_share = drift_result.get('drift_share', 0.0)
                n_drifted = drift_result.get('number_of_drifted_columns', 0)
                n_total = drift_result.get('number_of_columns', len(FEATURE_COLUMNS))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if drift_detected:
                        st.metric("Dataset Drift", "⚠️ DETECTED", delta="Review recommended")
                    else:
                        st.metric("Dataset Drift", "✅ None", delta="Model OK")
                
                with col2:
                    st.metric("Drift Share", f"{drift_share:.1%}", 
                              delta=f"{n_drifted}/{n_total} features")
                
                with col3:
                    st.metric("Production Samples", len(prod_df))
        
        st.divider()
        
        # Report generation buttons
        st.subheader("📈 Detailed Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Generate Drift Report", type="primary"):
                with st.spinner("Generating data drift report..."):
                    full_report = Report(metrics=[DataDriftPreset()])
                    full_report.run(
                        reference_data=ref_df,
                        current_data=prod_df,
                        column_mapping=column_mapping
                    )
                    report_html = full_report.get_html()
                    st.components.v1.html(report_html, height=800, scrolling=True)
        
        with col2:
            if 'prediction' in prod_df_with_pred.columns:
                if st.button("📊 Generate Classification Report", type="secondary"):
                    with st.spinner("Generating classification report..."):
                        # Column mapping for classification
                        class_column_mapping = ColumnMapping(
                            target='label',
                            prediction='prediction',
                            numerical_features=FEATURE_COLUMNS
                        )
                        class_report = Report(metrics=[ClassificationPreset()])
                        class_report.run(
                            reference_data=None,
                            current_data=prod_df_with_pred,
                            column_mapping=class_column_mapping
                        )
                        class_html = class_report.get_html()
                        st.components.v1.html(class_html, height=800, scrolling=True)
    
    except ImportError:
        st.warning("⚠️ Evidently library not installed. Showing basic statistics instead.")
        _render_basic_drift_stats(ref_df, prod_df)
    
    # Genre Distribution Comparison
    st.divider()
    st.subheader("🎵 Genre Distribution Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Reference Data**")
        if 'label' in ref_df.columns:
            ref_counts = ref_df['label'].value_counts()
            st.bar_chart(ref_counts)
    
    with col2:
        st.write("**Production Data**")
        if 'label' in prod_df.columns:
            prod_counts = prod_df['label'].value_counts()
            st.bar_chart(prod_counts)


def _render_basic_drift_stats(ref_df, prod_df):
    """Render basic drift statistics without Evidently."""
    st.write("**Feature Statistics Comparison:**")
    
    # Calculate mean differences for key features
    key_features = ['tempo', 'rms_mean', 'spectral_centroid_mean', 'chroma_stft_mean']
    
    comparison_data = []
    for feat in key_features:
        if feat in ref_df.columns and feat in prod_df.columns:
            ref_mean = ref_df[feat].mean()
            prod_mean = prod_df[feat].mean()
            diff_pct = ((prod_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
            comparison_data.append({
                'Feature': feat,
                'Reference Mean': f"{ref_mean:.4f}",
                'Production Mean': f"{prod_mean:.4f}",
                'Difference': f"{diff_pct:+.1f}%"
            })
    
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def render_prediction_page():
    """Render the main prediction page."""
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio")
        st.write("Upload a music file (WAV, MP3, OGG, FLAC)")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Supported formats: WAV, MP3, OGG, FLAC"
        )
        
        if uploaded_file is not None:
            # Store in session state
            audio_bytes = uploaded_file.read()
            st.session_state['audio_bytes'] = audio_bytes
            st.session_state['filename'] = uploaded_file.name
            
            # Play audio
            st.audio(audio_bytes, format='audio/wav')
            
            # Predict button
            if st.button("🔮 Classify Genre", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio..."):
                    try:
                        result = predict_genre(audio_bytes)
                        st.session_state['prediction'] = result
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("📊 Analysis")
        
        if 'audio_bytes' in st.session_state:
            try:
                # Show waveform
                with st.spinner("Generating visualization..."):
                    fig = create_waveform_plot(st.session_state['audio_bytes'])
                    st.pyplot(fig)
            except ImportError:
                st.info("Install librosa and matplotlib for visualization")
            except Exception as e:
                st.warning(f"Could not generate visualization: {e}")
    
    # Prediction Results
    if 'prediction' in st.session_state:
        result = st.session_state['prediction']
        
        st.divider()
        st.header("🎯 Prediction Result")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            genre = result['genre']
            confidence = result['confidence']
            emoji = GENRE_EMOJIS.get(genre, '🎵')
            
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 4rem;">{emoji}</div>
                <div class="genre-label">{genre.upper()}</div>
                <div class="confidence-label">{confidence:.1%} confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            # Probability chart
            try:
                fig = create_probability_chart(result['probabilities'])
                st.pyplot(fig)
            except Exception as e:
                # Fallback to simple bar chart
                probs_df = pd.DataFrame({
                    'Genre': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                st.bar_chart(probs_df.set_index('Genre'))
        
        # Feedback Section
        st.divider()
        st.header("📝 Feedback")
        st.write("Was the prediction correct? Help improve the model by providing feedback!")
        
        with st.form("feedback_form"):
            col_fb1, col_fb2 = st.columns([2, 1])
            
            with col_fb1:
                actual_genre = st.selectbox(
                    "What's the actual genre?",
                    options=list(GENRE_EMOJIS.keys()),
                    format_func=lambda x: f"{GENRE_EMOJIS[x]} {x.title()}",
                    index=list(GENRE_EMOJIS.keys()).index(result['genre'])
                )
            
            with col_fb2:
                submit_feedback_btn = st.form_submit_button(
                    "Submit Feedback",
                    type="secondary",
                    use_container_width=True
                )
            
            if submit_feedback_btn:
                if 'audio_bytes' in st.session_state:
                    with st.spinner("Submitting feedback..."):
                        try:
                            fb_result = submit_feedback(
                                st.session_state['audio_bytes'],
                                result['genre'],
                                actual_genre
                            )
                            
                            if fb_result['retrained']:
                                st.success(f"✅ Feedback recorded! Model was retrained with {fb_result['feedback_count']} samples.")
                            else:
                                st.success(f"✅ Feedback recorded! ({fb_result['feedback_count']} total samples)")
                        except Exception as e:
                            st.error(f"Error submitting feedback: {str(e)}")
                else:
                    st.warning("Please upload an audio file first")


def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses machine learning to classify music into 10 genres:
        - 🎸 Blues & Rock
        - 🎻 Classical
        - 🤠 Country
        - 🕺 Disco
        - 🎤 Hip-Hop
        - 🎷 Jazz
        - 🤘 Metal
        - 🎵 Pop
        - 🌴 Reggae
        """)
        
        st.divider()
        
        # API Status
        st.header("🔌 API Status")
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Unavailable")
            st.info("Make sure the API container is running:\n`docker compose -f serving/docker-compose.yml up`")
    
    # Tabs for navigation
    tab1, tab2 = st.tabs(["🎵 Classify Music", "📊 Data Drift"])
    
    with tab1:
        render_prediction_page()
    
    with tab2:
        render_drift_page()


if __name__ == "__main__":
    main()
