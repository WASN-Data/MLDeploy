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

# Genre emoji mapping for visual appeal (GTZAN - 9 genres, no reggae)
GENRE_EMOJIS = {
    'blues': '🎷',
    'classical': '🎻',
    'country': '🤠',
    'disco': '🕺',
    'hiphop': '🎤',
    'jazz': '🎺',
    'metal': '🤘',
    'pop': '🎵',
    'rock': '🎸'
}

GENRE_COLORS = {
    'blues': '#4169E1',
    'classical': '#8B4513',
    'country': '#DAA520',
    'disco': '#FF69B4',
    'hiphop': '#9400D3',
    'jazz': '#FF8C00',
    'metal': '#2F4F4F',
    'pop': '#FF1493',
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

# Custom CSS for styling - polished look with better fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .main-header {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .genre-label {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.01em;
    }
    
    .confidence-label {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Card-like sections */
    .stExpander {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Better buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Feedback success message styling */
    .feedback-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-weight: 500;
        text-align: center;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.5rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px !important;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
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
    
    # Create figure with three subplots for more visual impact
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # Waveform
    axes[0].set_facecolor('#1a1a2e')
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#4ecdc4', alpha=0.8)
    axes[0].set_title('Waveform', fontsize=14, fontweight='bold', color='white', pad=10)
    axes[0].set_xlabel('')
    axes[0].tick_params(colors='white', labelsize=9)
    for spine in axes[0].spines.values():
        spine.set_color('#333')
    
    # Mel spectrogram
    axes[1].set_facecolor('#1a1a2e')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axes[1], cmap='magma')
    axes[1].set_title('Mel Spectrogram', fontsize=14, fontweight='bold', color='white', pad=10)
    axes[1].tick_params(colors='white', labelsize=9)
    for spine in axes[1].spines.values():
        spine.set_color('#333')
    
    # Chromagram - shows pitch/harmony content (very colorful!)
    axes[2].set_facecolor('#1a1a2e')
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr, ax=axes[2], cmap='coolwarm')
    axes[2].set_title('Chromagram (Pitch Content)', fontsize=14, fontweight='bold', color='white', pad=10)
    axes[2].tick_params(colors='white', labelsize=9)
    for spine in axes[2].spines.values():
        spine.set_color('#333')
    
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout(pad=2.0)
    
    return fig


def create_probability_chart(probabilities: dict):
    """Create horizontal bar chart for probabilities with better formatting."""
    import matplotlib.pyplot as plt
    
    # Sort by probability (ascending for barh so highest is at top)
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1]))
    
    genres = list(sorted_probs.keys())
    probs = list(sorted_probs.values())
    colors = [GENRE_COLORS.get(g, '#808080') for g in genres]
    
    # Use plain genre names (emojis cause font issues in Docker)
    labels = [g.title() for g in genres]
    
    # Larger figure with more width for labels
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Create bars with rounded appearance
    bars = ax.barh(range(len(genres)), probs, color=colors, edgecolor='none', height=0.7)
    
    # Add subtle gradient effect to bars
    for bar in bars:
        bar.set_alpha(0.9)
    
    # Set y-ticks with genre labels
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(labels, fontsize=11, color='white', fontweight='500')
    
    # Adjust margins
    plt.subplots_adjust(left=0.22, right=0.95)
    
    # X-axis formatting
    ax.set_xlabel('Probability', fontsize=12, color='white', fontweight='500', labelpad=10)
    ax.set_xlim(0, 1.15)  # Extra space for percentage labels
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', left=False)  # No y-axis ticks
    
    # Clean spines
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, color='white')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        # Label position - inside if bar is big enough, outside otherwise
        if prob > 0.15:
            ax.text(prob - 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', ha='right', fontsize=10, 
                    color='white', fontweight='bold')
        else:
            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', ha='left', fontsize=10, 
                    color='white', fontweight='500')
    
    # Add padding on left for labels
    plt.subplots_adjust(left=0.20)
    
    return fig


def get_feature_columns():
    """Get feature column names (excluding label/prediction)."""
    return [
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


def get_common_feature_columns(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> list:
    """Get feature columns that exist in both reference and production data."""
    exclude_cols = {'label', 'prediction', 'target', 'timestamp', 'filename'}
    ref_cols = set(ref_df.columns) - exclude_cols
    prod_cols = set(prod_df.columns) - exclude_cols
    common = list(ref_cols & prod_cols)
    return sorted(common)


def load_drift_data(keep_prediction: bool = False):
    """Load reference and production data for drift analysis."""
    data_dir = "/data" if os.path.exists("/data") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    ref_path = os.path.join(data_dir, 'ref_data.csv')
    prod_path = os.path.join(data_dir, 'prod_data.csv')
    
    ref_df = None
    prod_df = None
    
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
    
    if os.path.exists(prod_path) and os.path.getsize(prod_path) > 0:
        try:
            prod_df = pd.read_csv(prod_path)
            if not keep_prediction and 'prediction' in prod_df.columns:
                prod_df = prod_df.drop(columns=['prediction'])
        except pd.errors.EmptyDataError:
            prod_df = None
    
    return ref_df, prod_df


def get_classification_metrics_simple(prod_df: pd.DataFrame) -> dict:
    """Calculate classification metrics from production data."""
    if prod_df is None or 'prediction' not in prod_df.columns or 'label' not in prod_df.columns:
        return None
    
    y_true = prod_df['label']
    y_pred = prod_df['prediction']
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'sample_count': len(prod_df),
    }


def render_drift_tab():
    """Render the data drift monitoring and model health tab."""
    st.header("📊 Model Monitoring Dashboard")
    
    st.write("""
    This page tracks:
    1. **Model Health**: Classification metrics (F1, precision, recall, balanced accuracy)
    2. **Data Drift**: When production data differs from training data
    """)
    
    # Load data
    ref_df, prod_df_with_pred = load_drift_data(keep_prediction=True)
    ref_df, prod_df = load_drift_data(keep_prediction=False)
    
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
    st.subheader("🔍 Data Drift Analysis")
    
    # Find common feature columns dynamically
    feature_columns = get_common_feature_columns(ref_df, prod_df)
    if len(feature_columns) == 0:
        st.error("❌ No common feature columns between reference and production data!")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reference Samples", len(ref_df))
    with col2:
        st.metric("Production Samples", len(prod_df))
    with col3:
        st.metric("Features Analyzed", len(feature_columns))
    
    st.divider()
    
    # Generate drift report button
    if st.button("🔍 Generate Drift Report", type="primary", use_container_width=True):
        with st.spinner("Analyzing data drift..."):
            try:
                from evidently.report import Report
                from evidently.metric_preset import DataDriftPreset
                
                # Prepare datasets
                ref_features = ref_df[feature_columns].copy()
                prod_features = prod_df[[c for c in feature_columns if c in prod_df.columns]].copy()
                
                # Generate drift report (Evidently 0.4.x API)
                drift_report = Report(metrics=[DataDriftPreset()])
                drift_report.run(reference_data=ref_features, current_data=prod_features)
                
                # Display results
                st.subheader("📊 Drift Analysis Results")
                
                # Get drift stats
                report_dict = drift_report.as_dict()
                
                # Display summary (0.4.x structure)
                metrics = report_dict.get("metrics", [])
                if metrics:
                    first_metric = metrics[0]
                    result = first_metric.get("result", {})
                    
                    drift_share = result.get("share_of_drifted_columns", result.get("drift_share", 0))
                    n_drifted = result.get("number_of_drifted_columns", 0)
                    n_columns = result.get("number_of_columns", len(feature_columns))
                    dataset_drift = result.get("dataset_drift", False)
                    
                    if dataset_drift or drift_share > 0.3:
                        st.error(f"⚠️ **Data Drift Detected!** ({n_drifted}/{n_columns} features drifted = {drift_share:.1%})")
                    else:
                        st.success(f"✅ **No Significant Drift** ({n_drifted}/{n_columns} features drifted = {drift_share:.1%})")
                    
                    # Show drifted columns
                    drift_by_columns = result.get("drift_by_columns", {})
                    drifted_cols = [col for col, info in drift_by_columns.items() if info.get("drift_detected", False)]
                    if drifted_cols:
                        st.write("**Drifted Features:**")
                        for feat in drifted_cols[:10]:
                            st.write(f"- {feat}")
                
                # Save and offer download
                data_dir = "/data" if os.path.exists("/data") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
                report_path = os.path.join(data_dir, "drift_report.html")
                drift_report.save_html(report_path)
                st.success(f"📄 Full report saved!")
                
                with open(report_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.download_button(
                    "⬇️ Download Full Report",
                    data=html_content,
                    file_name="drift_report.html",
                    mime="text/html"
                )
                
            except ImportError:
                st.error("❌ Evidently not installed. Run: `pip install evidently`")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Genre Distribution Comparison
    st.divider()
    st.subheader("🎵 Genre Distribution Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Reference Data (Training)**")
        if 'label' in ref_df.columns:
            ref_counts = ref_df['label'].value_counts()
            st.bar_chart(ref_counts)
    
    with col2:
        st.write("**Production Data (Feedback)**")
        if 'label' in prod_df.columns:
            prod_counts = prod_df['label'].value_counts()
            st.bar_chart(prod_counts)


def render_classify_tab():
    """Render the main classification tab."""
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
            # Read file bytes
            audio_bytes = uploaded_file.read()
            
            # Handle Streamlit's re-render behavior
            if len(audio_bytes) == 0 and 'audio_bytes' in st.session_state and st.session_state.get('filename') == uploaded_file.name:
                audio_bytes = st.session_state['audio_bytes']
            
            if len(audio_bytes) > 0:
                file_id = f"{uploaded_file.name}_{len(audio_bytes)}"
                
                if st.session_state.get('file_id') != file_id:
                    st.session_state['feedback_submitted'] = False
                    st.session_state.pop('prediction', None)
                    st.session_state['file_id'] = file_id
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
            try:
                fig = create_probability_chart(result['probabilities'])
                st.pyplot(fig)
            except Exception as e:
                probs_df = pd.DataFrame({
                    'Genre': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                st.bar_chart(probs_df.set_index('Genre'))
        
        # Feedback Section
        st.divider()
        st.header("📝 Feedback")
        st.write("Was the prediction correct? Help improve the model by providing feedback!")
        
        if 'feedback_submitted' not in st.session_state:
            st.session_state.feedback_submitted = False
            st.session_state.feedback_message = ""
            st.session_state.feedback_type = ""
        
        with st.form("feedback_form", clear_on_submit=False):
            col_fb1, col_fb2 = st.columns([3, 1])
            
            with col_fb1:
                actual_genre = st.selectbox(
                    "What's the actual genre?",
                    options=list(GENRE_EMOJIS.keys()),
                    format_func=lambda x: f"{GENRE_EMOJIS[x]} {x.title()}",
                    index=list(GENRE_EMOJIS.keys()).index(result['genre']) if result['genre'] in GENRE_EMOJIS else 0,
                    label_visibility="collapsed"
                )
            
            with col_fb2:
                submit_feedback_btn = st.form_submit_button(
                    "✓ Submit Feedback",
                    type="primary",
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
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_type = "retrained"
                            st.session_state.feedback_message = f"Model was retrained with {fb_result['feedback_count']} samples!"
                        else:
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_type = "success"
                            st.session_state.feedback_message = f"{fb_result['feedback_count']} total samples collected"
                    except Exception as e:
                        st.session_state.feedback_submitted = True
                        st.session_state.feedback_type = "error"
                        st.session_state.feedback_message = str(e)
            else:
                st.warning("Please upload an audio file first")
        
        # Display feedback result message
        if st.session_state.feedback_submitted:
            if st.session_state.feedback_type == "retrained":
                st.markdown(f'''
                    <div class="feedback-success">
                        🎉 {st.session_state.feedback_message}
                    </div>
                ''', unsafe_allow_html=True)
            elif st.session_state.feedback_type == "success":
                st.markdown(f'''
                    <div class="feedback-success">
                        ✅ Feedback recorded! {st.session_state.feedback_message}
                    </div>
                ''', unsafe_allow_html=True)
            elif st.session_state.feedback_type == "error":
                st.error(f"Error: {st.session_state.feedback_message}")


def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **ML-powered music classification**  
        Trained on GTZAN dataset (900 samples)
        """)
        
        st.markdown("**Supported Genres:**")
        # Display genres in a clean grid
        col1, col2, col3 = st.columns(3)
        genres_list = list(GENRE_EMOJIS.items())
        for i, (genre, emoji) in enumerate(genres_list):
            col = [col1, col2, col3][i % 3]
            with col:
                st.markdown(f"{emoji} {genre.title()}")
        
        st.divider()
        
        # API Status
        st.header("🔌 API Status")
        if check_api_health():
            st.success("✅ Connected")
        else:
            st.error("❌ API Unavailable")
            st.caption("Run: `docker compose -f serving/docker-compose.yml up`")
    
    # Tabs for navigation
    tab1, tab2 = st.tabs(["🎵 Classify Music", "📊 Model Monitoring"])
    
    with tab1:
        render_classify_tab()
    
    with tab2:
        render_drift_tab()


if __name__ == "__main__":
    main()
