"""
Streamlit Web Application for Music Genre Classification.
Features:
- Audio file upload
- Waveform visualization
- Genre prediction with confidence
- Feedback submission for model retraining
"""

import streamlit as st
import requests
import numpy as np
import io

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

# ============================================================================
# MAIN APP
# ============================================================================

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
                import pandas as pd
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


if __name__ == "__main__":
    main()
