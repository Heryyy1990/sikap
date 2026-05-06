"""
SIKAP - Sistem Klasifikasi Arsip Pintar
Main Streamlit Application
"""
import streamlit as st
import time
from src.metadata_handler import load_metadata
from src.classifier import run_classification_pipeline
from src.ui_components import (
    render_header, render_input_area, render_inti_surat,
    render_recommendations, render_sidebar_info
)

# =====================================================================
# Page Config & Header
# =====================================================================
render_header()

# =====================================================================
# Load Data (Cached)
# =====================================================================
@st.cache_resource(show_spinner="Memuat metadata...")
def load_data():
    df = load_metadata()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat metadata: {e}")
    st.stop()

# =====================================================================
# Sidebar
# =====================================================================
render_sidebar_info(df)

# =====================================================================
# Main UI
# =====================================================================
user_input = render_input_area()
process_btn = st.button("🚀 Proses Klasifikasi", type="primary", use_container_width=True)

if process_btn and user_input.strip():
    # Reset state
    if 'results' in st.session_state:
        del st.session_state['results']
    
    start_time = time.time()
    
    # Jalankan pipeline
    results = run_classification_pipeline(user_input.strip(), df)
    
    elapsed = time.time() - start_time
    
    # Simpan ke session state
    st.session_state['results'] = results
    st.session_state['elapsed'] = elapsed

# Tampilkan hasil jika ada di session state
if 'results' in st.session_state:
    results = st.session_state['results']
    elapsed = st.session_state.get('elapsed', 0)
    
    if results.get('error'):
        st.error(f"❌ Error: {results['error']}")
    else:
        # Inti Surat
        if results.get('inti_surat'):
            render_inti_surat(results['inti_surat'])
        
        # Rekomendasi
        render_recommendations(
            results.get('final_recommendations', []),
            df,
            results.get('primer'),
            results.get('sekunder'),
            results.get('gemini_verification')
        )
        
        st.caption(f"⏱️ Waktu proses: {elapsed:.2f} detik")
