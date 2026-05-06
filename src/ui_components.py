"""
Komponen UI untuk Streamlit — tampilan hasil yang profesional.
"""
import streamlit as st
import pandas as pd
from src.metadata_handler import get_code_tree_html, get_code_info

def render_header():
    """Render judul dan deskripsi aplikasi."""
    st.set_page_config(
        page_title="SIKAP - Sistem Klasifikasi Arsip Pintar",
        page_icon="📁",
        layout="wide",
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📁 SIKAP")
        st.caption("**S**istem **I**ntelijen **K**lasifikasi **A**rsip **P**emerintahan")
    with col2:
        st.markdown("---")
        st.markdown("⚡ *Hybrid LLM + Embedding*")

def render_input_area():
    """Render area input surat."""
    return st.text_area(
        "📝 Masukkan Isi Surat / Perihal / Uraian",
        placeholder="Contoh: Permohonan pencairan dana BOS untuk pembelian laptop guru...",
        height=150,
    )

def render_inti_surat(inti: str):
    """Render hasil ekstraksi inti surat."""
    st.markdown("### 🧠 Inti Surat")
    st.info(inti)

def render_recommendations(recommendations: list, df: pd.DataFrame, 
                          primer: str, sekunder: str, verification: dict = None):
    """Render 3 rekomendasi terbaik dengan detail."""
    st.markdown("---")
    st.markdown("### 📊 Hasil Klasifikasi")
    
    # Tampilkan primer & sekunder yang terpilih
    if primer and sekunder:
        p_info = get_code_info(df, primer)
        s_info = get_code_info(df, sekunder)
        col_p, col_s = st.columns(2)
        with col_p:
            st.metric("Kode Primer", primer, delta=p_info['uraian'] if p_info else "")
        with col_s:
            st.metric("Kode Sekunder", sekunder, delta=s_info['uraian'] if s_info else "")
    
    # Verifikasi Gemini
    if verification and verification.get('kode_terpilih'):
        st.success(f"✅ Gemini memverifikasi: **{verification['kode_terpilih']}** (confidence: {verification['confidence']}%)")
        if verification.get('alasan'):
            st.caption(f"*Alasan:* {verification['alasan']}")
    
    st.markdown("---")
    st.markdown("### 🏆 Top 3 Rekomendasi")
    
    if not recommendations:
        st.warning("Tidak ada rekomendasi yang memenuhi threshold.")
        return
    
    cols = st.columns(len(recommendations))
    
    for i, (col, rec) in enumerate(zip(cols, recommendations)):
        with col:
            # Determine border color based on rank
            border_color = ["#4CAF50", "#2196F3", "#FF9800"][i] if i < 3 else "#9E9E9E"
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "📌"
            
            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 12px; padding: 16px; height: 100%;">
                <h2 style="margin: 0 0 8px 0;">{rank_emoji} #{i+1}</h2>
                <h3 style="color: {border_color}; margin: 0 0 12px 0; font-family: monospace; font-size: 1.2em;">{rec['kode']}</h3>
                <p style="font-weight: bold; margin: 0 0 8px 0;">{rec['uraian']}</p>
                <div style="font-size: 0.85em; color: #666; margin-bottom: 12px;">{rec.get('penjelasan', '')[:150]}...</div>
                <div style="background: #f0f0f0; border-radius: 8px; padding: 4px 12px;">
                    Similarity: <b>{rec['similarity']:.2%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tampilkan detail tree
    with st.expander("🌳 Lihat Hierarki Kode", expanded=False):
        for rec in recommendations[:3]:
            st.markdown(f"**{rec['kode']}** — {rec['uraian']}")
            st.markdown(get_code_tree_html(df, rec['kode']), unsafe_allow_html=True)
            st.markdown("---")
    
    # Tampilkan penjelasan Gemini
    if verification and verification.get('alasan'):
        with st.expander("💬 Penjelasan Gemini", expanded=True):
            st.markdown(verification['alasan'])

def render_sidebar_info(df: pd.DataFrame):
    """Render informasi di sidebar."""
    with st.sidebar:
        st.markdown("## ℹ️ Tentang SIKAP")
        st.markdown("""
        **SIKAP** adalah sistem klasifikasi arsip cerdas yang menggabungkan:
        - 🧠 **Gemini Flash** untuk reasoning
        - 🔢 **Embedding + FAISS** untuk vector search
        - 🌳 **Hierarchical Classification** untuk akurasi tinggi
        
        **Dataset:** {} kode klasifikasi
        
        **Metode:** Hybrid LLM + Embedding
        """.format(len(df)))
        
        st.markdown("---")
        st.markdown("### 📋 Statistik Dataset")
        for level in [1, 2, 3, 4]:
            count = len(df[df['level'] == level])
            st.metric(f"Level {level}", f"{count} kode")
