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
    
    # Tampilkan primer & sekunder
    if primer and sekunder:
        p_info = get_code_info(df, primer)
        s_info = get_code_info(df, sekunder)
        col_p, col_s = st.columns(2)
        with col_p:
            st.metric("Kode Primer", primer, delta=p_info['uraian'] if p_info else "")
        with col_s:
            st.metric("Kode Sekunder", sekunder, delta=s_info['uraian'] if s_info else "")
    
    # Verifikasi (jika ada)
    if verification and verification.get('kode_terpilih'):
        st.success(f"✅ Gemini memverifikasi: **{verification['kode_terpilih']}** (confidence: {verification.get('confidence', '?')}%)")
        if verification.get('alasan'):
            st.caption(f"*Alasan:* {verification['alasan']}")
    
    st.markdown("---")
    st.markdown("### 🏆 Top 3 Rekomendasi")
    
    if not recommendations:
        st.warning("Tidak ada rekomendasi yang memenuhi threshold.")
        return
    
    cols = st.columns(len(recommendations))
    
    for i, col in enumerate(cols):
        rec = recommendations[i]
        # Ambil nilai similarity, fallback ke 0.0
        sim_value = float(rec.get('similarity', 0.0))
        
        # Warna border berdasarkan peringkat
        border_color = ["#4CAF50", "#2196F3", "#FF9800"][i] if i < 3 else "#9E9E9E"
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "📌"
        
        with col:
            st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 12px; padding: 16px; height: 100%;">
                <h2 style="margin: 0 0 8px 0;">{rank_emoji} #{i+1}</h2>
                <h3 style="color: {border_color}; margin: 0 0 12px 0; font-family: monospace; font-size: 1.2em;">{rec.get('kode', '?')}</h3>
                <p style="font-weight: bold; margin: 0 0 8px 0;">{rec.get('uraian', '?')}</p>
                <div style="font-size: 0.85em; color: #666; margin-bottom: 12px;">{str(rec.get('penjelasan', ''))[:150]}...</div>
                <div style="background: #f0f0f0; border-radius: 8px; padding: 4px 12px;">
                    Similarity: <b>{sim_value:.2%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tree hierarki
    with st.expander("🌳 Lihat Hierarki Kode", expanded=False):
        for rec in recommendations[:3]:
            st.markdown(f"**{rec.get('kode','?')}** — {rec.get('uraian','?')}")
            tree_html = get_code_tree_html(df, rec.get('kode',''))
            if tree_html:
                st.markdown(tree_html, unsafe_allow_html=True)
            st.markdown("---")

def render_sidebar_info(df: pd.DataFrame):
    """Render informasi di sidebar."""
    with st.sidebar:
        st.markdown("## ℹ️ Tentang SIKAP")
        st.markdown(f"""
        **SIKAP** adalah sistem klasifikasi arsip cerdas yang menggabungkan:
        - 🧠 **Gemini Flash** untuk reasoning
        - 🔢 **MiniLM + FAISS** untuk vector search
        - 🌳 **Hierarchical Classification** untuk akurasi tinggi
        
        **Dataset:** {len(df)} kode klasifikasi
        
        **Metode:** Hybrid LLM + Embedding
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Statistik Dataset")
        for level in [1, 2, 3, 4]:
            count = len(df[df['level'] == level])
            st.metric(f"Level {level}", f"{count} kode")
