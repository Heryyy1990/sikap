"""
Orchestrator utama: menggabungkan LLM dan Embedding dalam pipeline klasifikasi.
"""
import streamlit as st
import pandas as pd
from src.llm_handler import extract_surat_inti, classify_primary_secondary, verify_final_code
from src.embedding_handler import encode_text, search_by_parent, load_faiss_index
from src.metadata_handler import get_code_info, get_children, get_code_tree_html
from src.config import TOP_K_OUTPUT, SIMILARITY_THRESHOLD

def run_classification_pipeline(teks_surat: str, df: pd.DataFrame) -> dict:
    """
    Menjalankan pipeline klasifikasi lengkap.
    
    Returns:
        dict dengan semua hasil untuk ditampilkan di UI.
    """
    results = {
        'inti_surat': '',
        'primer': None,
        'sekunder': None,
        'tersier_candidates': [],
        'kuartier_candidates': [],
        'final_recommendations': [],
        'gemini_verification': None,
        'error': None
    }
    
    # =====================================================================
    # TAHAP 1: Ekstrak Inti Surat (Gemini)
    # =====================================================================
    with st.spinner("🧠 Mengekstrak inti surat..."):
        try:
            inti_surat = extract_surat_inti(teks_surat)
            results['inti_surat'] = inti_surat
        except Exception as e:
            results['error'] = f"Gagal ekstrak inti: {str(e)}"
            return results
    
    # =====================================================================
    # TAHAP 2: Klasifikasi Primer + Sekunder (Gemini)
    # =====================================================================
    with st.spinner("📂 Menentukan kode primer & sekunder..."):
        try:
            # Siapkan daftar kode level 1 & 2 untuk prompt
            level1_codes = df[df['level'] == 1][['kode', 'uraian']].to_dict('records')
            level2_summary = df[df['level'] == 2][['kode', 'uraian']].to_dict('records')
            
            code_list = "PRIMER (level 1):\n"
            for c in level1_codes:
                code_list += f"- {c['kode']}: {c['uraian']}\n"
            code_list += "\nSEKUNDER (level 2) yang tersedia:\n"
            for c in level2_summary[:50]:  # Batasi agar prompt tidak terlalu panjang
                code_list += f"- {c['kode']}: {c['uraian']}\n"
            
            ps_result = classify_primary_secondary(inti_surat, code_list)
            results['primer'] = ps_result.get('primer')
            results['sekunder'] = ps_result.get('sekunder')
            results['alasan_ps'] = ps_result.get('alasan', '')
        except Exception as e:
            results['error'] = f"Gagal klasifikasi primer/sekunder: {str(e)}"
            return results
    
    # =====================================================================
    # TAHAP 3: Cari Tersier (FAISS + MiniLM)
    # =====================================================================
    with st.spinner("🔍 Mencari kode tersier..."):
        try:
            query_vec = encode_text(inti_surat)
            tersier_results = search_by_parent(
                df, query_vec, 
                parent_code=results['sekunder'], 
                level=3, 
                top_k=5
            )
            results['tersier_candidates'] = tersier_results
        except Exception as e:
            results['error'] = f"Gagal cari tersier: {str(e)}"
            return results
    
    # =====================================================================
    # TAHAP 4: Cari Kuartier (FAISS + MiniLM) dari Tersier terbaik
    # =====================================================================
    with st.spinner("🎯 Mencari kode kuartier..."):
        try:
            kuartier_all = []
            # Untuk setiap kandidat tersier, cari kuartier-nya
            for tersier in tersier_results[:3]:  # Top-3 tersier
                kuartier = search_by_parent(
                    df, query_vec,
                    parent_code=tersier['kode'],
                    level=4,
                    top_k=3
                )
                for k in kuartier:
                    k['tersier_parent'] = tersier['kode']
                kuartier_all.extend(kuartier)
            
            # Urutkan kuartier berdasarkan similarity
            kuartier_all.sort(key=lambda x: x['similarity'], reverse=True)
            results['kuartier_candidates'] = kuartier_all
        except Exception as e:
            # Fallback: tidak ada kuartier, gunakan tersier
            results['kuartier_candidates'] = []
    
    # =====================================================================
    # TAHAP 5: Susun Rekomendasi & Verifikasi
    # =====================================================================
    with st.spinner("✅ Menyusun rekomendasi akhir..."):
        final_candidates = []
        
        if results['kuartier_candidates']:
            # Prioritaskan kuartier dengan similarity tinggi
            high_sim = [k for k in results['kuartier_candidates'] if k['similarity'] >= SIMILARITY_THRESHOLD]
            if len(high_sim) >= TOP_K_OUTPUT:
                final_candidates = high_sim[:TOP_K_OUTPUT]
            else:
                # Campur kuartier + tersier
                final_candidates = high_sim
                remaining = TOP_K_OUTPUT - len(final_candidates)
                for t in results['tersier_candidates']:
                    if remaining <= 0:
                        break
                    if t not in final_candidates:
                        final_candidates.append(t)
                        remaining -= 1
        else:
            # Fallback ke tersier
            final_candidates = results['tersier_candidates'][:TOP_K_OUTPUT]
        
        results['final_recommendations'] = final_candidates
    
    # =====================================================================
    # TAHAP 6: Verifikasi Gemini (opsional, jika confidence rendah)
    # =====================================================================
    if results['final_recommendations'] and len(results['final_recommendations']) > 0:
        top_sim = results['final_recommendations'][0]['similarity']
        if top_sim < 0.60:  # Threshold untuk trigger verifikasi
            with st.spinner("🤖 Gemini memverifikasi hasil..."):
                try:
                    # Siapkan kandidat untuk prompt
                    cand_text = ""
                    for i, c in enumerate(results['final_recommendations'][:3]):
                        info = get_code_info(df, c['kode'])
                        cand_text += f"\n{i+1}) KODE: {c['kode']}\n   URAIAN: {info['uraian'] if info else '?'}\n   PENJELASAN: {info['penjelasan'][:200] if info else '?'}\n"
                    
                    verification = verify_final_code(inti_surat, cand_text)
                    results['gemini_verification'] = verification
                except Exception:
                    pass  # Verifikasi gagal tidak masalah
    
    return results
