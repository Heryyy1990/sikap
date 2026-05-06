"""
Orchestrator utama: pipeline full Gemini (tanpa FAISS/embedding).
"""
import streamlit as st
import pandas as pd
from src.llm_handler import (
    extract_surat_inti,
    classify_primary_secondary,
    pick_best_level,
    verify_final_code,
)
from src.metadata_handler import get_code_info, get_children, get_code_tree_html
from src.config import TOP_K_OUTPUT, SIMILARITY_THRESHOLD


def run_classification_pipeline(teks_surat: str, df: pd.DataFrame) -> dict:
    results = {
        'inti_surat': '',
        'primer': None,
        'sekunder': None,
        'tersier_terpilih': None,
        'kuartier_terpilih': None,
        'final_recommendations': [],
        'alasan_ps': '',
        'alasan_tersier': '',
        'alasan_kuartier': '',
        'error': None,
    }

    # 1. INTI SURAT
    with st.spinner("🧠 Mengekstrak inti surat..."):
        try:
            inti_surat = extract_surat_inti(teks_surat)
            results['inti_surat'] = inti_surat
        except Exception as e:
            results['error'] = f"Gagal ekstrak inti: {e}"
            return results

    # 2. PRIMER + SEKUNDER
    with st.spinner("📂 Menentukan primer & sekunder..."):
        try:
            level1_codes = df[df['level'] == 1][['kode', 'uraian']].to_dict('records')
            level2_codes = df[df['level'] == 2][['kode', 'uraian']].to_dict('records')
            code_list = "PRIMER:\n" + "\n".join([f"- {c['kode']}: {c['uraian']}" for c in level1_codes])
            code_list += "\n\nSEKUNDER (contoh 30 pertama):\n" + "\n".join([f"- {c['kode']}: {c['uraian']}" for c in level2_codes[:30]])
            ps = classify_primary_secondary(inti_surat, code_list)
            results['primer'] = ps.get('primer')
            results['sekunder'] = ps.get('sekunder')
            results['alasan_ps'] = ps.get('alasan', '')
        except Exception as e:
            results['error'] = f"Gagal primer/sekunder: {e}"
            return results

    sekunder_code = results['sekunder']
    if not sekunder_code:
        results['error'] = "Sekunder tidak ditemukan"
        return results

    # 3. PILIH TERSIER (Level 3)
    with st.spinner("🔍 Memilih kode tersier..."):
        children_l3 = get_children(df, sekunder_code, level=3)
        if not children_l3.empty:
            candidates_l3 = children_l3[['kode','uraian','penjelasan']].to_dict('records')
            try:
                ter_pick = pick_best_level(inti_surat, candidates_l3, "Tersier")
                if ter_pick.get('kode'):
                    results['tersier_terpilih'] = ter_pick['kode']
                    results['alasan_tersier'] = ter_pick.get('alasan', '')
            except Exception as e:
                results['error'] = f"Gagal pilih tersier: {e}"
                # Fallback: gunakan tersier pertama
                results['tersier_terpilih'] = children_l3.iloc[0]['kode']

    tersier_code = results['tersier_terpilih']
    if not tersier_code:
        # Fallback: langsung gunakan sekunder sebagai rekomendasi
        results['final_recommendations'] = [get_code_info(df, sekunder_code)]
        return results

    # 4. PILIH KUARTIER (Level 4)
    kuartier_terpilih = None
    children_l4 = get_children(df, tersier_code, level=4)
    if not children_l4.empty:
        with st.spinner("🎯 Memilih kode kuartier..."):
            candidates_l4 = children_l4[['kode','uraian','penjelasan']].to_dict('records')
            try:
                kua_pick = pick_best_level(inti_surat, candidates_l4, "Kuartier")
                if kua_pick.get('kode'):
                    kuartier_terpilih = kua_pick['kode']
                    results['alasan_kuartier'] = kua_pick.get('alasan', '')
            except Exception:
                pass

    # 5. SUSUN REKOMENDASI
    final_list = []
    
    def add_info(kode, similarity=0.85):
        info = get_code_info(df, kode)
        if info:
            info['similarity'] = similarity
            final_list.append(info)
    
    if kuartier_terpilih:
        add_info(kuartier_terpilih, similarity=0.95)
        # Tambahkan 2 kuartier lain sebagai alternatif
        other_kuartier = children_l4[children_l4['kode'] != kuartier_terpilih]
        for _, row in other_kuartier.head(2).iterrows():
            add_info(row['kode'], similarity=0.80)
    else:
        # Fallback ke tersier
        add_info(tersier_code, similarity=0.90)
        other_tersier = children_l3[children_l3['kode'] != tersier_code]
        for _, row in other_tersier.head(2).iterrows():
            add_info(row['kode'], similarity=0.75)

    results['final_recommendations'] = final_list[:TOP_K_OUTPUT]
    return results
