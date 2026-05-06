"""
Orchestrator utama: Hybrid LLM (Gemini) + Embedding (MiniLM/FAISS).
"""
import streamlit as st
import pandas as pd
from src.llm_handler import (
    extract_surat_inti,
    classify_primary_secondary,
)
from src.embedding_handler import encode_text, search_by_parent
from src.metadata_handler import get_code_info, get_children, get_code_tree_html
from src.config import TOP_K_OUTPUT, SIMILARITY_THRESHOLD


def run_classification_pipeline(teks_surat: str, df: pd.DataFrame) -> dict:
    results = {
        'inti_surat': '',
        'primer': None,
        'sekunder': None,
        'final_recommendations': [],
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

    # 2. PRIMER + SEKUNDER (Gemini)
    with st.spinner("📂 Menentukan primer & sekunder..."):
        try:
            level1_codes = df[df['level'] == 1][['kode', 'uraian']].to_dict('records')
            level2_codes = df[df['level'] == 2][['kode', 'uraian']].to_dict('records')
            code_list = "PRIMER:\n" + "\n".join([f"- {c['kode']}: {c['uraian']}" for c in level1_codes])
            code_list += "\n\nSEKUNDER (30 pertama):\n" + "\n".join([f"- {c['kode']}: {c['uraian']}" for c in level2_codes[:30]])
            ps = classify_primary_secondary(inti_surat, code_list)
            results['primer'] = ps.get('primer')
            results['sekunder'] = ps.get('sekunder')
        except Exception as e:
            results['error'] = f"Gagal primer/sekunder: {e}"
            return results

    if not results['sekunder']:
        results['error'] = "Sekunder tidak ditemukan"
        return results

    # 3. TERSIER & KUARTIER (MiniLM + FAISS)
    with st.spinner("🔍 Mencari kode tersier & kuartier..."):
        try:
            query_vec = encode_text(inti_surat)
            all_candidates = []

            # Cari tersier di bawah sekunder
            tersier = search_by_parent(df, query_vec, results['sekunder'], level=3, top_k=5)
            for t in tersier:
                all_candidates.append(t)
                # Cari kuartier di bawah tersier ini
                kuartier = search_by_parent(df, query_vec, t['kode'], level=4, top_k=3)
                for k in kuartier:
                    k['tersier_parent'] = t['kode']
                    all_candidates.append(k)

            # Urutkan berdasarkan similarity
            all_candidates.sort(key=lambda x: x['similarity'], reverse=True)
            results['final_recommendations'] = all_candidates[:TOP_K_OUTPUT]

        except Exception as e:
            results['error'] = f"Gagal cari tersier/kuartier: {e}"
            return results

    # 4. Fallback jika kosong
    if not results['final_recommendations']:
        results['final_recommendations'] = [get_code_info(df, results['sekunder'])]

    return results
