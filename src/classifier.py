"""
Orchestrator utama: Hybrid LLM (Gemini) + Embedding (MiniLM/FAISS).
"""
import streamlit as st
import pandas as pd
from src.llm_handler import extract_surat_inti, classify_primary_secondary
from src.embedding_handler import encode_text, search_by_parent
from src.metadata_handler import get_code_info, get_children
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
            # Siapkan daftar kode untuk prompt
            level1_codes = df[df['level'] == 1][['kode', 'uraian']].to_dict('records')
            level2_codes = df[df['level'] == 2][['kode', 'uraian']].to_dict('records')
            code_list = "PRIMER:\n" + "\n".join(
                [f"- {c['kode']}: {c['uraian']}" for c in level1_codes]
            )
            code_list += "\n\nSEKUNDER (contoh 40 pertama):\n" + "\n".join(
                [f"- {c['kode']}: {c['uraian']}" for c in level2_codes[:40]]
            )
            ps = classify_primary_secondary(inti_surat, code_list)
            results['primer'] = ps.get('primer')
            results['sekunder'] = ps.get('sekunder')
        except Exception as e:
            results['error'] = f"Gagal primer/sekunder: {e}"
            return results

    sekunder = results['sekunder']
    if not sekunder:
        results['error'] = "Kode sekunder tidak ditemukan."
        return results

    # 3. TERSIER & KUARTIER via MiniLM + FAISS
    with st.spinner("🔍 Mencari kode tersier & kuartier..."):
        try:
            query_vec = encode_text(inti_surat)
            all_candidates = []

            # Cari tersier di bawah sekunder
            tersier_list = search_by_parent(df, query_vec, sekunder, level=3, top_k=5)
            if tersier_list:
                # Untuk setiap tersier, cari kuartier-nya
                for t in tersier_list[:3]:
                    all_candidates.append(t)  # simpan tersier sebagai kandidat
                    kuartier_list = search_by_parent(df, query_vec, t['kode'], level=4, top_k=3)
                    for k in kuartier_list:
                        k['tersier_parent'] = t['kode']
                        all_candidates.append(k)
            else:
                # Jika tidak ada tersier, fallback ke anak level 3? Seharusnya ada.
                # Coba ambil semua level 3 anak sekunder tanpa filter similarity
                children = get_children(df, sekunder, level=3)
                for _, row in children.head(5).iterrows():
                    all_candidates.append({
                        'kode': row['kode'],
                        'uraian': row['uraian'],
                        'penjelasan': row.get('penjelasan', ''),
                        'level': row['level'],
                        'similarity': 0.5  # nilai default rendah
                    })

            # Urutkan seluruh kandidat (kuartier diutamakan, tapi campur)
            # Kuartier diberi bobot lebih tinggi secara implisit karena biasanya memiliki similarity lebih tinggi
            all_candidates.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)

            # Ambil top-3, pastikan minimal satu kuartier jika ada
            final = []
            kuartier_added = 0
            for c in all_candidates:
                if c['level'] == 4 and kuartier_added < 2:
                    final.append(c)
                    kuartier_added += 1
                elif c['level'] != 4 and len(final) < 3:
                    final.append(c)
                if len(final) >= 3:
                    break
            # Jika belum 3, tambahkan dari sisa
            if len(final) < 3:
                for c in all_candidates:
                    if c not in final:
                        final.append(c)
                    if len(final) >= 3:
                        break

            results['final_recommendations'] = final[:TOP_K_OUTPUT]

        except Exception as e:
            results['error'] = f"Gagal cari tersier/kuartier: {e}"
            return results

    # 4. Fallback terakhir: jika masih kosong, rekomendasikan sekunder saja
    if not results['final_recommendations']:
        info = get_code_info(df, sekunder)
        if info:
            info['similarity'] = 0.5
            results['final_recommendations'] = [info]
        else:
            results['error'] = "Data sekunder tidak ditemukan di metadata."

    return results
