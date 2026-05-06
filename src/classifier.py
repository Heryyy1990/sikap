import streamlit as st
import pandas as pd
from src.llm_handler import extract_surat_inti, classify_primary_secondary
from src.embedding_handler import encode_text, search_by_parent
from src.metadata_handler import get_code_info, get_children
from src.config import TOP_K_OUTPUT

def run_classification_pipeline(teks_surat: str, df: pd.DataFrame) -> dict:
    res = {'inti_surat':'','primer':None,'sekunder':None,'final_recommendations':[],'error':None}

    # 1. Inti
    with st.spinner("🧠 Mengekstrak inti surat..."):
        try:
            res['inti_surat'] = extract_surat_inti(teks_surat)
        except Exception as e:
            res['error'] = f"Gagal inti: {e}"
            return res

    # 2. Primer + Sekunder
    with st.spinner("📂 Menentukan primer & sekunder..."):
        try:
            lev1 = df[df['level']==1][['kode','uraian']].to_dict('records')
            lev2 = df[df['level']==2][['kode','uraian']].to_dict('records')
            code_list = "PRIMER:\n" + "\n".join([f"- {c['kode']}: {c['uraian']}" for c in lev1])
            code_list += "\n\nSEKUNDER (contoh 40):\n" + "\n".join([f"- {c['kode']}: {c['uraian']}" for c in lev2[:40]])
            ps = classify_primary_secondary(res['inti_surat'], code_list)
            res['primer'] = ps.get('primer')
            res['sekunder'] = ps.get('sekunder')
        except Exception as e:
            res['error'] = f"Gagal primer/sekunder: {e}"
            return res

    sek = res['sekunder']
    if not sek:
        res['error'] = "Sekunder tidak ditemukan."
        return res

    # 3. Kandidat (FAISS lalu fallback metadata)
    with st.spinner("🔍 Mencari tersier & kuartier..."):
        try:
            vec = encode_text(res['inti_surat'])
            pool = []

            # FAISS
            t_faiss = search_by_parent(df, vec, sek, level=3, top_k=5)
            if t_faiss:
                for t in t_faiss[:3]:
                    pool.append(t)
                    k_faiss = search_by_parent(df, vec, t['kode'], level=4, top_k=3)
                    for k in k_faiss:
                        k['tersier_parent'] = t['kode']
                        pool.append(k)
            else:
                # Fallback lengkap dari metadata
                t_meta = get_children(df, sek, level=3)
                for _, row in t_meta.iterrows():
                    pool.append({'kode':row['kode'],'uraian':row['uraian'],
                                 'penjelasan':row.get('penjelasan',''),'level':3,'similarity':0.35})
                    k_meta = get_children(df, row['kode'], level=4)
                    for _, kr in k_meta.iterrows():
                        pool.append({'kode':kr['kode'],'uraian':kr['uraian'],
                                     'penjelasan':kr.get('penjelasan',''),'level':4,'similarity':0.45})

            # Sort & pilih (prioritas kuartier)
            kuar = sorted([p for p in pool if p['level']==4], key=lambda x:x.get('similarity',0), reverse=True)
            ters = sorted([p for p in pool if p['level']==3], key=lambda x:x.get('similarity',0), reverse=True)
            final = []
            final.extend(kuar[:2])
            for t in ters:
                if len(final)>=3: break
                final.append(t)
            if len(final)<3:
                final.extend(kuar[2:3])
            res['final_recommendations'] = final[:TOP_K_OUTPUT]

        except Exception as e:
            res['error'] = f"Gagal cari kandidat: {e}"
            return res

    # 4. Fallback absolut
    if not res['final_recommendations']:
        info = get_code_info(df, sek)
        if info:
            info['similarity'] = 0.2
            res['final_recommendations'] = [info]

    return res
