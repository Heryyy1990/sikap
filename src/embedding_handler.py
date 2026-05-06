"""
Handler untuk embedding dan FAISS vector search.
Menggunakan all-MiniLM-L6-v2 (lokal) + FAISS index yang sudah ada.
"""
import numpy as np
import faiss
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, TOP_K_FAISS

DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Load model embedding all-MiniLM-L6-v2 (lokal)."""
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_faiss_index() -> faiss.Index:
    """Load FAISS index dari file."""
    index_path = DATA_DIR / "vector_sikap_minilm.faiss"
    if not index_path.exists():
        st.error(f"FAISS index tidak ditemukan di {index_path}")
        st.stop()
    return faiss.read_index(str(index_path))

def encode_text(text: str) -> np.ndarray:
    """Encode text ke embedding vector."""
    model = load_embedding_model()
    embedding = model.encode([text], normalize_embeddings=True)
    return embedding.astype(np.float32)

def search_similar(index: faiss.Index, query_vec: np.ndarray,
                   filter_indices: list = None, top_k: int = TOP_K_FAISS) -> tuple:
    """
    Cari kode terdekat dengan filter opsional.
    """
    if filter_indices is not None and len(filter_indices) > 0:
        # Cari lebih banyak lalu filter manual
        distances, indices = index.search(query_vec, top_k * 3)
        mask = np.isin(indices[0], filter_indices)
        filtered_dist = distances[0][mask][:top_k]
        filtered_idx = indices[0][mask][:top_k]
        return filtered_dist.reshape(1, -1), filtered_idx.reshape(1, -1)
    else:
        distances, indices = index.search(query_vec, top_k)
    return distances, indices

def search_by_parent(df, query_vec: np.ndarray, parent_code: str,
                     level: int, top_k: int = TOP_K_FAISS) -> list:
    """
    Cari kode di level tertentu yang merupakan child dari parent_code.
    """
    mask = (df['level'] == level) & (df['kode'].str.startswith(parent_code))
    subset_df = df[mask]

    if subset_df.empty:
        return []

    subset_indices = subset_df.index.tolist()
    index = load_faiss_index()
    distances, indices = search_similar(index, query_vec, subset_indices, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        similarity = 1.0 - float(dist)
        results.append({
            'kode': row['kode'],
            'uraian': row['uraian'],
            'penjelasan': row['penjelasan'],
            'level': row['level'],
            'similarity': max(0, similarity),
        })
    return results
