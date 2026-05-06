"""
Handler untuk memuat metadata, membangun hierarchical tree, 
dan menyediakan fungsi navigasi kode.
"""
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_resource
def load_metadata() -> pd.DataFrame:
    """Load metadata DataFrame dari file pickle."""
    with open(DATA_DIR / "metadata_sikap.pkl", "rb") as f:
        df = pickle.load(f)
    # PASTIKAN LEVEL BERTIPE INTEGER
    df['level'] = df['level'].astype(int)
    return df

@st.cache_resource
def build_hierarchy_tree(df: pd.DataFrame) -> dict:
    """
    Membangun tree navigasi: 
    tree[parent_code] = [list_of_child_codes]
    """
    tree = {}
    for _, row in df.iterrows():
        code = row['kode']
        level = row['level']
        parent = _get_parent_code(code, level)
        if parent not in tree:
            tree[parent] = []
        if code not in tree[parent]:
            tree[parent].append(code)
    return tree

def _get_parent_code(code: str, level: int) -> str:
    """Ekstrak parent code dari kode berdasarkan level."""
    if level == 1:
        return "ROOT"
    parts = code.split('.')
    if level == 2:
        return parts[0]  # Parent adalah primer
    elif level == 3:
        return '.'.join(parts[:2])  # Parent adalah sekunder
    elif level >= 4:
        return '.'.join(parts[:3])  # Parent adalah tersier
    return "ROOT"

def get_children(df: pd.DataFrame, parent_code: str, level: int) -> pd.DataFrame:
    """Dapatkan semua child code dari parent tertentu pada level tertentu."""
    return df[(df['level'] == level) & (df['kode'].str.startswith(parent_code))]

def get_code_info(df: pd.DataFrame, code: str) -> dict:
    """Dapatkan informasi lengkap untuk satu kode."""
    row = df[df['kode'] == code]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        'kode': row['kode'],
        'uraian': row['uraian'],
        'penjelasan': row['penjelasan'],
        'konteks': row.get('konteks', ''),
        'level': row['level']
    }

def get_code_lineage(df: pd.DataFrame, code: str) -> list:
    """
    Dapatkan garis keturunan kode dari primer → sekunder → tersier → kuartier.
    Returns list of dicts dengan info setiap level.
    """
    parts = code.split('.')
    lineage = []
    # Primer
    primer_code = parts[0]
    lineage.append(get_code_info(df, primer_code))
    # Sekunder (jika ada)
    if len(parts) >= 2:
        sec_code = '.'.join(parts[:2])
        lineage.append(get_code_info(df, sec_code))
    # Tersier (jika ada)
    if len(parts) >= 3:
        ter_code = '.'.join(parts[:3])
        lineage.append(get_code_info(df, ter_code))
    # Kuartier (jika ada)
    if len(parts) >= 4:
        lineage.append(get_code_info(df, code))
    return [l for l in lineage if l is not None]

def get_code_tree_html(df: pd.DataFrame, code: str) -> str:
    """Generate HTML untuk menampilkan hierarchical tree."""
    lineage = get_code_lineage(df, code)
    if not lineage:
        return ""
    
    html = '<div style="font-family: monospace; line-height: 1.8; padding: 10px;">'
    for i, node in enumerate(lineage):
        indent = "&nbsp;&nbsp;" * i
        connector = "└─ " if i > 0 else ""
        html += f'{indent}{connector}<b>{node["kode"]}</b> — {node["uraian"]}<br>'
    html += '</div>'
    return html
