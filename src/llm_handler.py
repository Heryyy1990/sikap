"""
Handler untuk semua panggilan ke Gemini API.
Menggunakan Google GenAI SDK dengan Gemini Flash 2.0.
"""
import json
import re
import streamlit as st
import google.generativeai as genai
from src.config import GEMINI_MODEL, PRIMER_CODES, PRIMER_NAMES

def _get_client():
    """Initialize Gemini client dengan API key dari secrets."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY tidak ditemukan di Streamlit Secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

@st.cache_data(ttl=3600, show_spinner=False)
def extract_surat_inti(teks_surat: str) -> str:
    """
    Tahap 1: Ekstrak inti makna surat.
    Output: 1-3 kalimat yang merangkum esensi surat.
    """
    model = _get_client()
    prompt = f"""Tugas: Baca surat dinas ini dan tuliskan INTI MAKNANYA dalam 2-3 kalimat singkat.
Fokus pada: SIAPA yang meminta, APA yang diminta, TUJUANNYA untuk apa.
Gunakan bahasa Indonesia formal. JANGAN menambah informasi yang tidak ada.

SURAT:
{teks_surat}

INTI SURAT:"""
    
    response = model.generate_content(prompt)
    return response.text.strip()

@st.cache_data(ttl=3600, show_spinner=False)
def classify_primary_secondary(inti_surat: str, df_codes: str) -> dict:
    """
    Tahap 2: Tentukan kode PRIMER (level 1) dan SEKUNDER (level 2).
    Input: inti surat + daftar kode primer & sekunder yang tersedia.
    Output: dict dengan 'primer' dan 'sekunder'.
    """
    model = _get_client()
    prompt = f"""Tugas: Klasifikasikan surat dinas ini ke kode PRIMER dan SEKUNDER.

INTI SURAT:
{inti_surat}

PILIH KODE PRIMER (level 1) dari daftar:
{df_codes}

PILIH KODE SEKUNDER (level 2) yang paling spesifik dari pilihan di atas.
Jika Anda ragu, pilih yang paling mendekati.

FORMAT JAWABAN (JSON saja, tanpa teks lain):
{{"primer": "XXX", "sekunder": "XXX.XX", "alasan": "..."}}"""

    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # Parse JSON dari response
    try:
        # Bersihkan markdown code blocks
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"primer": "000", "sekunder": "000.1", "alasan": "Gagal parse"}
    except json.JSONDecodeError:
        return {"primer": "000", "sekunder": "000.1", "alasan": "Gagal parse"}

@st.cache_data(ttl=3600, show_spinner=False)
def verify_final_code(inti_surat: str, candidates: str) -> dict:
    """
    Tahap Verifikasi: Gemini memeriksa apakah kode kuartier terbaik sudah tepat.
    Input: inti surat + 3 kandidat (dengan uraian & penjelasan).
    Output: dict dengan 'kode_terpilih', 'confidence', 'alasan'.
    """
    model = _get_client()
    prompt = f"""Tugas: Anda adalah arsiparis senior. Verifikasi kode klasifikasi terbaik untuk surat ini.

INTI SURAT:
{inti_surat}

KANDIDAT KODE:
{candidates}

Pilih SATU kode terbaik. Berikan CONFIDENCE score 0-100.
Format JSON:
{{"kode_terpilih": "XXX.XX", "confidence": 85, "alasan": "..."}}"""

    response = model.generate_content(prompt)
    text = response.text.strip()
    
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"kode_terpilih": None, "confidence": 0, "alasan": "Gagal parse"}
    except json.JSONDecodeError:
        return {"kode_terpilih": None, "confidence": 0, "alasan": "Gagal parse"}
