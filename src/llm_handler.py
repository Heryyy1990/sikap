"""
Handler untuk semua panggilan ke Gemini API.
Menggunakan Google GenAI SDK dengan Gemini Flash 2.0.
"""
import json
import re
import time
import streamlit as st
import google.generativeai as genai
from src.config import GEMINI_MODEL

def _get_client():
    """Initialize Gemini client dengan API key dari secrets."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY tidak ditemukan di Streamlit Secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

def safe_generate(model, prompt, retries=3):
    """Panggil Gemini dengan retry dan backoff."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            if "429" in str(e):
                wait = 15 * (attempt + 1)
                st.warning(f"Rate limit. Menunggu {wait} detik...")
                time.sleep(wait)
            else:
                raise e
    return None

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
    
    response = safe_generate(model, prompt)
    if response:
        return response.text.strip()
    return "Gagal mengekstrak inti surat."

@st.cache_data(ttl=3600, show_spinner=False)
def classify_primary_secondary(inti_surat: str, df_codes: str) -> dict:
    """
    Tahap 2: Tentukan kode PRIMER (level 1) dan SEKUNDER (level 2).
    Prompt dipertegas agar fokus pada TUJUAN UTAMA, bukan kata kunci sekunder.
    """
    model = _get_client()
    prompt = f"""Tugas: Klasifikasikan surat dinas ini ke kode PRIMER (level 1) dan SEKUNDER (level 2) yang PALING TEPAT berdasarkan TUJUAN UTAMA surat.

INTI SURAT:
{inti_surat}

PENTING:
- Fokus pada TUJUAN UTAMA surat, bukan kata-kata yang kebetulan muncul.
- Contoh: jika surat tentang "sertifikasi tanah untuk pembangunan perpustakaan", tujuan utamanya adalah SERTIFIKASI TANAH, maka pilih kode terkait PERTANAHAN, bukan PERPUSTAKAAN.
- Pilih kode yang PALING SPESIFIK menggambarkan inti permohonan.

PILIH KODE PRIMER (level 1) dari daftar:
{df_codes}

PILIH KODE SEKUNDER (level 2) yang paling spesifik dari pilihan di atas.
Jika Anda ragu, pilih yang paling mendekati.

FORMAT JAWABAN (JSON saja, tanpa teks lain):
{{"primer": "XXX", "sekunder": "XXX.XX", "alasan": "..."}}"""

    response = safe_generate(model, prompt)
    if not response:
        return {"primer": "500", "sekunder": "500.17", "alasan": "Gagal memanggil Gemini, fallback ke pertanahan"}

    text = response.text.strip()
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {"primer": "500", "sekunder": "500.17", "alasan": "Gagal parse, fallback ke pertanahan"}
